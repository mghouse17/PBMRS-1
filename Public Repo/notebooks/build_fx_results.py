"""Produce every declared FX specification from validated simulation caches."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pbmrs_core.calibration import (
    acf_r2_profile,
    evaluate_profile_bank,
    load_npz_cache,
    sim_cache_key,
    wilson_interval,
)
from pbmrs_core.commodities import bootstrap_statistic
from pbmrs_core.fx import (
    discover_definitions,
    fetch_fx,
    fetch_fx_positioning,
    validate_raw_tree,
)
from pbmrs_core.fx_analysis import (
    detect_short_covering,
    event_contrast,
    excess_kurtosis,
    rolling_inference,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks/fx_cache"


def code_files():
    paths = list((ROOT / "src/pbmrs_core").glob("*.py"))
    paths += [ROOT / "configs/base.yaml", ROOT / "configs/fx_study.json"]
    paths += [
        ROOT / "notebooks" / name
        for name in (
            "build_fx_data.py",
            "build_fx_simulations.py",
            "build_fx_results.py",
            "build_fx_notebook.py",
        )
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in sorted(paths)
    }


def build():
    frozen = json.loads((CACHE / "preregistration.json").read_text())
    d = frozen["design"]
    sim = json.loads((CACHE / "simulation_manifest.json").read_text())
    if "completed_at_utc" not in sim:
        raise RuntimeError("production simulations are not yet complete")
    raw = validate_raw_tree(CACHE / "raw")
    if {k: v["sha256"] for k, v in raw.items()} != sim["raw_hashes"]:
        raise RuntimeError("raw data changed since simulation calibration")

    def cache(name):
        path = CACHE / f"{name}.npz"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != sim["cache_sha256"][path.name]
        ):
            raise RuntimeError(f"simulation payload changed: {path}")
        loaded = load_npz_cache(path, sim["cache_keys"][path.name])
        if loaded is None:
            raise RuntimeError(f"invalid simulation key: {path}")
        return loaded

    def evaluate(profile, bank, J, T):
        return evaluate_profile_bank(
            profile,
            bank,
            J=J,
            beta=sim["config"]["beta"],
            n_cal=d["n_cal"],
            alpha=d["primary"]["alpha"],
            valid_n=T,
            variant="FX",
        )

    results = {
        "pairs": {},
        "specifications": [],
        "controls": [],
        "power": [],
        "horizons": [],
        "gaussian": [],
        "walk_forward": [],
    }
    definitions = discover_definitions(CACHE / "raw")
    series_by_pair, rolling_by_pair, events_by_pair = {}, {}, {}
    for pair in ("JPY", "EUR"):
        series = fetch_fx(pair, cache_dir=CACHE / "raw", start=d["start"], end=d["end"])
        series_by_pair[pair] = series
        r, dates = series.log_returns, series.return_dates

        def fingerprint(x):
            return np.r_[
                acf_r2_profile(x, nlags=d["nlags"]), x.std(ddof=1), excess_kurtosis(x)
            ]

        boot = bootstrap_statistic(
            r,
            fingerprint,
            n_boot=d["n_bootstrap"],
            seed=d["seeds"]["bootstrap"],
            mean_block=21,
        )
        grid = sim["eligible"][pair]
        info = {
            "n_returns": len(r),
            "first": str(dates[0]),
            "last": str(dates[-1]),
            "fingerprint": {
                "point": boot.point.tolist(),
                "lo": boot.lo.tolist(),
                "hi": boot.hi.tolist(),
            },
            "positioning": {},
            "rolling": {},
            "kurtosis": [],
        }
        cots = {}
        for definition in definitions:
            cot, contract = fetch_fx_positioning(
                definition,
                pair,
                cache_dir=CACHE / "raw",
                start=d["start"],
                end=d["end"],
            )
            cots[definition.key] = cot
            info["positioning"][definition.key] = {
                "category": definition.category,
                "contract": contract,
                "reports": len(cot.dates),
                "last_date": str(cot.dates[-1]),
                "last_net_oi": float(cot.net_over_oi[-1]),
                "mean_net_oi": float(cot.net_over_oi.mean()),
            }
        primary_cot = cots[d["positioning_primary"]]
        events = detect_short_covering(
            r, dates, primary_cot.dates, primary_cot.net_over_oi, d["event"]
        )
        info["events"] = events
        events_by_pair[pair] = events
        rolling_by_pair[pair] = {}
        for T in d["windows"]:
            banks = np.stack([cache(f"{pair}_bank_J{J:.8f}")[f"T{T}"] for J in grid])
            rolling = rolling_inference(
                r,
                dates,
                banks,
                grid,
                [True] * len(grid),
                window=T,
                step=d["step"],
                n_cal=d["n_cal"],
                alpha=d["primary"]["alpha"],
                beta=sim["config"]["beta"],
            )
            info["rolling"][str(T)] = rolling
            rolling_by_pair[pair][T] = rolling
            # Gaussian control is evaluated beside the main rolling inference.
            gaussian = [
                acf_r2_profile(np.random.default_rng(s).normal(size=T))
                for s in range(d["seeds"]["gaussian"][0], d["seeds"]["gaussian"][1] + 1)
            ]
            gaussian_accept = np.array(
                [
                    [
                        evaluate(a, bank, J, T).p_value >= d["primary"]["alpha"]
                        for J, bank in zip(grid, banks)
                    ]
                    for a in gaussian
                ]
            )
            results["gaussian"].append(
                {
                    "pair": pair,
                    "window": T,
                    "n": len(gaussian),
                    "any_non_rejection_rate": float(gaussian_accept.any(axis=1).mean()),
                    "per_J_non_rejection": gaussian_accept.mean(axis=0).tolist(),
                    "J": grid,
                }
            )
            for J in (0.4, sim["comparison_J"][pair]):
                bank = cache(f"{pair}_bank_J{J:.8f}")[f"T{T}"]
                pseudo = cache(f"{pair}_pseudo_J{J:.8f}")[f"T{T}"]
                count = sum(
                    evaluate(a, bank, J, T).p_value < d["primary"]["alpha"]
                    for a in pseudo
                )
                lo, hi = wilson_interval(count, len(pseudo))
                results["controls"].append(
                    {
                        "pair": pair,
                        "window": T,
                        "J": J,
                        "n": len(pseudo),
                        "rejections": count,
                        "rate": count / len(pseudo),
                        "lo": lo,
                        "hi": hi,
                        "covers_nominal": lo <= d["primary"]["alpha"] <= hi,
                    }
                )
            for year in sorted(
                {row["date"][:4] for row in rolling if row["date"] >= d["split"]}
            ):
                subset = [row for row in rolling if row["date"].startswith(year)]
                results["walk_forward"].append(
                    {
                        "pair": pair,
                        "window": T,
                        "year": year,
                        "n_windows": len(subset),
                        "mean_J": float(np.mean([x["J_hat"] for x in subset])),
                        "empty_set_fraction": float(
                            np.mean([x["empty_set"] for x in subset])
                        ),
                        "training_cutoff": d["split"],
                        "nuisance_policy": "frozen initial training",
                    }
                )
        for T in d["power_lengths"]:
            bank = cache(f"{pair}_bank_J{0.4:.8f}")[f"T{T}"]
            truth = sim["comparison_J"][pair]
            pseudo = cache(f"{pair}_pseudo_J{truth:.8f}")[f"T{T}"]
            count = sum(
                evaluate(a, bank, 0.4, T).p_value < d["primary"]["alpha"]
                for a in pseudo
            )
            lo, hi = wilson_interval(count, len(pseudo))
            results["power"].append(
                {
                    "pair": pair,
                    "T": T,
                    "null_J": 0.4,
                    "truth_J": truth,
                    "n": len(pseudo),
                    "rejections": count,
                    "power": count / len(pseudo),
                    "lo": lo,
                    "hi": hi,
                }
            )
        for J in (0.4, sim["comparison_J"][pair]):
            moments = cache(f"{pair}_pseudo_J{J:.8f}")["moments"]
            info["kurtosis"].append(
                {
                    "J": J,
                    "mean": float(moments[:, 1].mean()),
                    "path_p025": float(np.percentile(moments[:, 1], 2.5)),
                    "path_p975": float(np.percentile(moments[:, 1], 97.5)),
                    "empirical": float(boot.point[-1]),
                    "gap": float(boot.point[-1] - moments[:, 1].mean()),
                }
            )
            h = cache(f"{pair}_horizon_J{J:.8f}")["horizon"]
            qualifying, recovered = int(h[:, 4].sum()), int(h[:, 5].sum())
            lo, hi = (
                wilson_interval(recovered, qualifying) if qualifying else (None, None)
            )
            results["horizons"].append(
                {
                    "pair": pair,
                    "J": J,
                    "horizon": 21,
                    "n": len(h),
                    "mdd_mean": float(h[:, 0].mean()),
                    "mdd_p95": float(np.percentile(h[:, 0], 95)),
                    "terminal_mean": float(h[:, 1].mean()),
                    "ending_drawdown_mean": float(h[:, 2].mean()),
                    "qualifying_5pct": qualifying,
                    "recovered_5pct": recovered,
                    "recovery_rate": recovered / qualifying if qualifying else None,
                    "recovery_lo": lo,
                    "recovery_hi": hi,
                    "median_recovery_sessions_if_observed": float(np.nanmedian(h[:, 3]))
                    if np.isfinite(h[:, 3]).any()
                    else None,
                }
            )
        results["pairs"][pair] = info
        print(pair, "rolling, controls and diagnostics complete", flush=True)
    for pair in ("JPY", "EUR"):
        series = series_by_pair[pair]
        for label_pair in ("JPY",) if pair == "JPY" else ("JPY", "EUR"):
            # EUR on JPY dates is the direct falsification; own EUR labels are also reported.
            if label_pair == pair:
                events = events_by_pair[pair]
            else:
                events = []
                for event in events_by_pair[label_pair]:
                    start = int(
                        np.searchsorted(
                            series.return_dates,
                            np.datetime64(event["interval_start"]),
                            side="right",
                        )
                    )
                    end = (
                        int(
                            np.searchsorted(
                                series.return_dates,
                                np.datetime64(event["date"]),
                                side="right",
                            )
                        )
                        - 1
                    )
                    events.append(
                        {**event, "start_position": start, "end_position": end}
                    )
            for T in d["windows"]:
                for segment in ("training", "validation"):
                    contrast = event_contrast(
                        series.log_returns,
                        series.return_dates,
                        rolling_by_pair[pair][T],
                        events,
                        d,
                        segment=segment,
                        seed=d["seeds"]["permutation"],
                    )
                    contrast.update(
                        pair=pair,
                        window=T,
                        label_pair=label_pair,
                        primary=(
                            pair == label_pair == "JPY"
                            and T == 500
                            and segment == "validation"
                        ),
                    )
                    roll = rolling_by_pair[pair][T]
                    endpoints = np.array([r["end_position"] for r in roll])
                    for event in contrast["matched"]:
                        start = event["start_position"]
                        indices = (
                            np.searchsorted(
                                endpoints,
                                np.arange(start - d["primary"]["K"], start),
                                side="right",
                            )
                            - 1
                        )
                        event["pre_event_empty_set_fraction"] = float(
                            np.mean([roll[i]["empty_set"] for i in indices])
                        )
                    results["specifications"].append(contrast)
    sources = code_files()
    manifest = {
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "design_sha256": frozen["design_sha256"],
        "simulation_root_key": sim["root_key"],
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "pbmrs",
                "numpy",
                "pandas",
                "matplotlib",
                "nbformat",
                "nbclient",
            )
        },
    }
    manifest["result_key"] = sim_cache_key(**manifest)
    body = json.dumps(results, indent=2, allow_nan=False).encode()
    (CACHE / "results.json").write_bytes(body)
    manifest["results_sha256"] = hashlib.sha256(body).hexdigest()
    (CACHE / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    primary = next(r for r in results["specifications"] if r["primary"])
    print(
        "Primary:",
        {
            k: v
            for k, v in primary.items()
            if k not in ("matched", "excluded", "permutation")
        },
        flush=True,
    )


if __name__ == "__main__":
    build()
