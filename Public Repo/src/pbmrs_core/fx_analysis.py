"""Minimum-distance FX inference and predeclared retrospective event tests."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .calibration import acf_r2_profile, evaluate_profile_bank, wilson_interval


def solve_scale(target, measure, *, lo=0.00001, hi=0.015, rtol=0.005, maxiter=24):
    """Bracketed solve of total simulated SD; measure must use fixed seeds."""
    if not np.isfinite(target) or target <= 0 or not 0 < lo < hi:
        raise ValueError("invalid volatility target or bracket")
    trace = []

    def evaluate(sigma):
        sd = float(measure(sigma))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError("invalid simulated volatility")
        trace.append({"sigma_eps": sigma, "total_sd": sd})
        return sd

    if not evaluate(lo) < target < evaluate(hi):
        raise ValueError("total volatility target is outside the simulation bracket")
    for _ in range(maxiter):
        middle = (lo + hi) / 2
        sd = evaluate(middle)
        if abs(sd / target - 1) <= rtol:
            return {
                "sigma_eps": middle,
                "target_sd": target,
                "matched_sd": sd,
                "relative_error": abs(sd / target - 1),
                "trace": trace,
            }
        if sd < target:
            lo = middle
        else:
            hi = middle
    raise RuntimeError("volatility solver failed to converge")


def minimum_distance(profile, banks, grid, eligible, *, beta, n_cal, alpha, valid_n):
    """Point estimate minimizes D_sum; inversion retains the exact discrete set."""
    grid, eligible = np.asarray(grid), np.asarray(eligible, dtype=bool)
    if grid.ndim != 1 or eligible.shape != grid.shape or not eligible.any():
        raise ValueError("inference requires a nonempty eligible grid")
    if len(banks) != len(grid) or np.any(np.diff(grid) <= 0):
        raise ValueError("banks must align with a strictly increasing grid")
    rows = [
        asdict(
            evaluate_profile_bank(
                profile,
                bank,
                J=J,
                beta=beta,
                n_cal=n_cal,
                alpha=alpha,
                valid_n=valid_n,
                variant="FX_full",
            )
        )
        for J, bank in zip(grid, banks)
    ]
    distance = np.array([r["distance"] for r in rows])
    p = np.array([r["p_value"] for r in rows])
    jhat = float(grid[np.argmin(np.where(eligible, distance, np.inf))])
    accepted = eligible & (p >= alpha)
    return {
        "J_hat": jhat,
        "confidence_set": grid[accepted].tolist(),
        "empty_set": not accepted.any(),
        "rows": rows,
    }


def rolling_inference(
    returns, dates, banks, grid, eligible, *, window, step, n_cal, alpha, beta, nlags=8
):
    returns = np.asarray(returns)
    if len(returns) != len(dates) or window <= nlags or step < 1:
        raise ValueError("invalid rolling dimensions")
    output = []
    for end in range(window - 1, len(returns), step):
        sample = returns[end - window + 1 : end + 1]
        result = minimum_distance(
            acf_r2_profile(sample, nlags=nlags),
            banks,
            grid,
            eligible,
            beta=beta,
            n_cal=n_cal,
            alpha=alpha,
            valid_n=window,
        )
        output.append(
            {
                "end_position": end,
                "date": str(dates[end]),
                "realized_sd": float(sample.std(ddof=1)),
                **result,
            }
        )
    return output


def detect_short_covering(returns, dates, cot_dates, net_over_oi, rules):
    """Label short-covering plus appreciation, using only pre-interval SD.

    Long-minus-short increases when net shorts unwind. Dates are report
    observation dates, not release dates; these are retrospective labels.
    """
    returns, dates = np.asarray(returns), np.asarray(dates, dtype="datetime64[D]")
    cot_dates = np.asarray(cot_dates, dtype="datetime64[D]")
    net = np.asarray(net_over_oi)
    if len(returns) != len(dates) or len(cot_dates) != len(net):
        raise ValueError("unaligned event inputs")
    if np.any(np.diff(dates).astype(int) <= 0) or np.any(
        np.diff(cot_dates).astype(int) <= 0
    ):
        raise ValueError("event dates must increase")
    events = []
    previous_end = -(10**9)
    for i in range(1, len(cot_dates)):
        a = int(np.searchsorted(dates, cot_dates[i - 1], side="right"))
        b = int(np.searchsorted(dates, cot_dates[i], side="right"))
        lookback = rules["volatility_lookback"]
        if a < lookback or b <= a or b > len(returns) or cot_dates[i] > dates[-1]:
            continue
        if (
            int((cot_dates[i] - cot_dates[i - 1]).astype(int))
            > rules["max_report_gap_days"]
        ):
            continue
        sd = float(returns[a - lookback : a].std(ddof=1))
        appreciation = float(returns[a:b].sum())
        short_before = -float(net[i - 1])
        short_fall = float(net[i] - net[i - 1])
        z = appreciation / (sd * np.sqrt(b - a)) if sd > 0 else 0.0
        if (
            short_before >= rules["prior_net_short_min"]
            and short_fall >= rules["net_short_fall_min"]
            and z >= rules["appreciation_sd_min"]
            and a - previous_end >= rules["minimum_spacing_sessions"]
        ):
            events.append(
                {
                    "start_position": a,
                    "end_position": b - 1,
                    "date": str(cot_dates[i]),
                    "interval_start": str(cot_dates[i - 1]),
                    "net_short_before": short_before,
                    "net_short_fall": short_fall,
                    "appreciation": appreciation,
                    "appreciation_z": z,
                }
            )
            previous_end = b - 1
    return events


def pre_event_score(rolling, start, K):
    """Average the last-known monthly estimate over K strictly prior sessions."""
    endpoints = np.array([r["end_position"] for r in rolling])
    ix = np.searchsorted(endpoints, np.arange(start - K, start), side="right") - 1
    if len(ix) != K or np.any(ix < 0):
        return None
    return float(np.mean([rolling[i]["J_hat"] for i in ix]))


def event_contrast(returns, dates, rolling, events, design, *, segment, seed):
    """Matched event-label permutation with year-cluster bootstrap intervals.

    Matching is a retrospective association design, not randomized treatment.
    The permutation null needs exchangeability within matched sets; overlapping
    windows and reused controls weaken that approximation and are disclosed.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    K, rules = design["primary"]["K"], design["event"]
    split = np.datetime64(design["split"])
    lookback, spacing = rules["volatility_lookback"], rules["minimum_spacing_sessions"]
    selected = [
        e
        for e in events
        if (dates[e["start_position"]] >= split) == (segment == "validation")
    ]
    candidates = range(
        max(lookback, rolling[0]["end_position"] + K + 1), len(dates), design["step"]
    )
    matched, excluded = [], []
    for event in selected:
        a = event["start_position"]
        score = pre_event_score(rolling, a, K)
        sd = np.std(returns[a - lookback : a], ddof=1)
        year = str(dates[a])[:4]
        pool = []
        for c in candidates:
            if str(dates[c])[:4] != year:
                continue
            if any(abs(c - e["start_position"]) <= spacing for e in events):
                continue
            control_score = pre_event_score(rolling, c, K)
            if control_score is None:
                continue
            control_sd = np.std(returns[c - lookback : c], ddof=1)
            delta = abs(np.log(control_sd / sd))
            if delta <= rules["max_log_vol_distance"]:
                pool.append((delta, c, control_score))
        pool.sort()
        controls = pool[: rules["controls_per_event"]]
        if score is None or len(controls) < rules["controls_per_event"]:
            excluded.append(
                {**event, "reason": "insufficient prehistory or matched controls"}
            )
            continue
        matched.append(
            {
                **event,
                "year": year,
                "score": score,
                "control_positions": [c[1] for c in controls],
                "control_scores": [c[2] for c in controls],
                "difference": score - float(np.mean([c[2] for c in controls])),
            }
        )
    result = {
        "segment": segment,
        "n_labelled": len(selected),
        "n_matched": len(matched),
        "matched": matched,
        "excluded": excluded,
        "effect": None,
        "lo": None,
        "hi": None,
        "p_value": None,
        "mc_lo": None,
        "mc_hi": None,
        "status": "insufficient_events",
        "permutation": [],
    }
    if len(matched) < 3:
        return result
    matrix = np.array([[m["score"], *m["control_scores"]] for m in matched])
    differences = np.array([m["difference"] for m in matched])
    effect = float(differences.mean())
    rng = np.random.default_rng(seed)
    choices = rng.integers(matrix.shape[1], size=(design["n_permutation"], len(matrix)))
    chosen = matrix[np.arange(len(matrix))[None, :], choices]
    null = (
        chosen - (matrix.sum(axis=1)[None, :] - chosen) / (matrix.shape[1] - 1)
    ).mean(axis=1)
    exceed = int(np.count_nonzero(null >= effect))
    p = (1 + exceed) / (1 + len(null))
    years = sorted({m["year"] for m in matched})
    clusters = [differences[[m["year"] == y for m in matched]] for y in years]
    boot = [
        float(
            np.concatenate(
                [clusters[i] for i in rng.integers(len(years), size=len(years))]
            ).mean()
        )
        for _ in range(design["n_bootstrap"])
    ]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mc_lo, mc_hi = wilson_interval(exceed, len(null))
    result.update(
        effect=effect,
        lo=float(lo),
        hi=float(hi),
        p_value=p,
        mc_lo=mc_lo,
        mc_hi=mc_hi,
        n_year_clusters=len(years),
        status="positive" if p < design["primary"]["alpha"] else "null",
        permutation=null.tolist(),
    )
    return result


def excess_kurtosis(values):
    centered = np.asarray(values) - np.mean(values)
    return float(np.mean(centered**4) / np.mean(centered**2) ** 2 - 3)


def load_study(root):
    """Validate offline study data, source fingerprints, design and NPZ bytes."""
    import hashlib
    import json
    from pathlib import Path

    from .calibration import load_npz_cache, sim_cache_key
    from .fx import validate_raw_tree

    root = Path(root)
    cache = root / "notebooks/fx_cache"
    frozen = json.loads((cache / "preregistration.json").read_text())
    manifest = json.loads((cache / "analysis_manifest.json").read_text())
    simulation = json.loads((cache / "simulation_manifest.json").read_text())
    design_hash = hashlib.sha256(
        (root / "configs/fx_study.json").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    if (
        not design_hash
        == frozen["design_sha256"]
        == manifest["design_sha256"]
        == simulation["design_sha256"]
    ):
        raise ValueError("FX design hash mismatch")
    if frozen["design"] != json.loads((root / "configs/fx_study.json").read_text()):
        raise ValueError("FX frozen design differs from configuration")
    for rel, digest in manifest["sources"].items():
        if (
            hashlib.sha256(
                (root / rel).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            != digest
        ):
            raise ValueError(f"FX source changed: {rel}; rebuild results")
    expected = sim_cache_key(
        **{
            k: simulation[k]
            for k in ("code_hash", "design_sha256", "config", "raw_hashes")
        }
    )
    if (
        expected != simulation["root_key"]
        or expected != manifest["simulation_root_key"]
    ):
        raise ValueError("FX simulation manifest key mismatch")
    raw = validate_raw_tree(cache / "raw")
    if {k: v["sha256"] for k, v in raw.items()} != simulation["raw_hashes"]:
        raise ValueError("FX raw input hashes changed")
    for name, key in simulation["cache_keys"].items():
        path = cache / name
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != simulation["cache_sha256"][name]
        ):
            raise ValueError(f"FX simulation payload hash mismatch: {name}")
        if load_npz_cache(path, key) is None:
            raise ValueError(f"FX simulation cache key mismatch: {name}")
    body = (cache / "results.json").read_bytes()
    if hashlib.sha256(body).hexdigest() != manifest["results_sha256"]:
        raise ValueError("FX result payload hash mismatch")
    return frozen, simulation, manifest, json.loads(body)
