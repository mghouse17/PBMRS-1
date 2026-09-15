"""Run the separately registered continuous FX stress design."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pbmrs_core.calibration import load_npz_cache
from pbmrs_core.fx import fetch_fx, validate_raw_tree
from pbmrs_core.fx_analysis import rolling_inference
from pbmrs_core.fx_continuous import (
    OUTCOMES,
    block_bootstrap_regression,
    exploratory_event_contrast,
    forward_stress_panel,
    volatility_stratified_quintiles,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks/fx_cache"
CONTINUOUS_RAW = ROOT / "notebooks/fx_continuous_cache/raw"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    started = datetime.now(timezone.utc).isoformat()
    frozen = json.loads((CACHE / "continuous_preregistration.json").read_text())
    d = frozen["design"]
    if frozen["registered_at_utc"] >= started:
        raise RuntimeError("continuous computation must begin after registration")
    event_frozen = json.loads((CACHE / "preregistration.json").read_text())
    event_results = json.loads((CACHE / "results.json").read_text())
    simulation = json.loads((CACHE / "simulation_manifest.json").read_text())

    def bank(pair, grid):
        payloads = []
        for J in grid:
            name = f"{pair}_bank_J{J:.8f}.npz"
            path = CACHE / name
            if sha256(path) != simulation["cache_sha256"][name]:
                raise RuntimeError(f"simulation bank bytes changed: {name}")
            payload = load_npz_cache(path, simulation["cache_keys"][name])
            if payload is None:
                raise RuntimeError(f"simulation bank key changed: {name}")
            payloads.append(payload["T500"])
        return np.stack(payloads)

    results = {
        "design_sha256": frozen["design_sha256"],
        "registered_event_design_sha256": event_frozen["design_sha256"],
        "estimand_change_reason": (
            "The registered event study did not execute because it had too few matched "
            "episodes. This separately registered continuous estimand was chosen because "
            "the old design had no sample, not because its results were unfavourable."
        ),
        "series": {},
        "regressions": [],
        "quintiles": {},
    }
    panels = {}
    for pair in d["sample"]["pairs"]:
        series = fetch_fx(
            pair,
            cache_dir=CONTINUOUS_RAW,
            start=d["sample"]["start"],
            end=d["sample"]["end"],
        )
        grid = simulation["eligible"][pair]
        rolling = rolling_inference(
            series.log_returns,
            series.return_dates,
            bank(pair, grid),
            grid,
            [True] * len(grid),
            window=d["estimator"]["window"],
            step=d["estimator"]["step"],
            n_cal=event_frozen["design"]["n_cal"],
            alpha=event_frozen["design"]["primary"]["alpha"],
            beta=simulation["config"]["beta"],
            nlags=d["estimator"]["nlags"],
        )
        panel = forward_stress_panel(
            series.log_returns,
            series.return_dates,
            rolling,
            horizon=d["forward_horizon"],
        )
        panels[pair] = panel
        results["series"][pair] = {
            "first_return_date": str(series.return_dates[0]),
            "last_return_date": str(series.return_dates[-1]),
            "n_returns": len(series.log_returns),
            "n_estimates": len(panel),
            "daily_sd": float(series.log_returns.std(ddof=1)),
            "panel": panel,
        }
        for outcome in OUTCOMES:
            results["regressions"].append(
                {
                    "series": pair,
                    **block_bootstrap_regression(
                        panel,
                        outcome,
                        block_length=d["bootstrap"]["block_length_steps"],
                        n_resamples=d["bootstrap"]["n_resamples"],
                        seed=d["bootstrap"]["seeds"][pair],
                        interval=d["regression"]["interval"],
                    ),
                }
            )
        results["quintiles"][pair] = volatility_stratified_quintiles(
            panel, d["primary"]["outcome"]
        )
        print(pair, len(series.log_returns), "returns,", len(panel), "estimates")

    jpy_series = results["series"]["JPY"]
    gaussian_returns = np.random.default_rng(
        d["negative_control"]["series_seed"]
    ).normal(scale=jpy_series["daily_sd"], size=jpy_series["n_returns"])
    jpy_dates = fetch_fx(
        "JPY",
        cache_dir=CONTINUOUS_RAW,
        start=d["sample"]["start"],
        end=d["sample"]["end"],
    ).return_dates
    grid = simulation["eligible"]["JPY"]
    gaussian_rolling = rolling_inference(
        gaussian_returns,
        jpy_dates,
        bank("JPY", grid),
        grid,
        [True] * len(grid),
        window=d["estimator"]["window"],
        step=d["estimator"]["step"],
        n_cal=event_frozen["design"]["n_cal"],
        alpha=event_frozen["design"]["primary"]["alpha"],
        beta=simulation["config"]["beta"],
        nlags=d["estimator"]["nlags"],
    )
    gaussian_panel = forward_stress_panel(
        gaussian_returns,
        jpy_dates,
        gaussian_rolling,
        horizon=d["forward_horizon"],
    )
    results["series"]["Gaussian"] = {
        "first_return_date": str(jpy_dates[0]),
        "last_return_date": str(jpy_dates[-1]),
        "n_returns": len(gaussian_returns),
        "n_estimates": len(gaussian_panel),
        "daily_sd": float(gaussian_returns.std(ddof=1)),
        "matched_pair": "JPY",
        "panel": gaussian_panel,
    }
    for outcome in OUTCOMES:
        results["regressions"].append(
            {
                "series": "Gaussian",
                **block_bootstrap_regression(
                    gaussian_panel,
                    outcome,
                    block_length=d["bootstrap"]["block_length_steps"],
                    n_resamples=d["bootstrap"]["n_resamples"],
                    seed=d["bootstrap"]["seeds"]["Gaussian"],
                    interval=d["regression"]["interval"],
                ),
            }
        )
    results["quintiles"]["Gaussian"] = volatility_stratified_quintiles(
        gaussian_panel, d["primary"]["outcome"]
    )

    old_design = event_frozen["design"]
    jpy_old = fetch_fx(
        "JPY",
        cache_dir=CACHE / "raw",
        start=old_design["start"],
        end=old_design["end"],
    )
    results["post_hoc_exploratory_event"] = exploratory_event_contrast(
        jpy_old.log_returns,
        jpy_old.return_dates,
        event_results["pairs"]["JPY"]["rolling"]["500"],
        event_results["pairs"]["JPY"]["events"],
        split=old_design["split"],
        K=old_design["primary"]["K"],
        step=old_design["step"],
        volatility_lookback=old_design["event"]["volatility_lookback"],
        spacing=old_design["event"]["minimum_spacing_sessions"],
    )

    body = json.dumps(results, indent=2, allow_nan=False).encode()
    result_path = CACHE / "continuous_results.json"
    result_path.write_bytes(body)
    manifest = {
        "design_sha256": frozen["design_sha256"],
        "registered_at_utc": frozen["registered_at_utc"],
        "started_at_utc": started,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_sha256": hashlib.sha256(body).hexdigest(),
        "event_design_sha256": event_frozen["design_sha256"],
        "event_results_sha256": sha256(CACHE / "results.json"),
        "simulation_root_key": simulation["root_key"],
        "raw_hashes": {
            name: entry["sha256"]
            for name, entry in validate_raw_tree(CONTINUOUS_RAW).items()
        },
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for path in (
                ROOT / "configs/fx_continuous_test.json",
                ROOT / "src/pbmrs_core/fx_continuous.py",
                ROOT / "notebooks/build_fx_continuous_results.py",
            )
        },
    }
    (CACHE / "continuous_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Continuous results:", manifest["results_sha256"])


if __name__ == "__main__":
    build()
