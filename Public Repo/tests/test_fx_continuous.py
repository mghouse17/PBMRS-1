from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pbmrs_core.fx_analysis import load_study
from pbmrs_core.fx_continuous import (
    block_bootstrap_regression,
    forward_stress_panel,
    load_continuous_study,
    volatility_stratified_quintiles,
)

ROOT = Path(__file__).resolve().parents[1]


def test_forward_panel_uses_only_the_next_exact_horizon():
    returns = np.array([0.01, -0.02, 0.03, -0.01, -0.04, 0.02, 0.05, -0.01, 0.02])
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-10"))
    rolling = [
        {"end_position": 3, "J_hat": 0.4, "realized_sd": 0.02},
        {"end_position": 5, "J_hat": 0.7, "realized_sd": 0.03},
    ]
    panel = forward_stress_panel(returns, dates, rolling, horizon=2)
    assert [row["forward_end_date"] for row in panel] == ["2020-01-06", "2020-01-08"]
    assert panel[0]["forward_realized_volatility"] == pytest.approx(
        np.std([-0.04, 0.02], ddof=1)
    )
    assert panel[0]["forward_absolute_terminal_return"] == pytest.approx(0.02)
    prices = np.exp([0.0, -0.04, -0.02])
    expected_mdd = np.max(1 - prices / np.maximum.accumulate(prices))
    assert panel[0]["forward_max_drawdown"] == pytest.approx(expected_mdd)


def test_block_bootstrap_regression_is_deterministic_and_controls_volatility():
    n = 120
    j = np.resize(np.array([0.3, 0.45, 0.6, 0.78, 0.85]), n)
    vol = 0.004 + np.arange(n) * 0.00001
    residual = np.sin(np.arange(n)) * 1e-5
    panel = [
        {
            "J_hat": j[i],
            "current_realized_volatility": vol[i],
            "forward_realized_volatility": 0.01 + 0.2 * j[i] + 0.3 * np.log(vol[i]) + residual[i],
        }
        for i in range(n)
    ]
    first = block_bootstrap_regression(
        panel, "forward_realized_volatility", block_length=12, n_resamples=199, seed=42
    )
    second = block_bootstrap_regression(
        panel, "forward_realized_volatility", block_length=12, n_resamples=199, seed=42
    )
    assert first == second
    assert first["b_J_hat"] == pytest.approx(0.2, abs=1e-4)
    assert first["b_lo"] <= first["b_J_hat"] <= first["b_hi"]


def test_volatility_stratified_table_preserves_every_observation():
    panel = [
        {
            "J_hat": float(i % 7),
            "current_realized_volatility": float(i + 1),
            "forward_realized_volatility": float(i * i),
        }
        for i in range(31)
    ]
    rows = volatility_stratified_quintiles(panel, "forward_realized_volatility")
    assert len(rows) == 15
    assert sum(row["n"] for row in rows) == len(panel)


def test_continuous_registration_results_and_event_dependency_validate():
    event_frozen, _, event_manifest, _ = load_study(ROOT)
    frozen, manifest, results = load_continuous_study(ROOT)
    assert frozen["registered_at_utc"] < manifest["started_at_utc"] < manifest["computed_at_utc"]
    assert manifest["event_design_sha256"] == event_frozen["design_sha256"]
    assert manifest["event_results_sha256"] == event_manifest["results_sha256"]
    assert hashlib.sha256((ROOT / "notebooks/fx_cache/results.json").read_bytes()).hexdigest() == event_manifest["results_sha256"]
    assert {row["series"] for row in results["regressions"]} == {"JPY", "EUR", "Gaussian"}
    assert len(results["regressions"]) == 9
    for name in ("JPY", "EUR", "Gaussian"):
        info = results["series"][name]
        assert info["n_returns"] == 5939 and info["n_estimates"] == 259
        positions = np.array([row["end_position"] for row in info["panel"]])
        assert np.all(np.diff(positions) == 21)
    primary = next(
        row for row in results["regressions"]
        if row["series"] == "JPY" and row["outcome"] == frozen["design"]["primary"]["outcome"]
    )
    assert primary["b_lo"] <= 0 <= primary["b_hi"]
    assert primary["block_length_steps"] == 24 and primary["n_resamples"] == 4999
    exploratory = results["post_hoc_exploratory_event"]
    assert exploratory["n_labelled"] == exploratory["n_matched"] == 5
    assert any(row["date"] == "2026-08-04" for row in exploratory["matched"])
    assert exploratory["lo"] <= exploratory["effect_descriptive"] <= exploratory["hi"]


def test_continuous_manifest_has_only_isolated_fred_inputs():
    _, manifest, results = load_continuous_study(ROOT)
    assert set(manifest["raw_hashes"]) == {
        "fred_DEXJPUS_2003-01-01_2026-09-15.csv",
        "fred_DEXUSEU_2003-01-01_2026-09-15.csv",
    }
    assert results["design_sha256"] == manifest["design_sha256"]
