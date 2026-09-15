from __future__ import annotations

import numpy as np
import pytest
from pbmrs_core.commodities import PriceSeries
from pbmrs_core.fx import usd_per_currency
from pbmrs_core.fx_analysis import (
    detect_short_covering,
    minimum_distance,
    pre_event_score,
    solve_scale,
)


def test_currency_orientation_and_source():
    dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    jpy = PriceSeries("DEXJPUS", "", "", "", dates, np.array([110.0, 100.0, 105.0]))
    oriented = usd_per_currency(jpy)
    np.testing.assert_allclose(oriented.log_returns, -jpy.log_returns)
    assert oriented.log_returns[0] > 0
    assert "Federal Reserve Board" in oriented.source
    eur = PriceSeries("DEXUSEU", "", "", "", dates, np.array([1.1, 1.2, 1.15]))
    np.testing.assert_array_equal(usd_per_currency(eur).close, eur.close)


def test_scale_solve_targets_total_sd_not_innovation_sd():
    result = solve_scale(0.006, lambda s: np.sqrt(s * s + 0.003**2))
    assert result["relative_error"] <= 0.005
    assert result["sigma_eps"] < 0.006
    with pytest.raises(ValueError, match="outside"):
        solve_scale(0.002, lambda s: np.sqrt(s * s + 0.003**2))


def test_distance_not_largest_pvalue_drives_estimator(monkeypatch):
    from pbmrs_core.calibration import AdequacyRow

    def fake(profile, bank, **kwargs):
        j = kwargs["J"]
        return AdequacyRow(
            j,
            1.2 * j,
            1 if j == 0.4 else 5,
            0.2 if j == 0.4 else 0.9,
            0.1,
            1.0,
            "non_rejected",
            False,
            "D_sum",
            5,
            9,
            10,
            "test",
        )

    monkeypatch.setattr("pbmrs_core.fx_analysis.evaluate_profile_bank", fake)
    result = minimum_distance(
        np.ones(8),
        [None, None],
        [0.4, 0.8],
        [True, True],
        beta=1.2,
        n_cal=5,
        alpha=0.1,
        valid_n=10,
    )
    assert result["J_hat"] == 0.4
    assert result["confidence_set"] == [0.4, 0.8]


def test_empty_inversion_set_is_preserved():
    rng = np.random.default_rng(9)
    banks = rng.normal(0, 0.01, (2, 100, 8))
    result = minimum_distance(
        np.ones(8),
        banks,
        [0.4, 0.8],
        [True, False],
        beta=1.2,
        n_cal=50,
        alpha=0.1,
        valid_n=500,
    )
    assert result["J_hat"] == 0.4
    assert result["empty_set"] and result["confidence_set"] == []


def test_pre_event_score_never_reads_event_or_future_estimates():
    rolling = [
        {"end_position": i, "J_hat": j} for i, j in [(10, 0.4), (20, 0.5), (30, 0.9)]
    ]
    assert pre_event_score(rolling, 30, 5) == 0.5
    assert pre_event_score(rolling, 11, 5) is None


def test_short_covering_requires_net_longs_rise_and_appreciation():
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-02-10"))
    returns = np.resize(np.array([0.001, -0.001]), len(dates))
    returns[21:26] = 0.01
    rules = {
        "volatility_lookback": 10,
        "max_report_gap_days": 10,
        "prior_net_short_min": 0.05,
        "net_short_fall_min": 0.05,
        "appreciation_sd_min": 1.5,
        "minimum_spacing_sessions": 10,
    }
    cot_dates = dates[[20, 25]]
    events = detect_short_covering(returns, dates, cot_dates, [-0.2, -0.1], rules)
    assert len(events) == 1 and events[0]["start_position"] == 21
    assert detect_short_covering(returns, dates, cot_dates, [-0.1, -0.2], rules) == []
    assert detect_short_covering(-returns, dates, cot_dates, [-0.2, -0.1], rules) == []


def test_fx_adapter_has_one_shared_acf_implementation():
    from pbmrs_core import acf_r2_profile, fx_analysis

    assert fx_analysis.acf_r2_profile is acf_r2_profile


def test_event_permutation_deterministic_and_reports_unmatched():
    from pbmrs_core.fx_analysis import event_contrast

    dates = np.arange(np.datetime64("2019-01-01"), np.datetime64("2022-01-01"))
    returns = np.resize(np.array([0.01, -0.01]), len(dates))
    rolling = [
        {"end_position": i, "J_hat": 0.4 + 0.01 * (i % 7)}
        for i in range(20, len(dates), 5)
    ]
    events = [{"start_position": i, "date": str(dates[i])} for i in (200, 500, 900)]
    design = {
        "primary": {"K": 5, "alpha": 0.1},
        "split": "2018-01-01",
        "step": 5,
        "event": {
            "volatility_lookback": 10,
            "minimum_spacing_sessions": 20,
            "controls_per_event": 5,
            "max_log_vol_distance": 0.25,
        },
        "n_permutation": 99,
        "n_bootstrap": 100,
    }
    first = event_contrast(
        returns, dates, rolling, events, design, segment="validation", seed=9
    )
    second = event_contrast(
        returns, dates, rolling, events, design, segment="validation", seed=9
    )
    assert first == second
    assert first["n_matched"] == 3 and first["n_year_clusters"] == 3
    expected = (
        1 + np.count_nonzero(np.array(first["permutation"]) >= first["effect"])
    ) / 100
    assert first["p_value"] == expected
    sparse = event_contrast(
        returns, dates, rolling, events[:1], design, segment="validation", seed=9
    )
    assert sparse["p_value"] is None and sparse["status"] == "insufficient_events"
