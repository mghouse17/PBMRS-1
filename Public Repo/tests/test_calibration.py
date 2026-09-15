from __future__ import annotations

import numpy as np
import pytest

from pbmrs_core import SimResult
from pbmrs_core.calibration import (
    acf_r2_profile,
    compute_horizon_stats,
    evaluate_profile_bank,
    load_npz_cache,
    save_npz_cache,
    sim_cache_key,
    wilson_interval,
)


def _result_with_returns(returns: np.ndarray) -> SimResult:
    n = len(returns)
    z = np.zeros(n)
    states = np.zeros(n + 1)
    prices = np.exp(np.r_[0.0, np.cumsum(returns)])
    return SimResult(states, np.ones(n + 1), np.ones(n + 1), z, z, returns, z, prices)


def _manual_fixed_grid(values: np.ndarray, mask: np.ndarray, lag: int) -> float:
    centered = np.full(values.shape, np.nan)
    centered[mask] = values[mask] - values[mask].mean()
    squared = centered**2
    squared[mask] -= squared[mask].mean()
    pairs = mask[:-lag] & mask[lag:]
    return float(np.dot(squared[:-lag][pairs], squared[lag:][pairs]) / np.dot(squared[mask], squared[mask]))


def test_known_fixed_grid_acf_and_compressed_deletion_differ():
    values = np.array([0.2, -0.1, 0.3, 1.4, -0.2, 0.4, -0.5, 0.1])
    mask = np.ones(len(values), dtype=bool)
    mask[3] = False
    profile = acf_r2_profile(values, nlags=3, valid_mask=mask)
    expected = np.array([_manual_fixed_grid(values, mask, lag) for lag in range(1, 4)])
    np.testing.assert_allclose(profile, expected)
    compressed = acf_r2_profile(values[mask], nlags=3)
    assert not np.allclose(profile, compressed)


def test_mask_is_positional_and_applies_identically_to_simulated_paths():
    mask = np.array([True, True, False, True, True, True])
    empirical = np.array([0.1, -0.2, 99.0, 0.3, -0.1, 0.2])
    simulated = np.array([0.2, -0.4, -500.0, 0.6, -0.2, 0.4])
    np.testing.assert_allclose(
        acf_r2_profile(empirical, nlags=2, valid_mask=mask),
        acf_r2_profile(simulated, nlags=2, valid_mask=mask),
    )


@pytest.mark.parametrize("mask", [np.ones(3), np.ones(5)])
def test_acf_rejects_bad_mask_shapes(mask):
    with pytest.raises(ValueError, match="same shape"):
        acf_r2_profile(np.arange(4.0), valid_mask=mask)


def test_acf_rejects_invalid_lags_too_few_and_zero_variance():
    with pytest.raises(ValueError, match="at least three"):
        acf_r2_profile([1.0, 2.0, 3.0], valid_mask=np.array([True, False, True]))
    with pytest.raises(ValueError, match="zero or invalid"):
        acf_r2_profile([1.0, 1.0, 1.0, 1.0], nlags=2)
    with pytest.raises(ValueError, match="nlags"):
        acf_r2_profile([1.0, 2.0, 3.0, 4.0], nlags=4)


def test_plus_one_p_value_wilson_interval_and_borderline_decision():
    profiles = np.array([[0.0], [0.2], [0.1], [0.3], [0.5], [0.7]])
    row = evaluate_profile_bank(
        np.array([0.4]), profiles, J=0.4, beta=1.2, n_cal=2,
        alpha=0.5, statistic="D_sum", valid_n=144, variant="full",
    )
    center = profiles[:2].mean(axis=0)
    scale = profiles[:2].std(axis=0, ddof=1)
    empirical_distance = np.square((np.array([0.4]) - center) / scale).sum()
    null_distance = np.square((profiles[2:] - center) / scale).sum(axis=1)
    exceedances = int(np.count_nonzero(null_distance >= empirical_distance))
    assert row.p_value == pytest.approx((1 + exceedances) / 5)
    assert (row.mc_lo, row.mc_hi) == pytest.approx(wilson_interval(exceedances, 4))
    assert row.borderline == (row.mc_lo <= 0.5 <= row.mc_hi)


def test_horizon_excludes_burn_and_uses_exact_21_returns():
    burn = np.full(5, 10.0)
    evaluation = np.array([np.log(0.90), np.log(1.0 / 0.90)] + [0.0] * 19)
    trailing = np.full(3, -10.0)
    result = _result_with_returns(np.r_[burn, evaluation, trailing])
    stats = compute_horizon_stats([result], burn=5, horizon=21, recovery_drawdown=0.05)
    assert stats.horizon == 21
    assert stats.max_drawdown[0] == pytest.approx(0.10)
    assert stats.terminal_log_return[0] == pytest.approx(0.0)
    assert stats.ending_drawdown[0] == pytest.approx(0.0)
    assert stats.qualifying_drawdown[0]
    assert stats.recovered[0]
    assert stats.recovery_steps[0] == 1


def test_horizon_rejects_short_path():
    with pytest.raises(ValueError, match="shorter"):
        compute_horizon_stats([_result_with_returns(np.zeros(20))], burn=1, horizon=21)


def test_simulation_cache_key_tracks_all_declared_inputs_and_detects_corruption(tmp_path):
    base = {
        "code": "abc",
        "config": {"J": 0.4},
        "data": "hash",
        "mask": [3],
        "grid": [0.4],
        "seeds": [0, 1],
    }
    key = sim_cache_key(**base)
    for name, value in base.items():
        changed = dict(base)
        changed[name] = str(value) + "-changed"
        assert sim_cache_key(**changed) != key
    path = tmp_path / "result.npz"
    save_npz_cache(path, key, metadata={"purpose": "test"}, values=np.arange(3))
    cached = load_npz_cache(path, key)
    np.testing.assert_array_equal(cached["values"], np.arange(3))
    assert load_npz_cache(path, "wrong-key") is None
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid simulation cache"):
        load_npz_cache(path, key)
