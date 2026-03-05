"""
tests/test_sim.py — PBMRS unit, regression, and edge-case tests

Coverage map
------------
test_compute_flow_*          Eq. 1   (Issue 10)
test_compute_return_*        Eq. 2   (Issue 10)
test_update_volatility_*     Eq. 3   (Issue 10)
test_update_liquidity_*      Eq. 4   (Issue 10)
test_compute_field_*         Eq. 5   (Issue 10)
test_update_agents_*         Eq. 6   (Issue 10)
test_simconfig_validation_*  Issue 1 / Issue 12
test_check_invariants_*      Issue 4 / Issue 12
test_constraint_*            Issue 12 (enforcement under stress)
test_regression_*            Issue 11 (frozen canonical values)
test_simresult_*             Issue 2
test_run_ensemble_*          Issue 13
"""

import pytest
import numpy as np
import dataclasses
import sys
sys.path.insert(0, '.')

from sim_v2 import (
    SimConfig, SimResult, _SimCache,
    run_sim, run_ensemble, check_invariants,
    compute_flow, compute_return,
    update_volatility, update_liquidity,
    compute_field, update_agents,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_cfg():
    """Default valid SimConfig — baseline used across many tests."""
    return SimConfig(seed=42, timesteps=200, n_agents=200, q0=0.005)


@pytest.fixture
def base_result(base_cfg):
    return run_sim(base_cfg)


@pytest.fixture
def agent_rng():
    return np.random.default_rng(0)


@pytest.fixture
def agent_buffers():
    s     = np.ones(100)
    draws = np.empty(100)
    return s, draws


# ══════════════════════════════════════════════════════════════════════════════
# Eq. 1 — compute_flow
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_flow_normal():
    assert compute_flow(1.0, 0.5)  == pytest.approx(0.5)
    assert compute_flow(1.0, -0.3) == pytest.approx(-0.3)


def test_compute_flow_zero_magnetization():
    """Zero herding produces zero flow regardless of scale."""
    assert compute_flow(999.0, 0.0) == pytest.approx(0.0)


def test_compute_flow_scales_with_flow_scale():
    """Doubling flow_scale doubles Qt."""
    assert compute_flow(2.0, 0.5) == pytest.approx(2 * compute_flow(1.0, 0.5))


# ══════════════════════════════════════════════════════════════════════════════
# Eq. 2 — compute_return
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_return_no_noise_deterministic():
    """With eps=0: rt = drift + lam*(Qt/lt). Hand-computable: 0+0.05*(1/0.5)=0.1"""
    rt = compute_return(drift=0.0, lam=0.05, Qt=1.0, lt=0.5,
                        vt=1.0, sqrt_dt=1.0, sigma_eps=0.01, eps=0.0)
    assert rt == pytest.approx(0.1, abs=1e-12)


def test_compute_return_low_liquidity_amplifies_impact():
    """Halving lt doubles the price impact (1/lt relationship)."""
    rt_high = compute_return(0.0, 0.05, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    rt_low  = compute_return(0.0, 0.05, 1.0, 0.5, 1.0, 1.0, 0.0, 0.0)
    assert rt_low == pytest.approx(2 * rt_high, abs=1e-12)


def test_compute_return_drift_applied():
    """Non-zero drift shifts return by mu0*dt."""
    rt_no   = compute_return(0.0,  0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    rt_with = compute_return(0.01, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    assert (rt_with - rt_no) == pytest.approx(0.01, abs=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Eq. 3 — update_volatility
# ══════════════════════════════════════════════════════════════════════════════

def test_update_volatility_fixed_point():
    """If vt=theta_v and rt=mt=0, volatility stays at theta_v."""
    kv = 0.05
    v_next = update_volatility(1 - kv, kv * 1.0, 0.85, 0.001,
                                vt=1.0, rt=0.0, mt=0.0, min_vol=0.0)
    assert v_next == pytest.approx(1.0, abs=1e-12)


def test_update_volatility_shock_raises_vol():
    """A large return shock raises volatility above the fixed point."""
    kv = 0.05
    v_calm    = update_volatility(1-kv, kv*1.0, 0.85, 0.001, 1.0, 0.0, 0.0, 0.0)
    v_shocked = update_volatility(1-kv, kv*1.0, 0.85, 0.001, 1.0, 1.0, 0.0, 0.0)
    assert v_shocked > v_calm




def test_update_liquidity_baseline_vol_no_depletion():
    """At vt=theta_v, the gamma_l term is zero — calm vol does not drain liquidity."""
    kl = 0.03
    # theta_v = vt = 1.0 → excess_vol = max(1-1, 0) = 0 → only EWMA and flow matter
    l_at_baseline_vol = update_liquidity(1-kl, kl*1.0, 0.0, 0.999, 1.0, 0.0, 1.0, 1.0, 1e-6)
    l_at_zero_vol     = update_liquidity(1-kl, kl*1.0, 0.0, 0.999, 1.0, 0.0, 0.0, 1.0, 1e-6)
    # gamma_l=0.999 but vt-theta_v=0 vs -1 → same result since max(x,0) kills negative excess
    assert l_at_baseline_vol == pytest.approx(l_at_zero_vol, abs=1e-12)


def test_update_liquidity_excess_vol_depletes():
    """Excess volatility (vt > theta_v) depletes liquidity; deficit (vt < theta_v) does not."""
    kl = 0.03
    l_excess = update_liquidity(1-kl, kl*1.0, 0.0, 0.05, 1.0, 0.0, 2.0, 1.0, 1e-6)  # excess=1
    l_defcit = update_liquidity(1-kl, kl*1.0, 0.0, 0.05, 1.0, 0.0, 0.5, 1.0, 1e-6)  # deficit → 0
    assert l_excess < 1.0         # drained by excess vol
    assert l_defcit == pytest.approx(1.0, abs=1e-12)  # deficit treated as 0; only EWMA

def test_update_volatility_clamped_to_min_vol():
    """Volatility is clamped to min_vol if equation produces a lower value."""
    v_next = update_volatility(0.0, 0.0, 0.0, 0.0,
                                vt=0.0, rt=0.0, mt=0.0, min_vol=0.5)
    assert v_next == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Eq. 4 — update_liquidity
# ══════════════════════════════════════════════════════════════════════════════

def test_update_liquidity_fixed_point():
    """If lt=l0 and Qt=vt=0, liquidity stays at l0."""
    kl = 0.03
    l_next = update_liquidity(1 - kl, kl * 1.0, 0.005, 0.002,
                               lt=1.0, Qt=0.0, vt=1.0, theta_v=1.0, min_liquidity=1e-6)
    assert l_next == pytest.approx(1.0, abs=1e-12)


def test_update_liquidity_flow_depletes():
    """Higher order flow reduces liquidity (depletion direction correct)."""
    kl = 0.03
    l_no   = update_liquidity(1-kl, kl*1.0, 0.005, 0.002, 1.0, 0.0, 1.0, 1.0, 1e-6)  # Qt=0, vt=theta_v
    l_flow = update_liquidity(1-kl, kl*1.0, 0.005, 0.002, 1.0, 1.0, 1.0, 1.0, 1e-6)  # Qt=1
    assert l_flow < l_no


def test_update_liquidity_vol_depletes():
    """Higher volatility reduces liquidity independently of flow."""
    kl = 0.03
    l_low  = update_liquidity(1-kl, kl*1.0, 0.005, 0.002, 1.0, 0.0, 0.0,  1e-6)
    l_high = update_liquidity(1-kl, kl*1.0, 0.005, 0.002, 1.0, 0.0, 5.0,  1e-6)
    assert l_high < l_low


def test_update_liquidity_clamped_to_min():
    """Liquidity must never fall below min_liquidity under extreme depletion."""
    l_next = update_liquidity(0.0, 0.0, 1.0, 1.0,
                               lt=0.001, Qt=100.0, vt=100.0, theta_v=1.0, min_liquidity=1e-6)
    assert l_next >= 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# Eq. 5 — compute_field
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_field_neutral_state():
    """At rt=0, vt=0, lt=l0, field equals alpha_0."""
    ht = compute_field(1.0, 0.001, 0.1, 0.05,
                        rt=0.0, vt=0.0, lt=1.0, l0=1.0)
    assert ht == pytest.approx(0.05, abs=1e-12)


def test_compute_field_positive_return_raises_field():
    """Positive return increases field (trend-following via alpha_r)."""
    ht_zero = compute_field(1.0, 0.001, 0.1, 0.0, 0.0,  0.0, 1.0, 1.0)
    ht_pos  = compute_field(1.0, 0.001, 0.1, 0.0, 0.05, 0.0, 1.0, 1.0)
    assert ht_pos > ht_zero


def test_compute_field_liquidity_stress_lowers_field():
    """Low liquidity (lt < l0) raises liq_stress and lowers the field."""
    ht_full = compute_field(1.0, 0.001, 0.1, 0.0, 0.0, 1.0, 1.0, 1.0)
    ht_half = compute_field(1.0, 0.001, 0.1, 0.0, 0.0, 1.0, 0.5, 1.0)
    assert ht_half < ht_full


# ══════════════════════════════════════════════════════════════════════════════
# Eq. 6 — update_agents
# ══════════════════════════════════════════════════════════════════════════════

def test_update_agents_all_risk_on(agent_rng, agent_buffers):
    """logit=50 → p_on≈1 → all agents become +1."""
    s, draws = agent_buffers
    new_m = update_agents(agent_rng, beta=1.0, J=0.0,
                           mt=0.0, ht=50.0, s=s, draws=draws)
    assert new_m == pytest.approx(1.0, abs=1e-6)
    assert np.all(s == 1.0)


def test_update_agents_all_risk_off(agent_rng, agent_buffers):
    """logit=-50 → p_on≈0 → all agents become -1."""
    s, draws = agent_buffers
    new_m = update_agents(agent_rng, beta=1.0, J=0.0,
                           mt=0.0, ht=-50.0, s=s, draws=draws)
    assert new_m == pytest.approx(-1.0, abs=1e-6)
    assert np.all(s == -1.0)


def test_update_agents_new_m_equals_mean_s(agent_rng, agent_buffers):
    """Return value must equal mean(s) after mutation."""
    s, draws = agent_buffers
    new_m = update_agents(agent_rng, beta=1.2, J=0.5,
                           mt=0.1, ht=0.05, s=s, draws=draws)
    assert new_m == pytest.approx(float(np.mean(s)), abs=1e-12)


def test_update_agents_s_contains_only_valid_states(agent_rng, agent_buffers):
    """All entries of s must be exactly +1 or -1."""
    s, draws = agent_buffers
    update_agents(agent_rng, beta=1.2, J=0.5,
                   mt=0.2, ht=0.0, s=s, draws=draws)
    assert np.all((s == 1.0) | (s == -1.0))


# ══════════════════════════════════════════════════════════════════════════════
# SimConfig validation  (Issue 1 / Issue 12)
# ══════════════════════════════════════════════════════════════════════════════

def test_simconfig_valid_default():
    cfg = SimConfig()
    assert cfg.J * cfg.beta < 1.0


def test_simconfig_warns_supercritical():
    """J*beta >= 1 now warns instead of raising — allows near/above-critical exploration."""
    with pytest.warns(UserWarning, match="supercritical"):
        SimConfig(J=0.9, beta=1.2)


def test_simconfig_rejects_zero_min_liquidity():
    with pytest.raises(ValueError, match="min_liquidity must be > 0"):
        SimConfig(min_liquidity=0.0)


def test_simconfig_rejects_negative_min_liquidity():
    with pytest.raises(ValueError, match="min_liquidity must be > 0"):
        SimConfig(min_liquidity=-1e-6)


def test_simconfig_rejects_kappa_v_out_of_range():
    with pytest.raises(ValueError, match="kappa_v must be in"):
        SimConfig(kappa_v=1.5)


def test_simconfig_rejects_kappa_l_out_of_range():
    with pytest.raises(ValueError, match="kappa_l must be in"):
        SimConfig(kappa_l=0.0)


def test_simconfig_rejects_zero_timesteps():
    with pytest.raises(ValueError, match="timesteps must be > 0"):
        SimConfig(timesteps=0)


def test_simconfig_rejects_zero_n_agents():
    with pytest.raises(ValueError, match="n_agents must be > 0"):
        SimConfig(n_agents=0)


def test_simconfig_reports_all_violations_at_once():
    """Multiple bad params produce one ValueError listing all issues."""
    with pytest.raises(ValueError) as exc_info:
        SimConfig(kappa_v=2.0, kappa_l=-0.1, min_liquidity=0.0)
    msg = str(exc_info.value)
    assert "kappa_v" in msg
    assert "kappa_l" in msg
    assert "min_liquidity" in msg


def test_simconfig_warns_on_extreme_flow_scale():
    """q0 * n_agents far from 1.0 should warn but not raise."""
    with pytest.warns(UserWarning, match="q0 \\* n_agents"):
        SimConfig(n_agents=100, q0=0.0001)


# ══════════════════════════════════════════════════════════════════════════════
# check_invariants  (Issue 4 / Issue 12)
# ══════════════════════════════════════════════════════════════════════════════

def test_check_invariants_passes_on_good_run(base_cfg, base_result):
    check_invariants(base_result, base_cfg)


def test_check_invariants_raises_on_nan_in_x(base_cfg, base_result):
    x_bad = base_result.x.copy()
    x_bad[50] = float("nan")
    bad = base_result._replace(x=x_bad)
    with pytest.raises(ValueError, match="NaN detected in x"):
        check_invariants(bad, base_cfg)


def test_check_invariants_raises_on_negative_liquidity(base_cfg, base_result):
    l_bad = base_result.l.copy()
    l_bad[10] = -0.5
    bad = base_result._replace(l=l_bad)
    with pytest.raises(ValueError, match="non-positive"):
        check_invariants(bad, base_cfg)


def test_check_invariants_raises_on_m_exceeding_one(base_cfg, base_result):
    m_bad = base_result.m.copy()
    m_bad[5] = 1.5
    bad = base_result._replace(m=m_bad)
    with pytest.raises(ValueError, match=r"\|m\|"):
        check_invariants(bad, base_cfg)


def test_check_invariants_silent_on_success(base_cfg, base_result, capsys):
    """Must produce zero stdout output on a passing run."""
    check_invariants(base_result, base_cfg)
    assert capsys.readouterr().out == ""


# ══════════════════════════════════════════════════════════════════════════════
# Constraint enforcement under stress  (Issue 12)
# ══════════════════════════════════════════════════════════════════════════════

def test_volatility_never_below_min_vol():
    cfg = SimConfig(seed=7, timesteps=500, n_agents=200, q0=0.005, min_vol=0.1)
    out = run_sim(cfg)
    assert np.all(out.v >= cfg.min_vol - 1e-12)


def test_liquidity_never_zero_or_below_under_stress():
    cfg = SimConfig(seed=7, timesteps=500, n_agents=200, q0=0.005,
                    eta_l=0.1, gamma_l=0.05)
    out = run_sim(cfg)
    assert np.all(out.l > 0.0)
    assert np.all(out.l >= cfg.min_liquidity - 1e-12)


def test_magnetization_bounded_near_critical():
    cfg = SimConfig(seed=99, timesteps=300, n_agents=200, q0=0.005,
                    J=0.79, beta=1.2)   # J*beta = 0.948
    out = run_sim(cfg)
    assert np.all(np.abs(out.m) <= 1.0 + 1e-9)


def test_no_nans_infs_long_run():
    cfg = SimConfig(seed=1, timesteps=5000, n_agents=500, q0=0.002)
    out = run_sim(cfg)
    for name, arr in [("x", out.x), ("v", out.v), ("l", out.l), ("r", out.r)]:
        assert not np.any(np.isnan(arr)), f"NaN in {name}"
        assert not np.any(np.isinf(arr)), f"Inf in {name}"


# ══════════════════════════════════════════════════════════════════════════════
# Regression  (Issue 11)
# ══════════════════════════════════════════════════════════════════════════════
# Frozen from: SimConfig(seed=42, timesteps=200, n_agents=200, q0=0.005)
# Run date: 2026-03-03  |  sim.py: post-review refactor
# To update intentionally: re-run print block in test_print_regression_values
# below, paste new values, and commit with a message explaining the change.

REGRESSION_CFG = SimConfig(seed=42, timesteps=200, n_agents=200, q0=0.005)

FROZEN = {
    "x_final":    -0.13184288,
    "v_mean":      1.00580374,
    "l_min":       0.95901573,
    "m_max":       0.18000000,
    "r_std":       0.01032860,
    "prices_min":  0.85343207,
}


def test_regression_frozen_values():
    """Same config + seed must reproduce identical summary stats to 7dp."""
    out = run_sim(REGRESSION_CFG)
    assert out.x[-1]        == pytest.approx(FROZEN["x_final"],   abs=1e-7)
    assert out.v.mean()     == pytest.approx(FROZEN["v_mean"],     abs=1e-7)
    assert out.l.min()      == pytest.approx(FROZEN["l_min"],      abs=1e-7)
    assert out.m.max()      == pytest.approx(FROZEN["m_max"],      abs=1e-7)
    assert out.r.std()      == pytest.approx(FROZEN["r_std"],      abs=1e-7)
    assert out.prices.min() == pytest.approx(FROZEN["prices_min"], abs=1e-7)


def test_regression_different_seed_different_path():
    """A different seed must NOT match the frozen path."""
    out = run_sim(dataclasses.replace(REGRESSION_CFG, seed=99))
    assert out.x[-1] != pytest.approx(FROZEN["x_final"], abs=1e-4)


def test_regression_prices_exp_x_exact():
    """prices = exp(x) must hold exactly at all time steps."""
    out = run_sim(REGRESSION_CFG)
    assert np.allclose(out.prices, np.exp(out.x), atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# SimResult NamedTuple  (Issue 2)
# ══════════════════════════════════════════════════════════════════════════════

def test_simresult_is_named_tuple(base_result):
    assert isinstance(base_result, tuple)
    assert hasattr(base_result, "x")
    assert hasattr(base_result, "prices")


def test_simresult_field_shapes(base_cfg, base_result):
    T = base_cfg.timesteps
    assert base_result.x.shape      == (T + 1,)
    assert base_result.v.shape      == (T + 1,)
    assert base_result.l.shape      == (T + 1,)
    assert base_result.m.shape      == (T + 1,)
    assert base_result.Q.shape      == (T,)
    assert base_result.r.shape      == (T,)
    assert base_result.h.shape      == (T,)
    assert base_result.prices.shape == (T + 1,)


# ══════════════════════════════════════════════════════════════════════════════
# run_ensemble  (Issue 13)
# ══════════════════════════════════════════════════════════════════════════════

def test_run_ensemble_returns_correct_count(base_cfg):
    results = run_ensemble(base_cfg, n_runs=4)
    assert len(results) == 4
    assert all(isinstance(r, SimResult) for r in results)


def test_run_ensemble_runs_are_independent(base_cfg):
    """Different seeds produce different paths."""
    results = run_ensemble(base_cfg, n_runs=3, seeds=[0, 1, 2])
    assert not np.allclose(results[0].x, results[1].x)
    assert not np.allclose(results[1].x, results[2].x)


def test_run_ensemble_matches_direct_run(base_cfg):
    """run_ensemble(seed=s) must match run_sim(cfg with seed=s) exactly."""
    results = run_ensemble(base_cfg, n_runs=1, seeds=[77])
    direct  = run_sim(dataclasses.replace(base_cfg, seed=77))
    assert np.allclose(results[0].x, direct.x)


def test_run_ensemble_rejects_mismatched_seeds(base_cfg):
    with pytest.raises(ValueError, match="len\\(seeds\\)"):
        run_ensemble(base_cfg, n_runs=3, seeds=[1, 2])


def test_run_ensemble_all_invariants_hold(base_cfg):
    for result in run_ensemble(base_cfg, n_runs=5):
        check_invariants(result, base_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# New diagnostic functions  (changes 4-5)
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, '.')
from pbmrs_core.diagnostics import (
    acf, acf_squared_returns, magnetization_persistence, tail_stats,
)


def test_acf_all_ones_series():
    """ACF of a constant series should be 1.0 at all lags (or 0 by convention when var=0)."""
    # constant → variance=0, our implementation returns 0.0 array
    result = acf(np.ones(100), nlags=5)
    assert result.shape == (6,)
    assert result[0] == pytest.approx(0.0)  # var=0 case


def test_acf_white_noise():
    """ACF of white noise at lag>0 should be near zero (not exactly — finite sample)."""
    rng = np.random.default_rng(0)
    wn  = rng.standard_normal(5000)
    ac  = acf(wn, nlags=20)
    assert ac[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.abs(ac[1:]) < 0.1)  # loose tolerance for finite sample


def test_acf_squared_returns_shape(base_result):
    ac = acf_squared_returns(base_result.r, nlags=20)
    assert ac.shape == (21,)
    assert ac[0] == pytest.approx(1.0, abs=1e-12)


def test_magnetization_persistence_keys(base_result):
    mp = magnetization_persistence(base_result.m, threshold=0.2, nlags=10)
    assert "acf_abs_m"     in mp
    assert "herd_fraction" in mp
    assert "herd_threshold" in mp
    assert mp["herd_threshold"] == pytest.approx(0.2)
    assert 0.0 <= mp["herd_fraction"] <= 1.0
    assert mp["acf_abs_m"].shape == (11,)
    assert mp["acf_abs_m"][0] == pytest.approx(1.0, abs=1e-2)  # loose: small series


def test_tail_stats_keys(base_cfg):
    results = run_ensemble(base_cfg, n_runs=5)
    ts = tail_stats(results, l0=base_cfg.l0)
    for key in ["n_runs", "mdd_mean", "mdd_std", "mdd_p95",
                "excess_kurtosis", "liq_stressed_frac",
                "mean_abs_m_mean", "mean_abs_m_std"]:
        assert key in ts, f"Missing key: {key}"
    assert ts["n_runs"] == 5
    assert 0.0 <= ts["mdd_mean"] <= 1.0
    assert 0.0 <= ts["liq_stressed_frac"] <= 1.0