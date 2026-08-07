import numpy as np

from pbmrs_core import (
    SimConfig,
    Scenario,
    ScenarioSpec,
    build_default_scenario,
    load_config,
    run_sim,
    run_ensemble,
    max_drawdown,
    tail_stats,
)


def test_package_imports_and_basic_run():
    cfg = SimConfig(seed=7, timesteps=50, n_agents=80, q0=0.01)
    out = run_sim(cfg)

    assert out.prices.shape == (51,)
    assert out.r.shape == (50,)
    assert np.isfinite(out.prices).all()
    assert np.isfinite(out.r).all()


def test_ensemble_and_diagnostics():
    cfg = SimConfig(seed=3, timesteps=80, n_agents=100, q0=0.005)
    results = run_ensemble(cfg, n_runs=3, seeds=[0, 1, 2])
    stats = tail_stats(results, l0=cfg.l0)

    assert len(results) == 3
    assert stats["n_runs"] == 3
    assert np.isfinite(stats["mdd_mean"])
    assert stats["liq_stressed_frac"] >= 0.0


def test_scenario_and_config_loading():
    scenario = build_default_scenario("flash_crash")
    assert isinstance(scenario, Scenario)
    assert scenario.name == "flash_crash"

    cfg = load_config("configs/base.yaml")
    assert cfg is not None
    assert cfg["simulation"]["timesteps"] > 0
