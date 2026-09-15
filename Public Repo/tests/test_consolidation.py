from __future__ import annotations

import importlib
import warnings

import numpy as np

import pbmrs_core
import pbmrs_core.simulation as canonical_simulation
from pbmrs_core import SimConfig, run_sim


def test_package_version_and_installed_phase_map_import():
    assert pbmrs_core.__version__ == "0.2.2"
    assert callable(pbmrs_core.phase_map)


def test_legacy_modules_export_canonical_object_identity():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module("pbmrs_core.sim")
        root_legacy = importlib.import_module("sim_v2")
    assert legacy.SimConfig is pbmrs_core.SimConfig
    assert legacy.run_sim is pbmrs_core.run_sim
    assert root_legacy.SimConfig is pbmrs_core.SimConfig
    assert root_legacy.run_sim is pbmrs_core.run_sim


def test_seeded_pre_consolidation_regression_values_are_unchanged():
    result = run_sim(SimConfig(seed=42, timesteps=200, n_agents=200, q0=0.005))
    actual = np.array([
        result.x[-1], result.v.mean(), result.l.min(), result.m.max(),
        result.r.std(), result.prices.min(),
    ])
    expected = np.array([
        -0.1318428812473228,
        1.0058037393226595,
        0.9590157263638346,
        0.18,
        0.010328601537328319,
        0.8534320674916147,
    ])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


def test_liquidity_update_receives_v_t_not_v_t_plus_one(monkeypatch):
    observed = []
    real_update = canonical_simulation.update_liquidity

    def spy(*args, **kwargs):
        observed.append(args[6])
        return real_update(*args, **kwargs)

    monkeypatch.setattr(canonical_simulation, "update_liquidity", spy)
    cfg = SimConfig(seed=2, timesteps=1, n_agents=100, q0=0.01, v0=2.0)
    result = run_sim(cfg)
    assert observed == [result.v[0]]
    assert observed[0] != result.v[1]


def test_enhanced_configuration_warnings_are_exposed():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SimConfig(impact_eps=1.0)
    assert any("effectively floored" in str(item.message) for item in caught)
