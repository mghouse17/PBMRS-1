"""Compatibility wrapper for historical root-level imports."""

import warnings

warnings.warn(
    "sim_v2 is deprecated; import simulator APIs from pbmrs_core",
    DeprecationWarning,
    stacklevel=2,
)

from pbmrs_core import (
    NEAR_CRITICAL_MT2_HEURISTIC,
    SimConfig,
    SimResult,
    check_invariants,
    compute_field,
    compute_flow,
    compute_return,
    run_ensemble,
    run_sim,
    update_agents,
    update_liquidity,
    update_volatility,
)
from pbmrs_core.simulation import _SimCache


__all__ = [
    "SimConfig",
    "NEAR_CRITICAL_MT2_HEURISTIC",
    "SimResult",
    "_SimCache",
    "run_sim",
    "run_ensemble",
    "check_invariants",
    "compute_flow",
    "compute_return",
    "update_volatility",
    "update_liquidity",
    "compute_field",
    "update_agents",
]
