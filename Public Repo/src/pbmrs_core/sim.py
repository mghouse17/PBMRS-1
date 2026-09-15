"""Compatibility imports for the pre-0.2.2 monolithic simulator.

Use public imports from pbmrs_core in new code. This module remains for one
release so older notebooks keep working while sharing the canonical objects.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "pbmrs_core.sim is deprecated; import simulator APIs from pbmrs_core",
    DeprecationWarning,
    stacklevel=2,
)

from .math import (
    compute_field,
    compute_flow,
    compute_return,
    update_agents,
    update_liquidity,
    update_volatility,
)
from .models import NEAR_CRITICAL_MT2_HEURISTIC, SimConfig, SimResult
from .simulation import _SimCache, check_invariants, run_ensemble, run_sim

__all__ = [
    "NEAR_CRITICAL_MT2_HEURISTIC",
    "SimConfig",
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
