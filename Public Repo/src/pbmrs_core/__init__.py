from importlib.metadata import PackageNotFoundError, version

from .config import ConfigError, load_config
from .models import (
    NEAR_CRITICAL_MT2_HEURISTIC,
    SimConfig,
    SimResult,
    Scenario,
    ScenarioSpec,
)
from .simulation import run_sim, run_ensemble, check_invariants
from .analysis import phase_map
from .calibration import (
    AdequacyResult,
    AdequacyRow,
    HorizonStats,
    PowerRow,
    StabilityRow,
    acf_r2_profile,
    assess_stability,
    compute_horizon_stats,
    estimate_point_power,
    invert_j_grid,
    lag_contributions,
)
from .diagnostics import (
    acf,
    acf_squared_returns,
    drawdown,
    fragility_index,
    magnetization_persistence,
    max_drawdown,
    recovery_time,
    regime_labels,
    tail_stats,
)
from .math import (
    compute_field,
    compute_flow,
    compute_return,
    update_agents,
    update_liquidity,
    update_volatility,
)
from .scenarios import build_default_scenario

try:
    __version__ = version("pbmrs")
except PackageNotFoundError:
    __version__ = "0.2.2"

__all__ = [
    "SimConfig",
    "SimResult",
    "Scenario",
    "ScenarioSpec",
    "ConfigError",
    "load_config",
    "build_default_scenario",
    "phase_map",
    "AdequacyResult",
    "AdequacyRow",
    "HorizonStats",
    "PowerRow",
    "StabilityRow",
    "acf_r2_profile",
    "assess_stability",
    "compute_horizon_stats",
    "estimate_point_power",
    "invert_j_grid",
    "lag_contributions",
    "NEAR_CRITICAL_MT2_HEURISTIC",
    "__version__",
    "run_sim",
    "run_ensemble",
    "check_invariants",
    "compute_flow",
    "compute_return",
    "update_volatility",
    "update_liquidity",
    "compute_field",
    "update_agents",
    "drawdown",
    "max_drawdown",
    "recovery_time",
    "fragility_index",
    "regime_labels",
    "acf",
    "acf_squared_returns",
    "magnetization_persistence",
    "tail_stats",
]
