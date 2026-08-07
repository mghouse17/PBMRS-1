from pbmrs_core import (
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


class _SimCache:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


__all__ = [
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
