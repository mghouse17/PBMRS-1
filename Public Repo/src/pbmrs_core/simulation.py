from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .math import (
    compute_field,
    compute_flow,
    compute_return,
    update_agents,
    update_liquidity,
    update_volatility,
)
from .models import SimConfig, SimResult


@dataclass(frozen=True)
class _SimCache:
    """Immutable scalars shared across paths in an ensemble."""

    flow_scale: float
    drift: float
    sqrt_dt: float
    one_minus_kv: float
    kv_target: float
    one_minus_kl: float
    kl_l0: float
    l0_init: float

    @classmethod
    def from_config(cls, cfg: SimConfig) -> _SimCache:
        return cls(
            flow_scale=cfg.q0 * cfg.n_agents,
            drift=cfg.mu0 * cfg.dt,
            sqrt_dt=float(np.sqrt(cfg.dt)),
            one_minus_kv=1.0 - cfg.kappa_v,
            kv_target=cfg.kappa_v * cfg.theta_v,
            one_minus_kl=1.0 - cfg.kappa_l,
            kl_l0=cfg.kappa_l * cfg.l0,
            l0_init=max(cfg.l0, cfg.min_liquidity),
        )


def _run_sim_with_cache(cfg: SimConfig, cache: _SimCache) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    T = cfg.timesteps

    x = np.zeros(T + 1)
    v = np.zeros(T + 1)
    l = np.zeros(T + 1)
    m_arr = np.zeros(T + 1)
    Q_arr = np.zeros(T)
    r_arr = np.zeros(T)
    h_arr = np.zeros(T)

    s = rng.choice([-1.0, 1.0], size=cfg.n_agents)
    draws = np.empty(cfg.n_agents)

    x[0] = cfg.x0
    v[0] = max(cfg.v0, cfg.min_vol)
    l[0] = cache.l0_init
    m_arr[0] = float(np.mean(s))

    flow_scale = cache.flow_scale
    drift = cache.drift
    sqrt_dt = cache.sqrt_dt
    one_minus_kv = cache.one_minus_kv
    kv_target = cache.kv_target
    one_minus_kl = cache.one_minus_kl
    kl_l0 = cache.kl_l0

    for t in range(T):
        mt = m_arr[t]
        Qt = compute_flow(flow_scale, mt)
        eps = rng.standard_normal()
        rt = compute_return(
            drift,
            cfg.lam,
            Qt,
            l[t],
            v[t],
            sqrt_dt,
            cfg.sigma_eps,
            eps,
            cfg.impact_eps,
        )
        x[t + 1] = x[t] + rt
        v[t + 1] = update_volatility(
            one_minus_kv,
            kv_target,
            cfg.eta_v,
            cfg.gamma_v,
            v[t],
            rt,
            mt,
            cfg.min_vol,
        )
        l[t + 1] = update_liquidity(
            one_minus_kl,
            kl_l0,
            cfg.eta_l,
            cfg.gamma_l,
            l[t],
            Qt,
            v[t],
            cfg.theta_v,
            cfg.min_liquidity,
        )
        ht = compute_field(
            cfg.alpha_r,
            cfg.alpha_v,
            cfg.alpha_l,
            cfg.alpha_0,
            rt,
            v[t + 1],
            l[t + 1],
            cfg.l0,
        )
        new_m = update_agents(rng, cfg.beta, cfg.J, mt, ht, s, draws)

        m_arr[t + 1] = new_m
        Q_arr[t] = Qt
        r_arr[t] = rt
        h_arr[t] = ht

    return SimResult(
        x=x,
        v=v,
        l=l,
        m=m_arr,
        Q=Q_arr,
        r=r_arr,
        h=h_arr,
        prices=np.exp(x),
    )


def run_sim(cfg: SimConfig) -> SimResult:
    return _run_sim_with_cache(cfg, _SimCache.from_config(cfg))


def run_ensemble(cfg: SimConfig, n_runs: int, seeds: list[int] | None = None) -> list[SimResult]:
    if seeds is None:
        seeds = list(range(n_runs))
    if len(seeds) != n_runs:
        raise ValueError("len(seeds) must equal n_runs")
    cache = _SimCache.from_config(cfg)
    return [_run_sim_with_cache(dataclasses.replace(cfg, seed=s), cache) for s in seeds]


def check_invariants(result: SimResult, cfg: SimConfig) -> None:
    errors = []
    for name, arr in [("x", result.x), ("v", result.v), ("l", result.l), ("m", result.m), ("Q", result.Q), ("r", result.r)]:
        if np.isnan(arr).any():
            errors.append(f"NaN detected in {name}")
        if np.isinf(arr).any():
            errors.append(f"Inf detected in {name}")
    if np.any(result.v < cfg.min_vol - 1e-12):
        errors.append("volatility fell below min_vol")
    if np.any(result.l <= 0.0):
        errors.append("liquidity contains non-positive values")
    if np.any(result.l < cfg.min_liquidity - 1e-12):
        errors.append("liquidity fell below min_liquidity")
    if np.abs(result.m).max() > 1.0 + 1e-9:
        errors.append("|m| exceeded 1.0")
    if errors:
        raise ValueError("\n".join(errors))
