from __future__ import annotations

import math

import numpy as np


def compute_flow(flow_scale: float, mt: float) -> float:
    """Aggregate order flow from magnetization."""
    return flow_scale * mt


def compute_return(
    drift: float,
    lam: float,
    Qt: float,
    lt: float,
    vt: float,
    sqrt_dt: float,
    sigma_eps: float,
    eps: float,
    impact_eps: float = 0.0,
) -> float:
    """Return equation with liquidity-adjusted price impact and stochastic noise."""
    eff_liq = max(lt, impact_eps)
    return drift + lam * (Qt / eff_liq) + math.sqrt(vt) * sqrt_dt * sigma_eps * eps


def update_volatility(
    one_minus_kv: float,
    kv_target: float,
    eta_v: float,
    gamma_v: float,
    vt: float,
    rt: float,
    mt: float,
    min_vol: float,
) -> float:
    """Volatility evolution with mean reversion, return feedback, and crowding."""
    v_next = one_minus_kv * vt + kv_target + eta_v * rt**2 + gamma_v * mt**2
    return max(v_next, min_vol)


def update_liquidity(
    one_minus_kl: float,
    kl_l0: float,
    eta_l: float,
    gamma_l: float,
    lt: float,
    Qt: float,
    vt: float,
    theta_v: float,
    min_liquidity: float = 1e-6,
) -> float:
    """Liquidity evolution with replenishment, flow depletion, and volatility stress."""
    excess_vol = max(vt - theta_v, 0.0)
    l_next = one_minus_kl * lt + kl_l0 - eta_l * abs(Qt) - gamma_l * excess_vol
    return max(l_next, min_liquidity)


def compute_field(
    alpha_r: float,
    alpha_v: float,
    alpha_l: float,
    alpha_0: float,
    rt: float,
    vt: float,
    lt: float,
    l0: float,
) -> float:
    """Market field incorporating trend, volatility aversion, and liquidity stress."""
    liq_stress = (l0 / lt) - 1.0
    return alpha_r * rt - alpha_v * vt - alpha_l * liq_stress + alpha_0


def update_agents(
    rng: np.random.Generator,
    beta: float,
    J: float,
    mt: float,
    ht: float,
    s: np.ndarray,
    draws: np.ndarray,
) -> float:
    """Stochastic Ising-inspired agent update."""
    logit = np.clip(beta * (J * mt + ht), -50.0, 50.0)
    p_on = 1.0 / (1.0 + np.exp(-logit))
    rng.random(out=draws)
    n_on = int(np.sum(draws < p_on))
    # Mirror the legacy behavior of mutating the state buffer and returning the new mean.
    s[:] = np.where(draws < p_on, 1.0, -1.0)
    return float(np.mean(s))
