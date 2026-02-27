from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    seed: int
    timesteps: int
    dt: float

    n_agents: int
    beta: float

    kappa: float
    alpha: float
    sigma_eps: float

    lambda_v: float
    lambda_l: float
    eta_v: float
    eta_l: float

    min_vol: float
    min_liquidity: float

    x0: float
    v0: float
    l0: float


def _softplus(x: float) -> float:
    # numerically stable softplus
    if x > 30:
        return x
    return float(np.log1p(np.exp(x)))


def run_sim(cfg: SimConfig) -> dict[str, np.ndarray]:
    """
    Notebook-first PBMRS MVP:
    - Agents: spins s_i ∈ {-1, +1}
    - Magnetization m_t = mean(s)
    - Flow Q_t = kappa * m_t * l_t
    - Returns r_t = alpha * Q_t + sqrt(v_t) * sigma_eps * eps
    - Vol update v_{t+1} = (1-lv)*v_t + lv*(r_t^2) + eta_v, clamped to >= min_vol
    - Liq update l_{t+1} = (1-ll)*l_t + ll*(1/(1+v_t)) + eta_l, enforced > min_liquidity

    This is a *baseline* consistent with the PBMRS closed-loop idea.
    You can refine the exact functional forms later while keeping tests stable.
    """
    rng = np.random.default_rng(cfg.seed)

    # state arrays
    x = np.zeros(cfg.timesteps + 1, dtype=float)
    v = np.zeros(cfg.timesteps + 1, dtype=float)
    lq = np.zeros(cfg.timesteps + 1, dtype=float)
    m = np.zeros(cfg.timesteps + 1, dtype=float)
    Q = np.zeros(cfg.timesteps + 1, dtype=float)
    r = np.zeros(cfg.timesteps + 1, dtype=float)

    x[0], v[0], lq[0] = cfg.x0, cfg.v0, max(cfg.l0, cfg.min_liquidity)

    # spins
    s = rng.choice([-1, 1], size=cfg.n_agents)

    for t in range(cfg.timesteps):
        # --- agent update (simple field: trend + liquidity - volatility penalty) ---
        # trend proxy: last return r[t-1] (0 for t=0)
        trend = r[t - 1] if t > 0 else 0.0
        field = trend + 0.2 * np.log(lq[t]) - 0.2 * np.log(max(v[t], 1e-12))

        # logistic probability for +1
        p_up = 1.0 / (1.0 + np.exp(-cfg.beta * field))
        flips = rng.random(cfg.n_agents) < p_up
        s = np.where(flips, 1, -1)

        m[t] = float(np.mean(s))
        Q[t] = cfg.kappa * m[t] * lq[t]

        eps = rng.standard_normal()
        r[t] = cfg.alpha * Q[t] + np.sqrt(max(v[t], 0.0)) * cfg.sigma_eps * eps
        x[t + 1] = x[t] + r[t] * cfg.dt

        # volatility update
        v_next = (1.0 - cfg.lambda_v) * v[t] + cfg.lambda_v * (r[t] ** 2) + cfg.eta_v
        v[t + 1] = max(v_next, cfg.min_vol)

        # liquidity update (liquidity dries up when v is high)
        liq_target = 1.0 / (1.0 + v[t + 1])
        l_next = (1.0 - cfg.lambda_l) * lq[t] + cfg.lambda_l * liq_target + cfg.eta_l

        # enforce strictly positive liquidity
        lq[t + 1] = max(l_next, cfg.min_liquidity)

    # finalize last magnetization/flow values
    m[-1] = m[-2]
    Q[-1] = Q[-2]

    return {"x": x, "v": v, "l": lq, "m": m, "Q": Q, "r": r}