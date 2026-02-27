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


def run_sim(cfg: SimConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)

    x = np.zeros(cfg.timesteps + 1)
    v = np.zeros(cfg.timesteps + 1)
    lq = np.zeros(cfg.timesteps + 1)

    x[0] = cfg.x0
    v[0] = cfg.v0
    lq[0] = max(cfg.l0, cfg.min_liquidity)

    s = rng.choice([-1, 1], size=cfg.n_agents)

    for t in range(cfg.timesteps):
        m = np.mean(s)
        Q = cfg.kappa * m * lq[t]

        eps = rng.standard_normal()
        r = cfg.alpha * Q + np.sqrt(max(v[t], 0)) * cfg.sigma_eps * eps

        x[t + 1] = x[t] + r

        v_next = (1 - cfg.lambda_v) * v[t] + cfg.lambda_v * (r ** 2)
        v[t + 1] = max(v_next, cfg.min_vol)

        l_next = (1 - cfg.lambda_l) * lq[t] + cfg.lambda_l * (1 / (1 + v[t + 1]))
        lq[t + 1] = max(l_next, cfg.min_liquidity)

        # random flip update
        s = rng.choice([-1, 1], size=cfg.n_agents)

    return {"x": x, "v": v, "l": lq}