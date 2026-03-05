"""
sim.py — PBMRS Core Simulation Loop
"""
from __future__ import annotations
import warnings
import dataclasses
from dataclasses import dataclass
from typing import List, NamedTuple, Optional
import numpy as np


@dataclass
class SimConfig:
    seed:       int   = 42
    timesteps:  int   = 2000
    dt:         float = 1.0
    n_agents:   int   = 2000
    q0:         float = 5e-4
    beta:       float = 1.2
    J:          float = 0.5
    alpha_r:    float =  1.0
    alpha_v:    float =  0.001
    alpha_l:    float =  0.1
    alpha_0:    float =  0.0
    mu0:        float = 0.0
    lam:        float = 0.05
    sigma_eps:  float = 0.01
    kappa_v:    float = 0.05
    theta_v:    float = 1.0
    eta_v:      float = 0.85
    gamma_v:    float = 0.050  # raised: crowding now meaningfully amplifies vol (was 0.001)
    kappa_l:    float = 0.03
    l0:         float = 1.0
    eta_l:      float = 0.015  # raised: flow depletion noticeable at Qt~0.5 (was 0.005)
    gamma_l:    float = 0.030  # raised (excess-vol only after fix 1): equivalent stress response (was 0.002)
    # ── Price-impact floor ────────────────────────────────────────────────
    # Prevents Qt/(lt) from exploding when lt → min_liquidity.
    # Effective denominator: max(lt, impact_eps). Default 1% of l0=1.0.
    impact_eps:     float = 0.010
    min_vol:        float = 0.0
    min_liquidity:  float = 1e-6
    x0:     float = 0.0
    v0:     float = 1.0
    l0_init: float = 1.0

    def __post_init__(self) -> None:
        errors: List[str] = []
        if self.min_liquidity <= 0.0:
            errors.append(f"min_liquidity must be > 0 (got {self.min_liquidity})")
        if not (0.0 < self.kappa_v < 1.0):
            errors.append(f"kappa_v must be in (0, 1) for EWMA stability (got {self.kappa_v})")
        if not (0.0 < self.kappa_l < 1.0):
            errors.append(f"kappa_l must be in (0, 1) for EWMA stability (got {self.kappa_l})")
        if self.timesteps <= 0:
            errors.append(f"timesteps must be > 0 (got {self.timesteps})")
        if self.n_agents <= 0:
            errors.append(f"n_agents must be > 0 (got {self.n_agents})")
        if self.dt <= 0.0:
            errors.append(f"dt must be > 0 (got {self.dt})")
        if self.theta_v <= 0.0:
            errors.append(f"theta_v must be > 0 (got {self.theta_v})")
        if self.l0 <= 0.0:
            errors.append(f"l0 must be > 0 (got {self.l0})")
        if errors:
            raise ValueError(
                f"Invalid SimConfig ({len(errors)} error(s)):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        # Soft warnings — bad but not crash-inducing
        jb = self.J * self.beta
        if jb >= 1.0:
            warnings.warn(
                f"J * beta = {jb:.4f} >= 1.0 (supercritical). "
                f"Agents will spontaneously lock to m = ±1. "
                f"This is intentional if you are exploring near/above criticality; "
                f"logit clipping keeps numerics safe. "
                f"Subcritical boundary: J < {1.0/self.beta:.4f}.",
                UserWarning, stacklevel=2,
            )
        flow_scale = self.q0 * self.n_agents
        if not (0.5 <= flow_scale <= 2.0):
            warnings.warn(
                f"q0 * n_agents = {flow_scale:.3f} is outside [0.5, 2.0]. "
                f"Consider q0 = {1.0/self.n_agents:.2e}.",
                UserWarning, stacklevel=2,
            )


class SimResult(NamedTuple):
    x:      np.ndarray
    v:      np.ndarray
    l:      np.ndarray
    m:      np.ndarray
    Q:      np.ndarray
    r:      np.ndarray
    h:      np.ndarray
    prices: np.ndarray


@dataclass
class _SimCache:
    flow_scale:   float
    drift:        float
    sqrt_dt:      float
    one_minus_kv: float
    kv_target:    float
    one_minus_kl: float
    kl_l0:        float
    theta_v:      float   # needed by update_liquidity (excess-vol fix)
    impact_eps:   float   # needed by compute_return (impact floor fix)

    @classmethod
    def from_config(cls, cfg: SimConfig) -> "_SimCache":
        return cls(
            flow_scale   = cfg.q0 * cfg.n_agents,
            drift        = cfg.mu0 * cfg.dt,
            sqrt_dt      = float(np.sqrt(cfg.dt)),
            one_minus_kv = 1.0 - cfg.kappa_v,
            kv_target    = cfg.kappa_v * cfg.theta_v,
            one_minus_kl = 1.0 - cfg.kappa_l,
            kl_l0        = cfg.kappa_l * cfg.l0,
            theta_v      = cfg.theta_v,
            impact_eps   = cfg.impact_eps,
        )


# ── Pure equation functions ────────────────────────────────────────────────────

def compute_flow(flow_scale: float, mt: float) -> float:
    """Eq. 1: Qt = q0 * N * mt  (flow_scale = q0 * N)"""
    return flow_scale * mt


def compute_return(drift: float, lam: float, Qt: float, lt: float,
                   vt: float, sqrt_dt: float, sigma_eps: float, eps: float,
                   impact_eps: float = 0.0) -> float:
    """Eq. 2: rt = mu0*dt + lam*(Qt/max(lt,impact_eps)) + sqrt(vt*dt)*sigma_eps*eps

    impact_eps: soft floor on lt in the denominator (change 6).
    Prevents exploding impact when lt is clamped near min_liquidity.
    """
    eff_liq = max(lt, impact_eps)
    return drift + lam * (Qt / eff_liq) + np.sqrt(max(vt, 0.0)) * sqrt_dt * sigma_eps * eps


def update_volatility(one_minus_kv: float, kv_target: float,
                       eta_v: float, gamma_v: float,
                       vt: float, rt: float, mt: float, min_vol: float) -> float:
    """Eq. 3: vt+1 = (1-kv)*vt + kv*theta_v + eta_v*rt^2 + gamma_v*mt^2"""
    v_next = (one_minus_kv * vt + kv_target
              + eta_v * rt ** 2 + gamma_v * mt ** 2)
    return max(v_next, min_vol)


def update_liquidity(one_minus_kl: float, kl_l0: float,
                      eta_l: float, gamma_l: float,
                      lt: float, Qt: float, vt: float,
                      theta_v: float, min_liquidity: float) -> float:
    """Eq. 4: lt+1 = (1-kl)*lt + kl*l0 - eta_l*|Qt| - gamma_l*max(vt-theta_v, 0)

    Change 1: penalise EXCESS volatility (vt - theta_v) instead of absolute vt.
    Fixed point: if lt=l0 and Qt=0 and vt=theta_v, then lt+1 = l0 exactly.
    At baseline vol (vt=theta_v), the gamma_l term is zero — calm regimes stay calm.
    Only vol above the baseline drains liquidity.
    """
    excess_vol = max(vt - theta_v, 0.0)
    l_next = (one_minus_kl * lt + kl_l0
              - eta_l * abs(Qt)
              - gamma_l * excess_vol)
    return max(l_next, min_liquidity)


def compute_field(alpha_r: float, alpha_v: float, alpha_l: float, alpha_0: float,
                   rt: float, vt: float, lt: float, l0: float) -> float:
    """Eq. 5: ht = alpha_r*rt - alpha_v*vt - alpha_l*(l0/lt - 1) + alpha_0"""
    liq_stress = (l0 / lt) - 1.0
    return alpha_r * rt - alpha_v * vt - alpha_l * liq_stress + alpha_0


def update_agents(rng: np.random.Generator, beta: float, J: float,
                   mt: float, ht: float,
                   s: np.ndarray, draws: np.ndarray) -> float:
    """Eq. 6: P(si=+1) = sigma(beta*(J*mt + ht)). Mutates s, draws in-place."""
    logit = np.clip(beta * (J * mt + ht), -50.0, 50.0)
    p_on  = 1.0 / (1.0 + np.exp(-logit))
    rng.random(out=draws)
    s[:] = 1.0
    s[draws >= p_on] = -1.0
    return float(np.mean(s))


# ── run_sim ────────────────────────────────────────────────────────────────────

def run_sim(cfg: SimConfig, _cache: Optional[_SimCache] = None) -> SimResult:
    if _cache is None:
        _cache = _SimCache.from_config(cfg)

    rng = np.random.default_rng(cfg.seed)
    T   = cfg.timesteps

    x     = np.zeros(T + 1)
    v     = np.zeros(T + 1)
    l     = np.zeros(T + 1)
    m_arr = np.zeros(T + 1)
    Q_arr = np.zeros(T)
    r_arr = np.zeros(T)
    h_arr = np.zeros(T)

    s     = rng.choice([-1.0, 1.0], size=cfg.n_agents)
    draws = np.empty(cfg.n_agents)

    x[0]     = cfg.x0
    v[0]     = max(cfg.v0,      cfg.min_vol)
    l[0]     = max(cfg.l0_init, cfg.min_liquidity)
    m_arr[0] = float(np.mean(s))

    flow_scale   = _cache.flow_scale
    drift        = _cache.drift
    sqrt_dt      = _cache.sqrt_dt
    one_minus_kv = _cache.one_minus_kv
    kv_target    = _cache.kv_target
    one_minus_kl = _cache.one_minus_kl
    kl_l0        = _cache.kl_l0
    theta_v      = _cache.theta_v
    impact_eps   = _cache.impact_eps
    lam           = cfg.lam
    sigma_eps     = cfg.sigma_eps
    eta_v         = cfg.eta_v
    gamma_v       = cfg.gamma_v
    eta_l         = cfg.eta_l
    gamma_l       = cfg.gamma_l
    alpha_r       = cfg.alpha_r
    alpha_v       = cfg.alpha_v
    alpha_l       = cfg.alpha_l
    alpha_0       = cfg.alpha_0
    l0            = cfg.l0
    beta          = cfg.beta
    J             = cfg.J
    min_vol       = cfg.min_vol
    min_liquidity = cfg.min_liquidity

    for t in range(T):
        mt = m_arr[t]

        Qt     = compute_flow(flow_scale, mt)
        eps    = rng.standard_normal()
        rt     = compute_return(drift, lam, Qt, l[t], v[t], sqrt_dt, sigma_eps, eps,
                                impact_eps)
        x[t+1] = x[t] + rt

        v[t+1] = update_volatility(one_minus_kv, kv_target, eta_v, gamma_v,
                                    v[t], rt, mt, min_vol)
        l[t+1] = update_liquidity(one_minus_kl, kl_l0, eta_l, gamma_l,
                                   l[t], Qt, v[t], theta_v, min_liquidity)
        ht     = compute_field(alpha_r, alpha_v, alpha_l, alpha_0,
                               rt, v[t+1], l[t+1], l0)
        new_m  = update_agents(rng, beta, J, mt, ht, s, draws)

        m_arr[t+1] = new_m
        Q_arr[t]   = Qt
        r_arr[t]   = rt
        h_arr[t]   = ht

    return SimResult(x=x, v=v, l=l, m=m_arr, Q=Q_arr, r=r_arr, h=h_arr,
                     prices=np.exp(x))


# ── run_ensemble ───────────────────────────────────────────────────────────────

def run_ensemble(cfg: SimConfig, n_runs: int,
                  seeds: Optional[List[int]] = None) -> List[SimResult]:
    """
    Run n_runs independent paths from the same base config.
    _SimCache is built once and reused across all runs.
    # TODO: Phase 2 — vectorise across runs (batched RNG, stacked arrays).
    """
    if seeds is None:
        seeds = list(range(n_runs))
    if len(seeds) != n_runs:
        raise ValueError(f"len(seeds)={len(seeds)} must equal n_runs={n_runs}")
    cache = _SimCache.from_config(cfg)
    return [run_sim(dataclasses.replace(cfg, seed=s), _cache=cache) for s in seeds]


# ── check_invariants ───────────────────────────────────────────────────────────

def check_invariants(result: SimResult, cfg: SimConfig) -> None:
    """
    Validate hard model invariants. Raises ValueError listing ALL violations.
    Silent on success — safe to use in test suites without stdout pollution.
    """
    errors: List[str] = []

    for name, arr in [("x", result.x), ("v", result.v), ("l", result.l),
                       ("m", result.m), ("Q", result.Q), ("r", result.r)]:
        if np.any(np.isnan(arr)):
            errors.append(f"NaN detected in {name}")
        if np.any(np.isinf(arr)):
            errors.append(f"Inf detected in {name}")

    if np.any(result.v < cfg.min_vol - 1e-12):
        errors.append(f"v fell below min_vol={cfg.min_vol} (min: {result.v.min():.6f})")
    if np.any(result.l <= 0.0):
        errors.append(f"l contains non-positive values (min: {result.l.min():.2e})")
    if np.any(result.l < cfg.min_liquidity - 1e-12):
        errors.append(f"l fell below min_liquidity={cfg.min_liquidity} (min: {result.l.min():.2e})")
    max_abs_m = float(np.abs(result.m).max())
    if max_abs_m > 1.0 + 1e-9:
        errors.append(f"|m| exceeded 1.0 (max: {max_abs_m:.8f})")

    if errors:
        raise ValueError(
            f"Invariants violated ({len(errors)}):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )