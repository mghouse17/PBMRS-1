"""
sim.py — PBMRS Core Simulation Loop
Version: 0.2.2

Architecture changes vs 0.2.0:
  [Arch-1] _cache removed from run_sim public signature.
           run_ensemble uses internal _run_sim_with_cache().
  [Arch-2] l0_init removed; initial liquidity derived as max(l0, min_liquidity).
  [Arch-4] Cross-parameter stability warnings added to SimConfig.__post_init__.

Code quality changes vs 0.2.1:
  [CQ-6]  Liquidity update now uses v[t+1] instead of v[t] — volatility spike
          immediately depletes liquidity in the same step (tighter feedback).
  [CQ-8]  NEAR_CRITICAL_MT2_HEURISTIC exported as named constant.
"""
from __future__ import annotations
import math
import warnings
import dataclasses
from dataclasses import dataclass
from typing import List, NamedTuple, Optional
import numpy as np


# ── Module-level constants ────────────────────────────────────────────────────

# [CQ-8] Single source of truth for near-critical E[mt²] approximation.
# At J·β → 1⁻, mean-field Ising gives E[mt²] that grows from 0.
# 0.25 is a conservative empirical estimate from subcritical runs near J·β=0.85.
# Import this in notebooks/analysis code rather than repeating the literal.
NEAR_CRITICAL_MT2_HEURISTIC: float = 0.25


# ── SimConfig ─────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    seed:       int   = 42
    timesteps:  int   = 2000
    dt:         float = 1.0
    n_agents:   int   = 2000
    q0:         float = 5e-4
    beta:       float = 1.2
    J:          float = 0.5    # J*beta=0.60 (subcritical). Critical boundary: J < 1/beta.

    # Market field weights (Eq. 5)
    alpha_r:    float =  1.0   # trend-following weight
    alpha_v:    float =  0.001 # volatility-aversion weight
    alpha_l:    float =  0.1   # liquidity-stress weight
    alpha_0:    float =  0.0   # baseline behavioral bias

    # Return dynamics (Eq. 2)
    mu0:        float = 0.0
    lam:        float = 0.05
    sigma_eps:  float = 0.01

    # Volatility dynamics (Eq. 3)
    kappa_v:    float = 0.05
    theta_v:    float = 1.0
    eta_v:      float = 0.85   # ARCH amplification. Stability bound: eta_v < kappa_v / (typical rt²)
    gamma_v:    float = 0.050  # crowding amplification. Raises steady-state vol by gamma_v*E[mt²]/kappa_v

    # Liquidity dynamics (Eq. 4)
    kappa_l:    float = 0.03
    l0:         float = 1.0    # baseline AND initial liquidity (see Arch-2)
    eta_l:      float = 0.015  # flow depletion
    gamma_l:    float = 0.030  # excess-vol depletion (only fires above theta_v)

    # Price-impact floor (Eq. 2) — prevents Qt/lt exploding near min_liquidity
    impact_eps:     float = 0.010

    # Hard constraints
    min_vol:        float = 0.0
    min_liquidity:  float = 1e-6

    # Initial conditions
    x0:     float = 0.0
    v0:     float = 1.0
    # NOTE: l0_init removed (Arch-2). Initial liquidity = max(l0, min_liquidity).

    def __post_init__(self) -> None:
        # ── Hard errors ──────────────────────────────────────────────────────
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

        # ── Soft warnings — dangerous but not crash-inducing ─────────────────
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

        # [Arch-4] Cross-parameter stability warnings
        # Volatility EWMA stability: requires eta_v * E[rt²] << kappa_v
        # At typical rt_std ~ sigma_eps * sqrt(theta_v) ≈ 0.01, E[rt²] ≈ 1e-4
        # Rough bound: if eta_v > kappa_v / (sigma_eps² * theta_v), vol may not revert
        approx_rt2 = (self.sigma_eps ** 2) * self.theta_v
        eta_v_stability_bound = self.kappa_v / approx_rt2 if approx_rt2 > 0 else float('inf')
        if self.eta_v > eta_v_stability_bound:
            warnings.warn(
                f"eta_v={self.eta_v:.4f} may cause volatility to not mean-revert. "
                f"Rough stability bound: eta_v < kappa_v / (sigma_eps² * theta_v) "
                f"= {eta_v_stability_bound:.2f}. "
                f"Check that v_t remains bounded in long runs.",
                UserWarning, stacklevel=2,
            )

        # impact_eps > l0 silently disables the impact channel
        if self.impact_eps >= self.l0:
            warnings.warn(
                f"impact_eps={self.impact_eps} >= l0={self.l0}. "
                f"Price impact floor is at or above baseline liquidity — "
                f"the liquidity feedback channel is effectively disabled.",
                UserWarning, stacklevel=2,
            )

        # Steady-state vol elevation from crowding (at mean-field mt approximation)
        # At near-critical J, E[mt²] can be ~0.2–0.4. Warn if crowding raises vol > 50% above theta_v.
        crowding_vol_elevation = self.gamma_v * NEAR_CRITICAL_MT2_HEURISTIC / self.kappa_v
        if crowding_vol_elevation > 0.5 * self.theta_v:
            warnings.warn(
                f"At near-critical herding (E[mt²]~0.25), gamma_v={self.gamma_v} raises "
                f"steady-state volatility by ~{crowding_vol_elevation:.2f} above theta_v={self.theta_v} "
                f"({crowding_vol_elevation/self.theta_v:.0%} elevation). "
                f"This may dominate the return signal at high J.",
                UserWarning, stacklevel=2,
            )


# ── SimResult ─────────────────────────────────────────────────────────────────

class SimResult(NamedTuple):
    x:      np.ndarray   # log-price, shape (T+1,)
    v:      np.ndarray   # volatility, shape (T+1,)
    l:      np.ndarray   # liquidity, shape (T+1,)
    m:      np.ndarray   # magnetization, shape (T+1,)
    Q:      np.ndarray   # order flow, shape (T,)
    r:      np.ndarray   # returns, shape (T,)
    h:      np.ndarray   # market field, shape (T,)
    prices: np.ndarray   # price level exp(x), shape (T+1,)


# ── _SimCache (internal only) ─────────────────────────────────────────────────

@dataclass
class _SimCache:
    """
    Pre-computed scalars derived from SimConfig. Internal to this module.
    All fields are immutable floats — safe to share across runs in run_ensemble.
    Do NOT add mutable state (e.g. arrays) here without updating run_ensemble.
    """
    flow_scale:   float
    drift:        float
    sqrt_dt:      float
    one_minus_kv: float
    kv_target:    float
    one_minus_kl: float
    kl_l0:        float
    theta_v:      float
    impact_eps:   float
    l0_init:      float   # derived: max(cfg.l0, cfg.min_liquidity)

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
            l0_init      = max(cfg.l0, cfg.min_liquidity),  # [Arch-2]
        )


# ── Pure equation functions ────────────────────────────────────────────────────

def compute_flow(flow_scale: float, mt: float) -> float:
    """Eq. 1: Qt = q0 * N * mt  (flow_scale = q0 * N)"""
    return flow_scale * mt


def compute_return(drift: float, lam: float, Qt: float, lt: float,
                   vt: float, sqrt_dt: float, sigma_eps: float, eps: float,
                   impact_eps: float = 0.0) -> float:
    """Eq. 2: rt = mu0*dt + lam*(Qt/max(lt,impact_eps)) + sqrt(vt*dt)*sigma_eps*eps"""
    eff_liq = max(lt, impact_eps)
    # math.sqrt is ~3x faster than np.sqrt for Python scalars.
    # vt >= min_vol >= 0 is guaranteed by update_volatility — guard is redundant.
    # [CQ-14] Removed max(vt, 0.0); invariant documented here.
    return drift + lam * (Qt / eff_liq) + math.sqrt(vt) * sqrt_dt * sigma_eps * eps


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

    Penalises EXCESS volatility only (vt - theta_v), not absolute vol.
    Fixed point: if lt=l0, Qt=0, vt=theta_v → lt+1 = l0. Calm stays calm.
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


# ── _run_sim_with_cache (internal) ─────────────────────────────────────────────

def _run_sim_with_cache(cfg: SimConfig, cache: _SimCache) -> SimResult:
    """
    Internal implementation used by both run_sim and run_ensemble.
    Accepts a pre-built _SimCache to avoid redundant recomputation in ensembles.
    Not part of the public API.
    """
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
    v[0]     = max(cfg.v0, cfg.min_vol)
    l[0]     = cache.l0_init                 # [Arch-2]: derived from l0 in cache
    m_arr[0] = float(np.mean(s))

    # Unpack cache into locals for tight loop performance
    flow_scale   = cache.flow_scale
    drift        = cache.drift
    sqrt_dt      = cache.sqrt_dt
    one_minus_kv = cache.one_minus_kv
    kv_target    = cache.kv_target
    one_minus_kl = cache.one_minus_kl
    kl_l0        = cache.kl_l0
    theta_v      = cache.theta_v
    impact_eps   = cache.impact_eps

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
                                   l[t], Qt, v[t+1], theta_v, min_liquidity)  # [CQ-6] v[t+1] not v[t]
        ht     = compute_field(alpha_r, alpha_v, alpha_l, alpha_0,
                               rt, v[t+1], l[t+1], l0)
        new_m  = update_agents(rng, beta, J, mt, ht, s, draws)

        m_arr[t+1] = new_m
        Q_arr[t]   = Qt
        r_arr[t]   = rt
        h_arr[t]   = ht

    return SimResult(x=x, v=v, l=l, m=m_arr, Q=Q_arr, r=r_arr, h=h_arr,
                     prices=np.exp(x))


# ── run_sim (public) ──────────────────────────────────────────────────────────

def run_sim(cfg: SimConfig) -> SimResult:
    """
    Run a single simulation path.

    Parameters
    ----------
    cfg : SimConfig — fully determines the run (config + seed = reproducible path)

    Returns
    -------
    SimResult with all state variables and prices arrays.

    Notes
    -----
    [Arch-1] _cache is no longer a public parameter. Call run_ensemble() for
    multiple runs — it handles cache reuse internally.
    """
    cache = _SimCache.from_config(cfg)
    return _run_sim_with_cache(cfg, cache)


# ── run_ensemble (public) ─────────────────────────────────────────────────────

def run_ensemble(cfg: SimConfig, n_runs: int,
                  seeds: Optional[List[int]] = None) -> List[SimResult]:
    """
    Run n_runs independent paths from the same base config.

    Seeds default to [0, 1, ..., n_runs-1]. Pass explicit seeds for
    non-contiguous or reproducible subsets.

    _SimCache is built once and shared — safe because all cache fields are
    immutable floats (no mutable arrays). See _SimCache docstring.
    """
    if seeds is None:
        seeds = list(range(n_runs))
    if len(seeds) != n_runs:
        raise ValueError(f"len(seeds)={len(seeds)} must equal n_runs={n_runs}")
    cache = _SimCache.from_config(cfg)
    return [_run_sim_with_cache(dataclasses.replace(cfg, seed=s), cache) for s in seeds]


# ── check_invariants (public) ─────────────────────────────────────────────────

def check_invariants(result: SimResult, cfg: SimConfig) -> None:
    """
    Validate hard model invariants. Raises ValueError listing ALL violations.
    Silent on success — safe to call in test suites.
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