"""
sim.py — PBMRS Core Simulation Loop
Implements Equations (1)–(6) from the PBMRS model spec (Model_Definition_and_Explanation.pdf
and Project_Risk_README.pdf).

Closed feedback loop (README §4.7):
    (si,t) → mt → Qt → (rt, xt) → (vt, ℓt) → ht → (si,t+1)

═══════════════════════════════════════════════════════════════════════════════
CHANGES FROM PRIOR VERSION  (and why each matters)
═══════════════════════════════════════════════════════════════════════════════

Fix 1 — Order flow and return impact (Eq. 1–2)
  Old:  Q = kappa * m * lq[t]          # liquidity MULTIPLIED — wrong direction
        r = alpha * Q + noise
  New:  Qt = q0 * N * mt               # pure aggregate flow (Eq. 1)
        rt = mu0*dt + lam*(Qt/ℓt) + …  # liquidity in DENOMINATOR (Eq. 2)
  Why:  Low liquidity should AMPLIFY price impact, not reduce it.

Fix 2 — Volatility: missing mean-reversion target and crowding term (Eq. 3)
  Old:  v_next = (1-lv)*v + lv*r²         # decays to 0 in calm periods
  New:  v_next = (1-κv)*v + κv*θv + ηv*r² + γv*m²
  Why:  θv provides a non-zero volatility floor; γv*m² links herding to vol.

Fix 3 — Liquidity: missing flow-depletion and volatility-withdrawal (Eq. 4)
  Old:  l_next = (1-ll)*l + ll*(1/(1+v))  # ad-hoc; no independent depletion
  New:  l_next = (1-κl)*l + κl*ℓ0 - ηl*|Qt| - γl*vt
  Why:  Both heavy order flow and high volatility drain market depth independently.

Fix 4 — Market field (Eq. 5) — entirely absent in prior version
  New:  ht = αr*rt − αv*vt − αl*(ℓ0/ℓt−1) + α0
  Why:  Without ht the agent field has no information about market conditions.

Fix 5 — Agent update: feedback loop closed (Eq. 6)
  Old:  s = rng.choice([-1, 1], size=N)   # pure noise — no feedback at all
  New:  p = σ(β·(J·mt + ht))             # Ising-inspired logistic rule
        s = +1 with prob p, −1 otherwise
  Why:  The entire closed loop breaks without this.  β is now actually used.

Fix 6 — dt applied consistently (Eq. 2)
  Old:  dt defined in config but unused.
  New:  drift = mu0*dt; noise scale = sqrt(vt*dt).

═══════════════════════════════════════════════════════════════════════════════
PARAMETER SCALING NOTES
═══════════════════════════════════════════════════════════════════════════════

Qt = q0 * N * mt  makes raw flow O(q0·N).  The default q0 = 5e-4 is sized so
that Qt = mt ∈ [−1, +1] when N = 2000.  Scale q0 jointly with lam when
calibrating to a different agent count or real data (see README §6).

The Ising subcritical condition (no spontaneous herding) requires:
    J · β < 1
The default J = 0.5, β = 1.2 gives J·β = 0.60 (safely subcritical).
Increase J toward 1/β ≈ 0.83 for near-critical regime-shift dynamics.

alpha_v should be small relative to the volatility scale θv to avoid
permanently biasing agents toward risk-off in normal conditions.
Default alpha_v = 0.001 with θv = 1.0 gives |αv·vt| ≈ 0.001 at steady state.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed:       int   = 42
    timesteps:  int   = 2000
    dt:         float = 1.0

    # ── Agent layer (Eq. 1, 6) ───────────────────────────────────────────────
    n_agents:  int   = 2000
    # q0 · N ≈ 1.0 at default N=2000 → Qt = mt ∈ [-1, 1].
    # Rescale q0 if you change n_agents significantly.
    q0:        float = 5e-4    # per-agent order size scale

    beta:      float = 1.2     # inverse temperature / agent responsiveness
    J:         float = 0.5     # herding coupling   (J·β = 0.60 → subcritical)
                                # raise toward 1/β ≈ 0.83 for near-critical dynamics

    # ── Market field weights (Eq. 5) ─────────────────────────────────────────
    alpha_r:   float =  1.0    # trend-following weight (positive → buy after rally)
    alpha_v:   float =  0.001  # volatility-aversion weight (small: v is O(θv)=O(1))
    alpha_l:   float =  0.1    # liquidity-stress weight
    alpha_0:   float =  0.0    # baseline behavioral bias

    # ── Return dynamics (Eq. 2) ───────────────────────────────────────────────
    mu0:       float = 0.0     # baseline drift
    lam:       float = 0.05    # price impact strength λ
    sigma_eps: float = 0.01    # noise scale (≈ 1% per step ≙ ≈16% annualised)

    # ── Volatility dynamics (Eq. 3) ───────────────────────────────────────────
    kappa_v:   float = 0.05    # mean-reversion rate toward θv
    theta_v:   float = 1.0     # baseline (target) volatility
    eta_v:     float = 0.85    # return-shock coefficient   (weights rt²)
    gamma_v:   float = 0.001   # crowding coefficient        (weights mt²)

    # ── Liquidity dynamics (Eq. 4) ────────────────────────────────────────────
    kappa_l:   float = 0.03    # replenishment rate toward ℓ0
    l0:        float = 1.0     # baseline liquidity
    eta_l:     float = 0.005   # flow-depletion coefficient  (weights |Qt|)
    gamma_l:   float = 0.002   # volatility-withdrawal coeff (weights vt)

    # ── Constraints ───────────────────────────────────────────────────────────
    min_vol:        float = 0.0
    min_liquidity:  float = 1e-6

    # ── Initial conditions ────────────────────────────────────────────────────
    x0:  float = 0.0
    v0:  float = 1.0    # start at θv; converges to v_ss ≈ 1.002 under normal noise
    l0_init: float = 1.0  # separate from the parameter l0 to avoid name clash


def run_sim(cfg: SimConfig) -> dict:
    """
    Run one PBMRS simulation path.

    Returns
    -------
    dict with arrays:
        x  (T+1,) — log-price path
        v  (T+1,) — volatility path
        l  (T+1,) — liquidity path
        m  (T+1,) — magnetization mt  (crowding / herding measure)
        Q  (T,)   — order flow Qt per step
        r  (T,)   — log-return rt per step
        h  (T,)   — market field ht per step
    """
    rng = np.random.default_rng(cfg.seed)

    T  = cfg.timesteps
    dt = cfg.dt

    # ── Pre-allocate arrays ───────────────────────────────────────────────────
    x     = np.zeros(T + 1)
    v     = np.zeros(T + 1)
    l     = np.zeros(T + 1)
    m_arr = np.zeros(T + 1)
    Q_arr = np.zeros(T)
    r_arr = np.zeros(T)
    h_arr = np.zeros(T)

    # ── Initial conditions ────────────────────────────────────────────────────
    x[0] = cfg.x0
    v[0] = max(cfg.v0, cfg.min_vol)
    l[0] = max(cfg.l0_init, cfg.min_liquidity)

    # Agents start uniform random → E[mt] = 0
    s = rng.choice([-1.0, 1.0], size=cfg.n_agents)
    m_arr[0] = float(np.mean(s))

    # ── Main simulation loop ──────────────────────────────────────────────────
    for t in range(T):

        # ── Eq. 1: Magnetization and aggregate order flow ─────────────────────
        mt = float(np.mean(s))
        Qt = cfg.q0 * cfg.n_agents * mt    # Qt ∈ [−q0·N, +q0·N] = [−1, 1] by default

        # ── Eq. 2: Log-return ─────────────────────────────────────────────────
        #   rt = µ0·Δt  +  λ·(Qt/ℓt)  +  √(vt·Δt)·σ·εt
        eps = rng.standard_normal()
        rt  = (cfg.mu0 * dt
               + cfg.lam * (Qt / l[t])                      # impact ∝ 1/liquidity
               + np.sqrt(max(v[t], 0.0) * dt) * cfg.sigma_eps * eps)

        x[t + 1] = x[t] + rt

        # ── Eq. 3: Volatility update ──────────────────────────────────────────
        #   vt+1 = (1−κv)·vt + κv·θv + ηv·rt² + γv·mt²
        v_next = (  (1.0 - cfg.kappa_v) * v[t]
                  +  cfg.kappa_v * cfg.theta_v       # mean-revert to θv (non-zero floor)
                  +  cfg.eta_v   * rt ** 2           # return-shock amplification
                  +  cfg.gamma_v * mt ** 2)          # crowding-induced amplification
        v[t + 1] = max(v_next, cfg.min_vol)

        # ── Eq. 4: Liquidity update ───────────────────────────────────────────
        #   ℓt+1 = (1−κl)·ℓt + κl·ℓ0 − ηl·|Qt| − γl·vt
        l_next = (  (1.0 - cfg.kappa_l) * l[t]
                  +  cfg.kappa_l * cfg.l0            # replenish toward baseline
                  -  cfg.eta_l   * abs(Qt)           # flow-induced depletion
                  -  cfg.gamma_l * v[t])             # volatility-induced withdrawal
        l[t + 1] = max(l_next, cfg.min_liquidity)

        # ── Eq. 5: Market field ───────────────────────────────────────────────
        #   ht = αr·rt − αv·vt − αl·(ℓ0/ℓt − 1) + α0
        # Liquidity-stress term (ℓ0/ℓt − 1) is dimensionless and zero when ℓt = ℓ0
        liq_stress = (cfg.l0 / l[t + 1]) - 1.0
        ht = (  cfg.alpha_r *  rt
              - cfg.alpha_v *  v[t + 1]
              - cfg.alpha_l *  liq_stress
              +  cfg.alpha_0)

        # ── Eq. 6: Agent update — Ising-inspired logistic rule ────────────────
        #   P(si,t+1 = +1) = σ(β · (J·mt + ht))
        # J·mt is the herding pressure; ht is the market-state signal.
        logit = cfg.beta * (cfg.J * mt + ht)
        logit = np.clip(logit, -50.0, 50.0)          # prevent exp overflow
        p_on  = 1.0 / (1.0 + np.exp(-logit))

        draws = rng.random(size=cfg.n_agents)
        s = np.where(draws < p_on, 1.0, -1.0)        # {−1, +1} states

        # ── Record ────────────────────────────────────────────────────────────
        m_arr[t + 1] = float(np.mean(s))
        Q_arr[t]     = Qt
        r_arr[t]     = rt
        h_arr[t]     = ht

    return {"x": x, "v": v, "l": l, "m": m_arr, "Q": Q_arr, "r": r_arr, "h": h_arr}


# ── Invariant checks ──────────────────────────────────────────────────────────

def check_invariants(results: dict, cfg: SimConfig) -> None:
    """
    Assert hard model invariants from Model Spec §12.
    Raises AssertionError on violation. Safe to strip from production runs.
    """
    assert not np.any(np.isnan(results["x"])), "NaN detected in x"
    assert not np.any(np.isnan(results["v"])), "NaN detected in v"
    assert not np.any(np.isnan(results["l"])), "NaN detected in l"
    assert np.all(results["v"] >= 0.0),        "Invariant violated: v must be >= 0"
    assert np.all(results["l"] >  0.0),        "Invariant violated: l must be > 0"
    assert np.all(np.abs(results["m"]) <= 1.0 + 1e-9), \
        "Invariant violated: |m| must be <= 1"
    print("All invariants passed.")