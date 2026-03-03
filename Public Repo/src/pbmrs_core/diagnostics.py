"""
diagnostics.py — PBMRS Risk Diagnostics
Implements Equations (7)–(9) from the model spec.

Fix vs prior version:
  - drawdown / max_drawdown / recovery_time must receive PRICE-space arrays
    (i.e. St = exp(xt)), not raw log-price.
  - Callers should pass np.exp(out['x']).
  - The spec defines DD(t) = 1 - St/Mt where St = exp(xt).
    Applying the formula to log-prices produces incorrect drawdown values
    because the running-peak ratio in log-space is not the same as in price space.
"""

from __future__ import annotations
import numpy as np


# ── Eq. 7: Drawdown ───────────────────────────────────────────────────────────

def drawdown(prices: np.ndarray) -> np.ndarray:
    """
    Compute fractional drawdown at each time step.

    Parameters
    ----------
    prices : array of PRICE-level values St = exp(xt).
             Do NOT pass log-prices directly.

    Returns
    -------
    dd : np.ndarray  in [0, 1)
         DD(t) = 1 - St / max_{u<=t} Su
    """
    peak  = np.maximum.accumulate(prices)
    denom = np.where(peak == 0, 1.0, peak)   # guard divide-by-zero at t=0
    return 1.0 - (prices / denom)


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum drawdown over the full path."""
    return float(np.max(drawdown(prices)))


# ── Eq. 8: Recovery time ──────────────────────────────────────────────────────

def recovery_time(prices: np.ndarray, epsilon: float = 0.0) -> int | None:
    """
    Steps from the max-drawdown trough until prices recover to within
    (1 - epsilon) of the prior peak.

    Parameters
    ----------
    prices  : price-level array St = exp(xt)
    epsilon : tolerance; 0.0 means full recovery to the exact prior peak.
              Set e.g. 0.02 for '98% recovery'.

    Returns
    -------
    int   — number of steps from trough to recovery, or
    None  — if the path never recovers within the simulation window.
    """
    dd       = drawdown(prices)
    t_trough = int(np.argmax(dd))
    ref_peak = float(np.max(prices[: t_trough + 1]))
    target   = (1.0 - epsilon) * ref_peak

    for t in range(t_trough + 1, len(prices)):
        if prices[t] >= target:
            return t - t_trough
    return None


# ── Eq. 9: Fragility index ────────────────────────────────────────────────────

def fragility_index(
    m: np.ndarray,
    l: np.ndarray,
    v: np.ndarray,
    l0: float,
    theta_v: float,
    wm: float = 1.0,
    wl: float = 1.0,
    wv: float = 1.0,
) -> np.ndarray:
    """
    Composite fragility index (README Eq. 9):
        F(t) = wm*|mt| + wl*(l0/lt) + wv*(vt/theta_v)

    All three components are dimensionless.

    Parameters
    ----------
    m, l, v  : state-variable arrays from run_sim output
    l0       : baseline liquidity (SimConfig.l0)
    theta_v  : baseline volatility (SimConfig.theta_v)
    wm/wl/wv : weights (default 1.0 each)

    Returns
    -------
    F : np.ndarray, same length as input arrays
    """
    return wm * np.abs(m) + wl * (l0 / l) + wv * (v / theta_v)


def regime_labels(
    F: np.ndarray,
    p_stable: float = 33.0,
    p_unstable: float = 66.0,
) -> np.ndarray:
    """
    Assign regime labels from fragility index using percentile thresholds.

    Returns integer array:
        0 = Stable    (F < p_stable  percentile)
        1 = Fragile   (p_stable <= F < p_unstable percentile)
        2 = Unstable  (F >= p_unstable percentile)
    """
    lo = np.percentile(F, p_stable)
    hi = np.percentile(F, p_unstable)
    return np.where(F < lo, 0, np.where(F < hi, 1, 2))