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


# ── Stylized-fact diagnostics ─────────────────────────────────────────────────

def acf(series: np.ndarray, nlags: int) -> np.ndarray:
    """
    Sample autocorrelation function of a 1-D array.
    Returns ACF[0..nlags] (ACF[0] = 1.0 by definition).
    Uses unbiased denominator (n - k) to match statsmodels convention.
    """
    n   = len(series)
    s   = series - series.mean()
    var = float(np.dot(s, s))
    if var == 0.0:
        return np.zeros(nlags + 1)
    result = np.empty(nlags + 1)
    for k in range(nlags + 1):
        result[k] = float(np.dot(s[:n - k], s[k:])) / var
    return result


def acf_squared_returns(r: np.ndarray, nlags: int = 40) -> np.ndarray:
    """
    ACF of squared returns — the standard volatility-clustering diagnostic.

    Persistent positive ACF(r²) at small lags confirms ARCH-like clustering.
    A nearly flat ACF(r²) ≈ 0 indicates the simulator is producing i.i.d. noise.

    Parameters
    ----------
    r     : return series from SimResult.r
    nlags : number of lags to compute

    Returns
    -------
    acf_r2 : np.ndarray of shape (nlags+1,), ACF[0]=1.0
    """
    return acf(r ** 2, nlags)


def magnetization_persistence(
    m: np.ndarray,
    threshold: float = 0.3,
    nlags: int = 40,
) -> dict:
    """
    Two complementary measures of magnetization persistence.

    1. ACF of |m| — how long herding memory lasts.
       Decaying slowly → agents stay correlated for many steps.
       Fast decay → herding is transient.

    2. Fraction of time |m| > threshold — how often the market is
       in a 'crowd' state (sustained herding episode).

    Parameters
    ----------
    m         : magnetization series from SimResult.m
    threshold : |m| level considered 'herding' (default 0.3)
    nlags     : lags for ACF

    Returns
    -------
    dict with keys:
      'acf_abs_m'        : np.ndarray (nlags+1,)
      'herd_fraction'    : float, fraction of steps where |m| > threshold
      'herd_threshold'   : float, the threshold used
      'mean_abs_m'       : float
    """
    abs_m = np.abs(m)
    return {
        "acf_abs_m":      acf(abs_m, nlags),
        "herd_fraction":  float(np.mean(abs_m > threshold)),
        "herd_threshold": threshold,
        "mean_abs_m":     float(abs_m.mean()),
    }


def tail_stats(
    results,        # List[SimResult]
    liq_threshold: float = 0.8,
    l0: float = 1.0,
) -> dict:
    """
    Ensemble tail-risk summary across a list of SimResult objects.

    Computes:
    - Max drawdown distribution (mean, std, 95th percentile)
    - Excess kurtosis of pooled returns
    - Fraction of runs where liquidity fell below liq_threshold * l0
    - Mean magnetization persistence (mean |m| across runs)

    Parameters
    ----------
    results       : list of SimResult (from run_ensemble)
    liq_threshold : fraction of l0 below which liquidity is 'stressed' (default 0.8)
    l0            : baseline liquidity for threshold scaling

    Returns
    -------
    dict with summary statistics
    """
    mdds          = np.array([max_drawdown(res.prices) for res in results])
    all_r         = np.concatenate([res.r for res in results])
    liq_stressed  = [float(np.any(res.l < liq_threshold * l0)) for res in results]
    mean_abs_m    = np.array([float(np.abs(res.m).mean()) for res in results])

    n             = len(results)
    kurt          = float(_kurtosis(all_r))

    return {
        "n_runs":               n,
        "mdd_mean":             float(mdds.mean()),
        "mdd_std":              float(mdds.std()),
        "mdd_p95":              float(np.percentile(mdds, 95)),
        "excess_kurtosis":      kurt,
        "liq_stressed_frac":    float(np.mean(liq_stressed)),
        "liq_threshold":        liq_threshold * l0,
        "mean_abs_m_mean":      float(mean_abs_m.mean()),
        "mean_abs_m_std":       float(mean_abs_m.std()),
    }


def _kurtosis(r: np.ndarray) -> float:
    """Fisher excess kurtosis (0 for normal distribution)."""
    n  = len(r)
    mu = r.mean()
    s  = r.std()
    if s == 0.0:
        return 0.0
    return float(np.mean(((r - mu) / s) ** 4)) - 3.0