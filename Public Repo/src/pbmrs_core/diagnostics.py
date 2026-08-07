from __future__ import annotations

import numpy as np


def drawdown(prices: np.ndarray) -> np.ndarray:
    """Compute fractional drawdown for a price path."""
    peak = np.maximum.accumulate(prices)
    denom = np.where(peak == 0, 1.0, peak)
    return 1.0 - (prices / denom)


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum drawdown over the full path."""
    return float(np.max(drawdown(prices)))


def recovery_time(prices: np.ndarray, epsilon: float = 0.0) -> int | None:
    """Steps from trough to recovery, or None if the path never recovers."""
    dd = drawdown(prices)
    t_trough = int(np.argmax(dd))
    ref_peak = float(np.max(prices[: t_trough + 1]))
    target = (1.0 - epsilon) * ref_peak
    for t in range(t_trough + 1, len(prices)):
        if prices[t] >= target:
            return t - t_trough
    return None


def fragility_index(m: np.ndarray, l: np.ndarray, v: np.ndarray, l0: float, theta_v: float) -> np.ndarray:
    """Composite fragility index from crowding, liquidity stress, and volatility."""
    return np.abs(m) + (l0 / l) + (v / theta_v)


def regime_labels(F: np.ndarray, p_stable: float = 33.0, p_unstable: float = 66.0) -> np.ndarray:
    """Classify periods as stable, fragile, or unstable."""
    lo = np.percentile(F, p_stable)
    hi = np.percentile(F, p_unstable)
    return np.where(F < lo, 0, np.where(F < hi, 1, 2))


def acf(series: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Sample autocorrelation function of a 1-D array."""
    n = len(series)
    s = series - series.mean()
    var = float(np.dot(s, s))
    if var == 0.0:
        return np.zeros(nlags + 1)
    result = np.empty(nlags + 1)
    for k in range(nlags + 1):
        result[k] = float(np.dot(s[: n - k], s[k:])) / var
    return result


def acf_squared_returns(r: np.ndarray, nlags: int = 40) -> np.ndarray:
    """ACF of squared returns for volatility clustering diagnostics."""
    return acf(r**2, nlags)


def magnetization_persistence(m: np.ndarray, threshold: float = 0.3, nlags: int = 40) -> dict:
    """Summarize persistence of market herding and crowding."""
    abs_m = np.abs(m)
    return {
        "acf_abs_m": acf(abs_m, nlags),
        "herd_fraction": float(np.mean(abs_m > threshold)),
        "herd_threshold": threshold,
        "mean_abs_m": float(abs_m.mean()),
    }


def tail_stats(results, liq_threshold: float = 0.8, l0: float = 1.0) -> dict:
    """Ensemble tail-risk summary."""
    mdds = np.array([max_drawdown(res.prices) for res in results])
    all_r = np.concatenate([res.r for res in results])
    liq_stressed = [float(np.any(res.l < liq_threshold * l0)) for res in results]
    mean_abs_m = np.array([float(np.abs(res.m).mean()) for res in results])
    kurt = float(_kurtosis(all_r))
    return {
        "n_runs": len(results),
        "mdd_mean": float(mdds.mean()),
        "mdd_std": float(mdds.std()),
        "mdd_p95": float(np.percentile(mdds, 95)),
        "excess_kurtosis": kurt,
        "liq_stressed_frac": float(np.mean(liq_stressed)),
        "liq_threshold": liq_threshold * l0,
        "mean_abs_m_mean": float(mean_abs_m.mean()),
        "mean_abs_m_std": float(mean_abs_m.std()),
    }


def _kurtosis(r: np.ndarray) -> float:
    """Fisher excess kurtosis."""
    mu = r.mean()
    s = r.std()
    if s == 0.0:
        return 0.0
    return float(np.mean(((r - mu) / s) ** 4)) - 3.0