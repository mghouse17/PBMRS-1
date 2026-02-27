from __future__ import annotations
import numpy as np


def drawdown(x: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(x)
    # avoid divide-by-zero; if peak==0, drawdown definition depends on convention
    denom = np.where(peak == 0, 1.0, peak)
    return 1.0 - (x / denom)


def max_drawdown(x: np.ndarray) -> float:
    return float(np.max(drawdown(x)))


def recovery_time(x: np.ndarray) -> int | None:
    """
    Returns the number of steps from the max drawdown trough to recovery above prior peak.
    If never recovers, return None.
    """
    peak = np.maximum.accumulate(x)
    dd = 1.0 - (x / np.where(peak == 0, 1.0, peak))
    t_trough = int(np.argmax(dd))

    # peak right before trough (the reference level to recover)
    ref_peak = float(np.max(x[: t_trough + 1]))

    for t in range(t_trough + 1, len(x)):
        if x[t] >= ref_peak:
            return t - t_trough
    return None