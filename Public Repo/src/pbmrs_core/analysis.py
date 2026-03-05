"""
analysis.py — PBMRS Phase-Space Analysis

Tools for sweeping parameter pairs and mapping system behaviour.

Public API
----------
phase_map   — run a 2-D grid over any two SimConfig parameters,
              return a dict of metric matrices ready to plot with imshow/contourf.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sim         import SimConfig, SimResult, run_ensemble
from .diagnostics import (
    max_drawdown, recovery_time,
    acf_squared_returns, magnetization_persistence, tail_stats,
)


# ── phase_map ─────────────────────────────────────────────────────────────────

def phase_map(
    cfg_base:    SimConfig,
    param_x:     str,
    values_x:    List[float],
    param_y:     str,
    values_y:    List[float],
    n_runs:      int = 20,
    metrics:     Optional[List[str]] = None,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """
    Run a 2-D parameter grid and return metric matrices.

    For each (param_x, param_y) combination, run_ensemble is called with
    n_runs independent seeds.  Results are summarised into scalar metrics
    and stored in (len_y × len_x) matrices — i.e. `result['mdd_mean'][iy, ix]`
    corresponds to `(values_x[ix], values_y[iy])`.

    Parameters
    ----------
    cfg_base   : base SimConfig; the two swept parameters are overridden per cell.
    param_x    : name of the x-axis SimConfig field  (e.g. "J")
    values_x   : list of values for the x-axis parameter
    param_y    : name of the y-axis SimConfig field  (e.g. "gamma_v")
    values_y   : list of values for the y-axis parameter
    n_runs     : independent seeds per grid cell (more → smoother metrics)
    metrics    : subset of metric names to compute (default: all)
    seed_offset: base seed for all ensemble runs

    Returns
    -------
    dict with keys:
      'param_x', 'values_x'       — x-axis metadata
      'param_y', 'values_y'       — y-axis metadata
      'n_runs', 'cfg_base'        — run metadata
      For each metric name → np.ndarray of shape (len_y, len_x)

    Available metrics
    -----------------
    'mdd_mean'          — mean max drawdown across n_runs
    'mdd_p95'           — 95th-percentile max drawdown
    'recovery_mean'     — mean recovery time (steps; NaN if no recovery)
    'excess_kurtosis'   — pooled-return excess kurtosis
    'liq_stressed_frac' — fraction of runs where l fell below 0.8 * l0
    'herd_fraction'     — fraction of steps where |m| > 0.3
    'acf_r2_lag1'       — ACF(r²) at lag 1 (volatility-clustering signal)
    'mean_abs_m'        — mean |m| across n_runs
    """
    ALL_METRICS = [
        "mdd_mean", "mdd_p95", "recovery_mean",
        "excess_kurtosis", "liq_stressed_frac",
        "herd_fraction", "acf_r2_lag1", "mean_abs_m",
    ]
    if metrics is None:
        metrics = ALL_METRICS
    for m in metrics:
        if m not in ALL_METRICS:
            raise ValueError(f"Unknown metric '{m}'. Valid: {ALL_METRICS}")

    nx, ny = len(values_x), len(values_y)

    # Pre-allocate result matrices
    matrices: Dict[str, np.ndarray] = {
        m: np.full((ny, nx), np.nan) for m in metrics
    }

    total_cells = nx * ny
    cell = 0

    for iy, vy in enumerate(values_y):
        for ix, vx in enumerate(values_x):
            cell += 1
            print(f"  [{cell:3d}/{total_cells}]  {param_x}={vx:.4g}  {param_y}={vy:.4g}",
                  end="\r", flush=True)

            # Build config for this cell
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                cfg_cell = dataclasses.replace(
                    cfg_base,
                    **{param_x: vx, param_y: vy}
                )

            # Seeds: unique per cell so ensemble runs don't repeat
            cell_seed_base = seed_offset + (iy * nx + ix) * n_runs
            seeds = list(range(cell_seed_base, cell_seed_base + n_runs))

            results = run_ensemble(cfg_cell, n_runs=n_runs, seeds=seeds)

            # ── Compute metrics ──────────────────────────────────────────────

            if "mdd_mean" in metrics or "mdd_p95" in metrics:
                mdds = np.array([max_drawdown(res.prices) for res in results])
                if "mdd_mean" in metrics:
                    matrices["mdd_mean"][iy, ix] = float(mdds.mean())
                if "mdd_p95" in metrics:
                    matrices["mdd_p95"][iy, ix]  = float(np.percentile(mdds, 95))

            if "recovery_mean" in metrics:
                recs = [recovery_time(res.prices) for res in results]
                valid = [r for r in recs if r is not None]
                matrices["recovery_mean"][iy, ix] = (
                    float(np.mean(valid)) if valid else np.nan
                )

            if "excess_kurtosis" in metrics:
                all_r = np.concatenate([res.r for res in results])
                mu, s = all_r.mean(), all_r.std()
                kurt  = float(np.mean(((all_r - mu) / s) ** 4)) - 3.0 if s > 0 else 0.0
                matrices["excess_kurtosis"][iy, ix] = kurt

            if "liq_stressed_frac" in metrics:
                liq_floor = 0.8 * cfg_cell.l0
                stressed  = [float(np.any(res.l < liq_floor)) for res in results]
                matrices["liq_stressed_frac"][iy, ix] = float(np.mean(stressed))

            if "herd_fraction" in metrics or "mean_abs_m" in metrics:
                abs_m_means = np.array([float(np.abs(res.m).mean()) for res in results])
                herd_fracs  = np.array([
                    float(np.mean(np.abs(res.m) > 0.3)) for res in results
                ])
                if "herd_fraction" in metrics:
                    matrices["herd_fraction"][iy, ix] = float(herd_fracs.mean())
                if "mean_abs_m" in metrics:
                    matrices["mean_abs_m"][iy, ix]    = float(abs_m_means.mean())

            if "acf_r2_lag1" in metrics:
                acf_lag1s = []
                for res in results:
                    ac = acf_squared_returns(res.r, nlags=1)
                    acf_lag1s.append(ac[1])
                matrices["acf_r2_lag1"][iy, ix] = float(np.mean(acf_lag1s))

    print(f"\n  Done: {total_cells} cells × {n_runs} runs each = "
          f"{total_cells * n_runs} simulations")

    return {
        "param_x":  param_x,
        "values_x": values_x,
        "param_y":  param_y,
        "values_y": values_y,
        "n_runs":   n_runs,
        "cfg_base": cfg_base,
        **matrices,
    }