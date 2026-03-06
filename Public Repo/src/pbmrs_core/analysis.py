"""
analysis.py — PBMRS Phase-Space Analysis
Version: 0.2.2

Changes vs 0.2.1:
  [Perf-13] phase_map now accepts n_jobs parameter.
            n_jobs=1  → sequential (original behaviour, default)
            n_jobs=-1 → use all available CPUs
            n_jobs=k  → use k processes
            Uses concurrent.futures.ProcessPoolExecutor (stdlib, no new deps).
            Each grid cell is an independent work unit — embarrassingly parallel.

Public API
----------
phase_map   — run a 2-D grid over any two SimConfig parameters,
              return a dict of metric matrices ready to plot with imshow/contourf.
"""

from __future__ import annotations

import dataclasses
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np

from sim import SimConfig, run_ensemble
from diagnostics import (
    max_drawdown, recovery_time,
    acf_squared_returns, magnetization_persistence, tail_stats,
)


# ── Cell worker (must be top-level for pickling) ──────────────────────────────

def _run_cell(
    cfg_base:    SimConfig,
    param_x:     str,
    vx:          float,
    param_y:     str,
    vy:          float,
    ix:          int,
    iy:          int,
    n_runs:      int,
    seed_offset: int,
    metrics:     List[str],
    l0:          float,
) -> Tuple[int, int, Dict[str, float]]:
    """
    Worker for one (ix, iy) grid cell. Top-level so ProcessPoolExecutor can pickle it.
    Returns (ix, iy, scalar_metrics_dict).
    """
    cfg = dataclasses.replace(cfg_base, **{param_x: vx, param_y: vy})
    seeds = list(range(seed_offset, seed_offset + n_runs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress supercritical warnings in workers
        results = run_ensemble(cfg, n_runs=n_runs, seeds=seeds)

    cell: Dict[str, float] = {}

    if "mdd_mean" in metrics or "mdd_p95" in metrics:
        mdds = np.array([max_drawdown(r.prices) for r in results])
        if "mdd_mean" in metrics:
            cell["mdd_mean"] = float(mdds.mean())
        if "mdd_p95" in metrics:
            cell["mdd_p95"]  = float(np.percentile(mdds, 95))

    if "recovery_mean" in metrics:
        times = [recovery_time(r.prices) for r in results]
        finite = [t for t in times if t is not None]
        cell["recovery_mean"] = float(np.mean(finite)) if finite else float("nan")

    if "excess_kurtosis" in metrics:
        all_r = np.concatenate([r.r for r in results])
        std   = all_r.std()
        cell["excess_kurtosis"] = float(
            np.mean(((all_r - all_r.mean()) / std) ** 4) - 3.0
        ) if std > 0 else 0.0

    if "liq_stressed_frac" in metrics:
        threshold = 0.8 * l0
        stressed  = [float(np.any(r.l < threshold)) for r in results]
        cell["liq_stressed_frac"] = float(np.mean(stressed))

    if "herd_fraction" in metrics or "mean_abs_m" in metrics:
        abs_m_means = np.array([float(np.abs(r.m).mean()) for r in results])
        herd_fracs  = np.array([float(np.mean(np.abs(r.m) > 0.3)) for r in results])
        if "herd_fraction" in metrics:
            cell["herd_fraction"] = float(herd_fracs.mean())
        if "mean_abs_m" in metrics:
            cell["mean_abs_m"]    = float(abs_m_means.mean())

    if "acf_r2_lag1" in metrics:
        acf_lag1s = [acf_squared_returns(r.r, nlags=1)[1] for r in results]
        cell["acf_r2_lag1"] = float(np.mean(acf_lag1s))

    return ix, iy, cell


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
    n_jobs:      int = 1,
) -> Dict[str, Any]:
    """
    Run a 2-D parameter grid and return metric matrices.

    Parameters
    ----------
    cfg_base    : base SimConfig; the two swept parameters are overridden per cell.
    param_x     : name of the x-axis SimConfig field  (e.g. "J")
    values_x    : list of values for the x-axis parameter
    param_y     : name of the y-axis SimConfig field  (e.g. "gamma_v")
    values_y    : list of values for the y-axis parameter
    n_runs      : independent seeds per grid cell
    metrics     : subset of metric names to compute (default: all)
    seed_offset : base seed for all ensemble runs
    n_jobs      : [Perf-13] number of parallel worker processes.
                  1  = sequential (safe in notebooks, default)
                  -1 = os.cpu_count() processes
                  k  = exactly k processes
                  Note: use n_jobs=1 inside Jupyter to avoid fork issues on macOS.
                  Use n_jobs=-1 from scripts or after 'if __name__ == "__main__"'.

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
            raise ValueError(
                f"Unknown metric '{m}'. Valid: {ALL_METRICS}"
            )

    nx, ny      = len(values_x), len(values_y)
    total_cells = nx * ny
    matrices    = {m: np.full((ny, nx), np.nan) for m in metrics}

    # Resolve n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, n_jobs)

    l0 = cfg_base.l0

    # Build list of all (ix, iy, vx, vy) work items
    work_items = [
        (ix, iy, values_x[ix], values_y[iy])
        for iy in range(ny)
        for ix in range(nx)
    ]

    print(f"phase_map: {ny}×{nx} grid | {n_runs} runs/cell | "
          f"{total_cells} cells | n_jobs={n_jobs}")

    if n_jobs == 1:
        # ── Sequential path (safe in all environments) ────────────────────────
        for i, (ix, iy, vx, vy) in enumerate(work_items):
            if i % max(1, total_cells // 10) == 0:
                print(f"  cell {i+1}/{total_cells} …", end="\r", flush=True)
            _, _, cell = _run_cell(
                cfg_base, param_x, vx, param_y, vy,
                ix, iy, n_runs, seed_offset, metrics, l0,
            )
            for m, v in cell.items():
                matrices[m][iy, ix] = v

    else:
        # ── Parallel path (ProcessPoolExecutor) ───────────────────────────────
        # Each future maps to one grid cell. Results arrive out of order;
        # (ix, iy) from the return value places them correctly in matrices.
        futures = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for ix, iy, vx, vy in work_items:
                fut = pool.submit(
                    _run_cell,
                    cfg_base, param_x, vx, param_y, vy,
                    ix, iy, n_runs, seed_offset, metrics, l0,
                )
                futures[fut] = (ix, iy)

            completed = 0
            for fut in as_completed(futures):
                completed += 1
                if completed % max(1, total_cells // 10) == 0:
                    print(f"  {completed}/{total_cells} cells done …",
                          end="\r", flush=True)
                ix, iy, cell = fut.result()   # raises if worker raised
                for m, v in cell.items():
                    matrices[m][iy, ix] = v

    print(f"\n  Done: {total_cells} cells × {n_runs} runs = "
          f"{total_cells * n_runs:,} simulations")

    return {
        "param_x":  param_x,
        "values_x": values_x,
        "param_y":  param_y,
        "values_y": values_y,
        "n_runs":   n_runs,
        "cfg_base": cfg_base,
        **matrices,
    }