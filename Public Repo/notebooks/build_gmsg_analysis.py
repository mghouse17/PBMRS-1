"""Build the committed, offline GMSG analysis caches.

Run from the repository root with the project virtual environment. The default
settings are the predeclared production settings used by the application
notebook; no reduced or presentation-only path is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pbmrs_core import SimConfig, check_invariants, load_config, run_sim
from pbmrs_core.calibration import (
    acf_r2_profile,
    compute_horizon_stats,
    load_npz_cache,
    save_npz_cache,
    sim_cache_key,
    wilson_interval,
)
from pbmrs_core.commodities import fetch_fred_series, read_manifest

BURN = 500
T = 144
NLAGS = 8
ALPHA = 0.10
NCAL = 1000
NNULL = 4999
N_STABILITY = 500
N_PSEUDO = 500
J_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
          0.70, 0.75, 0.78, 0.80, 0.82, 1.0 / 1.2, 0.85)
SIGMA_GRID = (0.010, 0.015, 0.020, 0.030, 0.037)
POWER_T = (144, 250, 500, 1000, 2000)
VARIANTS = ("full", "wti_april_mask", "brent_april_mask")

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks" / "data_cache"


def _code_hash() -> str:
    digest = hashlib.sha256()
    for rel in (
        "src/pbmrs_core/models.py", "src/pbmrs_core/math.py",
        "src/pbmrs_core/simulation.py", "src/pbmrs_core/calibration.py",
        "src/pbmrs_core/commodities.py", "configs/base.yaml",
    ):
        digest.update(rel.encode())
        digest.update((ROOT / rel).read_bytes())
    return digest.hexdigest()


def _base_config() -> SimConfig:
    values = load_config(ROOT / "configs" / "base.yaml")["simulation"]
    cfg = SimConfig(**values)
    if cfg.alpha_r != 12.0:
        raise RuntimeError("base.yaml must retain alpha_r=12.0")
    return cfg


def _data_and_masks():
    wti = fetch_fred_series(
        "DCOILWTICO", cache_dir=CACHE, start="2026-01-01", end="2026-07-31"
    )
    brent = fetch_fred_series(
        "DCOILBRENTEU", cache_dir=CACHE, start="2026-01-01", end="2026-07-31"
    )
    wti_r = wti.log_returns
    brent_r = brent.log_returns[-T:]
    brent_dates = brent.return_dates[-T:]
    if len(wti_r) != T or len(brent_r) != T:
        raise RuntimeError("both inference series must expose exactly 144 positions")
    wti_mask = np.ones(T, dtype=bool)
    brent_mask = np.ones(T, dtype=bool)
    wti_match = np.flatnonzero(wti.return_dates == np.datetime64("2026-04-08"))
    brent_match = np.flatnonzero(brent_dates == np.datetime64("2026-04-08"))
    if len(wti_match) != 1 or len(brent_match) != 1:
        raise RuntimeError("8 April return must have one position in each fixed grid")
    wti_mask[wti_match[0]] = False
    brent_mask[brent_match[0]] = False
    return wti, brent, wti_r, brent_r, wti_mask, brent_mask


def _profile_task(args):
    cfg, J, masks = args
    profiles = np.empty((len(masks), NCAL + NNULL, NLAGS), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(NCAL + NNULL):
            result = run_sim(dataclasses.replace(cfg, J=J, seed=seed, timesteps=BURN + T))
            sample = result.r[-T:]
            for m, mask in enumerate(masks):
                profiles[m, seed] = acf_r2_profile(sample, nlags=NLAGS, valid_mask=mask)
    return J, profiles


def _stability_task(args):
    cfg, J, sigma = args
    count = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(50_000, 50_000 + N_STABILITY):
            trial = dataclasses.replace(
                cfg, J=J, sigma_eps=sigma, seed=seed, timesteps=BURN + 2000
            )
            try:
                result = run_sim(trial)
                tail = result.r[-2000:]
                bad = (not np.all(np.isfinite(tail))) or float(np.std(tail)) > 5 * sigma
                check_invariants(result, trial)
            except (FloatingPointError, OverflowError, ValueError):
                bad = True
            count += int(bad)
    lo, hi = wilson_interval(count, N_STABILITY)
    return J, sigma, count, count / N_STABILITY, lo, hi


def _distance_setup(bank):
    cal, null = bank[:NCAL], bank[NCAL:]
    center = cal.mean(axis=0)
    scale = np.maximum(cal.std(axis=0, ddof=1), 1e-12)
    null_d = np.square((null - center) / scale).sum(axis=1)
    return center, scale, null_d


def _rejected(profile, center, scale, null_d):
    distance = float(np.square((profile - center) / scale).sum())
    p_value = (1 + int(np.count_nonzero(null_d >= distance))) / (1 + NNULL)
    return p_value < ALPHA


def _power_task(args):
    cfg, sample_length = args
    offset = POWER_T.index(sample_length)
    seed0 = 100_000 + offset * 20_000
    bank = np.empty((NCAL + NNULL, NLAGS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, seed in enumerate(range(seed0, seed0 + NCAL + NNULL)):
            result = run_sim(dataclasses.replace(
                cfg, J=0.40, seed=seed, timesteps=BURN + sample_length
            ))
            bank[i] = acf_r2_profile(result.r[-sample_length:], nlags=NLAGS)
        center, scale, null_d = _distance_setup(bank)
        rejected = 0
        for seed in range(seed0 + 10_000, seed0 + 10_000 + N_PSEUDO):
            result = run_sim(dataclasses.replace(
                cfg, J=0.78, seed=seed, timesteps=BURN + sample_length
            ))
            rejected += int(_rejected(
                acf_r2_profile(result.r[-sample_length:], nlags=NLAGS),
                center, scale, null_d,
            ))
    lo, hi = wilson_interval(rejected, N_PSEUDO)
    return sample_length, rejected, rejected / N_PSEUDO, lo, hi


def _control_task(args):
    cfg, J = args
    seed0 = 250_000 + round(J * 10_000)
    bank = np.empty((NCAL + NNULL, NLAGS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, seed in enumerate(range(seed0, seed0 + NCAL + NNULL)):
            result = run_sim(dataclasses.replace(cfg, J=J, seed=seed, timesteps=BURN + T))
            bank[i] = acf_r2_profile(result.r[-T:], nlags=NLAGS)
        center, scale, null_d = _distance_setup(bank)
        rejected = 0
        for seed in range(seed0 + 10_000, seed0 + 10_000 + N_PSEUDO):
            result = run_sim(dataclasses.replace(cfg, J=J, seed=seed, timesteps=BURN + T))
            rejected += int(_rejected(
                acf_r2_profile(result.r[-T:], nlags=NLAGS), center, scale, null_d
            ))
    lo, hi = wilson_interval(rejected, N_PSEUDO)
    return J, rejected, rejected / N_PSEUDO, lo, hi


def _horizon_task(args):
    cfg, J = args
    seed0 = 400_000 + round(J * 10_000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = [run_sim(dataclasses.replace(
            cfg, J=J, seed=seed, timesteps=BURN + 21
        )) for seed in range(seed0, seed0 + 500)]
    stats = compute_horizon_stats(results, burn=BURN, horizon=21, recovery_drawdown=0.05)
    return J, stats


def _marginal_task(args):
    cfg, J, sample_length = args
    seed0 = 500_000 + round(J * 10_000) + sample_length * 10
    values = np.empty((500, sample_length))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, seed in enumerate(range(seed0, seed0 + 500)):
            result = run_sim(dataclasses.replace(
                cfg, J=J, seed=seed, timesteps=BURN + sample_length
            ))
            values[i] = result.r[-sample_length:]
    return J, sample_length, values


def _run_pool(label, fn, items, workers):
    print(f"{label}: {len(items)} tasks with {workers} workers", flush=True)
    completed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for number, future in enumerate(as_completed(futures), 1):
            completed.append(future.result())
            print(f"{label}: {number}/{len(items)}", flush=True)
    return completed


def _needs_build(path: Path, key: str, force: bool) -> bool:
    return force or load_npz_cache(path, key) is None


def build(force: bool = False, workers: int | None = None) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = _base_config()
    _, _, wti_r, brent_r, wti_mask, brent_mask = _data_and_masks()
    workers = workers or min(12, os.cpu_count() or 1)
    manifest = read_manifest(CACHE)
    common = {
        "code": _code_hash(), "config": dataclasses.asdict(cfg),
        "data_manifest": manifest, "grid": J_GRID,
    }

    profile_path = CACHE / "gmsg_profiles.npz"
    profile_key = sim_cache_key(
        **common, phase="adequacy", burn=BURN, T=T, nlags=NLAGS,
        ncal=NCAL, nnull=NNULL, seeds=(0, NCAL + NNULL - 1),
        masks={"wti": wti_mask, "brent": brent_mask},
    )
    if _needs_build(profile_path, profile_key, force):
        masks = (None, wti_mask, brent_mask)
        output = _run_pool(
            "adequacy", _profile_task,
            [(cfg, J, masks) for J in J_GRID], workers,
        )
        output.sort(key=lambda x: J_GRID.index(x[0]))
        arrays = np.stack([x[1] for x in output])
        save_npz_cache(
            profile_path, profile_key,
            metadata={"variants": VARIANTS, "seed_ranges": {"calibration": [0, 999], "null": [1000, 5998]}},
            J=np.asarray(J_GRID), profiles=arrays,
            wti_returns=wti_r, brent_returns=brent_r,
            wti_mask=wti_mask, brent_mask=brent_mask,
        )

    stability_path = CACHE / "gmsg_stability.npz"
    stability_key = sim_cache_key(
        **common, phase="stability", burn=BURN, evaluation=2000,
        n_runs=N_STABILITY, sigma=SIGMA_GRID, seeds=(50_000, 50_499),
    )
    if _needs_build(stability_path, stability_key, force):
        output = _run_pool(
            "stability", _stability_task,
            [(cfg, J, sigma) for sigma in SIGMA_GRID for J in J_GRID], workers,
        )
        output.sort(key=lambda x: (SIGMA_GRID.index(x[1]), J_GRID.index(x[0])))
        save_npz_cache(
            stability_path, stability_key,
            metadata={"pathological": "non-finite, invariant-breaking, or return SD > 5*sigma_eps"},
            rows=np.asarray(output, dtype=float),
        )

    power_path = CACHE / "gmsg_power.npz"
    power_key = sim_cache_key(
        **common, phase="power", null_J=0.40, truth_J=0.78,
        lengths=POWER_T, n_pseudo=N_PSEUDO, ncal=NCAL, nnull=NNULL,
    )
    if _needs_build(power_path, power_key, force):
        output = _run_pool(
            "power", _power_task, [(cfg, length) for length in POWER_T],
            min(workers, len(POWER_T)),
        )
        output.sort(key=lambda x: POWER_T.index(x[0]))
        save_npz_cache(power_path, power_key, rows=np.asarray(output, dtype=float))

    controls_path = CACHE / "gmsg_controls.npz"
    controls_key = sim_cache_key(
        **common, phase="controls", points=(0.40, 0.78),
        n_pseudo=N_PSEUDO, ncal=NCAL, nnull=NNULL,
    )
    if _needs_build(controls_path, controls_key, force):
        output = _run_pool("controls", _control_task, [(cfg, J) for J in (0.40, 0.78)], 2)
        output.sort(key=lambda x: x[0])
        save_npz_cache(controls_path, controls_key, rows=np.asarray(output, dtype=float))

    horizon_path = CACHE / "gmsg_horizon.npz"
    horizon_key = sim_cache_key(
        **common, phase="horizon", points=(0.40, 0.78), burn=BURN,
        horizon=21, n_runs=500, recovery_drawdown=0.05,
    )
    if _needs_build(horizon_path, horizon_key, force):
        output = _run_pool("horizon", _horizon_task, [(cfg, J) for J in (0.40, 0.78)], 2)
        output.sort(key=lambda x: x[0])
        arrays = {"J": np.asarray([x[0] for x in output])}
        for name in (
            "max_drawdown", "terminal_log_return", "ending_drawdown",
            "recovery_steps", "qualifying_drawdown", "recovered",
        ):
            arrays[name] = np.stack([getattr(x[1], name) for x in output])
        save_npz_cache(horizon_path, horizon_key, **arrays)

    marginal_path = CACHE / "gmsg_marginals.npz"
    marginal_key = sim_cache_key(
        **common, phase="marginals", points=(0.40, 0.78),
        lengths=(144, 2000), burn=BURN, n_runs=500,
    )
    if _needs_build(marginal_path, marginal_key, force):
        items = [(cfg, J, length) for J in (0.40, 0.78) for length in (144, 2000)]
        output = _run_pool("marginals", _marginal_task, items, 4)
        arrays = {f"J{J:.2f}_T{length}": values for J, length, values in output}
        save_npz_cache(marginal_path, marginal_key, **arrays)

    ledger = {
        "schema_version": 1,
        "settings": {
            "burn": BURN, "positions": T, "nlags": NLAGS, "alpha": ALPHA,
            "n_calibration": NCAL, "n_null": NNULL, "n_stability": N_STABILITY,
            "n_pseudo": N_PSEUDO, "J_grid": list(J_GRID),
            "sigma_eps_grid": list(SIGMA_GRID), "power_lengths": list(POWER_T),
        },
        "cache_keys": {
            "gmsg_profiles.npz": profile_key, "gmsg_stability.npz": stability_key,
            "gmsg_power.npz": power_key, "gmsg_controls.npz": controls_key,
            "gmsg_horizon.npz": horizon_key, "gmsg_marginals.npz": marginal_key,
        },
        "seed_ledger": {
            "adequacy_calibration": [0, 999], "adequacy_null": [1000, 5998],
            "stability": [50000, 50499], "power": "100000-series, disjoint by T",
            "controls": "254000 and 257800 series", "horizon": "404000 and 407800 series",
            "marginals": "500000-series, disjoint by J and T",
        },
        "code_hash": common["code"],
    }
    (CACHE / "analysis_manifest.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("analysis caches complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    build(force=args.force, workers=args.workers)
