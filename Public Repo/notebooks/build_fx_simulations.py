"""Compute FX caches with the unchanged canonical simulator; resume per task."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pbmrs_core import SimConfig, check_invariants, load_config, run_sim
from pbmrs_core.calibration import acf_r2_profile, compute_horizon_stats, load_npz_cache, save_npz_cache, sim_cache_key, wilson_interval
from pbmrs_core.fx import fetch_fx, validate_raw_tree
from pbmrs_core.fx_analysis import excess_kurtosis, solve_scale

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks/fx_cache"


def source_hash():
    digest = hashlib.sha256()
    for rel in ("src/pbmrs_core/models.py", "src/pbmrs_core/math.py", "src/pbmrs_core/simulation.py",
                "src/pbmrs_core/calibration.py", "src/pbmrs_core/config.py", "configs/base.yaml",
                "notebooks/build_fx_simulations.py"):
        digest.update(rel.encode())
        digest.update((ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _task(args):
    cfg, seeds, burn, lengths, mode = args
    profiles = {str(n): [] for n in lengths}
    bad, moments, horizon = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in seeds:
            trial = replace(cfg, seed=seed, timesteps=burn+max(lengths))
            result = run_sim(trial)
            sample = result.r[burn:]
            if mode == "stability":
                pathological = not np.all(np.isfinite(sample)) or np.std(sample) > 5*cfg.sigma_eps
                try:
                    check_invariants(result, trial)
                except ValueError:
                    pathological = True
                bad.append(int(pathological))
            elif mode == "measure":
                moments.append(float(sample.std(ddof=1)))
            elif mode == "horizon":
                stats = compute_horizon_stats([result], burn=burn, horizon=21)
                horizon.append([float(getattr(stats, name)[0]) for name in (
                    "max_drawdown", "terminal_log_return", "ending_drawdown",
                    "recovery_steps", "qualifying_drawdown", "recovered")])
            else:
                # One post-burn path supplies nested horizons. CRN dependence is
                # explicit; no inference combines them as independent samples.
                for n in lengths:
                    profiles[str(n)].append(acf_r2_profile(result.r[burn:burn+n], nlags=8))
                moments.append([float(sample.std(ddof=1)), excess_kurtosis(sample)])
    if mode == "stability":
        return {"bad": np.asarray(bad)}
    if mode == "measure":
        return {"sd": np.asarray(moments)}
    if mode == "horizon":
        return {"horizon": np.asarray(horizon)}
    return {**{f"T{n}": np.asarray(a) for n, a in profiles.items()}, "moments": np.asarray(moments)}


def chunks(pool, cfg, seed_range, burn, lengths, mode):
    seeds = list(range(seed_range[0], seed_range[1]+1))
    tasks = [(cfg, seeds[i:i+25], burn, lengths, mode) for i in range(0, len(seeds), 25)]
    pieces = list(pool.map(_task, tasks))
    return {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]}


def build(workers=12):
    frozen = json.loads((CACHE / "preregistration.json").read_text())
    d = frozen["design"]
    raw = validate_raw_tree(CACHE / "raw")
    cfg = SimConfig(**load_config(ROOT / "configs/base.yaml")["simulation"])
    assert cfg.alpha_r == 12.0
    common = {"code_hash": source_hash(), "design_sha256": frozen["design_sha256"],
              "config": asdict(cfg), "raw_hashes": {k: v["sha256"] for k, v in raw.items()}}
    root_key = sim_cache_key(**common)
    state_path = CACHE / "simulation_manifest.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["root_key"] != root_key:
            raise RuntimeError("simulation inputs changed; existing results must not be silently reused")
    else:
        state = {**common, "root_key": root_key, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "scale": {}, "stability": {}, "eligible": {}, "cache_keys": {}, "cache_sha256": {}}

    def persist():
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def cached(name, config, seeds, lengths, mode, pool):
        key = sim_cache_key(root=root_key, cfg=asdict(config), seeds=seeds, lengths=lengths, mode=mode)
        path = CACHE / f"{name}.npz"
        data = load_npz_cache(path, key)
        if data is None:
            print(f"computing {name}", flush=True)
            data = chunks(pool, config, seeds, d["burn"], lengths, mode)
            save_npz_cache(path, key, **data)
        state["cache_keys"][path.name] = key
        state["cache_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        persist()
        return data

    with ProcessPoolExecutor(max_workers=min(workers, os.cpu_count() or 1)) as pool:
        for pair in ("JPY", "EUR"):
            series = fetch_fx(pair, cache_dir=CACHE / "raw", start=d["start"], end=d["end"])
            train = series.log_returns[series.return_dates < np.datetime64(d["split"])]
            if pair not in state["scale"]:
                def measure(sigma):
                    result = chunks(pool, replace(cfg, J=d["sigma_reference_J"], sigma_eps=sigma),
                                    d["seeds"]["solver"], d["burn"], [2000], "measure")
                    sd = float(result["sd"].mean())
                    print(f"{pair} sigma={sigma:.8f}, total SD={sd:.8f}", flush=True)
                    return sd
                state["scale"][pair] = solve_scale(float(train.std(ddof=1)), measure,
                                                   rtol=d["sigma_solver_relative_tolerance"])
                state["scale"][pair].update(training_n=len(train), training_end=str(series.return_dates[len(train)-1]),
                                            reference_J=d["sigma_reference_J"])
                persist()
            pair_cfg = replace(cfg, sigma_eps=state["scale"][pair]["sigma_eps"])
            stable = []
            for J in d["J_grid"]:
                values = cached(f"{pair}_stability_J{J:.8f}", replace(pair_cfg, J=J),
                                d["seeds"]["stability"], [d["stability_length"]], "stability", pool)
                count = int(values["bad"].sum())
                lo, hi = wilson_interval(count, d["n_stability"])
                stable.append({"J": J, "count": count, "n": d["n_stability"],
                               "fraction": count/d["n_stability"], "lo": lo, "hi": hi})
                print(pair, "stability", J, count, flush=True)
            state["stability"][pair] = stable
            eligible = [r["J"] for r in stable if r["fraction"] <= d["eligibility_max_rate"]]
            state["eligible"][pair] = eligible
            if 0.4 not in eligible:
                raise RuntimeError("reference J failed FX stability screen")
            comparison = 0.78 if 0.78 in eligible else max(eligible)
            state.setdefault("comparison_J", {})[pair] = comparison
            persist()
            for J in eligible:
                lengths = d["power_lengths"] if J in (0.4, comparison) else d["windows"]
                cached(f"{pair}_bank_J{J:.8f}", replace(pair_cfg, J=J),
                       [d["seeds"]["calibration"][0], d["seeds"]["null"][1]], lengths, "profiles", pool)
            for J in (0.4, comparison):
                cached(f"{pair}_pseudo_J{J:.8f}", replace(pair_cfg, J=J),
                       d["seeds"]["pseudo"], d["power_lengths"], "profiles", pool)
                cached(f"{pair}_horizon_J{J:.8f}", replace(pair_cfg, J=J),
                       d["seeds"]["horizon"], [21], "horizon", pool)
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist()
    print("FX simulations complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    build(parser.parse_args().workers)
