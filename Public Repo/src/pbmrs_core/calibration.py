"""Finite-sample PBMRS calibration and horizon-analysis utilities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import max_drawdown, recovery_time
from .models import SimConfig, SimResult
from .simulation import check_invariants, run_ensemble


@dataclass(frozen=True)
class AdequacyRow:
    J: float
    J_beta: float
    distance: float
    p_value: float
    mc_lo: float
    mc_hi: float
    decision: str
    borderline: bool
    statistic: str
    n_cal: int
    n_null: int
    valid_n: int
    variant: str


@dataclass(frozen=True)
class AdequacyResult:
    rows: tuple[AdequacyRow, ...]
    tested_grid: tuple[float, ...]
    non_rejected: tuple[float, ...]
    alpha: float
    statistic: str
    variant: str
    config_hash: str
    mask_indices: tuple[int, ...]
    calibration_seeds: tuple[int, int]
    null_seeds: tuple[int, int]


@dataclass(frozen=True)
class StabilityRow:
    J: float
    J_beta: float
    sigma_eps: float
    n_runs: int
    pathological_count: int
    pathological_fraction: float
    lo: float
    hi: float


@dataclass(frozen=True)
class PowerRow:
    sample_length: int
    n_pseudo: int
    rejected: int
    power: float
    lo: float
    hi: float


@dataclass(frozen=True)
class HorizonStats:
    horizon: int
    recovery_drawdown: float
    max_drawdown: np.ndarray
    terminal_log_return: np.ndarray
    ending_drawdown: np.ndarray
    recovery_steps: np.ndarray
    qualifying_drawdown: np.ndarray
    recovered: np.ndarray

    def summary(self) -> dict[str, float]:
        qualifying = self.qualifying_drawdown
        recovery_rate = float(self.recovered[qualifying].mean()) if np.any(qualifying) else float("nan")
        return {
            "n_runs": float(len(self.max_drawdown)),
            "mdd_mean": float(self.max_drawdown.mean()),
            "mdd_p95": float(np.percentile(self.max_drawdown, 95)),
            "terminal_return_mean": float(self.terminal_log_return.mean()),
            "ending_drawdown_mean": float(self.ending_drawdown.mean()),
            "qualifying_paths": float(np.count_nonzero(qualifying)),
            "recovery_rate": recovery_rate,
        }


def _valid_mask(values: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if valid_mask is None:
        mask = np.ones(len(values), dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must have the same shape as returns")
    if np.count_nonzero(mask) < 3 or not np.all(np.isfinite(values[mask])):
        raise ValueError("valid returns must contain at least three finite observations")
    return mask


def acf_r2_profile(
    returns: Sequence[float],
    *,
    nlags: int = 8,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """ACF of demeaned squared returns while preserving masked grid positions."""

    values = np.asarray(returns, dtype=float)
    mask = _valid_mask(values, valid_mask)
    if nlags < 1 or nlags >= len(values):
        raise ValueError("nlags must be between one and len(returns)-1")
    centered = np.full(values.shape, np.nan)
    centered[mask] = values[mask] - values[mask].mean()
    amplitude = float(np.max(np.abs(centered[mask])))
    if not np.isfinite(amplitude) or amplitude == 0.0:
        raise ValueError("squared returns have zero or invalid variance")
    centered[mask] /= amplitude
    squared = centered**2
    squared[mask] -= squared[mask].mean()
    denominator = float(np.dot(squared[mask], squared[mask]))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("squared returns have zero or invalid variance")
    result = np.empty(nlags, dtype=float)
    for lag in range(1, nlags + 1):
        pairs = mask[:-lag] & mask[lag:]
        if not np.any(pairs):
            raise ValueError(f"no valid pairs at lag {lag}")
        result[lag - 1] = float(
            np.dot(squared[:-lag][pairs], squared[lag:][pairs]) / denominator
        )
    return result


def lag_contributions(
    returns: Sequence[float],
    lag: int,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    mask = _valid_mask(values, valid_mask)
    if lag < 1 or lag >= len(values):
        raise ValueError("invalid lag")
    centered = np.full(values.shape, np.nan)
    centered[mask] = values[mask] - values[mask].mean()
    squared = centered**2
    squared[mask] -= squared[mask].mean()
    out = np.full(len(values) - lag, np.nan)
    pairs = mask[:-lag] & mask[lag:]
    products = squared[:-lag][pairs] * squared[lag:][pairs]
    total = products.sum()
    out[pairs] = products / total if total != 0 else products
    return out


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("invalid binomial counts")
    if confidence != 0.95:
        raise ValueError("only the predeclared 95% interval is supported")
    z = 1.959963984540054
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _center_scale(cal: np.ndarray, statistic: str) -> tuple[np.ndarray, np.ndarray]:
    if statistic in {"D_sum", "D_max"}:
        center = cal.mean(axis=0)
        scale = cal.std(axis=0, ddof=1)
    elif statistic == "robust":
        center = np.median(cal, axis=0)
        scale = np.percentile(cal, 75, axis=0) - np.percentile(cal, 25, axis=0)
    else:
        raise ValueError("statistic must be D_sum, D_max, or robust")
    return center, np.maximum(scale, 1e-12)


def _distances(values: np.ndarray, center: np.ndarray, scale: np.ndarray, statistic: str) -> np.ndarray:
    z = np.abs((values - center) / scale)
    return z.max(axis=-1) if statistic == "D_max" else np.square(z).sum(axis=-1)


def evaluate_profile_bank(
    empirical_profile: np.ndarray,
    profiles: np.ndarray,
    *,
    J: float,
    beta: float,
    n_cal: int,
    alpha: float = 0.10,
    statistic: str = "D_sum",
    valid_n: int,
    variant: str,
) -> AdequacyRow:
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2 or len(values) <= n_cal:
        raise ValueError("profiles must contain disjoint calibration and null rows")
    cal, null = values[:n_cal], values[n_cal:]
    center, scale = _center_scale(cal, statistic)
    empirical_distance = float(_distances(np.asarray(empirical_profile)[None, :], center, scale, statistic)[0])
    null_distances = _distances(null, center, scale, statistic)
    exceedances = int(np.count_nonzero(null_distances >= empirical_distance))
    p_value = (1 + exceedances) / (1 + len(null))
    lo, hi = wilson_interval(exceedances, len(null))
    return AdequacyRow(
        J=float(J),
        J_beta=float(J * beta),
        distance=empirical_distance,
        p_value=p_value,
        mc_lo=lo,
        mc_hi=hi,
        decision="non_rejected" if p_value >= alpha else "rejected",
        borderline=lo <= alpha <= hi,
        statistic=statistic,
        n_cal=n_cal,
        n_null=len(null),
        valid_n=valid_n,
        variant=variant,
    )


def invert_j_grid(
    cfg_base: SimConfig,
    j_grid: Sequence[float],
    empirical_returns: Sequence[float],
    *,
    valid_masks: Mapping[str, np.ndarray | None] | None = None,
    burn: int = 500,
    sample_length: int = 144,
    nlags: int = 8,
    n_cal: int = 1000,
    n_null: int = 4999,
    alpha: float = 0.10,
    statistic: str = "D_sum",
    calibration_seed_start: int = 0,
) -> dict[str, AdequacyResult]:
    empirical = np.asarray(empirical_returns, dtype=float)
    if len(empirical) != sample_length:
        raise ValueError("empirical return length must equal sample_length")
    variants = dict(valid_masks or {"full": None})
    masks = {name: _valid_mask(empirical, mask) for name, mask in variants.items()}
    empirical_profiles = {
        name: acf_r2_profile(empirical, nlags=nlags, valid_mask=mask)
        for name, mask in masks.items()
    }
    rows: dict[str, list[AdequacyRow]] = {name: [] for name in masks}
    total = n_cal + n_null
    seeds = list(range(calibration_seed_start, calibration_seed_start + total))
    for J in j_grid:
        simulations = run_ensemble(
            replace(cfg_base, J=float(J), timesteps=burn + sample_length),
            n_runs=total,
            seeds=seeds,
        )
        returns = [result.r[-sample_length:] for result in simulations]
        for name, mask in masks.items():
            bank = np.asarray(
                [acf_r2_profile(r, nlags=nlags, valid_mask=mask) for r in returns]
            )
            rows[name].append(
                evaluate_profile_bank(
                    empirical_profiles[name],
                    bank,
                    J=float(J),
                    beta=cfg_base.beta,
                    n_cal=n_cal,
                    alpha=alpha,
                    statistic=statistic,
                    valid_n=int(mask.sum()),
                    variant=name,
                )
            )
    config_hash = sim_cache_key(config=asdict(cfg_base))
    results = {}
    for name, values in rows.items():
        results[name] = AdequacyResult(
            rows=tuple(values),
            tested_grid=tuple(float(x) for x in j_grid),
            non_rejected=tuple(row.J for row in values if row.decision == "non_rejected"),
            alpha=alpha,
            statistic=statistic,
            variant=name,
            config_hash=config_hash,
            mask_indices=tuple(int(x) for x in np.flatnonzero(~masks[name])),
            calibration_seeds=(seeds[0], seeds[n_cal - 1]),
            null_seeds=(seeds[n_cal], seeds[-1]),
        )
    return results


def assess_stability(
    cfg_base: SimConfig,
    j_grid: Sequence[float],
    sigma_values: Sequence[float],
    *,
    burn: int = 500,
    evaluation_length: int = 2000,
    n_runs: int = 500,
    seed_start: int = 50_000,
) -> tuple[StabilityRow, ...]:
    rows = []
    seeds = list(range(seed_start, seed_start + n_runs))
    for sigma in sigma_values:
        for J in j_grid:
            cfg = replace(cfg_base, J=float(J), sigma_eps=float(sigma), timesteps=burn + evaluation_length)
            pathological = 0
            for result in run_ensemble(cfg, n_runs=n_runs, seeds=seeds):
                tail = result.r[-evaluation_length:]
                bad = not np.all(np.isfinite(tail)) or float(tail.std()) > 5 * sigma
                try:
                    check_invariants(result, cfg)
                except ValueError:
                    bad = True
                pathological += int(bad)
            lo, hi = wilson_interval(pathological, n_runs)
            rows.append(
                StabilityRow(
                    float(J), float(J * cfg_base.beta), float(sigma), n_runs,
                    pathological, pathological / n_runs, lo, hi,
                )
            )
    return tuple(rows)


def compute_horizon_stats(
    results: Sequence[SimResult],
    *,
    burn: int = 500,
    horizon: int = 21,
    recovery_drawdown: float = 0.05,
) -> HorizonStats:
    if not results or burn < 0 or horizon < 1 or not 0 < recovery_drawdown < 1:
        raise ValueError("invalid horizon-statistic inputs")
    mdds, terminals, endings, recoveries, qualifies, recovered = [], [], [], [], [], []
    for result in results:
        if len(result.r) < burn + horizon:
            raise ValueError("simulation is shorter than burn plus horizon")
        r = np.asarray(result.r[burn : burn + horizon], dtype=float)
        prices = np.exp(np.r_[0.0, np.cumsum(r)])
        dd = 1.0 - prices / np.maximum.accumulate(prices)
        mdd = max_drawdown(prices)
        recovery = recovery_time(prices)
        qualifies.append(mdd >= recovery_drawdown)
        recovered.append(mdd >= recovery_drawdown and recovery is not None)
        recoveries.append(float(recovery) if recovery is not None else np.nan)
        mdds.append(mdd)
        terminals.append(float(r.sum()))
        endings.append(float(dd[-1]))
    return HorizonStats(
        horizon=horizon,
        recovery_drawdown=recovery_drawdown,
        max_drawdown=np.asarray(mdds),
        terminal_log_return=np.asarray(terminals),
        ending_drawdown=np.asarray(endings),
        recovery_steps=np.asarray(recoveries),
        qualifying_drawdown=np.asarray(qualifies, dtype=bool),
        recovered=np.asarray(recovered, dtype=bool),
    )


def estimate_point_power(
    cfg_base: SimConfig,
    *,
    null_J: float,
    truth_J: float,
    sample_lengths: Sequence[int],
    n_pseudo: int = 500,
    burn: int = 500,
    nlags: int = 8,
    n_cal: int = 1000,
    n_null: int = 4999,
    alpha: float = 0.10,
) -> tuple[PowerRow, ...]:
    rows = []
    for offset, length in enumerate(sample_lengths):
        base_seed = 100_000 + offset * 20_000
        null_runs = run_ensemble(
            replace(cfg_base, J=null_J, timesteps=burn + length),
            n_runs=n_cal + n_null,
            seeds=list(range(base_seed, base_seed + n_cal + n_null)),
        )
        bank = np.asarray([acf_r2_profile(x.r[-length:], nlags=nlags) for x in null_runs])
        cal, null = bank[:n_cal], bank[n_cal:]
        center, scale = _center_scale(cal, "D_sum")
        null_distances = _distances(null, center, scale, "D_sum")
        truth_runs = run_ensemble(
            replace(cfg_base, J=truth_J, timesteps=burn + length),
            n_runs=n_pseudo,
            seeds=list(range(base_seed + 10_000, base_seed + 10_000 + n_pseudo)),
        )
        rejected = 0
        for result in truth_runs:
            profile = acf_r2_profile(result.r[-length:], nlags=nlags)
            distance = float(_distances(profile[None, :], center, scale, "D_sum")[0])
            p = (1 + int(np.count_nonzero(null_distances >= distance))) / (1 + n_null)
            rejected += int(p < alpha)
        lo, hi = wilson_interval(rejected, n_pseudo)
        rows.append(PowerRow(int(length), n_pseudo, rejected, rejected / n_pseudo, lo, hi))
    return tuple(rows)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def sim_cache_key(**params: Any) -> str:
    payload = json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_npz_cache(path: str | Path, key: str, *, metadata: Mapping[str, Any] | None = None, **arrays: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_key=np.asarray(key),
            metadata_json=np.asarray(json.dumps(_jsonable(metadata or {}), sort_keys=True)),
            **arrays,
        )
    os.replace(tmp, target)


def load_npz_cache(path: str | Path, key: str) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        with np.load(target, allow_pickle=False) as data:
            if str(data["cache_key"].item()) != key:
                return None
            result = {name: data[name].copy() for name in data.files if name not in {"cache_key", "metadata_json"}}
            result["metadata"] = json.loads(str(data["metadata_json"].item()))
            return result
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid simulation cache: {target}") from exc
