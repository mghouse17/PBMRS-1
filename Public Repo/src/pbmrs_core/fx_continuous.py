"""Continuous FX stress tests, separate from the registered event study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OUTCOMES = (
    "forward_realized_volatility",
    "forward_max_drawdown",
    "forward_absolute_terminal_return",
)


def forward_stress_panel(returns, dates, rolling, *, horizon: int = 21):
    """Pair each rolling estimate with the immediately following stress window."""
    returns = np.asarray(returns, dtype=float)
    dates = np.asarray(dates, dtype="datetime64[D]")
    if len(returns) != len(dates) or horizon < 2 or not rolling:
        raise ValueError("invalid continuous-panel inputs")
    if not np.all(np.isfinite(returns)) or np.any(np.diff(dates).astype(int) <= 0):
        raise ValueError("continuous-panel inputs must be finite and date ordered")
    panel = []
    previous_end = -1
    for row in rolling:
        end = int(row["end_position"])
        if end <= previous_end:
            raise ValueError("rolling endpoints must increase")
        previous_end = end
        if end + horizon >= len(returns):
            continue
        forward = returns[end + 1 : end + 1 + horizon]
        prices = np.exp(np.r_[0.0, np.cumsum(forward)])
        drawdown = 1.0 - prices / np.maximum.accumulate(prices)
        current_vol = float(row["realized_sd"])
        if not np.isfinite(current_vol) or current_vol <= 0:
            raise ValueError("current realized volatility must be positive")
        panel.append(
            {
                "date": str(dates[end]),
                "end_position": end,
                "forward_end_date": str(dates[end + horizon]),
                "J_hat": float(row["J_hat"]),
                "current_realized_volatility": current_vol,
                "forward_realized_volatility": float(forward.std(ddof=1)),
                "forward_max_drawdown": float(drawdown.max()),
                "forward_absolute_terminal_return": float(abs(forward.sum())),
            }
        )
    if not panel:
        raise ValueError("continuous panel has no usable rows")
    return panel


def _coefficient(rows: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    design = np.column_stack(
        [np.ones(len(rows)), rows[:, 0], np.log(rows[:, 1])]
    )
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("continuous regression design is rank deficient")
    return np.linalg.lstsq(design, outcome, rcond=None)[0]


def block_bootstrap_regression(
    panel,
    outcome: str,
    *,
    block_length: int = 24,
    n_resamples: int = 4999,
    seed: int,
    interval: float = 0.95,
):
    """OLS with a circular moving-block pairs-bootstrap interval for b."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown forward-stress outcome: {outcome}")
    if not 0 < interval < 1 or n_resamples < 1:
        raise ValueError("invalid bootstrap interval or resample count")
    rows = np.asarray(
        [[row["J_hat"], row["current_realized_volatility"]] for row in panel],
        dtype=float,
    )
    values = np.asarray([row[outcome] for row in panel], dtype=float)
    n = len(values)
    if rows.shape != (n, 2) or n < 4 or not 1 <= block_length <= n:
        raise ValueError("invalid regression panel or block length")
    if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(values)):
        raise ValueError("regression inputs must be finite")
    beta = _coefficient(rows, values)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    boot = np.empty(n_resamples)
    offsets = np.arange(block_length)
    draw = 0
    invalid_resamples = 0
    max_attempts = max(100, n_resamples * 10)
    while draw < n_resamples:
        if draw + invalid_resamples >= max_attempts:
            raise RuntimeError("too many rank-deficient bootstrap resamples")
        starts = rng.integers(0, n, size=n_blocks)
        ix = ((starts[:, None] + offsets) % n).ravel()[:n]
        try:
            boot[draw] = _coefficient(rows[ix], values[ix])[1]
        except ValueError as error:
            if "rank deficient" not in str(error):
                raise
            invalid_resamples += 1
            continue
        draw += 1
    tail = (1.0 - interval) / 2.0
    lo, hi = np.quantile(boot, [tail, 1.0 - tail])
    return {
        "outcome": outcome,
        "n": n,
        "intercept": float(beta[0]),
        "b_J_hat": float(beta[1]),
        "c_log_current_volatility": float(beta[2]),
        "interval_level": interval,
        "b_lo": float(lo),
        "b_hi": float(hi),
        "block_length_steps": block_length,
        "n_resamples": n_resamples,
        "rank_deficient_resamples_redrawn": invalid_resamples,
        "seed": seed,
    }


def volatility_stratified_quintiles(panel, outcome: str):
    """Summarize outcomes by J quintile inside current-volatility terciles."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown forward-stress outcome: {outcome}")
    j = np.asarray([row["J_hat"] for row in panel], dtype=float)
    vol = np.asarray([row["current_realized_volatility"] for row in panel], dtype=float)
    y = np.asarray([row[outcome] for row in panel], dtype=float)

    def bins(values, count):
        order = np.argsort(values, kind="stable")
        labels = np.empty(len(values), dtype=int)
        for label, indices in enumerate(np.array_split(order, count), start=1):
            labels[indices] = label
        return labels

    tercile, quintile = bins(vol, 3), bins(j, 5)
    rows = []
    for v in range(1, 4):
        for q in range(1, 6):
            mask = (tercile == v) & (quintile == q)
            rows.append(
                {
                    "volatility_tercile": v,
                    "J_hat_quintile": q,
                    "n": int(mask.sum()),
                    "mean_forward_stress": float(y[mask].mean()) if mask.any() else None,
                }
            )
    return rows


def exploratory_event_contrast(
    returns,
    dates,
    rolling,
    events,
    *,
    split: str,
    K: int,
    step: int,
    volatility_lookback: int,
    spacing: int,
    controls_per_event: int = 5,
    year_band: int = 1,
    max_log_vol_distance: float = 0.50,
    n_permutation: int = 4999,
    n_bootstrap: int = 4999,
    seed: int = 783000,
):
    """Post-hoc widened matching; never substitutes for the registered test."""
    from .fx_analysis import pre_event_score

    returns = np.asarray(returns, dtype=float)
    dates = np.asarray(dates, dtype="datetime64[D]")
    selected = [
        event
        for event in events
        if dates[event["start_position"]] >= np.datetime64(split)
    ]
    candidates = range(
        max(volatility_lookback, rolling[0]["end_position"] + K + 1),
        len(dates),
        step,
    )
    matched, excluded = [], []
    for event in selected:
        start = event["start_position"]
        score = pre_event_score(rolling, start, K)
        event_vol = returns[start - volatility_lookback : start].std(ddof=1)
        event_year = int(str(dates[start])[:4])
        pool = []
        for candidate in candidates:
            if abs(int(str(dates[candidate])[:4]) - event_year) > year_band:
                continue
            if any(abs(candidate - e["start_position"]) <= spacing for e in events):
                continue
            control_score = pre_event_score(rolling, candidate, K)
            if control_score is None:
                continue
            control_vol = returns[
                candidate - volatility_lookback : candidate
            ].std(ddof=1)
            distance = abs(np.log(control_vol / event_vol))
            if distance <= max_log_vol_distance:
                pool.append((distance, candidate, control_score))
        pool.sort(key=lambda item: (item[0], item[1]))
        controls = pool[:controls_per_event]
        if score is None or not controls:
            excluded.append({**event, "reason": "no post-hoc caliper match"})
            continue
        matched.append(
            {
                **event,
                "year": str(dates[start])[:4],
                "score": score,
                "n_controls": len(controls),
                "control_positions": [item[1] for item in controls],
                "control_scores": [item[2] for item in controls],
                "max_selected_log_vol_distance": float(max(item[0] for item in controls)),
                "difference": score - float(np.mean([item[2] for item in controls])),
            }
        )
    differences = [row["difference"] for row in matched]
    result = {
        "status": "post_hoc_exploratory",
        "n_labelled": len(selected),
        "n_matched": len(matched),
        "effect_descriptive": float(np.mean(differences)) if differences else None,
        "matched": matched,
        "excluded": excluded,
        "matching": {
            "calendar_year_band": year_band,
            "volatility": f"prior {volatility_lookback}-session SD",
            "max_log_vol_distance": max_log_vol_distance,
            "controls_per_event_maximum": controls_per_event,
            "control_policy": "nearest available controls inside caliper",
        },
        "inference": "descriptive only; specified after observing registered sample failure",
        "lo": None,
        "hi": None,
        "p_value": None,
        "n_permutation": n_permutation,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
    if len(matched) < 3:
        return result
    matrix = np.asarray(
        [[row["score"], *row["control_scores"]] for row in matched], dtype=float
    )
    rng = np.random.default_rng(seed)
    choices = rng.integers(matrix.shape[1], size=(n_permutation, len(matrix)))
    chosen = matrix[np.arange(len(matrix))[None, :], choices]
    null = (
        chosen - (matrix.sum(axis=1)[None, :] - chosen) / (matrix.shape[1] - 1)
    ).mean(axis=1)
    effect = result["effect_descriptive"]
    result["p_value"] = float((1 + np.count_nonzero(null >= effect)) / (1 + len(null)))
    differences = np.asarray(differences)
    years = sorted({row["year"] for row in matched})
    clusters = [differences[[row["year"] == year for row in matched]] for year in years]
    boot = np.asarray(
        [
            np.concatenate(
                [clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]
            ).mean()
            for _ in range(n_bootstrap)
        ]
    )
    result["lo"], result["hi"] = [float(value) for value in np.percentile(boot, [2.5, 97.5])]
    result["inference"] = (
        "post-hoc one-sided matched-label permutation p-value and year-cluster "
        "bootstrap interval; no pre-registered interpretation"
    )
    return result


def load_continuous_study(root):
    """Validate the independent design, result bytes, timestamps and dependencies."""
    root = Path(root)
    cache = root / "notebooks/fx_cache"
    config = root / "configs/fx_continuous_test.json"
    frozen = json.loads((cache / "continuous_preregistration.json").read_text())
    manifest = json.loads((cache / "continuous_manifest.json").read_text())
    body = (cache / "continuous_results.json").read_bytes()
    digest = hashlib.sha256(config.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if not digest == frozen["design_sha256"] == manifest["design_sha256"]:
        raise ValueError("continuous design hash mismatch")
    if frozen["design"] != json.loads(config.read_text()):
        raise ValueError("continuous frozen design differs from configuration")
    if not (
        frozen["registered_at_utc"]
        < manifest["started_at_utc"]
        < manifest["computed_at_utc"]
    ):
        raise ValueError("continuous registration/computation timestamps are invalid")
    if hashlib.sha256(body).hexdigest() != manifest["results_sha256"]:
        raise ValueError("continuous result payload hash mismatch")
    for relative, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256(
            (root / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"continuous source changed: {relative}")
    from .fx import validate_raw_tree

    raw = validate_raw_tree(root / "notebooks/fx_continuous_cache/raw")
    if {name: entry["sha256"] for name, entry in raw.items()} != manifest["raw_hashes"]:
        raise ValueError("continuous raw input hashes changed")
    event_registration = json.loads((cache / "preregistration.json").read_text())
    event_results = (cache / "results.json").read_bytes()
    if event_registration["design_sha256"] != manifest["event_design_sha256"]:
        raise ValueError("registered event design dependency changed")
    if hashlib.sha256(event_results).hexdigest() != manifest["event_results_sha256"]:
        raise ValueError("registered event results dependency changed")
    return frozen, manifest, json.loads(body)
