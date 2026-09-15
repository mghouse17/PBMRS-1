"""Auditable commodity-market data access and empirical helpers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np


class DataFetchError(RuntimeError):
    """Raised when real upstream data cannot be fetched or validated."""


@dataclass(frozen=True)
class CFTCDefinition:
    key: str
    dataset_id: str
    label: str
    long_col: str
    short_col: str
    category: str


CFTC_DEFINITIONS = (
    CFTCDefinition(
        "disagg_combined",
        "kh3c-gbw2",
        "Disaggregated managed money, futures and options",
        "m_money_positions_long_all",
        "m_money_positions_short_all",
        "managed_money",
    ),
    CFTCDefinition(
        "disagg_futures",
        "72hh-3qpy",
        "Disaggregated managed money, futures only",
        "m_money_positions_long_all",
        "m_money_positions_short_all",
        "managed_money",
    ),
    CFTCDefinition(
        "legacy_combined",
        "jun7-fc8e",
        "Legacy non-commercial, futures and options",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "noncommercial",
    ),
    CFTCDefinition(
        "legacy_futures",
        "6dca-aqww",
        "Legacy non-commercial, futures only",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "noncommercial",
    ),
)


def _date_array(values: Sequence[object]) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[D]").copy()


def _freeze(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ReturnSeries:
    dates: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        dates = _date_array(self.dates)
        values = np.asarray(self.values, dtype=float).copy()
        if dates.ndim != 1 or values.ndim != 1 or len(dates) != len(values):
            raise ValueError("return dates and values must be aligned one-dimensional arrays")
        if len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("returns must be nonempty and finite")
        if np.any(np.diff(dates).astype("timedelta64[D]").astype(int) <= 0):
            raise ValueError("return dates must be strictly increasing")
        object.__setattr__(self, "dates", _freeze(dates))
        object.__setattr__(self, "values", _freeze(values))


@dataclass(frozen=True)
class PriceSeries:
    symbol: str
    source: str
    url: str
    retrieved_at: str
    dates: np.ndarray
    close: np.ndarray

    def __post_init__(self) -> None:
        dates = _date_array(self.dates)
        close = np.asarray(self.close, dtype=float).copy()
        if dates.ndim != 1 or close.ndim != 1 or len(dates) != len(close):
            raise ValueError("dates and close must be equal-length one-dimensional arrays")
        if len(close) < 2:
            raise ValueError("price series requires at least two observations")
        if np.any(np.diff(dates).astype("timedelta64[D]").astype(int) <= 0):
            raise ValueError("dates must be strictly increasing")
        if not np.all(np.isfinite(close)) or np.any(close <= 0):
            raise ValueError("close must contain finite positive values")
        object.__setattr__(self, "dates", _freeze(dates))
        object.__setattr__(self, "close", _freeze(close))

    def window(self, start: str | date, end: str | date) -> PriceSeries:
        lo, hi = np.datetime64(start, "D"), np.datetime64(end, "D")
        keep = (self.dates >= lo) & (self.dates <= hi)
        if np.count_nonzero(keep) < 2:
            raise ValueError(f"window {start} to {end} has fewer than two prices")
        return replace(self, dates=self.dates[keep], close=self.close[keep])

    @property
    def log_returns(self) -> np.ndarray:
        return np.diff(np.log(self.close))

    @property
    def return_dates(self) -> np.ndarray:
        return self.dates[1:]

    @property
    def returns(self) -> ReturnSeries:
        return ReturnSeries(self.return_dates, self.log_returns)


@dataclass(frozen=True)
class PositioningSeries:
    definition: CFTCDefinition
    url: str
    retrieved_at: str
    dates: np.ndarray
    longs: np.ndarray
    shorts: np.ndarray
    open_interest: np.ndarray

    def __post_init__(self) -> None:
        dates = _date_array(self.dates)
        arrays = tuple(
            np.asarray(x, dtype=float).copy()
            for x in (self.longs, self.shorts, self.open_interest)
        )
        if any(x.ndim != 1 or len(x) != len(dates) for x in arrays):
            raise ValueError("positioning arrays must align with dates")
        if len(dates) == 0 or np.any(np.diff(dates).astype("timedelta64[D]").astype(int) <= 0):
            raise ValueError("positioning dates must be nonempty and strictly increasing")
        if any(not np.all(np.isfinite(x)) for x in arrays):
            raise ValueError("positioning arrays must be finite")
        if np.any(arrays[2] <= 0):
            raise ValueError("open interest must be positive")
        object.__setattr__(self, "dates", _freeze(dates))
        object.__setattr__(self, "longs", _freeze(arrays[0]))
        object.__setattr__(self, "shorts", _freeze(arrays[1]))
        object.__setattr__(self, "open_interest", _freeze(arrays[2]))

    @property
    def net_over_oi(self) -> np.ndarray:
        return (self.longs - self.shorts) / self.open_interest

    def value_on(self, target: str | date) -> float:
        matches = np.flatnonzero(self.dates == np.datetime64(target, "D"))
        if len(matches) != 1:
            raise KeyError(f"no unique positioning observation on {target}")
        return float(self.net_over_oi[matches[0]])


@dataclass(frozen=True)
class BootstrapResult:
    point: np.ndarray
    samples: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    se: np.ndarray
    n_boot: int
    mean_block: float
    seed: int


@dataclass(frozen=True)
class ReconRow:
    anchor: str
    reported: float
    computed: float
    abs_pct_diff: float
    tolerance_pct: float
    passed: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def read_manifest(cache_dir: str | Path) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.exists():
        return {"schema_version": 1, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataFetchError(f"invalid cache manifest: {path}") from exc
    if data.get("schema_version") != 1 or not isinstance(data.get("entries"), dict):
        raise DataFetchError(f"unsupported cache manifest: {path}")
    return data


def _record_provenance(cache_dir: Path, name: str, url: str, path: Path, body: bytes) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(cache_dir)
    retrieved = _utc_now()
    manifest["entries"][name] = {
        "source_name": name,
        "url": url,
        "retrieved_at_utc": retrieved,
        "sha256": _sha256(body),
        "byte_count": len(body),
        "request_parameters": url.split("?", 1)[1] if "?" in url else "",
        "cache_file": path.name,
    }
    tmp = cache_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, cache_dir / "manifest.json")
    return retrieved


def fetch_bytes(
    url: str,
    cache_path: str | Path,
    *,
    refresh: bool = False,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30,
    cache_dir: str | Path | None = None,
) -> bytes:
    path = Path(cache_path)
    root = Path(cache_dir) if cache_dir is not None else path.parent
    if path.exists() and not refresh:
        body = path.read_bytes()
        manifest = read_manifest(root)
        entry = manifest["entries"].get(path.name)
        if entry is None:
            raise DataFetchError(f"cache manifest entry missing: {path}")
        if entry.get("sha256") != _sha256(body):
            raise DataFetchError(f"cache hash mismatch: {path}")
        return body
    request_headers = {"User-Agent": "PBMRS/0.2.2 research"}
    request_headers.update(headers or {})
    try:
        with urlopen(Request(url, headers=request_headers), timeout=timeout) as response:
            body = response.read()
    except Exception as exc:
        if path.exists() and not refresh:
            return path.read_bytes()
        raise DataFetchError(f"failed to fetch {url}") from exc
    if not body:
        raise DataFetchError(f"empty response from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)
    _record_provenance(root, path.name, url, path, body)
    return body


def _collapse_rows(dates: Sequence[object], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    latest: dict[np.datetime64, float] = {}
    for d, value in zip(_date_array(dates), values):
        if np.isfinite(value):
            latest[d] = float(value)
    ordered = sorted(latest.items(), key=lambda item: item[0])
    return _date_array([x[0] for x in ordered]), np.asarray([x[1] for x in ordered], dtype=float)


def parse_yahoo_chart(body: bytes, *, symbol: str, url: str = "", retrieved_at: str = "") -> PriceSeries:
    try:
        payload = json.loads(body)
        chart = payload["chart"]
        if chart.get("error") is not None:
            raise DataFetchError(f"Yahoo chart error: {chart['error']}")
        result = chart["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
        zone_name = result["meta"]["exchangeTimezoneName"]
        zone = ZoneInfo(zone_name)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ZoneInfoNotFoundError) as exc:
        raise DataFetchError("invalid Yahoo chart payload") from exc
    dates, values = [], []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        dates.append(datetime.fromtimestamp(int(ts), timezone.utc).astimezone(zone).date())
        values.append(float(close))
    d, v = _collapse_rows(dates, values)
    return PriceSeries(symbol, "Yahoo Finance", url, retrieved_at, d, v)


def fetch_yahoo_chart(
    symbol: str,
    start: str | date,
    end: str | date,
    *,
    cache_dir: str | Path,
    refresh: bool = False,
    interval: str = "1d",
) -> PriceSeries:
    start_date, end_date = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    p1 = int(datetime.combine(start_date, time(), timezone.utc).timestamp())
    p2 = int(datetime.combine(end_date + timedelta(days=1), time(), timezone.utc).timestamp())
    query = urlencode({"period1": p1, "period2": p2, "interval": interval, "events": "history"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    path = Path(cache_dir) / f"yahoo_{symbol.replace('=', '_')}_{start_date}_{end_date}.json"
    body = fetch_bytes(url, path, refresh=refresh, cache_dir=cache_dir)
    entry = read_manifest(cache_dir)["entries"].get(path.name, {})
    return parse_yahoo_chart(body, symbol=symbol, url=url, retrieved_at=entry.get("retrieved_at_utc", "cached"))


def parse_fred_csv(body: bytes, *, series_id: str, url: str = "", retrieved_at: str = "") -> PriceSeries:
    try:
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DataFetchError("invalid FRED CSV") from exc
    if not rows:
        raise DataFetchError("empty FRED CSV")
    date_key = "observation_date" if "observation_date" in rows[0] else "DATE"
    value_key = series_id if series_id in rows[0] else next((k for k in rows[0] if k != date_key), None)
    if date_key not in rows[0] or value_key is None:
        raise DataFetchError("unrecognized FRED headers")
    dates, values = [], []
    for row in rows:
        raw = (row.get(value_key) or "").strip()
        if raw in {"", "."}:
            continue
        try:
            dates.append(date.fromisoformat(row[date_key]))
            values.append(float(raw))
        except (TypeError, ValueError) as exc:
            raise DataFetchError(f"invalid FRED row: {row}") from exc
    d, v = _collapse_rows(dates, values)
    return PriceSeries(series_id, "FRED (EIA)", url, retrieved_at, d, v)


def fetch_fred_series(
    series_id: str,
    *,
    cache_dir: str | Path,
    refresh: bool = False,
    start: str | date | None = None,
    end: str | date | None = None,
) -> PriceSeries:
    query = {"id": series_id}
    if start is not None:
        query["cosd"] = str(start)
    if end is not None:
        query["coed"] = str(end)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode(query)
    suffix = f"_{start}_{end}" if start is not None or end is not None else ""
    path = Path(cache_dir) / f"fred_{series_id}{suffix}.csv"
    body = fetch_bytes(url, path, refresh=refresh, cache_dir=cache_dir)
    entry = read_manifest(cache_dir)["entries"].get(path.name, {})
    return parse_fred_csv(body, series_id=series_id, url=url, retrieved_at=entry.get("retrieved_at_utc", "cached"))


def parse_cftc_json(
    body: bytes,
    *,
    definition: CFTCDefinition,
    url: str = "",
    retrieved_at: str = "",
) -> PositioningSeries:
    try:
        rows = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DataFetchError("invalid CFTC JSON") from exc
    if not isinstance(rows, list):
        raise DataFetchError("CFTC payload must be a list")
    parsed: dict[np.datetime64, tuple[float, float, float]] = {}
    for row in rows:
        try:
            day = np.datetime64(str(row["report_date_as_yyyy_mm_dd"])[:10], "D")
            parsed[day] = (
                float(row[definition.long_col]),
                float(row[definition.short_col]),
                float(row["open_interest_all"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataFetchError(f"invalid CFTC row for {definition.key}") from exc
    ordered = sorted(parsed.items(), key=lambda item: item[0])
    dates = _date_array([x[0] for x in ordered])
    vals = np.asarray([x[1] for x in ordered], dtype=float)
    if vals.size == 0:
        vals = np.empty((0, 3))
    return PositioningSeries(definition, url, retrieved_at, dates, vals[:, 0], vals[:, 1], vals[:, 2])


def fetch_cftc_positioning(
    definition: CFTCDefinition,
    *,
    start: str,
    end: str,
    contract_code: str = "067651",
    cache_dir: str | Path,
    refresh: bool = False,
    limit: int = 5000,
) -> PositioningSeries:
    select = f"report_date_as_yyyy_mm_dd,{definition.long_col},{definition.short_col},open_interest_all"
    where = (
        f"cftc_contract_market_code='{contract_code}' AND "
        f"report_date_as_yyyy_mm_dd between '{start}T00:00:00.000' and '{end}T23:59:59.999'"
    )
    query = urlencode({"$select": select, "$where": where, "$order": "report_date_as_yyyy_mm_dd", "$limit": limit})
    url = f"https://publicreporting.cftc.gov/resource/{definition.dataset_id}.json?{query}"
    path = Path(cache_dir) / f"cftc_{definition.dataset_id}_{start}_{end}.json"
    body = fetch_bytes(url, path, refresh=refresh, cache_dir=cache_dir)
    entry = read_manifest(cache_dir)["entries"].get(path.name, {})
    return parse_cftc_json(body, definition=definition, url=url, retrieved_at=entry.get("retrieved_at_utc", "cached"))


def log_returns(close: Sequence[float], *, demean: bool = False) -> np.ndarray:
    values = np.asarray(close, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("close must be a finite positive one-dimensional series")
    result = np.diff(np.log(values))
    return result - result.mean() if demean else result


def stationary_bootstrap_indices(
    n: int,
    *,
    mean_block: float = 7.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    if n < 2 or mean_block < 1 or n_boot < 1:
        raise ValueError("invalid stationary-bootstrap dimensions")
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n), dtype=np.int64)
    restart_probability = 1.0 / mean_block
    for b in range(n_boot):
        idx = int(rng.integers(n))
        for t in range(n):
            if t == 0 or rng.random() < restart_probability:
                idx = int(rng.integers(n))
            else:
                idx = (idx + 1) % n
            out[b, t] = idx
    return out


def bootstrap_statistic(
    r: Sequence[float],
    stat_fn: Callable[[np.ndarray], np.ndarray | float],
    *,
    mean_block: float = 7.0,
    n_boot: int = 2000,
    seed: int = 0,
    lo_pct: float = 2.5,
    hi_pct: float = 97.5,
) -> BootstrapResult:
    values = np.asarray(r, dtype=float)
    indices = stationary_bootstrap_indices(len(values), mean_block=mean_block, n_boot=n_boot, seed=seed)
    point = np.asarray(stat_fn(values), dtype=float)
    samples = np.asarray([stat_fn(values[i]) for i in indices], dtype=float)
    return BootstrapResult(
        point=point,
        samples=samples,
        lo=np.percentile(samples, lo_pct, axis=0),
        hi=np.percentile(samples, hi_pct, axis=0),
        se=samples.std(axis=0, ddof=1),
        n_boot=n_boot,
        mean_block=mean_block,
        seed=seed,
    )


def nearest_on_or_before(dates: Sequence[object], target: str | date) -> int:
    values = _date_array(dates)
    index = int(np.searchsorted(values, np.datetime64(target, "D"), side="right") - 1)
    if index < 0:
        raise KeyError(f"no observation on or before {target}")
    return index


def pct_change_between(series: PriceSeries, d0: str | date, d1: str | date) -> float:
    i0 = nearest_on_or_before(series.dates, d0)
    i1 = nearest_on_or_before(series.dates, d1)
    return float(series.close[i1] / series.close[i0] - 1.0)


def reconcile(
    computed: Mapping[str, float],
    reported: Mapping[str, float],
    *,
    tolerance_pct: float = 2.0,
) -> list[ReconRow]:
    if computed.keys() != reported.keys():
        raise ValueError("computed and reported anchors must have identical keys")
    rows = []
    for key in computed:
        target = float(reported[key])
        actual = float(computed[key])
        diff = abs(actual - target) / abs(target) * 100 if target else abs(actual) * 100
        rows.append(ReconRow(key, target, actual, diff, tolerance_pct, diff <= tolerance_pct))
    return rows
