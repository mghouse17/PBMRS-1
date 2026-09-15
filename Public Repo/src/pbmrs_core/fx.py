"""FX conventions and CFTC discovery using the shared raw-response cache."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from .commodities import (
    CFTCDefinition,
    DataFetchError,
    PriceSeries,
    fetch_bytes,
    fetch_cftc_positioning,
    fetch_fred_series,
    read_manifest,
)

FX_SERIES = {"JPY": "DEXJPUS", "EUR": "DEXUSEU"}


def usd_per_currency(series: PriceSeries) -> PriceSeries:
    """Orient both verified series so positive returns mean currency strength."""
    if series.symbol not in FX_SERIES.values():
        raise ValueError("unverified FX series")
    close = 1.0 / series.close if series.symbol == "DEXJPUS" else series.close
    result = replace(series, close=close, source="FRED / Federal Reserve Board, H.10")
    sign = -1 if series.symbol == "DEXJPUS" else 1
    np.testing.assert_allclose(
        result.log_returns, sign * series.log_returns, atol=2e-15
    )
    return result


def fetch_fx(pair: str, *, cache_dir: Path, start: str, end: str) -> PriceSeries:
    return usd_per_currency(
        fetch_fred_series(
            FX_SERIES[pair],
            cache_dir=cache_dir,
            start=start,
            end=end,
        )
    )


def discover_definitions(cache_dir: Path) -> tuple[CFTCDefinition, ...]:
    """Discover official dataset IDs from the compact CFTC views catalogue."""
    url = "https://publicreporting.cftc.gov/api/views.json?limit=100"
    payload = json.loads(fetch_bytes(url, cache_dir / "cftc_views.json"))
    definitions = []
    for family, category, prefix in (
        ("Legacy", "noncommercial", "noncomm"),
        ("TFF", "leveraged_money", "lev_money"),
    ):
        for mode, title in (("futures", "Futures Only"), ("combined", "Combined")):
            expected = f"{family} - {title}"
            matches = [
                r
                for r in payload
                if r["name"].strip() == expected and r.get("viewType") == "tabular"
            ]
            if len(matches) != 1:
                raise DataFetchError(
                    f"expected one official dataset named {expected}: {matches}"
                )
            dataset = matches[0]["id"]
            suffix = "_all" if family == "Legacy" else ""
            long, short = (
                f"{prefix}_positions_long{suffix}",
                f"{prefix}_positions_short{suffix}",
            )
            definitions.append(
                CFTCDefinition(
                    f"{family.lower()}_{mode}",
                    dataset,
                    expected,
                    long,
                    short,
                    category,
                )
            )
    return tuple(definitions)


def discover_contract(definition: CFTCDefinition, pair: str, cache_dir: Path) -> dict:
    """Require an exact CME contract name from the official catalogue rows."""
    expected = {
        "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
        "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    }[pair]
    query = urlencode(
        {
            "$select": "distinct market_and_exchange_names,cftc_contract_market_code",
            "$where": f"market_and_exchange_names='{expected}'",
            "$limit": 100,
        }
    )
    url = f"https://publicreporting.cftc.gov/resource/{definition.dataset_id}.json?{query}"
    rows = json.loads(
        fetch_bytes(url, cache_dir / f"cftc_contract_{definition.key}_{pair}.json")
    )
    if len(rows) != 1 or rows[0].get("market_and_exchange_names") != expected:
        raise DataFetchError(f"ambiguous or missing contract for {pair}: {rows}")
    code = rows[0]["cftc_contract_market_code"]
    if not code.isdigit():
        raise DataFetchError("unexpected contract code")
    return rows[0]


def fetch_fx_positioning(definition, pair, *, cache_dir, start, end):
    contract = discover_contract(definition, pair, cache_dir)
    code = contract["cftc_contract_market_code"]
    # Shared fetcher keys by dataset/date: isolate contracts to avoid collisions.
    series = fetch_cftc_positioning(
        definition,
        start=start,
        end=end,
        contract_code=code,
        cache_dir=cache_dir / f"{pair}_{code}",
        limit=20000,
    )
    if len(series.dates) >= 20000:
        raise DataFetchError("CFTC query reached row limit")
    return series, contract


def validate_raw_tree(root: Path) -> dict:
    """Validate analysis inputs while allowing ignored discovery-only dumps."""
    import hashlib

    entries = {}
    for path in sorted(root.rglob("manifest.json")):
        for entry in read_manifest(path.parent)["entries"].values():
            file = path.parent / entry["cache_file"]
            optional_metadata = (
                file.name == "cftc_catalog.json"
                or file.name.startswith("cftc_metadata_")
            )
            if optional_metadata and not file.exists():
                # These large schema/catalogue responses were used only during
                # initial discovery. Their historical hashes remain in the
                # manifest/cache key, while compact cftc_views.json and actual
                # positioning payloads support the complete offline analysis.
                entries[file.relative_to(root).as_posix()] = entry
                continue
            body = file.read_bytes()
            if (
                len(body) != entry["byte_count"]
                or hashlib.sha256(body).hexdigest() != entry["sha256"]
            ):
                raise DataFetchError(f"cache hash or byte-count mismatch: {file}")
            entries[file.relative_to(root).as_posix()] = entry
    if not entries:
        raise DataFetchError("empty FX raw cache")
    return entries
