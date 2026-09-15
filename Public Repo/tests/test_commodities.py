from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from pbmrs_core.commodities import (
    CFTC_DEFINITIONS,
    DataFetchError,
    PriceSeries,
    fetch_bytes,
    parse_cftc_json,
    parse_fred_csv,
    parse_yahoo_chart,
    stationary_bootstrap_indices,
)


def test_price_series_rejects_invalid_prices_and_dates():
    with pytest.raises(ValueError, match="positive"):
        PriceSeries("x", "test", "", "", ["2024-01-01", "2024-01-02"], [1.0, 0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        PriceSeries("x", "test", "", "", ["2024-01-02", "2024-01-01"], [1.0, 2.0])


def test_price_and_return_records_are_deeply_immutable():
    series = PriceSeries(
        "x", "test", "", "", ["2024-01-01", "2024-01-02"], [1.0, 2.0]
    )
    with pytest.raises(ValueError):
        series.close[0] = 9.0
    with pytest.raises(ValueError):
        series.returns.values[0] = 9.0
    assert series.returns.dates.tolist() == [np.datetime64("2024-01-02")]


def test_fred_parser_drops_missing_and_resolves_duplicates():
    body = b"observation_date,DCOILWTICO\n2024-01-02,70\n2024-01-03,.\n2024-01-02,71\n2024-01-04,72\n"
    series = parse_fred_csv(body, series_id="DCOILWTICO")
    assert series.dates.tolist() == [np.datetime64("2024-01-02"), np.datetime64("2024-01-04")]
    np.testing.assert_allclose(series.close, [71.0, 72.0])
    assert series.return_dates.tolist() == [np.datetime64("2024-01-04")]


def test_fred_parser_accepts_legacy_headers_and_fails_loudly():
    series = parse_fred_csv(b"DATE,VALUE\n2024-01-01,10\n2024-01-02,11\n", series_id="DUMMY")
    np.testing.assert_allclose(series.close, [10.0, 11.0])
    with pytest.raises(DataFetchError):
        parse_fred_csv(b"wrong,headers\na,b\n", series_id="DUMMY")


def test_yahoo_parser_uses_exchange_timezone_and_deterministic_duplicates():
    timestamps = [
        int(datetime(2024, 3, 10, 4, 30, tzinfo=timezone.utc).timestamp()),
        int(datetime(2024, 3, 10, 6, 30, tzinfo=timezone.utc).timestamp()),
        int(datetime(2024, 3, 11, 4, 30, tzinfo=timezone.utc).timestamp()),
    ]
    payload = {
        "chart": {
            "error": None,
            "result": [{
        "timestamp": timestamps[:2] + [timestamps[1] + 21600, timestamps[2]],
                "meta": {"exchangeTimezoneName": "America/New_York"},
                    "indicators": {"quote": [{"close": [80.0, 81.0, 81.5, 82.0]}]},
            }],
        }
    }
    series = parse_yahoo_chart(json.dumps(payload).encode(), symbol="CL=F")
    assert series.dates.tolist() == [np.datetime64("2024-03-09"), np.datetime64("2024-03-10"), np.datetime64("2024-03-11")]
    np.testing.assert_allclose(series.close, [80.0, 81.5, 82.0])


def test_yahoo_parser_drops_null_and_rejects_upstream_error():
    payload = {
        "chart": {
            "error": None,
            "result": [{
                "timestamp": [0, 86400, 172800],
                "meta": {"exchangeTimezoneName": "UTC"},
                "indicators": {"quote": [{"close": [1.0, None, 2.0]}]},
            }],
        }
    }
    assert len(parse_yahoo_chart(json.dumps(payload).encode(), symbol="X").close) == 2
    with pytest.raises(DataFetchError, match="Yahoo chart error"):
        parse_yahoo_chart(json.dumps({"chart": {"error": {"code": "x"}}}).encode(), symbol="X")


def test_cftc_parser_and_open_interest_validation():
    definition = CFTC_DEFINITIONS[0]
    rows = [{
        "report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000",
        definition.long_col: "120",
        definition.short_col: "20",
        "open_interest_all": "1000",
    }]
    series = parse_cftc_json(json.dumps(rows).encode(), definition=definition)
    assert series.value_on("2024-01-02") == pytest.approx(0.1)
    rows[0]["open_interest_all"] = "0"
    with pytest.raises(ValueError, match="open interest"):
        parse_cftc_json(json.dumps(rows).encode(), definition=definition)


def test_cache_hit_validates_manifest_hash(tmp_path, monkeypatch):
    path = tmp_path / "raw.bin"
    body = b"trusted"
    path.write_bytes(body)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "entries": {"raw.bin": {"sha256": hashlib.sha256(body).hexdigest()}},
    }), encoding="utf-8")
    monkeypatch.setattr("pbmrs_core.commodities.urlopen", lambda *args, **kwargs: pytest.fail("network used"))
    assert fetch_bytes("https://example.invalid", path) == body
    path.write_bytes(b"tampered")
    with pytest.raises(DataFetchError, match="hash mismatch"):
        fetch_bytes("https://example.invalid", path)


def test_cache_hit_requires_manifest_entry(tmp_path):
    path = tmp_path / "orphan.bin"
    path.write_bytes(b"untracked")
    with pytest.raises(DataFetchError, match="entry missing"):
        fetch_bytes("https://example.invalid", path)


def test_refresh_replaces_bytes_and_updates_manifest(tmp_path, monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.body

    path = tmp_path / "raw.bin"
    monkeypatch.setattr(
        "pbmrs_core.commodities.urlopen", lambda *args, **kwargs: Response(b"first")
    )
    assert fetch_bytes("https://example.test/data?a=1", path, refresh=True) == b"first"
    monkeypatch.setattr(
        "pbmrs_core.commodities.urlopen", lambda *args, **kwargs: Response(b"second")
    )
    assert fetch_bytes("https://example.test/data?a=2", path, refresh=True) == b"second"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["entries"]["raw.bin"]
    assert entry["sha256"] == hashlib.sha256(b"second").hexdigest()
    assert entry["request_parameters"] == "a=2"


def test_stationary_bootstrap_is_deterministic_and_wraps():
    first = stationary_bootstrap_indices(5, mean_block=1e12, n_boot=5, seed=12)
    second = stationary_bootstrap_indices(5, mean_block=1e12, n_boot=5, seed=12)
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0) & (first < 5))
    assert any(np.any(np.diff(row) < 0) for row in first)
