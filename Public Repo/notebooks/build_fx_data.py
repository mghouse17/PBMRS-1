"""Freeze the design, discover contracts, and cache official FX responses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pbmrs_core.fx import (
    discover_definitions,
    fetch_fx,
    fetch_fx_positioning,
    validate_raw_tree,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks/fx_cache"


def freeze_design():
    body = (ROOT / "configs/fx_study.json").read_bytes()
    design = json.loads(body)
    key = hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest()
    path = CACHE / "preregistration.json"
    CACHE.mkdir(exist_ok=True)
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen["design_sha256"] != key:
            raise RuntimeError(
                "Design changed after registration; requires a new labelled study"
            )
    else:
        frozen = {
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "design_sha256": key,
            "design": design,
        }
        path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen


def build():
    frozen = freeze_design()
    design = frozen["design"]
    raw = CACHE / "raw"
    raw.mkdir(exist_ok=True)
    definitions = discover_definitions(raw)
    result = {"definitions": [asdict(d) for d in definitions], "pairs": {}}
    for pair in ("JPY", "EUR"):
        fx = fetch_fx(pair, cache_dir=raw, start=design["start"], end=design["end"])
        info = {
            "symbol": fx.symbol,
            "source": fx.source,
            "units": "USD per currency",
            "n_prices": len(fx.close),
            "first": str(fx.dates[0]),
            "last": str(fx.dates[-1]),
            "positioning": {},
        }
        for definition in definitions:
            cot, contract = fetch_fx_positioning(
                definition,
                pair,
                cache_dir=raw,
                start=design["start"],
                end=design["end"],
            )
            info["positioning"][definition.key] = {
                "contract": contract,
                "n_reports": len(cot.dates),
                "first": str(cot.dates[0]),
                "last": str(cot.dates[-1]),
            }
            print(pair, definition.key, contract, len(cot.dates), flush=True)
        result["pairs"][pair] = info
    result["raw_entries"] = validate_raw_tree(raw)
    (CACHE / "data_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("FX data ready", flush=True)


if __name__ == "__main__":
    build()
