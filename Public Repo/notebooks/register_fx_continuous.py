"""Freeze the independent continuous-test design before computing its estimates."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fx_continuous_test.json"
TARGET = ROOT / "notebooks/fx_cache/continuous_preregistration.json"


def normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def freeze() -> dict:
    body = normalized_bytes(CONFIG)
    design = json.loads(body)
    digest = hashlib.sha256(body).hexdigest()
    if TARGET.exists():
        frozen = json.loads(TARGET.read_text(encoding="utf-8"))
        if frozen["design_sha256"] != digest or frozen["design"] != design:
            raise RuntimeError("continuous design differs from its frozen registration")
        return frozen
    frozen = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": digest,
        "design": design,
    }
    TARGET.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen


if __name__ == "__main__":
    registration = freeze()
    print(
        "Continuous design registered:",
        registration["registered_at_utc"],
        registration["design_sha256"],
    )
