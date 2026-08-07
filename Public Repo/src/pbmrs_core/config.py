from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a configuration file cannot be parsed or validated."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file from disk."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")

    if "simulation" not in data:
        data = {"simulation": data}
    return data
