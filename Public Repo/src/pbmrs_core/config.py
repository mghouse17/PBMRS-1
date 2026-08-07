from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a configuration file cannot be parsed or validated."""


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve config paths from the current working directory or the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    search_roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for root in search_roots:
        if root is None:
            continue
        for base in [root, *root.parents]:
            resolved = base / candidate
            if resolved.exists():
                return resolved

    # Fall back to a recursive search under the workspace roots for convenience.
    for root in search_roots:
        if root is None:
            continue
        for found in root.rglob(candidate.as_posix()):
            if found.is_file():
                return found

    return candidate


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file from disk."""
    path = _resolve_config_path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")

    if "simulation" not in data:
        data = {"simulation": data}
    return data
