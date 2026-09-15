from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pbmrs_core.calibration import load_npz_cache
from pbmrs_core.commodities import fetch_fred_series, read_manifest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks" / "data_cache"


def test_committed_raw_cache_manifest_hashes_validate():
    manifest = read_manifest(CACHE)
    assert len(manifest["entries"]) == 6
    for entry in manifest["entries"].values():
        body = (CACHE / entry["cache_file"]).read_bytes()
        assert len(body) == entry["byte_count"]
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
    assert not list(CACHE.glob("yahoo_*.json"))


def test_yahoo_unavailability_cannot_affect_primary_offline_path(monkeypatch):
    monkeypatch.setattr(
        "pbmrs_core.commodities.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    wti = fetch_fred_series(
        "DCOILWTICO", cache_dir=CACHE, start="2026-01-01", end="2026-07-31"
    )
    assert len(wti.log_returns) == 144


def test_analysis_seed_ranges_are_disjoint_and_controls_cover_nominal_size():
    analysis = json.loads((CACHE / "analysis_manifest.json").read_text(encoding="utf-8"))
    cal = set(range(*[analysis["seed_ledger"]["adequacy_calibration"][0],
                      analysis["seed_ledger"]["adequacy_calibration"][1] + 1]))
    null = set(range(*[analysis["seed_ledger"]["adequacy_null"][0],
                       analysis["seed_ledger"]["adequacy_null"][1] + 1]))
    assert cal.isdisjoint(null)
    with np.load(CACHE / "gmsg_controls.npz", allow_pickle=False) as cached:
        rows = cached["rows"]
    assert np.all((rows[:, 3] <= 0.10) & (0.10 <= rows[:, 4]))


def test_every_simulation_cache_matches_its_analysis_manifest_key():
    analysis = json.loads((CACHE / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert set(analysis["cache_keys"]) == {
        "gmsg_profiles.npz",
        "gmsg_stability.npz",
        "gmsg_power.npz",
        "gmsg_controls.npz",
        "gmsg_horizon.npz",
        "gmsg_marginals.npz",
    }
    for name, key in analysis["cache_keys"].items():
        assert load_npz_cache(CACHE / name, key) is not None


def test_every_maintained_notebook_is_fully_executed_without_errors():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [p.name for p in notebooks] == [
        "00_pbmrs_mvp.ipynb",
        "01_pbmrs_phase_transition.ipynb",
        "02_wti_regime_calibration.ipynb",
    ]
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert code_cells
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        assert not [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]


def test_application_notebook_has_exactly_nine_main_exhibits_and_visible_guards():
    path = ROOT / "notebooks" / "02_wti_regime_calibration.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    tagged = [
        cell for cell in notebook["cells"]
        if "main-exhibit" in cell.get("metadata", {}).get("tags", [])
    ]
    assert [cell["metadata"]["exhibit"] for cell in tagged] == list(range(1, 10))
    source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
    for guard in ("cfg.alpha_r == 12.0", "positions", "n_calibration", "wti_mask.sum()"):
        assert guard in source
    prohibited = ("PBMRS is rejected", "fits better", "WTI forecast")
    assert not any(phrase in source for phrase in prohibited)
