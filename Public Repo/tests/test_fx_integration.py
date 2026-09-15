from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pbmrs_core.fx import (
    discover_definitions,
    fetch_fx,
    fetch_fx_positioning,
    validate_raw_tree,
)
from pbmrs_core.fx_analysis import load_study

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks/fx_cache"


def test_fx_raw_data_and_discovered_contracts_work_offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network access is forbidden in the offline study")

    monkeypatch.setattr("pbmrs_core.commodities.urlopen", no_network)
    entries = validate_raw_tree(CACHE / "raw")
    assert len(entries) >= 19
    definitions = discover_definitions(CACHE / "raw")
    assert {d.key for d in definitions} == {
        "legacy_futures",
        "legacy_combined",
        "tff_futures",
        "tff_combined",
    }
    d = json.loads((ROOT / "configs/fx_study.json").read_text())
    codes = {}
    for pair in ("JPY", "EUR"):
        series = fetch_fx(pair, cache_dir=CACHE / "raw", start=d["start"], end=d["end"])
        assert len(series.log_returns) > 3500
        for definition in definitions:
            cot, contract = fetch_fx_positioning(
                definition,
                pair,
                cache_dir=CACHE / "raw",
                start=d["start"],
                end=d["end"],
            )
            assert len(cot.dates) > 700
            codes.setdefault(pair, set()).add(contract["cftc_contract_market_code"])
    assert len(codes["JPY"]) == len(codes["EUR"]) == 1
    assert codes["JPY"].isdisjoint(codes["EUR"])


def test_fx_design_scales_seeds_and_every_result_validate():
    frozen, sim, manifest, results = load_study(ROOT)
    d = frozen["design"]
    assert (
        frozen["registered_at_utc"]
        < sim["started_at_utc"]
        < manifest["computed_at_utc"]
    )
    ranges = [
        set(range(*[d["seeds"][key][0], d["seeds"][key][1] + 1]))
        for key in (
            "solver",
            "stability",
            "calibration",
            "null",
            "pseudo",
            "horizon",
            "gaussian",
        )
    ]
    for i, seeds in enumerate(ranges):
        assert all(seeds.isdisjoint(other) for other in ranges[i + 1 :])
    for pair in ("JPY", "EUR"):
        assert (
            sim["scale"][pair]["relative_error"] <= d["sigma_solver_relative_tolerance"]
        )
        assert sim["scale"][pair]["training_end"] < d["split"]
        assert len(sim["stability"][pair]) == len(d["J_grid"])
        eligible = [x["J"] for x in sim["stability"][pair] if x["fraction"] <= 0.01]
        assert eligible == sim["eligible"][pair]
        for window in d["windows"]:
            for row in results["pairs"][pair]["rolling"][str(window)]:
                distances = [r["distance"] for r in row["rows"]]
                assert row["J_hat"] == eligible[int(np.argmin(distances))]
                assert row["confidence_set"] == [
                    r["J"] for r in row["rows"] if r["p_value"] >= 0.1
                ]
    assert len(results["controls"]) == 8
    assert all(r["n"] >= 300 for r in results["controls"])
    assert len(results["specifications"]) == 12
    assert sum(r["primary"] for r in results["specifications"]) == 1
    for spec in results["specifications"]:
        for event in spec["matched"]:
            assert all(
                abs(i - event["start_position"])
                > d["event"]["minimum_spacing_sessions"]
                for i in event["control_positions"]
            )
    assert all(r["horizon"] == 21 for r in results["horizons"])


def test_fx_notebook_executes_from_clean_kernel_without_upstream_network(tmp_path):
    import sys

    nbformat = pytest.importorskip(
        "nbformat", reason="install pbmrs[notebook] for execution tests"
    )
    nbclient = pytest.importorskip(
        "nbclient", reason="install pbmrs[notebook] for execution tests"
    )
    from jupyter_client import KernelManager

    notebook = nbformat.read(
        ROOT / "notebooks/03_fx_regime_calibration.ipynb", as_version=4
    )
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    code_cells[0].source = (
        "import pbmrs_core.commodities as _data\n"
        "def _no_network(*args, **kwargs):\n"
        "    raise AssertionError('upstream network used during offline notebook execution')\n"
        "_data.urlopen = _no_network\n" + code_cells[0].source
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = nbclient.NotebookClient(
        notebook,
        timeout=300,
        km=manager,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    )
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    counts = [c.execution_count for c in notebook.cells if c.cell_type == "code"]
    assert counts == list(range(1, len(counts) + 1))
    exhibits = [
        c for c in notebook.cells if "main-exhibit" in c.metadata.get("tags", [])
    ]
    assert [c.metadata.exhibit for c in exhibits] == list(range(1, 12))
    assert all(c.outputs for c in exhibits)
    nbformat.write(notebook, tmp_path / "executed_fx.ipynb")


def test_fx_raw_hashes_do_not_accept_newline_changes(tmp_path):
    import pytest

    from pbmrs_core.commodities import DataFetchError

    body = b"DATE,VALUE\n2020-01-01,1\n"
    (tmp_path / "response.csv").write_bytes(body.replace(b"\n", b"\r\n"))
    entry = {
        "cache_file": "response.csv",
        "byte_count": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "entries": {"response.csv": entry}})
    )
    with pytest.raises(DataFetchError):
        validate_raw_tree(tmp_path)


def test_discovery_only_metadata_may_be_absent_from_offline_checkout(tmp_path):
    entry = {
        "cache_file": "cftc_metadata_generated.json",
        "byte_count": 123,
        "sha256": "0" * 64,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "entries": {"metadata": entry}})
    )
    assert validate_raw_tree(tmp_path)["cftc_metadata_generated.json"] == entry
