"""Run tracking: deterministic ids, JSON files, deduplicated index. No data files."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bankcanary import tracking
from bankcanary.config import FixedSplit, Settings


def make_settings(tmp_path) -> Settings:
    split = FixedSplit(
        train_start="2002-03-31",
        train_end="2008-12-31",
        test_start="2010-03-31",
        test_end="2013-12-31",
    )
    return Settings(runs_dir=tmp_path / "runs", fixed_split=split)


def test_run_id_is_deterministic_and_ignores_key_order():
    a = {"horizon": 4, "params": {"num_leaves": 31, "learning_rate": 0.03}, "model": "gbdt"}
    b = {"model": "gbdt", "params": {"learning_rate": 0.03, "num_leaves": 31}, "horizon": 4}
    assert tracking.run_id("train", a) == tracking.run_id("train", b)
    assert tracking.run_id("train", a).startswith("train-4q-")
    assert len(tracking.run_id("train", a)) == len("train-4q-") + 10
    assert tracking.run_id("train", {**a, "horizon": 8}) != tracking.run_id("train", a)
    assert tracking.run_id("train", {**a, "params": {"num_leaves": 63}}) != tracking.run_id(
        "train", a
    )
    with pytest.raises(KeyError):
        tracking.run_id("train", {"model": "gbdt"})


def test_finish_writes_files_and_deduplicates_the_index(tmp_path):
    settings = make_settings(tmp_path)
    config = {"model": "gbdt", "horizon": 4, "params": {"num_leaves": np.int64(31)}}
    run = tracking.start_run("train", config, settings)
    run.log_metrics({"pr_auc": np.float64(0.25), "nan_metric": float("nan"), "n": np.int64(3)})
    run.log_metrics({"by_year": [{"year": 2010, "pr_auc": 0.2}]})
    out = run.finish()
    assert out == tmp_path / "runs" / "train" / run.run_id
    assert json.loads((out / "config.json").read_text())["params"] == {"num_leaves": 31}
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["pr_auc"] == 0.25 and metrics["nan_metric"] is None and metrics["n"] == 3
    assert metrics["by_year"] == [{"year": 2010, "pr_auc": 0.2}]
    # the same config finished again replaces the index row instead of appending one
    again = tracking.start_run("train", config, settings)
    again.log_metrics({"pr_auc": 0.3})
    again.finish()
    other = tracking.start_run("tune", {"horizon": 8, "C": 0.1}, settings)
    other.finish()
    index = tracking.read_index(settings)
    assert [r["run_id"] for r in index] == [run.run_id, other.run_id]
    assert index[0]["metrics"] == {"pr_auc": 0.3}  # scalars only, latest values
    assert index[1] == {"run_id": other.run_id, "name": "tune", "horizon": 8, "metrics": {}}
    assert tracking.find_metrics("train", config, settings) == {"pr_auc": 0.3}
    assert tracking.find_metrics("train", {**config, "horizon": 8}, settings) is None


def test_read_index_is_empty_before_any_run(tmp_path):
    assert tracking.read_index(make_settings(tmp_path)) == []


def test_finish_appends_and_read_index_dedupes_keeping_the_latest_row(tmp_path):
    settings = make_settings(tmp_path)
    config = {"model": "logit", "horizon": 4, "C": 0.1}
    for pr_auc in (0.1, 0.2, 0.3):
        run = tracking.start_run("train", config, settings)
        run.log_metrics({"pr_auc": pr_auc})
        run.finish()
    lines = (tmp_path / "runs" / "index.jsonl").read_text().splitlines()
    assert len(lines) == 3  # finish only appends; nothing is rewritten
    assert [r["metrics"]["pr_auc"] for r in tracking.read_index(settings)] == [0.3]


def test_rebuild_index_scans_run_dirs_sorted_by_id_and_is_idempotent(tmp_path):
    settings = make_settings(tmp_path)
    ids = []
    for c in (0.3, 0.1, 1.0):
        run = tracking.start_run("train", {"horizon": 4, "C": c}, settings)
        run.log_metrics({"pr_auc": c, "grid": [c]})
        run.finish()
        ids.append(run.run_id)
    other = tracking.start_run("tune", {"horizon": 8}, settings)
    other.finish()
    (other.dir / "metrics.json").unlink()  # a run that never logged metrics still lists
    index_path = tmp_path / "runs" / "index.jsonl"
    index_path.write_text("garbage line\n")  # the old index is not trusted
    assert tracking.rebuild_index(settings) == 4
    rows = tracking.read_index(settings)
    assert [r["run_id"] for r in rows] == sorted(ids + [other.run_id])
    assert all(r["metrics"].get("grid") is None for r in rows)  # scalars only
    assert {r["run_id"]: r["metrics"] for r in rows}[other.run_id] == {}
    first = index_path.read_bytes()
    tracking.rebuild_index(settings)
    assert index_path.read_bytes() == first
    assert not list((tmp_path / "runs").glob(".index.jsonl.*"))  # temp file renamed away


def test_runs_list_cli_prints_id_name_horizon_and_headline(tmp_path):
    from typer.testing import CliRunner

    from bankcanary.cli import app

    settings = make_settings(tmp_path)
    fit = tracking.start_run("walkforward", {"horizon": 4, "model": "gbdt"}, settings)
    fit.log_metrics({"pr_auc": 0.25, "n": 10})
    fit.finish()
    cal = tracking.start_run("calibrate", {"horizon": 8, "model": "logit"}, settings)
    cal.log_metrics({"brier_raw": 0.01, "brier_calibrated": 0.0123})
    cal.finish()
    runs_dir = str(tmp_path / "runs")
    result = CliRunner().invoke(app, ["runs", "list", "--runs-dir", runs_dir])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == [fit.run_id, "walkforward", "4q", "pr_auc=0.2500"]
    assert lines[1].split() == [cal.run_id, "calibrate", "8q", "brier_calibrated=0.0123"]
    only = CliRunner().invoke(app, ["runs", "list", "--runs-dir", runs_dir, "--name", "calibrate"])
    assert [line.split() for line in only.output.splitlines()] == [lines[1].split()]
    last = CliRunner().invoke(app, ["runs", "list", "--runs-dir", runs_dir, "--limit", "1"])
    assert [line.split() for line in last.output.splitlines()] == [lines[1].split()]
    rebuilt = CliRunner().invoke(app, ["runs", "rebuild-index", "--runs-dir", runs_dir])
    assert rebuilt.output.strip() == "index rebuilt: 2 runs"
    empty = CliRunner().invoke(app, ["runs", "list", "--runs-dir", str(tmp_path / "none")])
    assert empty.output.strip() == "no runs logged"
