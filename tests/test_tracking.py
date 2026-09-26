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


def test_prune_drops_unreferenced_and_older_generations_then_rebuilds_the_index(tmp_path):
    import os

    settings = make_settings(tmp_path)
    root = tracking.runs_dir(settings)

    def run(name, config):
        r = tracking.start_run(name, config, settings)
        r.log_metrics({"pr_auc": 0.1})
        r.finish()
        return r

    kept_wf = run("walkforward", {"model": "logit", "horizon": 4, "test_year": 2010})
    gone_wf = run("walkforward", {"model": "logit", "horizon": 4, "test_year": 2010, "C": 0.1})
    tune = run("tune_walkforward", {"model": "logit", "horizon": 4, "params": {"C": 1.0}})
    old = run("calibrate", {"model": "logit", "horizon": 4, "test_year": 2010, "n_rows": 10})
    os.utime(old.dir / "metrics.json", (1, 1))
    os.utime(old.dir / "config.json", (1, 1))
    new = run("calibrate", {"model": "logit", "horizon": 4, "test_year": 2010, "min_bin": 50})
    other = run("calibrate", {"model": "hazard", "horizon": 4, "test_year": 2010})
    train = run("train", {"model": "gbdt", "horizon": 4})
    (root / "explain" / "stray").mkdir(parents=True)
    # the subject ignores derived fields and wall-clock stamps, not the run's identity
    assert tracking.run_subject("calibrate", old.config) == tracking.run_subject(
        "calibrate", {**new.config, "started_at": "2026-01-01T00:00:00"}
    )
    assert tracking.run_subject("calibrate", old.config) != tracking.run_subject(
        "calibrate", other.config
    )
    referenced = {kept_wf.run_id}
    plan = tracking.prune_plan(settings, referenced)
    assert plan == [old.dir, tune.dir, gone_wf.dir]
    dry = tracking.prune(settings, referenced, dry_run=True)
    assert dry["index_rows"] is None and gone_wf.dir.exists() and old.dir.exists()
    assert dry["deleted"] == [str(p.relative_to(root)) for p in plan]
    result = tracking.prune(settings, referenced)
    assert result["deleted"] == dry["deleted"] and result["index_rows"] == 4
    assert not any(p.exists() for p in plan)
    assert all(r.dir.exists() for r in (kept_wf, new, other, train))
    assert (root / "explain" / "stray").exists()
    ids = {r["run_id"] for r in tracking.read_index(settings)}
    assert ids == {kept_wf.run_id, new.run_id, other.run_id, train.run_id}
    assert tracking.prune(settings, referenced)["deleted"] == []


def test_runs_prune_cli_reads_the_saved_models_and_honours_dry_run(tmp_path):
    from typer.testing import CliRunner

    from bankcanary.cli import app

    settings = make_settings(tmp_path)
    config = {"model": "logit", "horizon": 4, "test_year": 2010, "C": 0.1}
    kept = tracking.start_run("walkforward", config, settings)
    kept.finish()
    gone = tracking.start_run("walkforward", {**config, "C": 1.0}, settings)
    gone.finish()
    model_dir = tmp_path / "models" / "walkforward" / "2010" / "logit"
    model_dir.mkdir(parents=True)
    stamped = {**config, "model_hash": "0" * 40, "model_version": "logit-2008-12-31-0000000"}
    (model_dir / "config.json").write_text(json.dumps(stamped), encoding="utf-8")
    args = ["--runs-dir", str(settings.runs_dir), "--models-dir", str(tmp_path / "models")]
    dry = CliRunner().invoke(app, ["runs", "prune", "--dry-run", *args])
    assert dry.exit_code == 0, dry.output
    assert dry.output.splitlines() == [
        f"would delete walkforward/{gone.run_id}",
        "1 run(s) would delete; 2 referenced run ids",
    ]
    assert gone.dir.exists()
    real = CliRunner().invoke(app, ["runs", "prune", *args])
    assert real.exit_code == 0, real.output
    last = real.output.splitlines()[-1]
    assert last == "1 run(s) deleted; 2 referenced run ids; index rebuilt: 1 runs"
    assert kept.dir.exists() and not gone.dir.exists()
    assert [r["run_id"] for r in tracking.read_index(settings)] == [kept.run_id]
