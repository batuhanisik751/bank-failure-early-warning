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
