"""Sensitivity analyses on a synthetic v2 + panel-dates frame: no data files, no network."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from bankcanary import tracking
from bankcanary.evaluation import sensitivity as s
from bankcanary.labels.build import build_labels, horizon_columns
from bankcanary.splits import prediction_date
from tests.test_hazard import LAG, make_frame, make_settings

AS_OF = dt.date(2016, 6, 30)


def make_panel_frame(n_certs: int = 40, seed: int = 0) -> pd.DataFrame:
    """v2 features with panel dates and labels built by ``build_labels`` at the 60-day lag.

    Every fourth bank fails on a date spread over 2004-2014 and its ``noncurrent_ratio``
    jumps in the six quarters before, so both models have something to learn; every
    seventh bank merges away instead (an ``exit_date``), which censors the windows
    around it.
    """
    frame = make_frame(n_certs, seed).drop(columns=["fail_date", "rssdhcr"])
    frame = frame.drop(columns=s.label_columns(frame))
    rng = np.random.default_rng(seed)
    certs = sorted(frame["cert"].unique())
    fail = {
        c: pd.Timestamp("2004-01-15") + pd.Timedelta(days=int(rng.integers(0, 3900)))
        for c in certs
        if c % 4 == 0
    }
    exit_ = {
        c: pd.Timestamp("2005-01-15") + pd.Timedelta(days=int(rng.integers(0, 3600)))
        for c in certs
        if c % 7 == 0 and c not in fail
    }
    frame = frame.assign(
        avail_date=frame["repdte"] + pd.Timedelta(days=LAG),
        fail_date=frame["cert"].map(fail).astype("datetime64[ns]"),
        exit_date=frame["cert"].map(exit_).astype("datetime64[ns]"),
        rssdhcr=0,
    )
    lead = (frame["fail_date"] - frame["repdte"]).dt.days
    signal = ((lead > 0) & (lead <= 550)).fillna(False)
    frame.loc[signal, "noncurrent_ratio"] = frame.loc[signal, "noncurrent_ratio"] + 3.0
    labels = build_labels(frame, [4, 8], AS_OF)
    return frame.merge(labels.drop(columns=["window_start"]), on=["cert", "repdte"])


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_panel_frame()


def test_build_labels_lag_override_shifts_the_window_and_the_drops(frame):
    base = build_labels(frame, [4], AS_OF)
    later = build_labels(frame, [4], AS_OF, lag_days=90)
    end = horizon_columns(4)[1]
    assert ((later[end] - base[end]) == pd.Timedelta(days=30)).all()
    assert (later["window_start"] == frame["repdte"] + pd.Timedelta(days=90)).all()
    assert later["dropped_failed_before_avail"].sum() >= base["dropped_failed_before_avail"].sum()
    assert build_labels(frame, [4], AS_OF, lag_days=LAG)[end].equals(base[end])
    with pytest.raises(ValueError, match="non-negative"):
        build_labels(frame, [4], AS_OF, lag_days=-1)


def test_variants_enumerate_each_analysis_and_reject_unknown_names(tmp_path):
    settings = make_settings(tmp_path)
    by = {w: s.variants(w, settings) for w in s.ANALYSES}
    assert [v.name for v in by["horizon"]] == ["4q", "8q", "4q", "8q"]
    assert [v.name for v in by["censored"]] == ["kept", "dropped"] * 2
    assert [v.name for v in by["lag"]] == ["45d", "60d", "90d"] * 2
    assert [v.model for v in by["lag"]] == ["logit"] * 3 + ["gbdt"] * 3
    assert len(s.variants("all", settings)) == 14
    assert len(s.variants("all", settings, ["logit"])) == 7
    with pytest.raises(ValueError, match="unknown analysis"):
        s.variants("cox", settings)


def test_relabel_keeps_rows_and_rebuilds_only_the_label_columns(frame):
    out = s.relabel(frame, [4, 8], AS_OF, 90)
    assert len(out) == len(frame) and list(out["cert"]) == list(frame["cert"])
    assert set(out.columns) == set(frame.columns)
    end = horizon_columns(4)[1]
    assert ((out[end] - frame[end]) == pd.Timedelta(days=30)).all()
    assert out["noncurrent_ratio"].equals(frame["noncurrent_ratio"])
    assert s.relabel(frame, [4], AS_OF, LAG)[horizon_columns(4)[0]].equals(frame["y_4q"])


def test_split_masks_drop_censored_rows_from_both_sides(tmp_path, frame):
    settings = make_settings(tmp_path)
    kept = s.Variant("censored", "logit")
    dropped = s.Variant("censored", "logit", drop_censored=True)
    train_k, test_k = s.split_masks(frame, settings, kept)
    train_d, test_d = s.split_masks(frame, settings, dropped)
    censored = frame[horizon_columns(4)[2]]
    assert (train_k & censored).any() and (test_k & censored).any()
    assert not (train_d & censored).any() and not (test_d & censored).any()
    assert (train_k & ~censored).equals(train_d) and (test_k & ~censored).equals(test_d)


@pytest.mark.parametrize("model", s.MODELS)
def test_fit_variant_logs_a_deterministic_leak_free_run(tmp_path, frame, model):
    settings = make_settings(tmp_path)
    variant = s.Variant("horizon", model, horizon=8)
    result = s.fit_variant(frame, settings, variant, as_of_date=AS_OF)
    again = s.fit_variant(frame, settings, variant, as_of_date=AS_OF)
    assert result.run_dir == again.run_dir
    assert result.run_dir.name == tracking.run_id("sensitivity", result.config)
    assert result.run_dir.name.startswith("sensitivity-8q-")
    cfg = json.loads((result.run_dir / "config.json").read_text("utf-8"))
    assert cfg["analysis"] == "horizon" and cfg["variant"] == "8q" and cfg["model"] == model
    assert cfg["label"] == "y_8q" and cfg["availability_lag_days"] == LAG
    assert cfg["drop_censored"] is False and cfg["features_version"] == "v2"
    cutoff = prediction_date(settings.fixed_split.test_start, LAG)
    assert cfg["first_test_prediction_date"] == str(cutoff.date())
    assert pd.Timestamp(cfg["train_repdte_max"]) + pd.DateOffset(months=24) < cutoff
    met = json.loads((result.run_dir / "metrics.json").read_text("utf-8"))
    for key in ("pr_auc", "roc_auc", "recall_at_2pct", "recall_at_top100", "brier", "n"):
        assert key in met
    assert met["n"] == cfg["n_test"] and met["n_failures"] == cfg["positives_test"]
    assert 0.0 <= met["brier"] <= 1.0
    assert set(result.scores["year"]) == {2010, 2011, 2012, 2013}
    assert [r["run_id"] for r in tracking.read_index(settings)].count(result.run_dir.name) == 1


def test_lag_variant_relabels_and_moves_the_prediction_date(tmp_path, frame):
    settings = make_settings(tmp_path)
    end = horizon_columns(4)[1]
    short = s.fit_variant(frame, settings, s.Variant("lag", "logit", lag_days=45), AS_OF)
    long = s.fit_variant(frame, settings, s.Variant("lag", "logit", lag_days=90), AS_OF)
    assert short.config["availability_lag_days"] == 45
    assert long.config["availability_lag_days"] == 90
    for res, lag in ((short, 45), (long, 90)):
        assert res.config["first_test_prediction_date"] == str(
            prediction_date(settings.fixed_split.test_start, lag).date()
        )
        assert res.config["variant"] == f"{lag}d"
    assert short.run_dir != long.run_dir
    # Neither call rewrote the caller's labels: the 60-day windows are still in ``frame``.
    assert (frame[end] == frame["repdte"] + pd.Timedelta(days=LAG) + pd.DateOffset(months=12)).all()


def test_report_lists_tables_for_run_analyses_and_pending_for_the_rest(tmp_path, frame):
    settings = make_settings(tmp_path)
    for variant in s.variants("censored", settings, ["logit"]):
        s.fit_variant(frame, settings, variant, AS_OF)
    runs = s.read_runs(settings)
    table = s.analysis_table(runs, "censored")
    assert list(table["variant"]) == ["kept", "dropped"]
    assert table.loc[1, "n"] < table.loc[0, "n"]
    lines = s.interpretation(table, "censored", settings)
    assert len(lines) == 2 and lines[0].startswith("What moves") and "logit" in lines[0]
    path = s.write_sensitivity_report(settings, tmp_path / "reports" / "sensitivity.md")
    text = path.read_text("utf-8")
    assert "| model | variant |" in text and "| logit | dropped |" in text
    assert text.count("Pending: run `bankcanary sensitivity --which") == 2
    assert s.analysis_table(runs, "lag").empty
