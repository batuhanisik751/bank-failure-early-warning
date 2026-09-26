"""Synthetic tests of the refresh decision, its idempotency and the run record (no network)."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
from typer.testing import CliRunner

from bankcanary import refresh
from bankcanary.publish.core import pipeline_run_row

Q1 = dt.date(2026, 3, 31)
Q2 = dt.date(2026, 6, 30)


def test_new_quarter_when_api_is_ahead():
    d = refresh.decide(Q2, Q1)
    assert d.new_quarter and d.quarter == Q2 and "newer" in d.reason
    assert d.source == "database"


def test_no_op_when_published_is_current_or_ahead():
    assert not refresh.decide(Q2, Q2).new_quarter
    assert not refresh.decide(Q1, Q2).new_quarter
    assert refresh.decide(Q1, Q2).quarter is None


def test_first_publish_and_missing_probe():
    first = refresh.decide(Q2, None, source="warehouse")
    assert first.new_quarter and first.quarter == Q2 and "warehouse" in first.reason
    assert not refresh.decide(None, Q1).new_quarter


def test_force_quarter_wins_over_comparison():
    d = refresh.decide(Q2, Q2, force_quarter=Q1)
    assert d.new_quarter and d.quarter == Q1 and "forced" in d.reason


def test_decision_is_idempotent_and_pure():
    a = refresh.decide(Q2, Q1)
    b = refresh.decide(Q2, Q1)
    assert a == b
    assert refresh.decide(Q2, Q2) == refresh.decide(Q2, Q2)


def test_prior_quarter_steps_back_one_quarter_end():
    assert refresh.prior_quarter(Q2) == Q1
    assert refresh.prior_quarter(dt.date(2026, 3, 31)) == dt.date(2025, 12, 31)


def test_merge_quarter_versions_keeps_published_backtest_versions():
    fresh = pd.DataFrame(
        {
            "repdte": [Q1, Q2],
            "label": ["2026Q1", "2026Q2"],
            "n_banks": [10, 11],
            "model_year": pd.array([None, None], dtype="Int64"),
            "model_version": ["gbdt_mono-2022-12-31-new", "gbdt_mono-2022-12-31-new"],
        }
    )
    existing = pd.DataFrame(
        {"repdte": [Q1], "model_year": [2026], "model_version": ["gbdt_mono-2024-12-31-old"]}
    )
    out = refresh.merge_quarter_versions(fresh, existing)
    assert out.loc[0, "model_version"] == "gbdt_mono-2024-12-31-old"
    assert out.loc[0, "model_year"] == 2026
    assert out.loc[1, "model_version"] == "gbdt_mono-2022-12-31-new"
    assert pd.isna(out.loc[1, "model_year"])
    assert refresh.merge_quarter_versions(fresh, None).equals(fresh)


def test_run_record_row_carries_decision_and_rows(tmp_path):
    started = dt.datetime(2026, 9, 28, 9, 0, tzinfo=dt.UTC)
    finished = started + dt.timedelta(minutes=5)
    row = pipeline_run_row(
        "refresh-4q-abc", started, finished, "ok", {"scores": 9000}, "new quarter",
        latest_repdte=Q2, new_quarter=True,
    )  # fmt: skip
    assert row.loc[0, "new_quarter"] is True or bool(row.loc[0, "new_quarter"])
    assert row.loc[0, "latest_repdte"] == Q2
    assert json.loads(row.loc[0, "rows_written"]) == {"scores": 9000}
    assert list(row.columns) == [
        "run_id", "started_at", "finished_at", "status", "latest_repdte", "new_quarter",
        "rows_written", "log",
    ]  # fmt: skip


def test_github_output_written_only_when_requested(tmp_path):
    target = tmp_path / "out.txt"
    assert refresh.write_github_output(refresh.decide(Q2, Q1), "ok", str(target))
    text = target.read_text()
    assert "new_quarter=true" in text and "quarter=2026-06-30" in text and "status=ok" in text
    assert refresh.write_github_output(refresh.decide(Q2, Q1), "failed", str(target))
    assert "new_quarter=false" in target.read_text().splitlines()[-3]
    assert not refresh.write_github_output(refresh.decide(Q2, Q2), "ok", "")


def test_timings_text_and_total():
    t = refresh.Timings()
    t.add("panel", 12.34)
    t.add("labels", 3.0)
    assert t.text() == "panel 12 s, labels 3 s"
    assert t.total() == 15.3


def test_refresh_command_is_registered_and_validates_quarter():
    from bankcanary.cli import app

    result = CliRunner().invoke(app, ["refresh", "--help"])
    assert result.exit_code == 0 and "--force-quarter" in result.output
    bad = CliRunner().invoke(app, ["refresh", "--dry-run", "--force-quarter", "2026-13-99"])
    assert bad.exit_code != 0 and "YYYY-MM-DD" in bad.output
