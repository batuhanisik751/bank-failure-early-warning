"""Spec section 5 label rules on synthetic bank-quarters (no data files, no network)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from bankcanary.config import load_settings
from bankcanary.labels.build import (
    build_labels,
    build_labels_table,
    failures_as_of_date,
    horizon_columns,
    label_summary,
)
from bankcanary.storage import duckdb as warehouse
from bankcanary.storage.parquet import read_table, write_table

AS_OF = dt.date(2026, 9, 25)
LAG = pd.Timedelta(days=60)

# Report 2008-12-31 -> avail 2009-03-01 -> 4q window ends 2010-03-01, 8q 2011-03-01.
Q = pd.Timestamp("2008-12-31")
AVAIL = Q + LAG
END_4Q = pd.Timestamp("2010-03-01")
END_8Q = pd.Timestamp("2011-03-01")


def _row(cert, repdte, fail=None, exit_=None, assisted=False):
    return {
        "cert": cert,
        "repdte": pd.Timestamp(repdte),
        "avail_date": pd.Timestamp(repdte) + LAG,
        "fail_date": pd.Timestamp(fail) if fail else pd.NaT,
        "exit_date": pd.Timestamp(exit_) if exit_ else pd.NaT,
        "assisted": assisted,
    }


def synthetic_panel() -> pd.DataFrame:
    rows = [
        _row(1, Q, fail=AVAIL),  # failed on avail(q): dropped
        _row(2, Q, fail=AVAIL + pd.Timedelta(days=1)),  # avail + 1 day: y = 1
        _row(3, Q, fail=END_4Q),  # exactly on window_end_4q: y_4q = 1
        _row(4, Q, fail=END_4Q + pd.Timedelta(days=1)),  # window_end + 1: y_4q = 0
        _row(5, Q, exit_="2009-09-30"),  # merged inside the window: censored
        _row(6, "2026-06-30"),  # recent quarter: label incomplete
        _row(7, "2009-03-31", fail="2009-01-16"),  # report dated after the failure
        _row(8, Q, assisted=True),  # open-bank assistance only: y = 0
        _row(9, Q),  # open bank, nothing happens
        _row(10, Q, exit_=END_8Q + pd.Timedelta(days=1)),  # exit after both windows
        _row(11, "2007-12-31", fail="2009-02-28"),  # avail 2008-02-29 (leap day)
    ]
    frame = pd.DataFrame(rows)
    for col in ("repdte", "avail_date", "fail_date", "exit_date"):
        frame[col] = pd.to_datetime(frame[col]).astype("datetime64[ns]")
    return frame


@pytest.fixture(scope="module")
def labels() -> pd.DataFrame:
    return build_labels(synthetic_panel(), [4, 8], AS_OF).set_index("cert")


def test_failure_on_avail_day_is_dropped(labels):
    assert labels.loc[1, "dropped_failed_before_avail"]
    assert labels.loc[1, "y_4q"] == 0 and labels.loc[1, "y_8q"] == 0


def test_failure_one_day_after_avail_is_positive(labels):
    assert not labels.loc[2, "dropped_failed_before_avail"]
    assert labels.loc[2, "y_4q"] == 1 and labels.loc[2, "y_8q"] == 1


def test_failure_exactly_on_window_end_is_positive(labels):
    assert labels.loc[3, "window_end_4q"] == END_4Q
    assert labels.loc[3, "y_4q"] == 1


def test_failure_one_day_after_window_end_is_negative(labels):
    assert labels.loc[4, "y_4q"] == 0
    assert labels.loc[4, "y_8q"] == 1  # but inside the 8-quarter window
    assert not labels.loc[4, "censored_in_window_4q"]


def test_merged_bank_inside_window_is_censored(labels):
    assert labels.loc[5, "censored_in_window_4q"] and labels.loc[5, "censored_in_window_8q"]
    assert labels.loc[5, "y_4q"] == 0 and labels.loc[5, "y_8q"] == 0
    assert not labels.loc[10, "censored_in_window_8q"]  # exit after window_end_8q


def test_recent_quarter_is_not_label_complete(labels):
    assert not labels.loc[6, "label_complete_4q"] and not labels.loc[6, "label_complete_8q"]
    assert labels.loc[6, "y_4q"] == 0
    complete = labels.drop(index=6)
    assert complete["label_complete_4q"].all() and complete["label_complete_8q"].all()


def test_window_arithmetic_across_year_end(labels):
    assert labels.loc[1, "window_start"] == pd.Timestamp("2009-03-01")
    assert labels.loc[1, "window_end_4q"] == pd.Timestamp("2010-03-01")
    assert labels.loc[1, "window_end_8q"] == pd.Timestamp("2011-03-01")
    # 2007-12-31 + 60 days = leap day 2008-02-29; 12 months later clamps to 2009-02-28.
    assert labels.loc[11, "window_start"] == pd.Timestamp("2008-02-29")
    assert labels.loc[11, "window_end_4q"] == pd.Timestamp("2009-02-28")
    assert labels.loc[11, "y_4q"] == 1  # failure on that clamped window end counts


def test_report_dated_after_failure_is_dropped(labels):
    assert labels.loc[7, "dropped_failed_before_avail"]
    assert labels.loc[7, "y_4q"] == 0


def test_assisted_only_bank_is_negative_and_not_censored(labels):
    for h in (4, 8):
        y_col, _, cens_col, complete_col = horizon_columns(h)
        assert labels.loc[8, y_col] == 0
        assert not labels.loc[8, cens_col]
        assert labels.loc[8, complete_col]
    assert not labels.loc[8, "dropped_failed_before_avail"]


def test_open_bank_with_no_dates_is_all_false(labels):
    row = labels.loc[9]
    assert row["y_4q"] == 0 and row["y_8q"] == 0
    assert not row["dropped_failed_before_avail"]
    assert not row["censored_in_window_4q"] and not row["censored_in_window_8q"]


def test_horizons_are_consistent(labels):
    complete = labels[labels["label_complete_8q"] & ~labels["dropped_failed_before_avail"]]
    assert ((complete["y_4q"] == 0) | (complete["y_8q"] == 1)).all()
    assert (complete["window_end_8q"] > complete["window_end_4q"]).all()


def test_column_dtypes_and_order(labels):
    frame = labels.reset_index()
    expected = ["cert", "repdte", "window_start", "dropped_failed_before_avail"]
    expected += horizon_columns(4) + horizon_columns(8)
    assert list(frame.columns) == expected
    assert str(frame["y_4q"].dtype) == "int64"
    assert str(frame["window_end_4q"].dtype) == "datetime64[ns]"
    assert str(frame["censored_in_window_4q"].dtype) == "bool"
    assert str(frame["label_complete_8q"].dtype) == "bool"
    assert set(frame["y_8q"].unique()) <= {0, 1}


def test_missing_panel_column_raises():
    with pytest.raises(KeyError):
        build_labels(synthetic_panel().drop(columns=["exit_date"]), [4], AS_OF)


def test_label_summary_counts_only_usable_rows(labels):
    s = label_summary(labels.reset_index(), [4, 8])
    # 11 rows, 2 dropped (certs 1 and 7), 1 incomplete (cert 6): 8 usable.
    assert s["rows"] == 11 and s["dropped"] == 2
    assert s["rows_4q"] == 8 and s["pos_4q"] == 3 and s["rate_4q"] == pytest.approx(3 / 8)
    assert s["pos_8q"] == 4
    assert s["per_year"].loc[2008, "pos_4q"] == 2 and s["per_year"].loc[2007, "pos_4q"] == 1
    assert 2026 not in s["per_year"].index or s["per_year"].loc[2026, "rows_4q"] == 0


@pytest.fixture
def settings(tmp_path: Path):
    return load_settings().model_copy(update={"data_dir": tmp_path / "data"})


def test_failures_as_of_date_reads_cached_pull(settings):
    path = settings.data_dir / "raw" / "fdic" / "failures" / "all.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"query": {}, "fetched_at": "2026-09-25", "total": 0, "data": []}))
    assert failures_as_of_date(settings) == AS_OF


def test_build_labels_table_round_trip(settings):
    path = settings.data_dir / "raw" / "fdic" / "failures" / "all.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"fetched_at": "2026-09-25T10:00:00", "data": []}))
    write_table(synthetic_panel(), "panel", settings=settings)
    labels, summary = build_labels_table(settings)
    assert summary["as_of_date"] == AS_OF
    assert summary["duckdb_rows"] == 11 and len(labels) == 11
    back = read_table("labels", settings=settings)
    pd.testing.assert_frame_equal(back, labels)
    with warehouse.connect(settings, read_only=True) as con:
        n = con.execute("SELECT count(*) FROM labels WHERE y_4q = 1").fetchone()[0]
    assert n == 3
