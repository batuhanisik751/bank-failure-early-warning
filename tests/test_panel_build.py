"""Panel join on small synthetic frames, plus a write/read round trip in a tmp data_dir."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bankcanary.config import load_settings
from bankcanary.panel.build import assemble_panel, build_panel, panel_summary
from bankcanary.storage import duckdb as warehouse
from bankcanary.storage.parquet import read_table, write_table


@pytest.fixture
def settings(tmp_path: Path):
    return load_settings().model_copy(update={"data_dir": tmp_path / "data"})


def _dates(*values):
    return pd.to_datetime(list(values)).astype("datetime64[ns]")


def financials():
    # cert 1: two quarters, second with a gap in bkclass; cert 2: no holding company (0);
    # cert 3: not in institutions at all.
    return pd.DataFrame(
        {
            "cert": pd.array([1, 1, 2, 3], dtype="Int64"),
            "repdte": _dates("2009-03-31", "2009-06-30", "2009-03-31", "2009-03-31"),
            "asset": [100.0, 110.0, 50.0, 10.0],
            "bkclass": ["N", None, "SM", "NM"],
            "stalp": ["TX", "TX", None, "CA"],
            "rssdhcr": pd.array([None, 55, 0, None], dtype="Int64"),
            "fed_rssd": pd.array([None, None, None, None], dtype="Int64"),
            "estymd": _dates("1990-01-01", "1990-01-01", "1980-05-05", "2000-01-01"),
            "cb": [True, True, False, True],
        }
    )


def institutions():
    return pd.DataFrame(
        {
            "cert": pd.array([1, 2], dtype="Int64"),
            "estymd": _dates("1990-01-01", "1980-05-05"),
            "bkclass": ["NM", "SM"],
            "stalp": ["TX", "OK"],
            "rssdhcr": pd.array([99, None], dtype="Int64"),
            "fed_rssd": pd.array([1001, 1002], dtype="Int64"),
            "cb": [True, False],
            "latitude": [30.0, 35.0],
            "longitude": [-97.0, -96.0],
            "active": [False, True],
            "endefymd": _dates("2010-01-15", None),
        }
    )


def exits():
    return pd.DataFrame(
        {
            "cert": pd.array([1, 2], dtype="Int64"),
            "fail_date": _dates("2010-01-15", None),
            "assisted": [False, True],
            "exit_date": _dates(None, None),
            "exit_reason": pd.Series([None, None], dtype="str"),
        }
    )


def test_assemble_panel_join_rules():
    p = assemble_panel(financials(), institutions(), exits(), 60)
    assert len(p) == 4 and p.columns[:2].tolist() == ["cert", "repdte"]
    assert set(financials().columns) <= set(p.columns)
    assert (p["avail_date"] == p["repdte"] + pd.Timedelta(days=60)).all()
    row = p.set_index(["cert", "repdte"])
    q1, q2 = pd.Timestamp("2009-03-31"), pd.Timestamp("2009-06-30")
    # point-in-time values kept; gaps filled from the institution record
    assert row.loc[(1, q1), "bkclass"] == "N" and row.loc[(1, q2), "bkclass"] == "NM"
    assert row.loc[(2, q1), "stalp"] == "OK"
    # rssdhcr never filled: a null quarter means no holding company then
    assert pd.isna(row.loc[(1, q1), "rssdhcr"]) and not row.loc[(1, q1), "has_holding_company"]
    assert row.loc[(1, q2), "rssdhcr"] == 55 and bool(row.loc[(1, q2), "has_holding_company"])
    assert row.loc[(2, q1), "rssdhcr"] == 0 and not row.loc[(2, q1), "has_holding_company"]
    # identifiers and coordinates come from institutions
    assert row.loc[(1, q1), "fed_rssd"] == 1001 and row.loc[(1, q1), "latitude"] == 30.0
    # exit columns joined; missing institution / exits rows leave NA but keep the row
    assert row.loc[(1, q1), "fail_date"] == pd.Timestamp("2010-01-15")
    assert bool(row.loc[(2, q1), "assisted"]) and not row.loc[(3, q1), "assisted"]
    assert pd.isna(row.loc[(3, q1), "fed_rssd"]) and pd.isna(row.loc[(3, q1), "fail_date"])
    assert p["has_holding_company"].dtype == bool and p["assisted"].dtype == bool


def test_panel_summary_counts_matched_failures():
    p = assemble_panel(financials(), institutions(), exits(), 60)
    ex = exits()
    # add a failed bank with no prior report: matched share must drop to 1/2
    extra = pd.DataFrame(
        {"cert": pd.array([9], dtype="Int64"), "fail_date": _dates("2011-01-01"),
         "assisted": [False], "exit_date": _dates(None),
         "exit_reason": pd.Series([None], dtype="str")}
    )  # fmt: skip
    s = panel_summary(p, pd.concat([ex, extra], ignore_index=True), institutions())
    assert s["rows"] == 4 and s["banks"] == 3 and s["banks_missing_institution"] == 1
    assert s["failed_banks"] == 2 and s["failed_matched"] == 1 and s["failed_matched_share"] == 0.5
    assert s["exit_reasons"] == {"none": 3}


def test_build_panel_round_trip(settings):
    write_table(financials(), "financials_raw", settings=settings)
    write_table(institutions(), "institutions", settings=settings)
    failures = pd.DataFrame(
        {"cert": pd.array([1], dtype="Int64"), "fail_date": _dates("2010-01-15"),
         "restype": ["FAILURE"]}
    )  # fmt: skip
    write_table(failures, "failures", settings=settings)
    history = pd.DataFrame(
        {"cert": pd.array([2], dtype="Int64"), "transnum": pd.array([1], dtype="Int64"),
         "subject_cert": pd.array([2], dtype="Int64"), "effdate": _dates("2001-01-01"),
         "changecode": pd.array([110], dtype="Int64")}
    )  # fmt: skip
    write_table(history, "history", settings=settings)
    panel, summary = build_panel(settings)
    back = read_table("panel", settings=settings)
    assert len(back) == 4 and summary["duckdb_rows"] == 4 and summary["failed_matched"] == 1
    assert str(back["avail_date"].dtype) == "datetime64[ns]"
    with warehouse.connect(settings, read_only=True) as con:
        n = con.execute("select count(*) from panel where fail_date is not null").fetchone()[0]
    assert n == 2
