"""Data-quality report on a tiny synthetic warehouse (no network, no real data)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bankcanary.config import load_settings
from bankcanary.fields import DEFAULT_FIELD_MAP_PATH
from bankcanary.quality import report as dq
from bankcanary.storage.parquet import write_table

QUARTERS = ("2001-03-31", "2001-06-30", "2001-09-30", "2001-12-31", "2002-03-31")
Q = [pd.Timestamp(d) for d in QUARTERS]
NUMERIC = ("asset", "eq", "lnlsgr", "nclnls", "netinc", "dep", "bro", "ore", "lnatres")


def _financials() -> pd.DataFrame:
    rows = []
    for cert in (1, 2):
        for i, d in enumerate(Q):
            row = {"cert": cert, "repdte": d, "name": f"BANK {cert}", "stalp": "TX"}
            row.update({c: float(1000 * cert + i) for c in NUMERIC})
            row["eq"] = -5.0 if (cert == 1 and i == 4) else row["eq"]
            row["cblrind"] = None if d.year == 2001 else 0.0  # appears in 2002
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.astype({"cert": "Int64", "cblrind": "float64"})


def _labels(fin: pd.DataFrame) -> pd.DataFrame:
    df = fin[["cert", "repdte"]].copy()
    df["window_start"] = df["repdte"] + pd.Timedelta(days=60)
    df["dropped_failed_before_avail"] = False
    for h in (1, 4, 8):
        df[f"y_{h}q"] = ((df["cert"] == 1) & (df["repdte"] >= Q[2])).astype("int64")
        df[f"window_end_{h}q"] = df["window_start"] + pd.DateOffset(months=3 * h)
        df[f"censored_in_window_{h}q"] = (df["cert"] == 2) & (df["repdte"] == Q[4])
        df[f"label_complete_{h}q"] = df["repdte"] < Q[4]
    return df


@pytest.fixture
def settings(tmp_path: Path):
    s = load_settings().model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    fin = _financials()
    write_table(fin, "financials_raw", settings=s)
    write_table(fin[["cert", "repdte", "asset"]], "panel", settings=s)
    write_table(_labels(fin), "labels", settings=s)
    failures = pd.DataFrame(
        {
            "cert": [1, 9, 2],
            "fail_date": pd.to_datetime(["2002-06-30", "2001-06-30", "2001-09-30"]),
            "name": ["BANK 1", "GHOST BANK", "BANK 2"],
            "restype": ["FAILURE", "FAILURE", "ASSISTANCE"],
        }
    )
    write_table(failures, "failures", settings=s)
    return s


@pytest.fixture
def field_map(tmp_path: Path) -> Path:
    dst = tmp_path / "fields.yaml"
    shutil.copy(DEFAULT_FIELD_MAP_PATH, dst)
    return dst


@pytest.fixture
def con(settings):
    c = dq.open_warehouse(settings)
    yield c
    c.close()


def test_rows_per_quarter_is_one_row_per_year(con):
    table = dq.rows_per_quarter(con)
    assert list(table.columns) == ["year", "Q1", "Q2", "Q3", "Q4"]
    assert table.set_index("year").loc[2001].tolist() == [2, 2, 2, 2]
    assert table.set_index("year").loc[2002].tolist() == [2, 0, 0, 0]


def test_missingness_and_structural_changes(con):
    miss = dq.missingness_by_year(con)
    assert list(miss.columns) == [2001, 2002]
    assert miss.loc["cblrind", 2001] == 1.0 and miss.loc["cblrind", 2002] == 0.0
    assert miss.loc["asset"].tolist() == [0.0, 0.0]
    assert "rbc" not in miss.index  # columns absent from the table are skipped, not nulls
    changes = dq.structural_changes(miss)
    assert changes.to_dict("records") == [
        {"field": "cblrind", "year": 2002, "before": 1.0, "after": 0.0, "change": "appears"}
    ]


def test_first_available_dates(con):
    dates = dq.first_available(con)
    assert dates["asset"] == "2001-03-31" and dates["cblrind"] == "2002-03-31"


def test_failed_bank_matching_lists_unmatched(con):
    m = dq.failed_bank_matching(con)
    assert (m["failures"], m["matched"]) == (2, 1) and m["share"] == 0.5
    assert m["unmatched"]["cert"].tolist() == [9]
    assert m["unmatched"]["name"].tolist() == ["GHOST BANK"]


def test_outlier_summary_counts_negatives(con):
    out = dq.outlier_summary(con).set_index("field")
    assert list(out.index) == list(dq.OUTLIER_FIELDS)
    assert out.loc["eq", "negatives"] == 1 and out.loc["asset", "negatives"] == 0
    assert out.loc["asset", "p50"] == pytest.approx(1502.0)


def test_label_summary_by_year(con):
    table = dq.label_summary_by_year(con, 4).set_index("year")
    assert table.loc[2001, ["rows", "usable", "positives", "censored"]].tolist() == [8, 8, 2, 0]
    assert table.loc[2002, ["rows", "usable", "positives", "censored"]].tolist() == [2, 0, 0, 0]


def test_build_report_writes_markdown_and_field_map(settings, field_map):
    path, sections = dq.build_report(settings, field_map)
    text = path.read_text()
    assert path == settings.reports_dir / "data_quality.md"
    assert "| 2001 | 2 | 2 | 2 | 2 |" in text and "GHOST BANK" in text
    assert "Horizon 4 quarters" in text and "Horizon 8 quarters" in text
    doc = yaml.safe_load(field_map.read_text())
    by_col = {f["column"]: f for f in doc["fields"]}
    assert by_col["cblrind"]["first_available"] == "2002-03-31"
    assert by_col["asset"]["first_available"] == "2001-03-31"
    assert by_col["cert"]["first_available"] is None  # identity fields are not touched
    original = yaml.safe_load(DEFAULT_FIELD_MAP_PATH.read_text())
    assert [f["code"] for f in doc["fields"]] == [f["code"] for f in original["fields"]]
    assert sections["fields_updated"] >= 1
    again, sections2 = dq.build_report(settings, field_map)
    assert again.read_text() == text and sections2["fields_updated"] == 0
