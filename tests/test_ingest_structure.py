"""Ingest tests for failures, institutions and history: fixtures only, no network."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import tenacity

from bankcanary.config import load_settings
from bankcanary.ingest import parse_dates, to_int
from bankcanary.ingest.failures import check_benchmarks, clean_failures, failures_per_year
from bankcanary.ingest.history import clean_history, fetch_history
from bankcanary.ingest.institutions import clean_institutions
from bankcanary.sources.fdic import FdicClient, cache_path
from bankcanary.storage.parquet import read_table, write_table

FIXTURES = Path(__file__).parent / "fixtures" / "fdic"


def rows(name: str) -> list[dict]:
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return [r["data"] for r in json.load(fh)["data"]]


@pytest.fixture
def settings(tmp_path: Path):
    base = load_settings()
    fdic = base.fdic.model_copy(update={"requests_per_second": 10_000.0, "max_retries": 2})
    return base.model_copy(update={"data_dir": tmp_path / "data", "fdic": fdic})


def test_parse_dates_handles_all_fdic_formats():
    got = parse_dates(pd.Series(["3/10/2023", "03/12/2023", "2021-07-01T00:00:00", "", None,
                                 "12/31/9999"]))  # fmt: skip
    assert str(got.dtype) == "datetime64[ns]"
    assert got.iloc[0] == pd.Timestamp("2023-03-10") and got.iloc[2] == pd.Timestamp("2021-07-01")
    assert got.iloc[3:].isna().all()


def test_to_int_coerces_strings_and_blanks():
    got = to_int(pd.Series(["573401", "", None, 12, "7.0"]))
    assert str(got.dtype) == "Int64" and got.tolist()[:1] == [573401]
    assert got.isna().tolist() == [False, True, True, False, False]


def test_clean_failures_types_and_null_cert_rows():
    df = clean_failures(rows("failures_structure.json"))
    assert len(df) == 6 and df.columns[:2].tolist() == ["cert", "fail_date"]
    assert str(df["fail_date"].dtype) == "datetime64[ns]" and str(df["fail_year"].dtype) == "Int64"
    assert df["cert"].isna().sum() == 2  # pre-1966 failures without a certificate are kept
    svb = df[df["cert"] == 24735].iloc[0]
    assert svb["fail_date"] == pd.Timestamp("2023-03-10") and svb["restype"] == "FAILURE"
    assert svb["fail_year"] == 2023 and svb["pstalp"] == "CA" and svb["cost"] > 0
    assert set(df["restype"]) == {"FAILURE", "ASSISTANCE"}
    assert df["uninsdep"].dtype == "float64"
    assert failures_per_year(df).to_dict() == {2023: 2}
    with pytest.raises(AssertionError):
        check_benchmarks(df)


def test_clean_institutions_open_bank_sentinel_and_rssd_strings():
    df = clean_institutions(rows("institutions_structure.json"))
    assert df["cert"].tolist() == [4, 3511, 24735, 57053]
    wells = df[df["cert"] == 3511].iloc[0]
    assert pd.isna(wells["endefymd"]) and wells["active"] and not wells["inactive"]
    svb = df[df["cert"] == 24735].iloc[0]
    assert svb["endefymd"] == pd.Timestamp("2023-03-10") and not svb["active"]
    assert str(df["fed_rssd"].dtype) == "Int64" and str(df["rssdhcr"].dtype) == "Int64"
    assert str(df["estymd"].dtype) == "datetime64[ns]" and df["asset"].dtype == "float64"
    assert df["active"].dtype == bool and str(df["ultcert"].dtype) == "Int64"


def test_clean_institutions_treats_empty_strings_as_missing():
    df = clean_institutions(
        [{"CERT": "9", "ENDEFYMD": "", "FED_RSSD": "", "RSSDHCR": "", "ACTIVE": "1"}]
    )
    assert pd.isna(df.loc[0, "endefymd"]) and pd.isna(df.loc[0, "rssdhcr"])
    assert df.loc[0, "cert"] == 9 and bool(df.loc[0, "active"])


def test_clean_history_subject_cert_and_iso_dates():
    df = clean_history(rows("history_structure.json"))
    assert df.columns[:2].tolist() == ["cert", "transnum"]
    assert str(df["effdate"].dtype) == "datetime64[ns]" and str(df["changecode"].dtype) == "Int64"
    fail = df[df["changecode"] == 211]
    assert len(fail) == 2 and fail["cert"].isna().all()
    assert sorted(fail["subject_cert"].tolist()) == [24735, 57053]
    assert sorted(fail["acq_cert"].tolist()) == [59331, 59332]
    assert (
        df[df["changecode"] == 110]["subject_cert"].tolist()
        == df[df["changecode"] == 110]["cert"].tolist()
    )
    assert (df["changecode"] < 500).all()


def test_clean_history_drops_branch_events_and_exact_duplicates():
    row = {"CERT": 1, "TRANSNUM": 0, "CHANGECODE": 223, "EFFDATE": "2001-01-01T00:00:00"}
    df = clean_history([row, dict(row), {**row, "CHANGECODE": 510}])
    assert len(df) == 1 and df.loc[0, "changecode"] == 223


def test_fetch_history_uses_mock_transport_and_round_trips_parquet(settings):
    with open(FIXTURES / "history_structure.json", encoding="utf-8") as fh:
        page = json.load(fh)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["filters"] == "CHANGECODE:[1 TO 499]"
        assert request.url.params["sort_by"] == "TRANSNUM"
        return httpx.Response(200, json=page)

    client = FdicClient(settings, transport=httpx.MockTransport(handler), wait=tenacity.wait_none())
    with client:
        df = fetch_history(settings, client=client)
    assert cache_path(settings, "history", "all").exists()
    write_table(df, "history", settings=settings)
    back = read_table("history", settings=settings)
    assert len(back) == len(df) == len(page["data"])
    assert str(back["effdate"].dtype) == "datetime64[ns]"
    assert back["subject_cert"].tolist() == df["subject_cert"].tolist()
