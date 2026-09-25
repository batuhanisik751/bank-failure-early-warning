"""Financials ingest tests: quarter list, typing on a 3-row fixture, mocked pull. No network."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import pytest
import tenacity

from bankcanary.config import load_settings
from bankcanary.ingest.financials import (
    build_financials_raw,
    clean_financials,
    fetch_financials,
    quarter_counts,
    quarter_ends,
    repdte_code,
)
from bankcanary.sources.fdic import FdicClient, cache_path
from bankcanary.storage.parquet import read_table, write_table

FIXTURE = Path(__file__).parent / "fixtures" / "fdic" / "financials_page.json"


@pytest.fixture
def page() -> dict:
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def settings(tmp_path: Path):
    base = load_settings()
    fdic = base.fdic.model_copy(update={"requests_per_second": 10_000.0, "max_retries": 2})
    return base.model_copy(update={"data_dir": tmp_path / "data", "fdic": fdic})


def test_quarter_ends_snaps_and_covers_range():
    got = quarter_ends(dt.date(2001, 3, 31), dt.date(2002, 12, 31))
    assert got[0] == dt.date(2001, 3, 31) and got[-1] == dt.date(2002, 12, 31)
    assert len(got) == 8 and all(d.month in (3, 6, 9, 12) for d in got)
    assert quarter_ends(dt.date(2024, 2, 1), dt.date(2024, 7, 1)) == [
        dt.date(2024, 3, 31), dt.date(2024, 6, 30), dt.date(2024, 9, 30)]  # fmt: skip
    assert len(quarter_ends(dt.date(2001, 3, 31), dt.date(2026, 6, 30))) == 102
    assert repdte_code(dt.date(2026, 6, 30)) == "20260630"


def test_clean_financials_types_and_drops_empty_shells(page):
    rows = [r["data"] for r in page["data"]]
    df = clean_financials(rows)
    assert len(df) == 2 and df["cert"].tolist() == [3511, 24735]  # asset == 0 row dropped
    assert list(df.columns[:2]) == ["cert", "repdte"] and len(df.columns) == 107
    assert str(df["repdte"].dtype) == "datetime64[ns]"
    assert df["repdte"].iloc[0] == dt.datetime(2009, 3, 31)
    assert str(df["estymd"].dtype) == "datetime64[ns]"
    assert df["estymd"].iloc[1] == dt.datetime(1983, 10, 17)
    assert str(df["cert"].dtype) == "Int64" and str(df["rssdhcr"].dtype) == "Int64"
    assert str(df["cb"].dtype) == "bool" and df["cb"].tolist() == [False, False]
    assert str(df["name"].dtype) == "str" and df["stalp"].tolist() == ["SD", "CA"]
    for col in ("asset", "netinc", "rbct1j", "cblrind", "depsmb"):
        assert str(df[col].dtype) == "float64", col
    assert df["netinc"].iloc[0] == 2125000.0 and df["netinc"].isna().iloc[1]
    assert df["rbct1j"].isna().iloc[1] and df["depsmb"].isna().all()  # never sent -> NaN


def test_fetch_financials_caches_per_quarter_and_builds_table(settings, page):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        filters = request.url.params["filters"]
        calls.append(filters)
        assert request.url.params["sort_by"] == "CERT"
        assert "RBCT1J" in request.url.params["fields"].split(",")
        stamped = json.loads(json.dumps(page))  # same three banks, dated as requested
        for record in stamped["data"]:
            record["data"]["REPDTE"] = filters.split(":")[1]
        return httpx.Response(200, json=stamped)

    client = FdicClient(settings, transport=httpx.MockTransport(handler), wait=tenacity.wait_none())
    start, end = dt.date(2009, 3, 31), dt.date(2009, 6, 30)
    with client:
        counts = fetch_financials(settings, client=client, start=start, end=end)
        again = fetch_financials(settings, client=client, start=start, end=end)
    assert calls == ["REPDTE:20090331", "REPDTE:20090630"]  # second run served from cache
    assert counts == again == {start: 3, end: 3}
    assert cache_path(settings, "financials", "20090331").exists()

    df = build_financials_raw(settings)
    assert len(df) == 4 and quarter_counts(df).tolist() == [2, 2]
    write_table(df, "financials_raw", settings=settings)
    back = read_table("financials_raw", settings=settings)
    assert back["asset"].tolist() == df["asset"].tolist()
    assert str(back["repdte"].dtype) == "datetime64[ns]"
