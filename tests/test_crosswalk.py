"""crosswalk_rssd and the securities cross-check on synthetic frames; no data/ reads."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bankcanary.ingest import crosswalk as cw

FIXTURES = Path(__file__).parent / "fixtures" / "ffiec"


def institutions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cert": pd.array([57053, 24735, 99, 24735], dtype="Int64"),
            "name": ["Signature Bank", "Silicon Valley Bank", "No Rssd Bank", "dup"],
            "fed_rssd": pd.array([2942690, 802866, 0, 802866], dtype="Int64"),
            "stalp": ["NY", "CA", "TX", "CA"],
        }
    )


def financials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cert": pd.array([24735, 57053, 24735, 1], dtype="Int64"),
            "repdte": pd.to_datetime(["2022-12-31", "2022-12-31", "2022-09-30", "2022-12-31"]),
            "name": ["SILICON VALLEY BANK", "SIGNATURE BANK", "SILICON VALLEY BANK", "OTHER"],
            "scha": [91_327_000.0, 7_780_398.0, 1.0, 5.0],
            "schf": [76_168_000.0, 7_018_201.0, 1.0, 5.0],
            "scaa": [28_502_000.0, 20_826_240.0, 1.0, 5.0],
            "scaf": [25_976_000.0, 18_372_419.0, 1.0, 5.0],
            "depunins": [151_592_000.0, None, 1.0, 5.0],
        }
    )


def test_build_crosswalk_projects_and_nulls_zero_rssd():
    out = cw.build_crosswalk(institutions())
    assert out.columns.tolist() == ["cert", "fed_rssd", "name", "source"]
    assert out["cert"].tolist() == [99, 24735, 57053]
    assert out["fed_rssd"].isna().tolist() == [True, False, False]
    assert str(out["fed_rssd"].dtype) == "Int64"
    assert (out["source"] == "institutions").all()
    assert cw.missing_rssd_count(out) == 1


def test_securities_crosscheck_long_table_and_published_check():
    table = cw.securities_crosscheck(financials())
    assert len(table) == 10  # two banks x five items; other cert and quarter excluded
    assert table.columns.tolist() == [
        "cert", "name", "repdte", "item", "fdic_field", "mdrm", "fdic_value",
    ]  # fmt: skip
    svb = table.loc[table["cert"] == 24735].set_index("fdic_field")["fdic_value"]
    assert svb.to_dict() == cw.PUBLISHED_SVB_2022Q4
    assert (
        table.loc[(table["cert"] == 57053) & (table["fdic_field"] == "depunins"), "fdic_value"]
        .isna()
        .all()
    )
    cw.check_published(table)
    broken = table.assign(fdic_value=table["fdic_value"] + 1)
    with pytest.raises(AssertionError, match="scha"):
        cw.check_published(broken)
    with pytest.raises(AssertionError, match="no cross-check rows"):
        cw.check_published(table, cert=1)


@pytest.fixture
def bulk_zip(tmp_path: Path) -> Path:
    path = tmp_path / "call_single_period_2022-12-31.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for member in sorted(FIXTURES.glob("*.txt")):
            zf.write(member, arcname=member.name)
    return path


def test_add_ffiec_values_prefers_rcfd_then_rcon(bulk_zip: Path):
    xw = cw.build_crosswalk(institutions())
    table = cw.securities_crosscheck(financials(), repdtes=["2022-12-31", "2022-09-30"])
    out = cw.add_ffiec_values(table, bulk_zip, xw)
    svb = out.loc[(out["cert"] == 24735) & (out["repdte"] == "2022-12-31")]
    assert svb.set_index("fdic_field")["ffiec_code"].to_dict() == {
        "scha": "RCFD1754", "schf": "RCFD1771", "scaa": "RCFD1772", "scaf": "RCFD1773",
        "depunins": "RCON5597",
    }  # fmt: skip
    assert (svb["ffiec_value"] == svb["fdic_value"]).all()
    sig = out.loc[out["cert"] == 57053].set_index("fdic_field")
    assert sig.at["scha", "ffiec_code"] == "RCON1754" and sig.at["scha", "ffiec_value"] == 7_780_398
    # A quarter other than the ZIP's period gets no FFIEC value.
    other = out.loc[out["repdte"] == "2022-09-30"]
    assert other["ffiec_value"].isna().all() and other["ffiec_code"].isna().all()
    report = cw.render_report(out, n_certs=3, n_missing=1, zip_name=bulk_zip.name)
    assert "| `RCFD1754` | 91,327,000 | yes |" in report
    assert "| 1 | - |  |  |" in report  # the 2022-09-30 rows show no FFIEC value or match
    assert "3 certificates in `institutions`, 1 without an RSSD id" in report
    assert "## Why the FDIC fields are used directly" in report
    plain = cw.render_report(table, n_certs=3, n_missing=1, zip_name=None)
    assert "FFIEC item" not in plain and "No FFIEC bulk ZIP" in plain


def test_build_crosswalk_command_writes_table_and_report(tmp_path: Path, monkeypatch):
    from bankcanary.cli import app
    from bankcanary.config import load_settings
    from bankcanary.storage import duckdb as warehouse
    from bankcanary.storage.parquet import read_table, write_table

    settings = load_settings().model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    monkeypatch.setattr("bankcanary.config.load_settings", lambda: settings)
    write_table(institutions().drop_duplicates("cert"), "institutions", settings=settings)
    write_table(financials(), "financials_raw", settings=settings)
    result = CliRunner().invoke(app, ["build-crosswalk"])
    assert result.exit_code == 0, result.output
    assert "3 certs, 1 without an RSSD id" in result.output
    back = read_table("crosswalk_rssd", settings=settings)
    assert back["cert"].tolist() == [99, 24735, 57053]
    con = warehouse.connect(settings, read_only=True)
    assert con.execute("select count(*) from crosswalk_rssd").fetchone()[0] == 3
    con.close()
    report = (tmp_path / "reports" / "ffiec_crosscheck.md").read_text()
    assert "No FFIEC bulk ZIP" in report and "Silicon Valley Bank (24735)" in report
