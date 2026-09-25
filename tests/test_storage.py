"""Storage tests: tiny in-memory frames, a tmp_path data_dir, no network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bankcanary.config import load_settings
from bankcanary.storage import duckdb as warehouse
from bankcanary.storage.parquet import read_table, table_path, write_table


@pytest.fixture
def settings(tmp_path: Path):
    return load_settings().model_copy(update={"data_dir": tmp_path / "data"})


def sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["b", "a", "c"],
            "cert": pd.array([2, 1, 2], dtype="Int64"),
            "fail_date": pd.to_datetime(["2010-01-01", "2009-06-30", "2008-01-01"]),
            "cost": [1.5, None, 3.0],
            "fed_rssd": pd.array([None, 5, 6], dtype="Int64"),
        }
    )


def test_write_table_sorts_by_key_and_fixes_column_order(settings):
    path = write_table(sample(), "failures", settings=settings)
    assert path == table_path(settings, "failures")
    back = read_table("failures", settings=settings)
    assert back.columns.tolist() == ["cert", "fail_date", "name", "cost", "fed_rssd"]
    assert back["cert"].tolist() == [1, 2, 2]
    assert back["fail_date"].tolist()[1] == pd.Timestamp("2008-01-01")
    assert str(back["fail_date"].dtype) == "datetime64[ns]"
    assert str(back["fed_rssd"].dtype) == "Int64" and back["fed_rssd"].isna().sum() == 1


def test_write_table_is_byte_identical_on_rerun(settings):
    p1 = write_table(sample(), "institutions", key="cert", settings=settings)
    first = p1.read_bytes()
    p2 = write_table(sample().sample(frac=1, random_state=3), "institutions", key="cert",
                     settings=settings)  # fmt: skip
    assert p2.read_bytes() == first


def test_write_table_rejects_missing_key(settings):
    with pytest.raises(KeyError):
        write_table(sample().drop(columns=["fail_date"]), "failures", settings=settings)
    with pytest.raises(ValueError):
        write_table(sample(), "unknown_table", settings=settings)


def test_read_missing_table_raises(settings):
    with pytest.raises(FileNotFoundError):
        read_table("failures", settings=settings)


def test_replace_table_from_parquet_and_frame(settings):
    write_table(sample(), "failures", settings=settings)
    assert warehouse.replace_table("failures", settings=settings) == 3
    assert warehouse.replace_table("failures", settings=settings) == 3  # idempotent
    assert warehouse.replace_table("scratch", sample().head(2), settings=settings) == 2
    con = warehouse.connect(settings, read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM failures").fetchone()[0] == 3
        assert con.execute("SELECT min(cert) FROM scratch").fetchone()[0] == 1
        date = con.execute("SELECT min(fail_date)::DATE FROM failures").fetchone()[0]
        assert str(date) == "2008-01-01"
        assert warehouse.warehouse_path(settings).exists()
    finally:
        con.close()
