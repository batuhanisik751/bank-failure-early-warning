"""Typed Parquet tables under ``data/parquet/`` (CONTRACT section 3 and 5).

Every table is written through :func:`write_table`, which sorts rows by the table key
and fixes the column order so that re-running an ingest from cache produces a
byte-identical file. Parquet keeps the pandas dtypes (nullable ints, datetime64[ns]),
so a table read back has the same schema it was written with.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bankcanary.config import Settings

log = logging.getLogger(__name__)

#: Primary keys of the tables in CONTRACT section 5. ``write_table`` falls back to this
#: mapping when no ``key`` is passed, so callers only spell the key once.
TABLE_KEYS: dict[str, tuple[str, ...]] = {
    "failures": ("cert", "fail_date"),
    "institutions": ("cert",),
    "history": ("cert", "transnum"),
    "financials_raw": ("cert", "repdte"),
    "panel": ("cert", "repdte"),
    "labels": ("cert", "repdte"),
    "features_v1": ("cert", "repdte"),
}


def parquet_dir(settings: Settings) -> Path:
    return settings.data_dir / "parquet"


def table_path(settings: Settings, name: str) -> Path:
    return parquet_dir(settings) / f"{name}.parquet"


def _resolve_key(name: str, key: str | tuple[str, ...] | list[str] | None) -> list[str]:
    if key is None:
        if name not in TABLE_KEYS:
            raise ValueError(f"no key given and {name!r} is not a known table")
        return list(TABLE_KEYS[name])
    return [key] if isinstance(key, str) else list(key)


def write_table(
    df: pd.DataFrame,
    name: str,
    key: str | tuple[str, ...] | list[str] | None = None,
    settings: Settings | None = None,
) -> Path:
    """Write ``df`` as ``data/parquet/<name>.parquet`` deterministically.

    Rows are sorted by ``key`` (default: the table's key from CONTRACT section 5) with a
    stable sort, key columns are moved to the front and the rest keep their order, the
    index is dropped and the file is written with pyarrow using zstd compression.
    Returns the path written.
    """
    if settings is None:
        from bankcanary.config import load_settings

        settings = load_settings()
    key_cols = _resolve_key(name, key)
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"table {name!r} is missing key column(s) {missing}")
    ordered = key_cols + [c for c in df.columns if c not in key_cols]
    out = df.loc[:, ordered].sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    out = _nanosecond_dates(out)
    path = table_path(settings, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    log.info("wrote %s: %d rows x %d columns", path, len(out), len(out.columns))
    return path


def read_table(name: str, settings: Settings | None = None) -> pd.DataFrame:
    """Read ``data/parquet/<name>.parquet`` back with the dtypes it was written with."""
    if settings is None:
        from bankcanary.config import load_settings

        settings = load_settings()
    path = table_path(settings, name)
    if not path.exists():
        raise FileNotFoundError(f"table {name!r} has not been written yet ({path})")
    return _nanosecond_dates(pq.read_table(path).to_pandas())


def _nanosecond_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Force every datetime column to ``datetime64[ns]`` (pandas 3 infers ``[us]``)."""
    dates = {c: "datetime64[ns]" for c in df.columns if str(df[c].dtype).startswith("datetime64")}
    return df.astype(dates) if dates else df
