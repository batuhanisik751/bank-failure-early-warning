"""DuckDB warehouse at ``data/warehouse.duckdb``: a queryable copy of the Parquet tables.

Parquet is the canonical store; DuckDB exists so that joins and ad-hoc checks can be
written in SQL. :func:`replace_table` always rebuilds a table from scratch (from the
Parquet file or a frame), which keeps the warehouse in step with the files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from bankcanary.config import Settings
from bankcanary.storage.parquet import table_path

log = logging.getLogger(__name__)


def warehouse_path(settings: Settings) -> Path:
    return settings.data_dir / "warehouse.duckdb"


def connect(settings: Settings | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the warehouse database and return the connection."""
    if settings is None:
        from bankcanary.config import load_settings

        settings = load_settings()
    path = warehouse_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def replace_table(
    name: str,
    df_or_parquet: pd.DataFrame | Path | str | None = None,
    settings: Settings | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> int:
    """Create or replace warehouse table ``name`` and return its row count.

    The source is a DataFrame, a Parquet path, or (default) ``data/parquet/<name>.parquet``.
    Datetime columns become ``TIMESTAMP`` in DuckDB (nanosecond precision would not round
    trip as ``DATE``); downstream SQL casts with ``::DATE`` where a date is wanted.
    """
    if settings is None:
        from bankcanary.config import load_settings

        settings = load_settings()
    own = con is None
    con = con or connect(settings)
    try:
        if isinstance(df_or_parquet, pd.DataFrame):
            frame = df_or_parquet  # noqa: F841 - referenced by DuckDB via the local scope
            con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM frame')
        else:
            src = Path(df_or_parquet) if df_or_parquet is not None else table_path(settings, name)
            if not src.exists():
                raise FileNotFoundError(src)
            con.execute(
                f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM read_parquet(?)', [str(src)]
            )
        count = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        log.info("duckdb table %s replaced: %d rows", name, count)
        return int(count)
    finally:
        if own:
            con.close()
