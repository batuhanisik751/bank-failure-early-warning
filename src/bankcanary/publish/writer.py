"""Apply ``schema.sql`` and load frames: one transaction per table, COPY, ON CONFLICT upserts.

``replace`` mode (the publish job) truncates the table and COPYs every row inside one
transaction, so readers see either the old table or the new one. ``upsert`` mode COPYs
into a temporary table and merges with ``INSERT ... ON CONFLICT DO UPDATE``, which keeps
rows the frame does not mention (used for partial writes and by the tests).
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from bankcanary.publish import TABLE_KEYS

log = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).with_name("schema.sql")
COPY_CHUNK_ROWS = 100_000


def schema_sql() -> str:
    return SCHEMA_FILE.read_text(encoding="utf-8")


def apply_schema(conn: psycopg.Connection, schema: str = "public") -> None:
    """Create the schema (if not ``public``) and every table and index, idempotently."""
    with conn.transaction():
        if schema != "public":
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
        conn.execute(schema_sql())
    log.info("schema applied to %s", schema)


def _cell(value) -> str:
    """CSV text for COPY: empty = NULL, booleans as t/f, everything else via ``str``."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if value is pd.NaT or (isinstance(value, pd.Timestamp) and pd.isna(value)):
        return ""
    if isinstance(value, bool | np.bool_):
        return "t" if value else "f"
    text = str(value)
    if any(ch in text for ch in ',"\n\r\\'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _csv_chunks(frame: pd.DataFrame, chunk: int = COPY_CHUNK_ROWS):
    """Yield COPY-ready CSV text, ``chunk`` rows at a time (NA of any dtype becomes NULL)."""
    for start in range(0, len(frame), chunk):
        part = (
            frame.iloc[start : start + chunk]
            .astype(object)
            .where(frame.iloc[start : start + chunk].notna(), None)
        )
        buf = io.StringIO()
        for row in part.itertuples(index=False, name=None):
            buf.write(",".join(_cell(v) for v in row))
            buf.write("\n")
        yield buf.getvalue()


def _copy_into(conn: psycopg.Connection, target: sql.Composable, frame: pd.DataFrame) -> None:
    columns = sql.SQL(", ").join(sql.Identifier(c) for c in frame.columns)
    stmt = sql.SQL("COPY {} ({}) FROM STDIN (FORMAT csv, NULL '')").format(target, columns)
    with conn.cursor() as cur, cur.copy(stmt) as copy:
        for text in _csv_chunks(frame):
            copy.write(text)


def _qualified(schema: str, table: str) -> sql.Composable:
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Timestamps as ISO text (COPY parses them), nullable ints kept, index dropped."""
    out = frame.reset_index(drop=True).copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out


def write_table(
    conn: psycopg.Connection,
    table: str,
    frame: pd.DataFrame,
    mode: str = "replace",
    schema: str = "public",
) -> int:
    """Load ``frame`` into ``schema.table`` inside one transaction; returns the row count.

    ``replace`` truncates first; ``upsert`` merges on the table's primary key
    (:data:`bankcanary.publish.TABLE_KEYS`) so re-running with the same rows is a no-op.
    """
    if table not in TABLE_KEYS:
        raise ValueError(f"unknown table {table!r}")
    if mode not in ("replace", "upsert"):
        raise ValueError(f"mode must be 'replace' or 'upsert', got {mode!r}")
    data = _prepare(frame)
    target = _qualified(schema, table)
    with conn.transaction():
        if mode == "replace":
            conn.execute(sql.SQL("TRUNCATE TABLE {}").format(target))
            _copy_into(conn, target, data)
        else:
            tmp = sql.Identifier(f"_{table}_stage")
            conn.execute(
                sql.SQL("CREATE TEMPORARY TABLE {} (LIKE {}) ON COMMIT DROP").format(tmp, target)
            )
            _copy_into(conn, tmp, data)
            keys = [k for k in TABLE_KEYS[table]]
            updates = [c for c in data.columns if c not in keys]
            cols = sql.SQL(", ").join(sql.Identifier(c) for c in data.columns)
            conflict = sql.SQL(", ").join(sql.Identifier(k) for k in keys)
            if updates:
                action = sql.SQL("DO UPDATE SET ") + sql.SQL(", ").join(
                    sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in updates
                )
            else:
                action = sql.SQL("DO NOTHING")
            conn.execute(
                sql.SQL("INSERT INTO {} ({}) SELECT {} FROM {} ON CONFLICT ({}) {}").format(
                    target, cols, cols, tmp, conflict, action
                )
            )
    log.info("%s: %d rows (%s)", table, len(data), mode)
    return int(len(data))


def analyze(conn: psycopg.Connection, tables, schema: str = "public") -> None:
    """``ANALYZE`` each table so the planner sees the fresh row counts."""
    with conn.transaction():
        for table in tables:
            conn.execute(sql.SQL("ANALYZE {}").format(_qualified(schema, table)))
