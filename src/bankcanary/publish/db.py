"""Postgres connections for the publish job (psycopg 3, ``Secrets.database_url``).

The URL comes from the environment or the git-ignored ``.env`` and is never logged or
embedded anywhere; :func:`reachable` lets tests skip cleanly when no database answers.
"""

from __future__ import annotations

import logging

import psycopg

log = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 5


def database_url(url: str | None = None) -> str:
    """``url`` when given, else ``Secrets.database_url``; raises when neither is set."""
    if url:
        return url
    from bankcanary.config import load_secrets

    found = load_secrets().database_url
    if not found:
        raise RuntimeError("DATABASE_URL is not set (environment or .env)")
    return found


def connect(url: str | None = None, autocommit: bool = False) -> psycopg.Connection:
    """Open a connection; the caller owns commits (one transaction per table)."""
    return psycopg.connect(
        database_url(url), autocommit=autocommit, connect_timeout=CONNECT_TIMEOUT_SECONDS
    )


def reachable(url: str | None = None) -> bool:
    """``True`` when a connection can be opened and ``SELECT 1`` answers."""
    try:
        with connect(url, autocommit=True) as conn:
            conn.execute("SELECT 1").fetchone()
    except (RuntimeError, psycopg.Error, OSError) as exc:
        log.debug("database not reachable: %s", type(exc).__name__)
        return False
    return True


def database_size_bytes(conn: psycopg.Connection) -> int:
    """``pg_database_size`` of the current database (CONTRACT 16: total < 400 MB)."""
    row = conn.execute("SELECT pg_database_size(current_database())").fetchone()
    return int(row[0]) if row else 0


def table_counts(conn: psycopg.Connection, tables: list[str], schema: str = "public") -> dict:
    """Exact ``COUNT(*)`` per table (the tables are our own identifiers, never user input)."""
    counts: dict[str, int] = {}
    for table in tables:
        query = psycopg.sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
            psycopg.sql.Identifier(schema), psycopg.sql.Identifier(table)
        )
        row = conn.execute(query).fetchone()
        counts[table] = int(row[0]) if row else 0
    return counts


def table_sizes(conn: psycopg.Connection, tables: list[str], schema: str = "public") -> dict:
    """``pg_total_relation_size`` (heap + indexes) per table, in bytes."""
    sizes: dict[str, int] = {}
    for table in tables:
        row = conn.execute(
            "SELECT pg_total_relation_size(%s::regclass)", (f"{schema}.{table}",)
        ).fetchone()
        sizes[table] = int(row[0]) if row else 0
    return sizes
