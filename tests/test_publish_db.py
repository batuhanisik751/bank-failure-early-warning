"""Schema + writer round trip against a scratch schema; skips without a reachable database.

``DATABASE_URL`` comes from the environment or the git-ignored ``.env`` (never from the
test); every table lands in ``publish_test_<pid>`` and the schema is dropped afterwards.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from bankcanary.publish import CORE_TABLES, TABLE_KEYS, db, writer


def _url() -> str | None:
    """``DATABASE_URL`` from the environment, else the git-ignored ``.env`` via ``Secrets``."""
    try:
        return db.database_url(os.environ.get("DATABASE_URL"))
    except RuntimeError:
        return None


URL = _url()
pytestmark = pytest.mark.skipif(
    URL is None or not db.reachable(URL), reason="DATABASE_URL unset or database not reachable"
)


@pytest.fixture
def scratch():
    schema = f"publish_test_{os.getpid()}"
    with db.connect(URL, autocommit=True) as conn:
        try:
            yield conn, schema
        finally:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def _tables(conn, schema):
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (schema,)
    ).fetchall()
    return {r[0] for r in rows}


def test_schema_applies_idempotently_and_creates_every_table(scratch):
    conn, schema = scratch
    writer.apply_schema(conn, schema)
    writer.apply_schema(conn, schema)
    assert set(TABLE_KEYS) <= _tables(conn, schema)
    comment = conn.execute(
        "SELECT obj_description(%s::regclass, 'pg_class')", (f"{schema}.scores",)
    ).fetchone()[0]
    assert "not investment advice" in comment


def test_upsert_then_replace_round_trip(scratch):
    conn, schema = scratch
    writer.apply_schema(conn, schema)
    frame = pd.DataFrame(
        {
            "cert": [1, 2],
            "fail_date": pd.to_datetime(["2023-03-10", "2023-05-01"]).date,
            "name": ["A, Inc", 'B "quoted"'],
            "city": ["X", None],
            "state": ["CA", "NY"],
            "restype1": ["PA", "PA"],
            "cost": [1.5, float("nan")],
            "qbfasset": [10.0, 20.0],
            "qbfdep": [8.0, 16.0],
        }
    )
    assert writer.write_table(conn, "failures", frame, mode="upsert", schema=schema) == 2
    frame.loc[0, "cost"] = 9.0
    writer.write_table(conn, "failures", frame.iloc[:1], mode="upsert", schema=schema)
    rows = conn.execute(
        f'SELECT cert, name, city, cost FROM "{schema}".failures ORDER BY cert'
    ).fetchall()
    assert rows == [(1, "A, Inc", "X", 9.0), (2, 'B "quoted"', None, None)]
    writer.write_table(conn, "failures", frame.iloc[1:], mode="replace", schema=schema)
    assert db.table_counts(conn, ["failures"], schema) == {"failures": 1}
    assert db.database_size_bytes(conn) > 0


def test_published_scores_rank_each_quarter_from_one():
    """When the public schema holds scores, every (repdte, model) is ranked 1..n."""
    with db.connect(URL, autocommit=True) as conn:
        if not {"scores"} <= _tables(conn, "public"):
            pytest.skip("public.scores not published")
        bad = conn.execute(
            "SELECT count(*) FROM (SELECT repdte, model, count(*) AS n, min(rank) AS lo, "
            "max(rank) AS hi FROM scores GROUP BY 1, 2) g WHERE lo <> 1 OR hi <> n"
        ).fetchone()[0]
        assert bad == 0
        assert set(CORE_TABLES) <= _tables(conn, "public")
