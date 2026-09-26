"""Acceptance hooks against the published database (CONTRACT 19); skip without it.

``DATABASE_URL`` comes from the environment or the git-ignored ``.env``. The time-machine
hook reads the 2009 ``gbdt_mono`` rows of ``public.scores`` (the quarters whose
``model_year`` is 2009), takes their outcome labels from the warehouse ``labels`` table
and checks that recall at the top 2 percent, pooled over the four quarters, equals the
``walkforward_metrics`` row for that model and year.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from bankcanary.publish import ALL_TABLES, db

YEAR = 2009
MODEL = "gbdt_mono"


def _url() -> str | None:
    try:
        return db.database_url(os.environ.get("DATABASE_URL"))
    except RuntimeError:
        return None


URL = _url()
pytestmark = pytest.mark.skipif(
    URL is None or not db.reachable(URL), reason="DATABASE_URL unset or database not reachable"
)


def _tables(conn) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    return {r[0] for r in rows}


def _labels() -> pd.DataFrame:
    from bankcanary.config import load_settings
    from bankcanary.storage.parquet import read_table, table_path

    settings = load_settings()
    if not table_path(settings, "labels").exists():
        pytest.skip("warehouse labels table not built")
    out = read_table("labels", settings)[["cert", "repdte", "y_4q"]]
    return out.assign(repdte=pd.to_datetime(out["repdte"]).dt.date)


def test_time_machine_2009_recall_at_2pct_matches_walkforward_metrics():
    from bankcanary.evaluation.metrics import evaluate

    with db.connect(URL, autocommit=True) as conn:
        if not {"scores", "quarters", "walkforward_metrics"} <= _tables(conn):
            pytest.skip("core tables not published")
        rows = conn.execute(
            "SELECT s.cert, s.repdte, s.score, s.model_version FROM scores s "
            "JOIN quarters q ON q.repdte = s.repdte WHERE s.model = %s AND q.model_year = %s",
            (MODEL, YEAR),
        ).fetchall()
        metric = conn.execute(
            "SELECT recall_at_2pct, n, n_failures FROM walkforward_metrics "
            "WHERE model = %s AND horizon = 4 AND test_year = %s",
            (MODEL, YEAR),
        ).fetchone()
        expected_failures = conn.execute(
            "SELECT sum(n_failures_next_4q) FROM quarters WHERE model_year = %s", (YEAR,)
        ).fetchone()[0]
    if not rows or metric is None:
        pytest.skip(f"{MODEL} {YEAR} not published")
    scores = pd.DataFrame(rows, columns=["cert", "repdte", "score", "model_version"])
    assert scores["repdte"].map(lambda d: d.year).eq(YEAR).all()
    assert scores["repdte"].nunique() == 4
    assert scores["model_version"].nunique() == 1
    assert scores["model_version"].iloc[0].startswith(f"{MODEL}-{YEAR - 2}-")
    merged = scores.merge(_labels(), on=["cert", "repdte"], how="left", validate="one_to_one")
    assert merged["y_4q"].notna().all()
    y = merged["y_4q"].to_numpy(dtype=float)
    assert len(merged) == metric[1]
    assert int(y.sum()) == metric[2] == int(expected_failures)
    recall = evaluate(y, merged["score"].to_numpy(dtype=float), tie_breaker=merged["cert"])
    assert abs(recall["recall_at_2pct"] - metric[0]) < 1e-6
    assert np.isfinite(metric[0])


def test_every_table_is_published_and_page_tables_reference_scores():
    with db.connect(URL, autocommit=True) as conn:
        if not set(ALL_TABLES) <= _tables(conn):
            pytest.skip("not every table exists")
        counts = db.table_counts(conn, list(ALL_TABLES))
        if any(counts[t] == 0 for t in ALL_TABLES):
            pytest.skip(f"empty tables: {[t for t in ALL_TABLES if counts[t] == 0]}")
        orphans = conn.execute(
            "SELECT count(*) FROM drivers d LEFT JOIN scores s ON s.cert = d.cert "
            "AND s.repdte = d.repdte AND s.model = d.model WHERE s.cert IS NULL"
        ).fetchone()[0]
        assert orphans == 0
        grid = conn.execute(
            "SELECT count(*) FROM (SELECT shock_bp, duration_years, count(*) AS n, "
            "min(rank) AS lo, max(rank) AS hi FROM rate_shock_scores GROUP BY 1, 2) g "
            "WHERE lo <> 1 OR hi <> n"
        ).fetchone()[0]
        assert grid == 0
        scenarios = conn.execute(
            "SELECT count(DISTINCT (shock_bp, duration_years)) FROM rate_shock_scores"
        ).fetchone()[0]
        assert scenarios == 20
        assert db.database_size_bytes(conn) < 400_000_000
