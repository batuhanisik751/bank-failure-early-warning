"""``bankcanary publish``: the warehouse and ``models/production/`` rendered as Postgres tables.

CONTRACT section 16 fixes the schema (``schema.sql``, Python owns the DDL) and section 17
the job. :mod:`core` builds one DataFrame per table from the Parquet warehouse, the run
records and the production model artefacts; :mod:`writer` applies the schema idempotently
and loads each frame inside one transaction; :mod:`db` opens the connection from
``Secrets.database_url``.

Every table is rebuilt from scratch on each run, so the database is a pure function of the
warehouse and the model artefacts. ``TABLE_KEYS`` lists the primary key of every table in
publish order (foreign-key-like columns point at tables written earlier).
"""

from __future__ import annotations

#: Primary key columns per table, in publish order. The tables that later steps fill
#: (``drivers``, the case study, the rate shock grid and ``map_quarters``) are created by
#: the schema but not written by this step.
TABLE_KEYS: dict[str, tuple[str, ...]] = {
    "model_versions": ("model_version",),
    "banks": ("cert",),
    "quarters": ("repdte",),
    "scores": ("cert", "repdte", "model"),
    "drivers": ("cert", "repdte", "model", "rank"),
    "ratios": ("cert", "repdte"),
    "peer_stats": ("repdte", "size_bucket", "region", "ratio"),
    "failures": ("cert", "fail_date"),
    "walkforward_metrics": ("model", "horizon", "test_year"),
    "case_study_2023": ("cert", "quarter", "view", "model"),
    "case_study_series": ("cert", "repdte"),
    "rate_shock_scores": ("cert", "shock_bp", "duration_years"),
    "map_quarters": ("repdte", "cert"),
    "pipeline_runs": ("run_id",),
}

#: Tables this step builds, in publish order.
CORE_TABLES: tuple[str, ...] = (
    "model_versions",
    "banks",
    "quarters",
    "scores",
    "ratios",
    "peer_stats",
    "failures",
    "walkforward_metrics",
)

DISCLAIMER = (
    "Educational project, not a credit rating, not investment advice, not a supervisory "
    "assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category."
)
