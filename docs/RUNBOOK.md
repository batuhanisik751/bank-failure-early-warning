# Runbook: publishing to Postgres

Educational project, not a credit rating, not investment advice, not a supervisory
assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.

## 1. Start the local database

```bash
docker compose up -d            # container bankcanary-postgres on port 5433
docker compose ps               # wait for "healthy"
```

`DATABASE_URL` lives in the git-ignored `.env` (`bankcanary.config.load_secrets()` reads
it; the web app reads `process.env`). Never commit `.env`, never `source` it, never paste
the URL into code, docs or commit messages. For a fresh checkout copy the compose
credentials into `.env` as `DATABASE_URL=postgresql://<user>:<password>@localhost:5433/bankcanary`.

## 2. Publish

```bash
uv run bankcanary publish --schema-only     # tables, indexes and comments only
uv run bankcanary publish --dry-run         # build every frame, print counts, write nothing
uv run bankcanary publish                   # full rebuild: about 60 s locally
uv run bankcanary publish --tables scores,quarters
```

What it does (CONTRACT 16-17): applies `src/bankcanary/publish/schema.sql` idempotently,
builds one frame per table from `data/parquet/*.parquet`, `runs/` and `models/production/`,
loads each table inside one transaction (truncate + COPY), runs `ANALYZE`, appends a
`pipeline_runs` row and a `runs/publish/<run_id>/` tracking record, and prints the
database size. The label-incomplete quarters (after the last complete walk-forward year)
are scored on the fly with `models/production/{gbdt_mono,hazard}/`.

Prerequisites: the warehouse tables `institutions`, `panel`, `labels`, `features_v2`,
`failures`, `walkforward_scores`, plus `models/walkforward/<year>/<model>/config.json`
for every test year (the `model_version` of a backtest year is
`<model>-<train_end_repdte>-<sha of the commit that added its run record>`).

## 3. Verify

```bash
uv run python -c "
from bankcanary.publish import CORE_TABLES, db
with db.connect() as c:
    print(db.table_counts(c, list(CORE_TABLES) + ['pipeline_runs']))
    print(round(db.database_size_bytes(c) / 1e6, 1), 'MB')
"
DATABASE_URL=... uv run pytest tests/test_publish_db.py -q   # scratch schema round trip
```

Expected after a full publish: `scores` about 905 k rows (two models x every quarter
2008Q1 onward), `ratios` about 711 k, `peer_stats` about 29 k, `banks` about 28 k,
`failures` about 4 k, `quarters` about 100, `walkforward_metrics` 36 (17 years + pooled
per model), `model_versions` 34. The database must stay under 400 MB (Neon free tier).

## 4. Troubleshooting

- `No module named bankcanary`: `uv run --no-sync python scripts/fix_venv.py` once.
- Port 5433 is deliberate: another project's Postgres holds 5432.
- `DATABASE_URL is not set`: the `.env` file is missing or the variable is absent.
- A failed table load rolls back that table only; re-run `publish` (idempotent).
- Neon: the owner sets `DATABASE_URL` (pooled, `sslmode=verify-full`) as a GitHub
  secret and runs the same command from the refresh workflow; nothing here deploys.
