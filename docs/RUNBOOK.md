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
uv run bankcanary publish                   # full rebuild, all 13 tables: about 100 s locally
uv run bankcanary publish --tables scores,quarters
uv run bankcanary publish --tables drivers,map_quarters,rate_shock_scores   # about 10 s
uv run bankcanary publish --tables case_study_2023,case_study_series        # about 30 s
```

What it does (CONTRACT 16-17): applies `src/bankcanary/publish/schema.sql` idempotently,
builds one frame per table from `data/parquet/*.parquet`, `runs/` and `models/production/`,
loads each table inside one transaction (truncate + COPY), runs `ANALYZE`, appends a
`pipeline_runs` row and a `runs/publish/<run_id>/` tracking record, and prints the
database size. The label-incomplete quarters (after the last complete walk-forward year)
are scored on the fly with `models/production/{gbdt_mono,hazard}/`.

The page tables (`src/bankcanary/publish/pages.py`) come after the core ones: `drivers`
filters the warehouse `drivers` table (`bankcanary explain --all` must have run) to the
CONTRACT subset and keeps the five largest contributions per bank-quarter; `map_quarters`
joins `scores` to the coordinates in `banks`; `rate_shock_scores` re-scores the latest
quarter under 4 shocks x 5 durations with the production `gbdt_mono`; the two case study
tables re-run `bankcanary.evaluation.case_study_2023` (four fits, no run record). A subset
request builds the core frames a page table needs in memory without writing them.

Prerequisites: the warehouse tables `institutions`, `panel`, `labels`, `features_v2`,
`failures`, `walkforward_scores`, `drivers`, plus `models/walkforward/<year>/<model>/config.json`
for every test year (the `model_version` of a backtest year is
`<model>-<train_end_repdte>-<sha of the commit that added its run record>`).

## 3. Verify

```bash
uv run python -c "
from bankcanary.publish import ALL_TABLES, db
with db.connect() as c:
    print(db.table_counts(c, list(ALL_TABLES) + ['pipeline_runs']))
    print(round(db.database_size_bytes(c) / 1e6, 1), 'MB')
"
uv run pytest tests/test_publish_db.py tests/test_publish.py   # scratch round trip + hooks
```

`tests/test_publish.py` is the CONTRACT 19 time-machine hook: the 2009 `gbdt_mono` rows of
`scores`, pooled over the four quarters and labelled from the warehouse `labels` table,
must reproduce `walkforward_metrics.recall_at_2pct` for 2009 within 1e-6; it also checks
that every table is filled, every driver row has its score row and every rate-shock
scenario ranks 1..n. Both files skip when `DATABASE_URL` is unset or unreachable.

Expected after a full publish: `scores` about 905 k rows (two models x every quarter
2008Q1 onward), `drivers` about 288 k, `map_quarters` about 448 k, `ratios` about 453 k
(2008Q1 onward), `rate_shock_scores` 86,260 (20 scenarios x the latest quarter's banks),
`peer_stats` about 29 k, `banks` about 28 k, `failures` about 4 k, `quarters` about 100,
`walkforward_metrics` 36, `model_versions` 34, `case_study_2023` 28, `case_study_series`
37. The database must stay under 400 MB (Neon free tier); it is 399.6 MB, so see the
levers in `docs/DECISIONS.md` before the next quarter lands.

## 4. Troubleshooting

- `No module named bankcanary`: `uv run --no-sync python scripts/fix_venv.py` once.
- Port 5433 is deliberate: another project's Postgres holds 5432.
- `DATABASE_URL is not set`: the `.env` file is missing or the variable is absent.
- A failed table load rolls back that table only; re-run `publish` (idempotent).
- A column type changed in `schema.sql` does not reach an existing table (`CREATE TABLE IF
  NOT EXISTS`): `DROP TABLE <t>` locally, then `publish --tables <t>` recreates and fills it.
- Neon: the owner sets `DATABASE_URL` (pooled, `sslmode=verify-full`) as a GitHub
  secret and runs the same command from the refresh workflow; nothing here deploys.

## 5. Scheduled refresh (GitHub Actions)

`bankcanary refresh [--force-quarter YYYY-MM-DD] [--dry-run]` probes the FDIC API for the
newest `REPDTE`, compares it with `quarters.max(repdte)` in Postgres (the warehouse when
the database is unreachable; the run log says so) and, when a newer quarter exists,
ingests it, rebuilds `panel`, `labels`, `macro_state` and `features_v2` from the raw
cache, scores it with `models/production/`, explains it with SHAP and upserts the rows
that quarter adds (`quarters`, `scores`, `drivers`, `ratios`, `peer_stats`, the last four
quarters of `map_quarters`; `banks`, `failures` and `rate_shock_scores` are rebuilt whole).
Historical walk-forward rows are never touched, so the job needs no walk-forward
artefacts. Every run, new quarter or not, writes one `pipeline_runs` row and a
`runs/refresh/` record; `--dry-run` only prints the decision.

`.github/workflows/refresh.yml` runs it every Monday 09:00 UTC and on demand (the
`force_quarter` input republishes one quarter). `data/raw/fdic` and `data/raw/fred` are
restored with `actions/cache` keyed `raw-<latest REPDTE>`; a run with no cache at all
first pulls every quarter once (`bankcanary ingest --what all --no-build`, then
`build-macro --pull-only`). When a quarter was published the job POSTs the web app's
revalidate route. `retrain.yml` is manual only: it rebuilds the warehouse, runs the
walk-forward and calibration for the latest complete year (or the `year` input),
promotes with `scripts/promote_production_models.py` and opens a pull request touching
`models/production/` and the run records; nothing is merged automatically.

### Secrets the owner adds (repository → Settings → Secrets and variables → Actions)

| secret | used by | value |
|---|---|---|
| `DATABASE_URL` | refresh | the Neon **pooled** connection string with `sslmode=verify-full`; a role that may write every table (the web app's role stays read-only and lives in Vercel, not here) |
| `REVALIDATE_URL` | refresh | `https://<site>/api/revalidate` of the deployed web app; leave unset until the app is live and the step is skipped |
| `REVALIDATE_SECRET` | refresh | the same 32+ character random string the web app holds as `REVALIDATE_SECRET` (`openssl rand -hex 32`) |
| `FRED_API_KEY` | refresh, retrain | optional; without it the FRED client falls back to the public CSV endpoint |

`gh secret set DATABASE_URL --repo <owner>/<repo>` reads the value from a prompt without
putting it in shell history. Never paste a connection string into an issue, a workflow
file or a commit. The first scheduled run after adding `DATABASE_URL` takes about 20
minutes (cache warm-up); later no-op runs finish in about a minute and a new quarter in
about five. Locally the same command runs against the docker database from `.env`.

## 6. Web app (`web/`)

```bash
cd web
npm install                       # Node 22, npm 10
npx playwright install chromium   # once
npm run build                     # reads DATABASE_URL (copied from ../.env by next.config.ts)
npm run start                     # http://localhost:3100
npm run lint && npm run typecheck && npm test      # eslint, tsc, vitest (no database)
npm run test:e2e                  # Playwright + axe over the built app, starts/stops the server itself
```

- Port 3100 for `dev` and `start`, so it never collides with another project on 3000.
- `npm run build` prerenders the home page from the database, so the local container must
  be up and published (`bankcanary publish`); the other pages and `/api/revalidate` do not
  read at build time.
- `npm test` includes the schema drift test: after any change to
  `src/bankcanary/publish/schema.sql`, mirror it in `web/lib/db/schema.ts` or the web CI
  job fails.
- After a publish, expire the web cache: `curl -X POST -H "Authorization: Bearer
  $REVALIDATE_SECRET" <site>/api/revalidate` (the refresh workflow does this with the
  `REVALIDATE_URL` and `REVALIDATE_SECRET` secrets). Locally the secret is optional; the
  route answers 503 when it is unset and 401 on a wrong token.
- Deploying (owner's action, nothing here deploys): a Vercel project with root directory
  `web`, environment variables `DATABASE_URL` (Neon pooled URL with
  `sslmode=verify-full`), `REVALIDATE_SECRET` (any long random string, the same value as
  the GitHub secret) and `NEXT_PUBLIC_SITE_URL` (the public origin). The build command is
  the default `npm run build`.
