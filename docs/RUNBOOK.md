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
- Neon: the owner sets `DATABASE_URL` (pooled, `sslmode=verify-full&sslrootcert=system`)
  as a GitHub secret and runs the same command from the refresh workflow; section 7 is
  the full checklist and explains the TLS parameters; nothing here deploys.

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

## 7. Deployment checklist (owner's actions; nothing in the repository deploys)

Order matters: the database first, the web app second, the GitHub secrets last, so the
first scheduled refresh finds a filled database and a live revalidate route.

1. **Create the Neon database through the Vercel marketplace.** Vercel dashboard →
   Storage → Create → Neon (free tier) → connect it to the project created in step 2 (or
   create the project first with root directory `web`). Neon's free tier caps the
   project at 0.5 GB and the database is 399.6 MB after a full publish, so do not add
   anything else to it. In the Neon console create a second role for the web app that
   may only read (`GRANT SELECT ON ALL TABLES IN SCHEMA public TO <reader>` plus
   `ALTER DEFAULT PRIVILEGES ... GRANT SELECT` for tables published later); the owner
   role that Neon created is the writer for `publish` and `refresh`.
2. **Vercel project.** Import the GitHub repository, set root directory `web`, framework
   Next.js, build command the default `npm run build`, Node 22. Environment variables
   (production and preview): `DATABASE_URL` = the Neon **pooled** connection string of the
   reader role with `?sslmode=verify-full` and nothing else after the `?` (the TLS note
   below says why), `REVALIDATE_SECRET` = `openssl rand -hex 32`, `NEXT_PUBLIC_SITE_URL` =
   the public origin. The first build fails until step 4 has filled the database, because
   the home page prerenders from it; redeploy after the publish.
3. **GitHub secrets** (section 5 table): `DATABASE_URL` = the pooled string of the
   **writer** role with `?sslmode=verify-full&sslrootcert=system`, `REVALIDATE_URL` =
   `https://<site>/api/revalidate`, `REVALIDATE_SECRET` = the same value as on Vercel,
   optionally `FRED_API_KEY`. Use `gh secret set <NAME> --repo <owner>/<repo>` and type the
   value at the prompt.
4. **Publish once from a machine that has the warehouse** (`data/parquet/`, `runs/`,
   `models/production/`; the refresh job never rebuilds history, so this seed is the only
   full load): `DATABASE_URL='<writer pooled string>' uv run bankcanary publish`. Pass the
   variable inline for that one command rather than editing `.env`, so the local
   container stays the default. About two minutes on a home connection.
5. **Verify** with the counts query of section 3 against the same URL (expect the row
   counts listed there and about 400 MB), then open the site and check the leaderboard
   shows the latest quarter, `/time-machine?quarter=2009Q1` loads, and
   `curl -X POST -H "Authorization: Bearer $REVALIDATE_SECRET" https://<site>/api/revalidate`
   answers `{"revalidated": true, ...}`.
6. **Run the refresh workflow by hand** (Actions → Refresh → Run workflow, no input). The
   first run warms the raw cache (about 20 minutes) and ends with `no new quarter`;
   `pipeline_runs` gains one row with status `ok`. From then on it runs every Monday.
7. Put the live URL in `README.md` (status line) and in the Vercel project's
   `NEXT_PUBLIC_SITE_URL` if a custom domain replaces the `vercel.app` one.

### TLS notes

- **psycopg (publish, refresh, tests)** hands the URL to libpq unchanged, so the query
  string decides. `sslmode=verify-full` checks the certificate chain *and* the host name,
  but libpq then needs a root store: `sslrootcert=system` (libpq 16+, which
  `psycopg[binary]` bundles) uses the operating system's CA bundle, which contains Neon's
  public CA. Without it verify-full fails with "root certificate file does not exist".
  Neon's default `sslmode=require` encrypts but does not verify; do not use it.
- **node-postgres (the web app)** merges the parsed URL over the explicit config, so
  `web/lib/db/client.ts` strips `ssl`, `sslmode`, `sslcert`, `sslkey`, `sslrootcert`,
  `sslnegotiation` and `uselibpqcompat` from any non-local URL (a warning names the dropped
  keys, never values) and always connects with `rejectUnauthorized: true`, verifying against
  Node's bundled CA store. Give Vercel `sslmode=verify-full` only: it is the documented
  contract (CONTRACT 18), and `sslrootcert=system` would just produce a warning at every
  cold start. A local `DATABASE_URL` (localhost, 127.0.0.1, ::1) stays plain.
- Neon's pooled host (`-pooler` in the host name) is required for the web app because
  serverless functions open many short connections; the writer may use either host.
