# Prototype 3 acceptance checklist

Spec: `PROJECT_SPEC.md`, "Prototype 3 — Production" (acceptance criteria at lines
363-368). Every criterion below was measured on 2026-09-26 at commit `7e7cfc3` against the
local Postgres container (`docker-compose.yml`, port 5433, database 393.1 MB after the
refresh below) that a full `uv run bankcanary publish` had filled the same day, and against
the web app built from the same commit (`npm run build`, Next.js 16, port 3100). The
deployment criterion is the owner's action and is marked pending with the exact remaining
steps. Educational project, not a credit rating, not investment advice, not a supervisory
assessment; FDIC insurance covers $250,000 per depositor, per bank, per ownership category.

## 1. A new quarter is picked up and published end-to-end without manual steps

- [x] **Dry run against the latest quarter (2026Q2, `repdte = 2026-06-30`).** In one
  transaction the quarter's rows were deleted from `scores`, `drivers`, `map_quarters` and
  `quarters` so that the plain command, not a flag, had to detect the gap:

  | table | rows deleted | rows after `refresh` |
  |---|---|---|
  | `scores` | 8,626 (4,313 `gbdt_mono` + 4,313 `hazard`) | 8,626 (4,313 + 4,313) |
  | `drivers` | 21,565 | 21,565 |
  | `map_quarters` | 4,257 | 4,257 |
  | `quarters` | 1 | 1 (`label 2026Q2, n_banks 4313, model_year null, model_version gbdt_mono-2022-12-31-bbc0230`) |

  `quarters.max(repdte)` fell to 2026-03-31. `uv run bankcanary refresh` (no options) then
  printed `latest REPDTE (API): 2026-06-30; published (database): 2026-03-31`,
  `decision: new quarter 2026-06-30`, `reason: API 2026-06-30 is newer than database
  2026-03-31`, and wrote `model_versions 2, banks 27834, quarters 102, scores 8626, drivers
  21565, ratios 4313, peer_stats 276, failures 3629, rate_shock_scores 86260, map_quarters
  21745` rows (the last three upsert the whole lookback window, so they exceed the
  per-quarter counts). Steps: probe 1 s, ingest structure 6 s, ingest financials 28 s,
  panel 6 s, labels 1 s, macro 54 s, features_v2 11 s, load warehouse 0 s, build frames
  7 s, publish 3 s; **wall time 119 s** (`date` before and after the command). Status
  `ok`, run `refresh-4q-0200d054cc`, database size 393.1 MB.
- [x] **`pipeline_runs` row.** `SELECT run_id, status, latest_repdte, new_quarter,
  started_at, finished_at FROM pipeline_runs WHERE run_id = 'refresh-4q-0200d054cc'` →
  `ok | 2026-06-30 | true | 2026-09-26 20:06:03+00 | 2026-09-26 20:08:01+00`, with
  `rows_written` holding the ten counts above. The matching tracking record is committed
  under `runs/refresh/refresh-4q-0200d054cc/` (`metrics.json` has the same steps).
- [x] **Idempotence.** The previous two runs in `pipeline_runs` show the other branch: a
  natural new-quarter run (`refresh-4q-948a5118e6`, 115 s) followed by a no-op
  (`refresh-4q-eb251f0116`, `new_quarter = false`, finished in the same second, wrote only
  its own row).

## 2. The time machine for 2009Q2 reproduces the walk-forward results exactly

- [x] **Published scores.** `scores` rows with `model = 'gbdt_mono'` for the four quarters
  whose `quarters.model_year = 2009` (`2009-03-31, 2009-06-30, 2009-09-30, 2009-12-31`):
  32,811 rows, a single `model_version = gbdt_mono-2007-12-31-bbc0230`, 679 failures
  within four quarters (`sum(quarters.n_failures_next_4q) = 679`). The 2009Q2 quarter
  alone has 8,247 rows, 165 of them in the `high` band (top 2%); rank 1 is Republic
  Federal Bank, N.A. (cert 22846, FL, probability 0.2857), rank 2 FirsTier Bank
  (57646, CO), rank 3 SAVIBANK (57978, WA).
- [x] **Recall at the top 2%, pooled over the year.** Script: join those 32,811 rows to
  the warehouse `labels` table (`y_4q`), call
  `bankcanary.evaluation.metrics.evaluate(y, score, tie_breaker=cert)` on the raw
  `score` column (the published rank order):

  | source | recall_at_2pct | n | n_failures |
  |---|---|---|---|
  | published `scores` (this pass) | `0.374079528718704` | 32,811 | 679 |
  | `walkforward_metrics` (`gbdt_mono`, horizon 4, `test_year = 2009`) | `0.374079528718704` | 32,811 | 679 |

  `==` in Python is `True`, absolute difference `0.0`; `walkforward_metrics.pr_auc` for
  the same row is 0.3414, the value the model card prints for `gbdt_mono` 2009 (37.41%
  recall, 0.3414 PR-AUC). `tests/test_publish.py::
  test_time_machine_2009_recall_at_2pct_matches_walkforward_metrics` makes the same
  check on every run with a database.
- [x] **Built app.** `GET /time-machine?quarter=2009Q2` on the built app (200, 553 KB)
  renders "2009Q2 as the model saw it · Banks scored 8,247 · Failed within 12 months 174
  · Recall at top 2% 68 of 174 (39.1%), top 165 of the ranking · Model year 2009, trained
  on data before 2009 · Model version `gbdt_mono-2007-12-31-bbc0230`", and its year
  panel says "Pooled over the 4 published quarters of the 2009 walk-forward year this
  page computes 254 of 679 (37.4%); the model card's walkforward_metrics row says 37.4%
  from 679 failures — identical". 254 / 679 = 0.37408, the number above. The displayed
  model version and year equal `quarters.model_year = 2009` and `quarters.model_version`
  of `2009-06-30`. `npx playwright test e2e/time_machine_map.spec.ts` (8 passed, 16 s,
  including "replays 2009Q2 and reproduces the published 2009 recall" and the axe check in
  both themes) exercises the same page against the built app.

## 3. Every number shown in the UI traces to a database row and a model version

Numbers were read from the HTML the built app served (`GET /` 150 KB, `GET /bank/8681`
192 KB) with the scripts and styles stripped, then matched to rows in the local database.
Every page and table carries `model_version` and the quarter in its caption. (Gap pass,
same day: the bank profile, the map status line and the case-study rank table were the
three places that did not print it; they do now, the case study by publish run id plus the
production version at that publish, because its refit rows have no version column.)

| rendered | page | database row(s) | model_version |
|---|---|---|---|
| "Banks scored 4,313" | `/` | `quarters` `repdte = 2026-06-30`: `n_banks = 4313`; equals `SELECT count(*) FROM scores WHERE model = 'gbdt_mono' AND repdte = '2026-06-30'` = 4,313 | `quarters.model_version = gbdt_mono-2022-12-31-bbc0230` |
| "High band 87" | `/` | `SELECT count(*) FROM scores WHERE model = 'gbdt_mono' AND repdte = '2026-06-30' AND band = 'high'` = 87 (top 2% of 4,313 by rank, `percentile` rule of CONTRACT 15) | `scores.model_version = gbdt_mono-2022-12-31-bbc0230` on each of the 87 rows |
| Rank 1 "Lamont Bank of St. John, Saint John, WA · cert 8681 · $51.12M · High · 40.0% · Change 0.00%" | `/` | `scores` `(cert 8681, repdte 2026-06-30, model gbdt_mono)`: `probability = 0.4, rank = 1, band = high, delta_prob_prior_q = 0.0` (the 2026-03-31 row has `probability = 0.4, rank = 2`, so the change is 0); `banks` `cert 8681`: `name, city, state, latest_assets = 51124.0` (thousands → $51.12M), `size_bucket = under_100m` | `scores.model_version = gbdt_mono-2022-12-31-bbc0230` |
| Driver "Change in texas_ratio over the last 4 quarters is +78.87 pp … contribution +11.09" and "hazard model 4.3%" | `/bank/8681` | `drivers` `(cert 8681, repdte 2026-06-30, model gbdt_mono, rank 1)`: `feature = d4q_texas_ratio, shap_value = 11.088307, feature_value = 0.78868645, direction = raises`; `scores` `(8681, 2026-06-30, hazard)`: `probability = 0.04347826, rank = 2, band = high` | `drivers` rows belong to `gbdt_mono-2022-12-31-bbc0230`; the hazard row carries `hazard-2023-12-31-6740dae` |

The profile page also shows ranks 2 and 3 of the same `drivers` query (`total_rbc_ratio`
7.34%, +5.34; `wholesale_funding_ratio` 0.20%, +2.17), and the leaderboard's driver
column is the same table restricted to the visible certs (`leaderboardDrivers`). Every
query lives in `web/lib/queries/*.ts` (CONTRACT 18); the client does no arithmetic
except selecting precomputed rate-shock rows.

## 4. Deployed publicly; README links to the live site — pending owner

- [x] **Deployment readiness verified locally.** `npm run build` exit 0 (every route builds:
  `/`, `/bank/[cert]`, `/map`, `/time-machine`, `/sitemap.xml` and the API routes
  dynamic; the rest static with 1 h revalidate). `.github/workflows/ci.yml` parses as
  YAML with jobs `python` and `web`; `refresh.yml` (`schedule` + `workflow_dispatch`,
  job `refresh`) and `retrain.yml` (job `retrain`) parse too. `docs/RUNBOOK.md` section 5
  lists every secret the workflows read — `DATABASE_URL`, `REVALIDATE_URL`,
  `REVALIDATE_SECRET`, optional `FRED_API_KEY` — and section 6 the three Vercel variables
  (`DATABASE_URL`, `REVALIDATE_SECRET`, `NEXT_PUBLIC_SITE_URL`).
- [ ] **Pending owner** (RUNBOOK section 7, in this order): (1) create the Neon database
  through the Vercel marketplace and a read-only role for the web app; (2) import the
  repository into Vercel with root directory `web`, Node 22, the three environment
  variables above (reader role, `sslmode=verify-full`); (3) add the four GitHub Actions
  secrets with `gh secret set`; (4) seed once from a machine with the warehouse:
  `DATABASE_URL='<writer pooled string>' uv run bankcanary publish`, then redeploy;
  (5) verify the counts of RUNBOOK section 3, open the leaderboard and
  `/time-machine?quarter=2009Q1`, and POST `/api/revalidate` with the bearer secret;
  (6) run the Refresh workflow by hand once (about 20 min cache warm-up, ends with
  `no new quarter`); (7) put the live URL in `README.md` and `NEXT_PUBLIC_SITE_URL`.
  Nothing in this repository deploys or touches an external account.

## Tests and hygiene (same commit)

- `uv run pytest`: **444 passed in 22 s** (the Postgres-backed tests ran against the local
  database instead of skipping). `npm test` in `web/`: 13 files, **114 passed** (vitest,
  no database). `npx tsc --noEmit`: exit 0. `npx playwright test e2e/time_machine_map.spec.ts`:
  8 passed against the built app (the Playwright `webServer` starts and stops it).
- `git log --format=%B | grep -iE 'co-authored|claude|anthropic|assistant|generated with'`
  prints nothing; `grep -riE 'claude|anthropic'` over `*.py *.md *.toml *.yaml *.yml *.ts
  *.tsx *.ipynb` (excluding `.venv`, `node_modules`, `.next`, `data`, `PROJECT_SPEC.md`)
  prints nothing.
- Before this commit `git status` showed only the refresh run record of criterion 1
  (`runs/refresh/refresh-4q-0200d054cc/`, `runs/index.jsonl`), which is committed with
  this checklist; `HEAD == origin/main` (`7e7cfc3`) before and after the push.
- Not re-run here (unchanged since the previous step): the full 65-test Playwright
  suite, the 6-test database-down suite and Lighthouse; their last results are in
  `docs/DECISIONS.md` (2026-09-26, E11).
