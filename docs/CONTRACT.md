# Build contract

The shared conventions every component follows. `PROJECT_SPEC.md` says *what* to build;
this file pins *how* the pieces fit so that modules written independently compose.
If a step needs to deviate, append the reason to `docs/DECISIONS.md` and update this file.

## 1. Toolchain

- Python 3.12 (`.python-version`), managed by `uv`. Run everything through `uv run …`.
  Dependencies are declared in `pyproject.toml`; add one with `uv add <pkg>` (or
  `uv add --dev <pkg>`) and commit `pyproject.toml` + `uv.lock` together.
- `ruff` (lint + format, line length 100), `pytest` (`tests/`), `pre-commit` hooks.
- **pandas 3.x** is installed: copy-on-write is always on, the default string dtype is
  `str`. Assign with `.loc[...]` or `.assign(...)`; never chained assignment.
- Package layout: `src/bankcanary/<area>/<module>.py`. Public functions get a docstring
  that explains the banking meaning, not just the code.
- Logging via `logging.getLogger(__name__)`; no `print` outside the CLI and scripts.

## 2. Settings and secrets

`bankcanary.config.load_settings()` reads `config/settings.yaml` into a validated
`Settings` object (paths anchored at the repo root). Keys: `data_dir`, `models_dir`,
`reports_dir`, `start_quarter`, `end_quarter`, `availability_lag_days`,
`horizons_quarters`, `fixed_split.{train_start,train_end,test_start,test_end}`,
`peer_asset_buckets_thousands`, `fdic.{base_url,docs_url,page_size,requests_per_second,
timeout_seconds,max_retries}`. Secrets (`FDIC_API_KEY`, `FRED_API_KEY`, `DATABASE_URL`)
come only from the environment / `.env` via `load_secrets()`. Never read `.env` any other way.

## 3. Storage layout (all git-ignored)

```
data/raw/fdic/docs/<name>.yaml            field-definition files from api.fdic.gov/banks/docs/
data/raw/fdic/<endpoint>/<name>.json      cached API responses (see §4)
data/parquet/<table>.parquet              typed tables (canonical outputs)
data/warehouse.duckdb                     DuckDB copy of the same tables, same names
models/<model_name>/                      fitted pipelines (joblib) + metrics.json + config.json
reports/                                  generated markdown + figures (committed when small)
```

`bankcanary.storage.parquet.write_table(df, name)` / `read_table(name)` and
`bankcanary.storage.duckdb.replace_table(name, df_or_parquet)` / `connect()` are the only
ways tables are written. `write_table` **sorts by the table key, fixes column order, and
writes with pyarrow (zstd)** so that re-running from cache is byte-identical.

## 4. FDIC client (`bankcanary.sources.fdic.FdicClient`)

- `get(endpoint, *, filters=None, fields=None, sort_by=None, sort_order="ASC",
  limit=None, cache_name=None, force=False) -> list[dict]`: returns the flattened `data`
  records of **all pages** (pagination via `offset`, page size from settings; offsets past
  10,000 work). Unknown field names are silently dropped by the API, so `get` warns when a
  requested field is missing from every returned record.
- Cache: one JSON file per call under `data/raw/fdic/<endpoint>/`. Name = `cache_name`
  when given (ingest code uses `<REPDTE>` for financials, `all` for full-table pulls), else
  a sha1 of the canonical query. Each file stores `{"query": …, "fetched_at": ISO date,
  "total": n, "data": [...]}`. Cached files are never re-downloaded unless `force=True`.
- Rate limit: token bucket at `fdic.requests_per_second`. Retries with exponential backoff
  (tenacity) on 429/5xx/timeouts up to `fdic.max_retries`.
- Tests use `httpx.MockTransport` with recorded fixtures in `tests/fixtures/`; no live calls.
- Field definitions: `fetch_definitions(name)` downloads
  `<docs_url><name>.yaml` (`risview_properties` = financials, `failure_properties`,
  `institution_properties`, `history_properties`, `location_properties`) into
  `data/raw/fdic/docs/` and returns the parsed `properties.data.properties` mapping.

## 5. Tables

All column names are `snake_case`. Dates are `datetime64[ns]` (pandas) / `DATE` (DuckDB).
Dollar amounts are **thousands of dollars** (the FDIC unit); ratios are as published (%).

| table | key | required columns |
|---|---|---|
| `failures` | `cert, fail_date` | `name, fail_year, restype ('FAILURE'/'ASSISTANCE'), restype1, cost, qbfasset, qbfdep, cityst, pstalp, chclass1` |
| `institutions` | `cert` | `name, stalp, city, bkclass, charter, estymd, endefymd (nullable), active (bool), fed_rssd, rssdhcr, asset, ultcert, newcert, latitude, longitude, procdate` |
| `history` | `cert, transnum` | `changecode (int), changecode_desc, effdate, acq_cert, out_cert, acq_uninum, out_uninum` — institution-level events only (`changecode < 500`) |
| `financials_raw` | `cert, repdte` | one lower-cased column per code in `config/fields.yaml`; `name, stalp, bkclass, rssdhcr` if present |
| `panel` | `cert, repdte` | every `financials_raw` column + `avail_date, fail_date, exit_date, exit_reason, assisted (bool), estymd, bkclass, stalp, rssdhcr, has_holding_company` |
| `labels` | `cert, repdte` | for each horizon H in {4, 8}: `y_{H}q, window_end_{H}q, censored_in_window_{H}q, label_complete_{H}q`; plus `dropped_failed_before_avail` (rows kept in the table, flagged, and excluded downstream) |
| `features_v1` | `cert, repdte` | feature columns exactly as named in the registry (§7) + `ytd_prev_missing` flags |

## 6. Panel, exits and labels

- `avail_date = repdte + availability_lag_days` (default 60).
- `fail_date`: from `failures` with `restype == 'FAILURE'` (earliest failure on or after the
  bank's `estymd`). `assisted = True` if the cert only has an `ASSISTANCE` record.
- `exit_date`: for inactive banks without a failure, `institutions.endefymd`; `exit_reason`
  from the institution-level `history` event on that date (`merger`, `voluntary_closing`,
  `absorption`, `consolidation`, `charter_change`, `other`), else `unknown`.
- `as_of_date`: the `fetched_at` date of the cached `failures` pull. Windows that end after
  it are `label_complete = False`. Using the cached date keeps rebuilds deterministic.
- Windows: `window_end = avail_date + DateOffset(months=3·H)`; the window is
  `(avail_date, window_end]`. Rules 1–6 of spec §5 apply verbatim; the required unit
  tests live in `tests/test_labels.py`.
- Rule 6.2 lives in **one** function, `bankcanary.splits.time_split.training_mask(labels,
  horizon, test_start_repdte)`, and every training path calls it.

## 7. Feature registry (`bankcanary.features.registry`)

Every feature is registered once with: `name, camels_group, formula, unit, explanation,
prototype ("P1"/"P2")`. `build_features(panel) -> DataFrame` evaluates the registry in
order. Names for P1 (spec §7.2): `equity_to_assets, tier1_leverage, total_rbc_ratio,
total_rbc_ratio_missing, tangible_equity_to_assets, noncurrent_ratio, npa_to_assets,
early_delinquency, nco_rate, reserve_coverage, texas_ratio, texas_ratio_capped,
asset_growth_4q, asset_growth_12q, loan_growth_4q, roa_q, nim_q, efficiency_ratio,
provision_rate, brokered_share, loans_to_deposits, liquid_assets_ratio,
wholesale_funding_ratio, construction_to_capital, cre_to_capital, share_construction,
share_nonfarm_nonres, share_multifamily, share_residential, share_ci, share_consumer,
share_agri, log_assets, bank_age_years, has_holding_company, bkclass_<code>` (one-hot).
Income items are de-accumulated by `features.ytd.deaccumulate()` first (spec §7.3):
quarterly = YTD(q) − YTD(previous quarter, same year); Q1 = YTD; missing prior quarter →
`ytd_prev_missing = True` and quarterly = YTD / quarters elapsed. Annualised = quarterly × 4.
Averages use the mean of the current and previous quarter-end (fallback: current).
Ratios guard zero/negative denominators (return NaN + a flag; Texas ratio → capped at 10
with `texas_ratio_capped = True`). Winsorisation is **not** applied here; it is a
pipeline step fitted on the training fold (§8).

## 8. Models and evaluation

- Model code lives in `bankcanary.models`; every estimator is an sklearn `Pipeline`
  (`Winsorizer(0.005, 0.995)` → `SimpleImputer(median, add_indicator)` → `StandardScaler`
  → estimator). Feature columns are taken from the registry; identifiers never enter.
- `models/<name>/` holds `pipeline.joblib`, `metrics.json`, `config.json`, `features.json`.
- `bankcanary.evaluation.metrics.evaluate(y_true, scores, k_fracs=(0.01,0.02,0.05),
  k_counts=(50,100)) -> dict` with `pr_auc, roc_auc, recall_at_1pct, recall_at_2pct,
  recall_at_5pct, recall_at_top50, recall_at_top100, n, n_failures`.
- Texas-ratio baseline = rank by `texas_ratio` descending. Small logistic = 6 features
  (`equity_to_assets, noncurrent_ratio, roa_q, brokered_share, construction_to_capital,
  log_assets`), L2 at `C=1.0`, `class_weight="balanced"`. Regularised logistic = all P1
  features, L2, **no class weighting**, `C = bankcanary.models.baselines.LOGIT_C` chosen by
  `scripts/tune_logit_c.py` on a validation slice inside the training period (rule 6.7;
  see `docs/DECISIONS.md`). Every training and evaluation run also reports the spec §5
  rule 3 sensitivity (censored rows dropped) and the rule 6 per-failure-event metrics
  (`metrics.evaluate_by_event`), stored in `metrics.json` under
  `sensitivity_censored_dropped` and `per_event`; `bankcanary evaluate` writes the
  figures to `reports/figures/`.

## 9. CLI (`bankcanary …`)

`ingest [--what failures|institutions|history|financials|all] [--start Q --end Q]
[--force]`, `build-panel`, `build-labels`, `build-features`, `train --model
{texas,logit_small,logit}`, `evaluate [--model …]`, `dq-report`. Every command is
idempotent and safe to re-run from cache.

## 10. Git and hygiene

- Commit after each spec step, with a plain imperative subject line (≤ 72 chars) and a short
  body if useful. **No co-author trailers and no tool attribution anywhere in the repo.**
- Never commit `data/`, `models/`, `.env`. Never `source` a file holding a connection string.
- Tests never touch the network. Long pulls run in chunks that finish in under two minutes.

---

## Prototype 2 additions

### 11. New tables

| table | key | columns |
|---|---|---|
| `crosswalk_rssd` | `cert` | `fed_rssd, name, source` (from `institutions.fed_rssd`) |
| `macro_state` | `stalp, avail_date` | `unemp_rate, unemp_change_4q, hpi_change_4q, t10y3m, fedfunds_change_4q` evaluated point-in-time at each panel `avail_date` (latest observation whose period end + publication lag ≤ avail_date; lags: state unemployment 45 days, state HPI 75 days, daily/monthly national rates 1 day) |
| `features_v2` | `cert, repdte` | every `features_v1` column + the P2 features (§12); `features_v1` is kept untouched for P1 reproducibility |
| `walkforward_scores` | `cert, repdte, horizon, model, test_year` | `score, score_calibrated, y, label_complete, censored` for every test-year model, from 2008 to the latest label-complete year |
| `drivers` | `cert, repdte, model, rank` | `feature, shap_value, feature_value, direction` — the five largest positive and five largest negative SHAP contributions per bank-quarter for the gradient-boosting walk-forward models and the latest production model |
| `runs/` (JSON files, committed) | — | `runs/<name>/<run_id>/{config.json, metrics.json}` + `runs/index.jsonl`; written by `bankcanary.tracking` for every `train`, `walkforward`, `calibrate`, `sensitivity` call; run ids are deterministic (`<name>-<horizon>q-<config hash>`), never timestamps |

### 12. P2 feature names (binding, `prototype = "P2"` in the registry)

Sensitivity: `afs_unrealized_to_tier1, htm_unrealized_to_tier1, unrealized_loss_to_tier1,
adjusted_tier1_leverage, securities_to_assets, htm_share_of_securities`.
Run risk: `uninsured_share, uninsured_to_liquid_assets, large_time_deposit_share`.
Trends: `d1q_<ratio>` and `d4q_<ratio>` for `noncurrent_ratio, texas_ratio, roa_q,
equity_to_assets, tier1_leverage, brokered_share, uninsured_share, unrealized_loss_to_tier1`;
persistence: `neg_roa_quarters_last_8, consecutive_loss_quarters, noncurrent_rising_quarters_last_4`.
Structure: `region_<code>` one-hot (FDIC regions derived from `stalp`), `is_community_bank`.
Macro (joined from `macro_state`): `macro_unemp_rate, macro_unemp_change_4q, macro_hpi_change_4q,
macro_t10y3m, macro_fedfunds_change_4q`.
Each module exposes `SPECS: list[FeatureSpec]` and `build(panel) -> DataFrame`; `registry.py`
concatenates module lists in a fixed order and `build.py` calls the builders in that order.

### 13. Models, backtest and explanations

- `bankcanary.models.gbdt`: gradient-boosted trees behind one factory, `make_gbdt(backend, monotone)`:
  backend `lightgbm` (`LGBMClassifier`) when `import lightgbm` succeeds, else `sklearn`
  (`HistGradientBoostingClassifier`, same histogram algorithm, native NaN, `monotonic_cst`).
  The chosen backend is written to `config/settings.yaml` (`models.gbdt.backend`) by step D6 and
  every later step reads it, so one run never mixes backends. Same preprocessing pipeline as the
  logits minus imputer/scaler (Winsorizer only; trees take NaN natively). Hyper-parameters
  chosen on the inner validation slice (reports 2007Q1–2008Q4, inner train windows closed
  before 2007-05-30), never on test years. Two configs are always trained and logged:
  `gbdt` (unconstrained) and `gbdt_mono` (monotone constraints from the registry's
  `monotone` field: +1 risk-increasing, −1 risk-decreasing, 0 none). **Decision Point 2 is
  presented to the owner with both sets of results; until then the config in
  `settings.models.gbdt.monotone` (default: the better inner-validation config) is used.**
- `bankcanary.models.hazard`: discrete-time hazard (Shumway 2001) = logistic regression on
  bank-quarter rows with event "fails within the next quarter" (`y_1q`, built by
  `labels.build` for horizon 1); 4q/8q probabilities by `1 − (1 − h)^H` assuming persistence
  of covariates (documented approximation); the optional Cox model is deferred (see DECISIONS: lifelines pins pandas < 3).
- `bankcanary.evaluation.walkforward`: for each test year Y from 2008 to the latest
  label-complete year, training rows = `training_mask(labels, H, first repdte of Y)`, test
  rows = report quarters in Y (label-complete, not dropped). Saves
  `models/walkforward/<Y>/<model>/` and appends to `walkforward_scores`. Runs one year per
  CLI call (`bankcanary walkforward --year Y [--model …]`) so no command exceeds two minutes.
- Calibration: per walk-forward year, isotonic regression fitted on the last complete year
  inside the training window (an inner model trained on the earlier years scores that slice);
  the map is then applied to the full-window model. Stored in `score_calibrated`.
- Metrics suite (`evaluation.metrics`): P1 metrics + `brier`, `lead_time_quarters` summary
  (median, share flagged ≥ 2 quarters ahead), cluster-bootstrap CIs by `cert` (200 draws;
  years with < 10 failures flagged `low_confidence`), per-year and pooled tables.
- SHAP: `bankcanary.explain.shap_drivers` with `TreeExplainer`; top-5 ± drivers per
  bank-quarter into `drivers`.

### 14. CLI additions

`build-macro`, `build-features --version v2`, `train --model {gbdt,gbdt_mono,hazard}`,
`walkforward --year Y --model M`, `calibrate`, `sensitivity`, `explain`, `runs list`.
All idempotent; all log a run record.
