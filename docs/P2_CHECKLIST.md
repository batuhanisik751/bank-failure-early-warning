# Prototype 2 acceptance checklist

Spec: `PROJECT_SPEC.md`, "Prototype 2 — Depth". Every criterion below was re-measured on
2026-09-25 against the cached FDIC pull of the same date (`failures` `fetched_at =
2026-09-25`), the committed run records under `runs/` (1,001 rows in `runs/index.jsonl`:
200 `walkforward`, 526 `tune_walkforward`, 166 `calibrate`, 49 `tune_gbdt`, 7 `tune_hazard`,
9 `train`, 14 `sensitivity`, 18 `explain`, 8 `metrics`, 4 `case_study_2023`; every run
directory the index names is committed) and the warehouse tables named next to each item.
`uv run pytest`: 402 passed in 22 s, no network, no `data/` reads. The written summary of
everything below is [`docs/model_card.md`](model_card.md). The acceptance pass at the end of
this file records the commands and numbers behind each tick.

- [x] **Walk-forward results exist for every year from 2008 to the latest complete year, with
  failure counts and CIs.** `walkforward_scores` holds 3,352,958 rows: 4q for `texas, logit,
  gbdt, gbdt_mono, hazard` over 17 test years 2008-2024 (426,019 bank-quarters, 2,103
  failures) and 8q for `logit, gbdt, hazard` over 16 test years 2008-2023 (407,621 rows,
  3,768 failures); 2024 is the last 4q-complete year and 2023 the last 8q-complete year under
  the 2026-09-25 `as_of_date`. `models/walkforward/<Y>/` exists for all 17 years with
  `pipeline.joblib, features.json, config.json, metrics.json, tuning.json,
  calibration.{joblib,json}` per model (16 `_8q` directories each for logit, gbdt and
  hazard). Both sides of Decision Point 2 are walked forward (`gbdt` and `gbdt_mono`, model
  card section 5). `reports/walkforward.md` lists every year with `n`,
  `failures`, PR-AUC and recall@2% with 200-draw cluster-bootstrap 95% intervals, Brier raw
  and calibrated, and a `low confidence` flag for years under 10 failures (2017, 2020, 2021,
  2024 at 4q; 2020 at 8q; 2021 has no 4q failure and undefined ranking metrics).

- [x] **The best model beats P1's regularised logistic on pooled walk-forward PR-AUC and
  recall@top-2%.** Met, and statistically separated, against the fair comparison: the P1
  learner (L2 logit, no class weighting) walked forward on the 43 `features_v1` columns over
  the same 426,019 4q rows, one fit per test year 2008-2024 with each year's `C` taken from
  the same year's v2 logit selection (chosen inside that year's training period; run records
  `runs/walkforward/` with `features_version = "v1"`, 17 records). It pools to PR-AUC 0.2194
  [0.191, 0.246] and recall@2% 0.5625 [0.531, 0.589] (ROC-AUC 0.9441, Brier 0.0043). The
  hazard pools to 0.3268 [0.294, 0.357] and 0.7038 [0.679, 0.730]; the intervals do not
  overlap on either metric, and the v2 logit (0.3056 / 0.6833) and booster (0.2647 / 0.6367)
  also sit above the v1 logit on both metrics (the booster's PR-AUC interval, [0.236, 0.297],
  touches the v1 logit's). Among the P2 learners the hazard and the v2 logit are not
  separated from each other (overlapping intervals); the Texas ratio keeps the best
  recall@2% (0.7437) with the worst PR-AUC (0.2606). The P1 fixed-split figure (0.3867 /
  0.7143, test 2010-2013) is a different test period and is not the comparison. At 8q the
  hazard is best (0.4113 / 0.6598; logit 0.2566 / 0.5207, gbdt 0.0887 / 0.2200).

- [x] **Median lead time for 2009-2012 failures is reported.** `reports/walkforward.md`,
  "Lead time" (`metrics.lead_time_summary`): of the 440 banks that failed in 2009-2012,
  the hazard flagged 398 (90.5%) in some quarter's top 2% before failure, median lead 5
  quarters, 88.4% at least 2 quarters ahead; logit 395 (89.8%), 5 quarters, 87.7%; gbdt 364
  (82.7%), 4.5 quarters, 81.6%; texas 372 (84.5%), 4 quarters, 82.5%. Over all 545 failures
  with a scored quarter before them the logit flags 89.4%, median 5, 87.5% two or more
  quarters ahead. The spec's target (most failures flagged >= 2 quarters ahead) holds.

- [x] **The 2023 notebook clearly shows how SVB/Signature/First Republic ranked under both
  models and explains why.** `notebooks/03_svb_2023_case_study.ipynb` (7 code cells, 0 error
  outputs, executed in place and re-executed on 2026-09-25 to a scratch copy in 33 s; `reports/svb_2023_case_study.md`; `runs/case_study_2023/`, four
  records) fits credit-only and rate-aware views of the logit and the booster on the 2022Q4
  cut (622,341 rows, 2,226 failures, reports through 2021Q3) and ranks all ~4,800 banks at
  2022Q3, 2022Q4 and 2023Q1. SVB at 2022Q4: credit-only logit rank 1677 / gbdt 1412,
  rate-aware logit 1854 / gbdt 245 (94.9th percentile). Signature: 113 / 1648 / 627 / 2634.
  First Republic: 1570 / 1167 / 1750 / 1416. The drivers table shows the booster's SVB score
  coming from `adjusted_tier1_leverage` = -0.33 (+2.1 log-odds) and the rate-aware logit
  cancelling it through `uninsured_share` and `macro_fedfunds_change_4q`, which carry safer
  signs over 2001-2021. The notebook also plots unrealised losses and uninsured share of the
  three banks against peer bands 2020-2023 (`reports/figures/svb_unrealized_losses.png`).

- [x] **Calibrated probabilities: reliability curve within reasonable tolerance on the test
  period; Brier score reported.** Brier is reported for every model, year and horizon, raw
  and calibrated (`reports/walkforward.md`, `runs/metrics/`). The isotonic map is fitted per
  year on the last complete label year inside the training window; the logit and the hazard
  score that slice with the year's own model, the boosters with an inner model fitted before
  it (`calibration.SLICE_SCORER_BY_MODEL`; the boosters' in-sample slice scores separate
  perfectly and carry no calibration information). Pooled 4q Brier raw / calibrated: logit
  0.0041 / 0.0038, hazard 0.0041 / 0.0038, gbdt 0.0043 / 0.0044, gbdt_mono 0.0045 / 0.0048;
  top-decile mean calibrated against observed: logit 0.042 / 0.045, hazard 0.053 / 0.043,
  gbdt 0.045 / 0.036, gbdt_mono 0.068 / 0.042. The earlier inner-model recipe for the linear
  models (pooled Brier 0.0189 and 0.0545, 2009 mean calibrated 0.19 and 0.47 against a rate
  of 0.021) is recorded in `docs/DECISIONS.md`. Remaining caveats: post-2013 maps rest on 0
  to 37 failures, the boosters over-predict the top decile, and the 8q hazard map
  over-predicts 2010-2011 (Brier 0.0070 raw against 0.0080 calibrated), and the 8q booster
  map is worse than raw (0.0088 against 0.0203) where the retuned 2011-2013 boosters sit on
  a coarser scale than the inner model that scored their slice.

- [x] **Model card written.** `docs/model_card.md`: intended use and non-use; data sources,
  coverage 2001Q1-2026Q2, units, the FDIC securities and uninsured fields and their FFIEC
  cross-check, macro sources; label and censoring; leakage protections; models and the
  LightGBM backend with its scikit-learn fallback; per-year and pooled walk-forward tables
  with failure counts and CIs at 4q and 8q; calibration; lead time; the 2023 case study;
  sensitivity and false-positive summaries; limitations; ethical considerations; references.

- [x] **Hygiene.** A case-insensitive grep of every commit message for co-author trailers and
  tool or vendor names prints nothing; the same grep over `*.py, *.md, *.toml, *.yaml, *.ipynb`
  (excluding `.venv`, `data`, `PROJECT_SPEC.md`) prints nothing. `data/`, `models/` and `.env`
  are git-ignored; `runs/` (JSON only) is committed, including the 328 `walkforward` and
  `tune_walkforward` run directories that the index already named but that had been left
  untracked before the acceptance pass.

## Feature list check (spec items 1-15)

| item | status | evidence |
|---|---|---|
| 1 FFIEC RC-B / RC-O ingestion with `IDRSSD <-> CERT` crosswalk | done, as a cross-check | the FDIC `/financials` fields `scaa, scaf, scha, schf, depunins` carry the same values (spec 3.2: prefer the FDIC API); `reports/ffiec_crosscheck.md` matches all 15 SVB/Signature/First Republic 2022Q4 items exactly; `crosswalk_rssd` (27,834 certs); the bulk downloader is optional |
| 2 rate / run-risk features | done | `unrealized_loss_to_tier1, adjusted_tier1_leverage, uninsured_share, securities_to_assets` plus five more in `features_v2` (`docs/FEATURES.md`) |
| 3 trend and persistence features | done | 16 `d1q_`/`d4q_` deltas, `neg_roa_quarters_last_8, consecutive_loss_quarters, noncurrent_rising_quarters_last_4` |
| 4 FRED macro features | done | `macro_state` (5,650 state-availability rows, point-in-time with publication lags); `macro_unemp_rate, macro_unemp_change_4q, macro_hpi_change_4q, macro_t10y3m, macro_fedfunds_change_4q` |
| 5 full feature set with registry | done | 83 registered features (43 P1 + 40 P2), each with group, formula, unit, explanation and monotone sign; `docs/FEATURES.md` generated from the registry |
| 6 LightGBM, hazard, optional Cox | done (Cox dropped) | `models.gbdt.make_gbdt` (LightGBM 4.7, sklearn fallback), `models.hazard`; `lifelines` pins pandas < 3 (`docs/DECISIONS.md`) |
| 7 walk-forward harness, one artefact per test year | done | `bankcanary walkforward --year Y`, `models/walkforward/<Y>/<model>[_8q]/`, 200 run records (`texas, logit, gbdt, gbdt_mono, hazard` at 4q; `logit, gbdt, hazard` at 8q) |
| 8 isotonic calibration and reliability plots | done | `bankcanary calibrate [--scorer full|inner]`, `score_calibrated` filled for 3,352,958 rows (Texas NaN), `reports/figures/reliability_{logit,gbdt,gbdt_mono,hazard}.png` |
| 9 SHAP per bank-quarter | done | `drivers` table: 4,525,760 rows over 452,576 bank-quarters, ten drivers each, for the 17 walk-forward boosters and the 2024 production model; `reports/shap_summary.md` |
| 10 metrics suite | done | `evaluation.metrics`: PR-AUC, ROC-AUC, recall@k, Brier, lead time, cluster-bootstrap CIs; per-year and pooled tables in `reports/walkforward.md` |
| 11 2023 case study notebook | done | `notebooks/03_svb_2023_case_study.ipynb` (see the criterion above) |
| 12 false-positive analysis | done | `notebooks/04_false_positives.ipynb`, `reports/false_positives.md`: 875 flagged non-failing bank-years 2009-2012, 26.4% failed later, 9.9% acquired within two years, 61.8% still open |
| 13 sensitivity analyses | done | `bankcanary sensitivity`, `reports/sensitivity.md`, `notebooks/05_sensitivity.ipynb`: 4q/8q, censored kept/dropped, lag 45/60/90 days, 14 run records |
| 14 experiment tracking | done (JSON, not MLflow) | `bankcanary.tracking`: `runs/<name>/<run_id>/{config,metrics}.json` + `runs/index.jsonl`, deterministic ids `<name>-<H>q-<config hash>`, read back with `tracking.read_index(settings)`; the `runs list` CLI named in the contract was not built (the index file is greppable) |
| 15 model card | done | `docs/model_card.md` |

Notebooks 02-05 are built by `scripts/make_notebook_0{2,3,4,5}.py` and executed in place
(`jupyter nbconvert --execute`); each runs in under a minute after the tables and `runs/`
exist and none retrains a walk-forward model. Open items for Prototype 3: settle Decision
Point 2 (the walk-forward evidence for both boosters is in the model card, section 5);
treat post-2021 scores as out of regime (model card, section 12).

## Acceptance pass (2026-09-25)

| check | command | result |
|---|---|---|
| walk-forward coverage | DuckDB: `walkforward_scores` grouped by `model` where `horizon = 4` | `texas, logit, gbdt, gbdt_mono, hazard`: 17 test years each, 2008-2024, 426,019 rows, 2,103 positives, `label_complete` on every row; `score_calibrated` filled except for `texas`; per-year positives 429, 679, 408, 231, 121, 73, 37, 24, 28, 5, 10, 18, 4, 0, 17, 10, 9; latest failure in `failures` 2026-07-17, so 2024 is the last 4q-complete year |
| best model vs P1 logit | `features_v1` walk-forward logit, 17 fits (4-8 s each), pooled with `metrics.evaluate` and `cluster_bootstrap_ci` | see criterion 2: hazard 0.3268 / 0.7038 against 0.2194 / 0.5625, intervals disjoint |
| lead time 2009-2012 | `reports/walkforward.md`, "Lead time" | hazard median 5 quarters, 88.4% flagged >= 2 quarters ahead (logit 5, 87.7%) |
| notebook 03 | `jupyter nbconvert --execute` to a scratch copy, 110 s cell timeout | 7 code cells, 0 error outputs; SVB 2022Q4 ranks 1677 / 1412 (credit-only logit / gbdt) and 1854 / 245 (rate-aware); section 5 gives the explanation |
| calibration | `reports/walkforward.md`, pooled table and "Calibration" | Brier raw / calibrated: hazard 0.0041 / 0.0038, logit 0.0041 / 0.0038, gbdt 0.0043 / 0.0044, gbdt_mono 0.0045 / 0.0048; `reports/figures/reliability_{logit,gbdt,gbdt_mono,hazard}.png` committed |
| Decision Point 2 | `reports/walkforward.md`, pooled 4q table | gbdt_mono 0.3138 [0.282, 0.345] / 0.7147 [0.692, 0.739] against gbdt 0.2647 [0.236, 0.297] / 0.6367 [0.603, 0.663]; inner validation 0.197 against 0.230 (`config/settings.yaml`) |
| hazard at 8q | `reports/walkforward.md`, "Horizon 8q" | 16 test years, PR-AUC 0.4113 [0.382, 0.442], recall@2% 0.6598 [0.634, 0.688] |
| model card | `grep '^#' docs/model_card.md` | 14 sections: intended use, data, label and censoring, leakage, models, walk-forward metrics (pooled and per year at 4q and 8q), calibration, lead time, 2023 case study, sensitivity, false positives, limitations, ethics, references |
| tests | `uv run pytest` | 402 passed in 22 s |
| determinism | `bankcanary build-labels` then `bankcanary build-features-v2`, `shasum -a 256` before and after | `labels.parquet` `fe0c3237...86286fdec` and `features_v2.parquet` `8871e61c...af93cd0685a` unchanged |
| hygiene | case-insensitive grep of every commit message for `co-authored`, tool and vendor names; the same grep over `*.py *.md *.toml *.yaml *.ipynb` excluding `.venv`, `data`, `PROJECT_SPEC.md` | both print nothing |
