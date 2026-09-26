# Prototype 2 acceptance checklist

Spec: `PROJECT_SPEC.md`, "Prototype 2 — Depth". Every criterion below was re-measured on
2026-09-25 against the cached FDIC pull of the same date (`failures` `fetched_at =
2026-09-25`) and re-measured on 2026-09-26 after the raw-feature booster refit, the
2006-2007 test years, the `logit_v1` refit and the trailing calibration recipe; the
committed run records under `runs/` (1,560 records in `runs/index.jsonl` after `runs prune`:
162 `walkforward`, 1,055 `tune_walkforward`, 153 `calibrate`, 98 `tune_gbdt`, 7
`tune_hazard`, 11 `train`, 14 `sensitivity`, 36 `explain`, 9 `metrics`, 4 `case_study_2023`,
7 `publish`, 4 `refresh`; every run directory the index names is committed) and the
warehouse tables named next to each item.
`uv run pytest`: 454 passed, no network, no `data/` reads. The written summary of
everything below is [`docs/model_card.md`](model_card.md). The acceptance pass at the end of
this file records the commands and numbers behind each tick.

- [x] **Walk-forward results exist for every year from 2008 to the latest complete year, with
  failure counts and CIs.** `walkforward_scores` holds 4,199,937 rows: 4q for `texas, logit,
  logit_v1, gbdt, gbdt_mono, hazard` over 19 test years 2006-2024 (496,179 bank-quarters,
  2,191 failures; `walkforward.FIRST_TEST_YEAR = 2006`) and 8q for `logit, gbdt, hazard`
  over 16 test years 2008-2023 (407,621 rows, 3,768 failures); 2024 is the last 4q-complete
  year and 2023 the last 8q-complete year under the 2026-09-25 `as_of_date`.
  `models/walkforward/<Y>/` exists for all 19 years with `pipeline.joblib, features.json,
  config.json (stamped with model_hash and model_version), metrics.json, tuning.json,
  calibration.{joblib,json}` per model (16 `_8q` directories each for logit, gbdt and
  hazard). Both sides of Decision Point 2 are walked forward (`gbdt` and `gbdt_mono`, model
  card section 5). `reports/walkforward.md` lists every year with `n`,
  `failures`, PR-AUC and recall@2% with 200-draw cluster-bootstrap 95% intervals, Brier raw
  and calibrated, and a `low confidence` flag for years under 10 failures (2006, 2017, 2020,
  2021, 2024 at 4q; 2020 at 8q; 2021 has no 4q failure and undefined ranking metrics).

- [ ] **The best model beats P1's regularised logistic on pooled walk-forward PR-AUC and
  recall@top-2%.** Met on PR-AUC, level on recall@2%. The fair comparison is `logit_v1`:
  the P1 learner (L2 logit, no class weighting) walked forward on the 43 `features_v1`
  columns with its own `C` re-selected for every test year on that year's nested
  validation slice, the recipe every other model gets (`models/walkforward/<Y>/logit_v1/`,
  run records with `features_version = "v1"`). Over the 19 test years 2006-2024 (496,179
  4q rows, 2,191 failures) it pools to PR-AUC 0.2801 [0.247, 0.315] and recall@2% 0.7047
  [0.671, 0.736] (ROC-AUC 0.9475, Brier 0.0037 raw / 0.0047 calibrated). The hazard pools
  to 0.3194 [0.286, 0.352] and 0.7033 [0.674, 0.733], the monotone booster to 0.3022
  [0.275, 0.335] and 0.6992 [0.671, 0.724], the v2 logit to 0.2953 [0.264, 0.330] and
  0.6787 [0.648, 0.710]: every P2 probability model except the unconstrained booster
  (0.2160 / 0.5728) sits above the v1 logit on PR-AUC, but no interval is disjoint from
  the v1 logit's, and on recall@2% the v1 logit is level with the hazard and above the v2
  logit. Restricted to 2008-2024 (426,019 rows, 2,103 failures) it pools to 0.2901 /
  0.6933 against the hazard's 0.3268 / 0.7038 and the monotone booster's 0.3131 / 0.7066;
  the earlier comparison (0.2194 / 0.5625) borrowed each year's `C` from the v2 logit's
  selection, which under-regularised the smaller feature set and overstated the gap. The
  Texas ratio keeps the best recall@2% (0.7485) with a low PR-AUC (0.2534). The P1
  fixed-split figure (0.3867 / 0.7143, test 2010-2013) is a different test period and is
  not the comparison. At 8q (2008-2023) the hazard is best (0.4113 / 0.6598; logit 0.2566 /
  0.5207, gbdt 0.1059 / 0.2585).

- [x] **Median lead time for 2009-2012 failures is reported.** `reports/walkforward.md`,
  "Lead time" (`metrics.lead_time_summary`): of the 440 banks that failed in 2009-2012,
  the hazard flagged 403 (91.6%) in some quarter's top 2% before failure, median lead 6
  quarters, 89.8% at least 2 quarters ahead; logit 398 (90.5%), 5 quarters, 88.4%;
  gbdt_mono 387 (88.0%), 5 quarters, 86.4%; logit_v1 387 (88.0%), 5 quarters, 86.4%; gbdt
  382 (86.8%), 5 quarters, 85.5%; texas 372 (84.5%), 5 quarters, 82.5%. Over all 552
  failures with a scored quarter before them the hazard flags 91.5%, median 6, 89.7% two
  or more quarters ahead. The spec's target (most failures flagged >= 2 quarters ahead) holds.

- [x] **The 2023 notebook clearly shows how SVB/Signature/First Republic ranked under both
  models and explains why.** `notebooks/03_svb_2023_case_study.ipynb` (7 code cells, 0 error
  outputs, executed in place and re-executed on 2026-09-25 to a scratch copy in 33 s; `reports/svb_2023_case_study.md`; `runs/case_study_2023/`, four
  records) fits credit-only and rate-aware views of the logit and the booster on the 2022Q4
  cut (622,341 rows, 2,226 failures, reports through 2021Q3) and ranks all ~4,800 banks at
  2022Q3, 2022Q4 and 2023Q1. SVB at 2022Q4 (raw-feature `gbdt_mono`): credit-only logit rank
  1677 / gbdt_mono 2089, rate-aware logit 1854 / gbdt_mono 2711 (43.2nd percentile). Signature:
  113 / 1008 / 627 / 301 (93.7th). First Republic: 1570 / 978 / 1750 / 1813. The drivers table
  shows `adjusted_tier1_leverage` = -0.33 as the booster's top SVB driver (+0.47 log-odds),
  outweighed by the credit and capital ratios (-1.06 together), and the rate-aware logit
  cancelling it through `uninsured_share` and `macro_fedfunds_change_4q`, which carry safer
  signs over 2001-2021. The notebook also plots unrealised losses and uninsured share of the
  three banks against peer bands 2020-2023 (`reports/figures/svb_unrealized_losses.png`).

- [x] **Calibrated probabilities: reliability curve within reasonable tolerance on the test
  period; Brier score reported.** Brier is reported for every model, year and horizon, raw
  and calibrated (`reports/walkforward.md`, `runs/metrics/`). The map for test year Y is a
  binned isotonic fit (`calibration.binned_isotonic`, bins of at least 50 rows, so every
  step is a rate observed on at least 50 banks) on the same model's out-of-sample
  walk-forward scores from the two most recent earlier test years whose windows closed
  before Y's first prediction date (`trailing`; 2006-2008 at 4q and 2008-2011 at 8q fall
  back to one in-window slice, `full` for the logits and `inner` for the boosters, flagged
  `fallback = true`). Pooled 4q Brier raw / calibrated over 2006-2024: hazard 0.0037 /
  0.0038, gbdt_mono 0.0041 / 0.0044, logit_v1 0.0037 / 0.0047, gbdt 0.0042 / 0.0049, logit
  0.0037 / 0.0050; top-decile mean calibrated against observed: hazard 0.054 / 0.039,
  gbdt_mono 0.067 / 0.037, logit_v1 0.080 / 0.040, gbdt 0.052 / 0.030, logit 0.093 / 0.040;
  no calibrated value is 1.0 (the largest is 0.5968) and the 2009Q2 plateau of 170 banks at
  exactly 0.5 is gone. The two earlier recipes (inner-model scoring for every learner, then
  one in-window year) and their numbers are in `docs/DECISIONS.md`. Remaining caveats: the
  2010-2012 maps are learned on crisis-year scores and over-predict (2010 mean calibrated
  0.053-0.095 against a rate of 0.013), which is why no pooled 4q Brier falls under
  calibration; post-2016 maps rest on 4 to 61 failures; the 8q hazard map over-predicts
  2010-2011 (0.0070 raw against 0.0082 calibrated); and the 8q booster map is worse than
  raw (0.0087 against 0.0312) because that model's raw scale drifts between years, so it
  should not be read as calibrated.

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
| 7 walk-forward harness, one artefact per test year | done | `bankcanary walkforward --year Y`, `models/walkforward/<Y>/<model>[_8q]/`, 162 referenced run records after `runs prune` (`texas, logit, logit_v1, gbdt, gbdt_mono, hazard` at 4q over 2006-2024; `logit, gbdt, hazard` at 8q over 2008-2023), each stamped with the content-addressed `model_version` |
| 8 isotonic calibration and reliability plots | done | `bankcanary calibrate [--scorer trailing|full|inner] [--min-bin N]`, `score_calibrated` filled for 4,199,937 rows (Texas NaN), `reports/figures/reliability_{logit,logit_v1,gbdt,gbdt_mono,hazard}.png` |
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
| walk-forward coverage | DuckDB: `walkforward_scores` grouped by `model` where `horizon = 4` | `texas, logit, logit_v1, gbdt, gbdt_mono, hazard`: 19 test years each, 2006-2024, 496,179 rows, 2,191 positives, `label_complete` on every row; `score_calibrated` filled except for `texas`; per-year positives 8, 80, 429, 679, 408, 231, 121, 73, 37, 24, 28, 5, 10, 18, 4, 0, 17, 10, 9; latest failure in `failures` 2026-07-17, so 2024 is the last 4q-complete year |
| best model vs P1 logit | `logit_v1`: `features_v1` walk-forward logit tuned per year, 19 fits (about 9 s each), pooled by `metrics-report` with `cluster_bootstrap_ci` | see criterion 2: hazard 0.3194 / 0.7033 against 0.2801 / 0.7047; PR-AUC intervals overlap, recall level |
| lead time 2009-2012 | `reports/walkforward.md`, "Lead time" | hazard median 5 quarters, 88.4% flagged >= 2 quarters ahead (logit 5, 87.7%) |
| notebook 03 | `jupyter nbconvert --execute` to a scratch copy, 110 s cell timeout | 7 code cells, 0 error outputs; SVB 2022Q4 ranks 1677 / 2089 (credit-only logit / gbdt_mono) and 1854 / 2711 (rate-aware); section 5 gives the explanation |
| calibration | `reports/walkforward.md`, pooled table and "Calibration" (2026-09-26, trailing binned maps) | Brier raw / calibrated: hazard 0.0037 / 0.0038, gbdt_mono 0.0041 / 0.0044, logit_v1 0.0037 / 0.0047, gbdt 0.0042 / 0.0049, logit 0.0037 / 0.0050; max `score_calibrated` 0.5968; `reports/figures/reliability_{logit,logit_v1,gbdt,gbdt_mono,hazard}.png` committed |
| Decision Point 2 | `reports/walkforward.md`, pooled 4q table (2006-2024) | gbdt_mono 0.3022 [0.275, 0.335] / 0.6992 [0.671, 0.724] against gbdt 0.2160 [0.191, 0.240] / 0.5728 [0.545, 0.600]; inner validation 0.214 against 0.158 on the raw features (`config/settings.yaml`) |
| hazard at 8q | `reports/walkforward.md`, "Horizon 8q" | 16 test years, PR-AUC 0.4113 [0.382, 0.442], recall@2% 0.6598 [0.634, 0.688] |
| model card | `grep '^#' docs/model_card.md` | 14 sections: intended use, data, label and censoring, leakage, models (with the model-version definition), walk-forward metrics (pooled and per year at 4q and 8q, the Prototype 1 comparison), calibration, lead time, 2023 case study, sensitivity, false positives, limitations, ethics, references |
| tests | `uv run pytest` | 454 passed (2026-09-26) |
| determinism | `bankcanary build-labels` then `bankcanary build-features-v2`, `shasum -a 256` before and after | `labels.parquet` `fe0c3237...86286fdec` and `features_v2.parquet` `8871e61c...af93cd0685a` unchanged |
| hygiene | case-insensitive grep of every commit message for `co-authored`, tool and vendor names; the same grep over `*.py *.md *.toml *.yaml *.ipynb` excluding `.venv`, `data`, `PROJECT_SPEC.md` | both print nothing |
