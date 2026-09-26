# Prototype 2 acceptance checklist

Spec: `PROJECT_SPEC.md`, "Prototype 2 — Depth". Every criterion below was re-measured on
2026-09-25 against the cached FDIC pull of the same date (`failures` `fetched_at =
2026-09-25`), the committed run records under `runs/` (618 rows in `runs/index.jsonl`:
150 `walkforward`, 278 `tune_walkforward`, 83 `calibrate`, 49 `tune_gbdt`, 7 `tune_hazard`,
9 `train`, 14 `sensitivity`, 18 `explain`, 6 `metrics`, 4 `case_study_2023`) and the
warehouse tables named next to each item. `uv run pytest`: 396 passed, no network, no
`data/` reads. The written summary of everything below is [`docs/model_card.md`](model_card.md).

- [x] **Walk-forward results exist for every year from 2008 to the latest complete year, with
  failure counts and CIs.** `walkforward_scores` holds 2,519,318 rows: 4q for `texas, logit,
  gbdt, hazard` over 17 test years 2008-2024 (426,019 bank-quarters, 2,103 failures) and 8q
  for `logit, gbdt` over 16 test years 2008-2023 (407,621 rows, 3,768 failures); 2024 is the
  last 4q-complete year and 2023 the last 8q-complete year under the 2026-09-25 `as_of_date`.
  `models/walkforward/<Y>/` exists for all 17 years with `pipeline.joblib, features.json,
  config.json, metrics.json, tuning.json, calibration.{joblib,json}` per model (16 `_8q`
  directories each for logit and gbdt). `reports/walkforward.md` lists every year with `n`,
  `failures`, PR-AUC and recall@2% with 200-draw cluster-bootstrap 95% intervals, Brier raw
  and calibrated, and a `low confidence` flag for years under 10 failures (2017, 2020, 2021,
  2024 at 4q; 2020 at 8q; 2021 has no 4q failure and undefined ranking metrics).

- [x] **The best model beats P1's regularised logistic on pooled walk-forward PR-AUC and
  recall@top-2%.** Met on point estimates, not statistically separated. On the same 426,019
  4q rows the hazard pools to PR-AUC 0.3261 [0.293, 0.357] and recall@2% 0.7042 [0.679,
  0.729] against the walk-forward logit's 0.3066 [0.276, 0.335] and 0.6843 [0.657, 0.713]
  (the P1 learner, per-year `C`, on the v2 features). The intervals overlap, so the backtest
  does not distinguish the two; the booster (0.2813 / 0.6434) is behind both, and the Texas
  ratio keeps the best recall@2% (0.7437) with the worst PR-AUC (0.2606). The P1 fixed-split
  figure (0.3867 / 0.7143, test 2010-2013) is a different test period and is not the
  comparison. At 8q the logit is best (0.2835 / 0.5488; gbdt 0.1169 / 0.2710).

- [x] **Median lead time for 2009-2012 failures is reported.** `reports/walkforward.md`,
  "Lead time" (`metrics.lead_time_summary`): of the 440 banks that failed in 2009-2012,
  the hazard flagged 398 (90.5%) in some quarter's top 2% before failure, median lead 5
  quarters, 88.4% at least 2 quarters ahead; logit 395 (89.8%), 5 quarters, 87.7%; gbdt 364
  (82.7%), 4.5 quarters, 81.6%; texas 372 (84.5%), 4 quarters, 82.5%. Over all 545 failures
  with a scored quarter before them the logit flags 89.4%, median 5, 87.5% two or more
  quarters ahead. The spec's target (most failures flagged >= 2 quarters ahead) holds.

- [x] **The 2023 notebook clearly shows how SVB/Signature/First Republic ranked under both
  models and explains why.** `notebooks/03_svb_2023_case_study.ipynb` (7 code cells, 0 error
  outputs, executed in place; `reports/svb_2023_case_study.md`; `runs/case_study_2023/`, four
  records) fits credit-only and rate-aware views of the logit and the booster on the 2022Q4
  cut (622,341 rows, 2,226 failures, reports through 2021Q3) and ranks all ~4,800 banks at
  2022Q3, 2022Q4 and 2023Q1. SVB at 2022Q4: credit-only logit rank 1677 / gbdt 1412,
  rate-aware logit 1854 / gbdt 245 (94.9th percentile). Signature: 113 / 1648 / 627 / 2634.
  First Republic: 1570 / 1167 / 1750 / 1416. The drivers table shows the booster's SVB score
  coming from `adjusted_tier1_leverage` = -0.33 (+2.1 log-odds) and the rate-aware logit
  cancelling it through `uninsured_share` and `macro_fedfunds_change_4q`, which carry safer
  signs over 2001-2021. The notebook also plots unrealised losses and uninsured share of the
  three banks against peer bands 2020-2023 (`reports/figures/svb_unrealized_losses.png`).

- [ ] **Calibrated probabilities: reliability curve within reasonable tolerance on the test
  period; Brier score reported.** Partly met. Brier is reported for every model, year and
  horizon, raw and calibrated (`reports/walkforward.md`, `runs/metrics/`). The isotonic map
  (fitted per year on the last complete label year inside the training window, from an inner
  model's scores) is within tolerance for the booster only: gbdt pooled 4q Brier 0.0043 raw
  and 0.0043 calibrated, top-decile mean calibrated 0.045 against an observed 0.036. For the
  logit and the hazard the map hurts (Brier 0.0041 -> 0.0189 and 0.0041 -> 0.0545; 2009
  logit mean calibrated 0.19 against a failure rate of 0.021) because the inner model's score
  scale differs from the full-window model's in the crisis years. The raw scores of the linear
  models track the observed rate in the lower nine deciles and under-predict the top decile
  by about a factor of two (logit 0.021 against 0.045). Left unticked: the calibrated output
  is not usable as a probability for two of the three learners; the fix (fit the map on a
  slice scored by the model being calibrated) is noted in the model card, section 7.

- [x] **Model card written.** `docs/model_card.md`: intended use and non-use; data sources,
  coverage 2001Q1-2026Q2, units, the FDIC securities and uninsured fields and their FFIEC
  cross-check, macro sources; label and censoring; leakage protections; models and the
  LightGBM backend with its scikit-learn fallback; per-year and pooled walk-forward tables
  with failure counts and CIs at 4q and 8q; calibration; lead time; the 2023 case study;
  sensitivity and false-positive summaries; limitations; ethical considerations; references.

- [x] **Hygiene.** A case-insensitive grep of every commit message for co-author trailers and
  tool or vendor names prints nothing; the same grep over `*.py, *.md, *.toml, *.yaml, *.ipynb`
  (excluding `.venv`, `data`, `PROJECT_SPEC.md`) prints nothing. `data/`, `models/` and `.env`
  are git-ignored; `runs/` (JSON only) is committed.

## Feature list check (spec items 1-15)

| item | status | evidence |
|---|---|---|
| 1 FFIEC RC-B / RC-O ingestion with `IDRSSD <-> CERT` crosswalk | done, as a cross-check | the FDIC `/financials` fields `scaa, scaf, scha, schf, depunins` carry the same values (spec 3.2: prefer the FDIC API); `reports/ffiec_crosscheck.md` matches all 15 SVB/Signature/First Republic 2022Q4 items exactly; `crosswalk_rssd` (27,834 certs); the bulk downloader is optional |
| 2 rate / run-risk features | done | `unrealized_loss_to_tier1, adjusted_tier1_leverage, uninsured_share, securities_to_assets` plus five more in `features_v2` (`docs/FEATURES.md`) |
| 3 trend and persistence features | done | 16 `d1q_`/`d4q_` deltas, `neg_roa_quarters_last_8, consecutive_loss_quarters, noncurrent_rising_quarters_last_4` |
| 4 FRED macro features | done | `macro_state` (5,650 state-availability rows, point-in-time with publication lags); `macro_unemp_rate, macro_unemp_change_4q, macro_hpi_change_4q, macro_t10y3m, macro_fedfunds_change_4q` |
| 5 full feature set with registry | done | 83 registered features (43 P1 + 40 P2), each with group, formula, unit, explanation and monotone sign; `docs/FEATURES.md` generated from the registry |
| 6 LightGBM, hazard, optional Cox | done (Cox dropped) | `models.gbdt.make_gbdt` (LightGBM 4.7, sklearn fallback), `models.hazard`; `lifelines` pins pandas < 3 (`docs/DECISIONS.md`) |
| 7 walk-forward harness, one artefact per test year | done | `bankcanary walkforward --year Y`, `models/walkforward/<Y>/<model>[_8q]/`, 150 run records |
| 8 isotonic calibration and reliability plots | done, with the caveat above | `bankcanary calibrate`, `score_calibrated` filled for 2,519,318 rows (Texas NaN), `reports/figures/reliability_{logit,gbdt,hazard}.png` |
| 9 SHAP per bank-quarter | done | `drivers` table: 4,525,760 rows over 452,576 bank-quarters, ten drivers each, for the 17 walk-forward boosters and the 2024 production model; `reports/shap_summary.md` |
| 10 metrics suite | done | `evaluation.metrics`: PR-AUC, ROC-AUC, recall@k, Brier, lead time, cluster-bootstrap CIs; per-year and pooled tables in `reports/walkforward.md` |
| 11 2023 case study notebook | done | `notebooks/03_svb_2023_case_study.ipynb` (see the criterion above) |
| 12 false-positive analysis | done | `notebooks/04_false_positives.ipynb`, `reports/false_positives.md`: 875 flagged non-failing bank-years 2009-2012, 26.4% failed later, 9.9% acquired within two years, 61.8% still open |
| 13 sensitivity analyses | done | `bankcanary sensitivity`, `reports/sensitivity.md`, `notebooks/05_sensitivity.ipynb`: 4q/8q, censored kept/dropped, lag 45/60/90 days, 14 run records |
| 14 experiment tracking | done (JSON, not MLflow) | `bankcanary.tracking`: `runs/<name>/<run_id>/{config,metrics}.json` + `runs/index.jsonl`, deterministic ids `<name>-<H>q-<config hash>`, read back with `tracking.read_index(settings)`; the `runs list` CLI named in the contract was not built (the index file is greppable) |
| 15 model card | done | `docs/model_card.md` |

Notebooks 02-05 are built by `scripts/make_notebook_0{2,3,4,5}.py` and executed in place
(`jupyter nbconvert --execute`); each runs in under a minute after the tables and `runs/`
exist and none retrains a walk-forward model. Open items for Prototype 3: refit the isotonic
maps on a slice scored by the calibrated model itself; treat post-2021 scores as out of
regime (model card, section 12).
