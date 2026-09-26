# Model card: BankCanary Prototype 2

Version: Prototype 2 ("Depth"), walk-forward models fitted 2026-09-25 and 2026-09-26 on the
FDIC pull of 2026-09-25 (the raw-feature boosters, the 2006-2007 test years, the `logit_v1`
comparison and the trailing calibration maps date from 2026-09-26). Companion documents: [`docs/FEATURES.md`](FEATURES.md) (every feature with its
formula and banking meaning), [`reports/walkforward.md`](../reports/walkforward.md) (the full
backtest tables), [`docs/DECISIONS.md`](DECISIONS.md) (dated design decisions) and the five
notebooks under [`notebooks/`](../notebooks/).

## 1. Intended use and non-use

BankCanary estimates, for every FDIC-insured bank and every quarterly report, the probability
that the bank fails (is closed by its chartering authority and placed in FDIC receivership)
within the next four or eight quarters, using only information that was public on the
prediction date. It exists to teach and to demonstrate: how a leakage-safe early-warning
backtest is built, what the classic CAMELS ratios can and cannot see, and why a model trained
on the 2008 crisis did not see the 2023 failures coming.

Intended uses: study of bank-failure prediction methodology; reproducible research; the
public dashboard of Prototype 3, which presents these scores with the caveats below; a
watch list that says which balance sheets look like the ones that failed before.

Not intended for: deposit or investment decisions, lending or counterparty decisions,
regulatory or supervisory use, credit ratings, any statement about an individual bank's
solvency. A high score means "this balance sheet resembles past failures", nothing more; a
low score is not a clean bill of health (the 2023 section shows a bank at the 65th percentile
failing four months later). The model has never seen supervisory information, examination
findings, management quality or deposit behaviour inside a quarter.

## 2. Data

Sources (all public):

- FDIC BankFind Suite API (`api.fdic.gov/banks`): `/financials` (quarterly Call Report
  summaries, 96 fields listed in `config/fields.yaml`), `/failures` (the failed-bank list),
  `/institutions` (charter facts, `fed_rssd`) and `/history` (mergers, closings and other
  structure events). Raw JSON responses are cached under `data/raw/fdic/` and rebuilt into
  Parquet and DuckDB tables byte-identically.
- FRED (`fred.stlouisfed.org`): state unemployment rates (`<ST>UR`), state house price
  indexes (`<ST>STHPI`), `FEDFUNDS`, `T10Y3M` and `DGS10`, pulled through the keyless CSV
  endpoint and evaluated point-in-time at every panel `avail_date` with publication lags of
  45 days (state unemployment), 75 days (state HPI) and one day (national rates).
- FFIEC Central Data Repository: used only as a cross-check. The FDIC `/financials` endpoint
  already carries the securities and uninsured-deposit fields the spec expected to need from
  raw Call Report schedules: `scaa`/`scaf` (available-for-sale securities at amortised cost
  and fair value), `scha`/`schf` (held-to-maturity, same pair) and `depunins` (estimated
  uninsured deposits, populated for about 98.5 percent of banks). `reports/ffiec_crosscheck.md`
  matches all fifteen SVB, Signature and First Republic values for 2022Q4 to the FFIEC MDRM
  items (`RCFD/RCON1754, 1771, 1772, 1773, RCON5597`) exactly, and SVB's row to its 2022
  Form 10-K. `crosswalk_rssd` (27,834 certs) maps `cert` to `fed_rssd` for anyone who does
  pull the bulk files.

Coverage: report dates 2001Q1 through 2026Q2 (`panel`: 710,691 bank-quarters, 11,243 banks).
Dollar amounts are thousands of dollars, the FDIC unit; regulatory ratios are as published
(percent); engineered ratios are plain fractions unless `docs/FEATURES.md` says otherwise.
Income items are year-to-date in the source and are de-accumulated to quarterly values before
any ratio is formed. `features_v2` holds 83 features: the 43 Prototype 1 CAMELS ratios plus 40
Prototype 2 additions (six interest-rate sensitivity features, three deposit-run features,
sixteen one- and four-quarter trends, three persistence counters, FDIC-region and
community-bank indicators, and five macro columns). Feature choice was reviewed in
`reports/features_v2_summary.md`; the registry records the expected monotone direction of
every feature.

The `failures` pull has `fetched_at = 2026-09-25`; that date is the `as_of_date` for label
completeness, so later refreshes do not change any historical label.

## 3. Label definition and censoring

A bank-quarter with report date `repdte` becomes available on `avail_date = repdte + 60 days`
(the availability lag; 45 and 90 days are tested in the sensitivity section). Its outcome
window is `(avail_date, avail_date + 3H months]` for horizon H in {4, 8} quarters (the hazard
model also uses H = 1). `y_Hq = 1` when the bank's `fail_date` (from `/failures`,
`restype = FAILURE`, earliest failure on or after the bank's establishment date) falls inside
the window. Assistance transactions are not failures (`assisted` is kept as a flag).

Censoring follows spec section 5: a bank that leaves by merger, voluntary closing or another
non-failure exit inside the window keeps `y = 0` and is marked `censored_in_window`; it is
kept in training and evaluation (the "kept" cell of the sensitivity analysis) because a
supervisor scoring that quarter did not know the bank would be sold. A window that ends after
the `as_of_date` of the failures pull is `label_complete = False` and is excluded from every
metric. Reports filed after a bank's failure (`dropped_failed_before_avail`) stay in the
table, flagged, and are excluded downstream. Failures of two banks of one holding company on
the same day are one event for the per-event metrics in `reports/p1_baselines.md`.

## 4. Leakage protections

- Availability lag: nothing is scored before its `avail_date`, and macro series are joined at
  the value published by that date, never the value for the calendar quarter.
- Outcome-window rule (spec 6.2): a training row for a prediction date `d` must have an
  outcome window that closed before `d`. `bankcanary.splits.time_split.training_mask` is the
  only implementation and every fit (fixed split, walk-forward, hazard, calibration inner
  models, case study, sensitivity) selects rows through it and then runs `assert_no_leakage`.
  For the 2009 walk-forward model this means training reports end at 2007Q4; for the 2022Q4
  case-study cut they end at 2021Q3.
- Inner-validation tuning (spec 6.7): every hyper-parameter (logit and hazard `C`, the
  booster grid) is chosen on a validation slice carved from the last eight report quarters of
  the training window itself, widened backwards until it holds five failures, with the inner
  model trained on windows that closed before the slice. Test years never inform a choice;
  the per-year choice is recorded in `models/walkforward/<Y>/<model>/tuning.json`.
- Fold-fitted preprocessing: the linear models' winsorisation limits (0.5th and 99.5th
  percentiles), imputation medians and scaler statistics are fitted inside the sklearn pipeline
  on the training rows of each fit, so test-year distributions never touch them; the boosters
  have no preprocessing step to leak through.
- Identifiers (`cert`, names, dates) never enter a model; the feature list is read from the
  registry and stored beside each pipeline (`features.json`).
- Tests (`tests/test_labels*.py`, `tests/test_time_split*.py`,
  `tests/test_walkforward*.py`) pin the window arithmetic, the rule 6.2 mask and the
  no-leakage assertion on synthetic frames; they never read `data/`.

## 5. Models

The linear models are sklearn pipelines: `Winsorizer(0.005, 0.995)`, then
`SimpleImputer(median, add_indicator)` and `StandardScaler`. The boosters take the **raw**
registry features (the pipeline is the bare `("model", estimator)` step: no winsoriser, no
imputer, no scaler). Trees split on rank order and a leaf's value is bounded by
`min_samples_leaf` rows, so an outlier cannot distort them the way it distorts a linear fit,
while the 0.5 percent clip erased the tail that carries the interest-rate signal (Silicon
Valley Bank at 2022-12-31: `unrealized_loss_to_tier1` -1.04 was shown to the trees as -0.19,
`adjusted_tier1_leverage` -0.33 as 3.94). The published SHAP `feature_value` is therefore
the bank's own ratio.

| name | learner | inputs | notes |
|---|---|---|---|
| `texas` | ranking by the Texas ratio (non-performing assets over tangible equity plus reserves) | one ratio | no fit; the classic single-number warning signal and the baseline everything must beat |
| `logit` | L2 logistic regression, no class weighting, `C` re-selected per test year on the inner slice | 83 `features_v2` columns | the Prototype 1 learner (Cole and White 2012 style) on the richer feature set |
| `logit_v1` | the same learner and tuning recipe | the 43 `features_v1` columns | the Prototype 1 model walked forward on its own features, so that the spec's "beats P1" criterion compares like with like (section 6); run records carry `features_version = "v1"` |
| `gbdt` | gradient-boosted trees, `make_gbdt(backend, monotone)` | 83 features | backend `lightgbm` 4.7 (`settings.models.gbdt.backend`); `HistGradientBoostingClassifier` is the fallback when LightGBM cannot load its OpenMP runtime, same histogram algorithm, native NaN, monotone constraints |
| `gbdt_mono` | the same booster under the registry's monotone signs (+1 risk-increasing, -1 risk-decreasing, 0 free) | 83 features | walked forward beside `gbdt` at 4q with its own per-year tuning (section 6); **the production model** since Decision Point 2 (2026-09-26): the constrained booster pools better (PR-AUC 0.3022 against 0.2160, recall@2% 0.6992 against 0.5728 over 2006-2024, both intervals disjoint) and, on the raw features, the inner-validation slice prefers it too (0.214 against 0.158; with winsorised inputs the slice had preferred the unconstrained one); `settings.models.gbdt.monotone = true`, the SHAP `drivers` table holds `gbdt_mono` rows only, and `models/production/gbdt_mono/` carries the 2024 walk-forward fit |
| `hazard` | discrete-time hazard (Shumway 2001): logistic regression on the one-quarter event `y_1q`, converted to H quarters by `1 - (1 - h)^H` | 83 features | assumes covariates persist over the horizon (a documented approximation); trained at its own 1q mask inside each walk-forward year |

**Production model.** `models/production/` holds the latest walk-forward fits, copied by
`scripts/promote_production_models.py` with a `model_version.json` each: `gbdt_mono` (test year
2024, trained on reports 2001Q1-2022Q4, version `gbdt_mono-2022-12-31-9e7bab6`) scores every
label-incomplete quarter and supplies the published drivers, and `hazard` (trained through
2023Q4, `hazard-2023-12-31-0cc72ac`) is the secondary score. A model version is
`<model>-<train_end_repdte>-<first 7 hex of the model hash>`, where the hash is the sha1 of a
deterministic serialisation of the fitted estimator (`walkforward.model_fingerprint`: the
LightGBM booster's model string, the coefficient and intercept bytes of a logistic model);
it is stamped into every walk-forward `config.json` and run record, changes exactly when a
refit changes a leaf or a coefficient, and owes nothing to git history (the earlier git-sha
suffix had kept one version across a refit). Each production directory also holds
`calibration.joblib`, the production map of section 7 fitted on the 2023-2024 walk-forward
scores, which the promotion keeps in place of the promoted year's own map. The 2024 `gbdt_mono` fit is among the
weakest of the series on its own test year (PR-AUC 0.11 and ROC-AUC 0.73 on 9 failures, a
low-confidence year); its mean |SHAP| leans on `total_rbc_ratio` (0.50), `macro_dgs10` (0.46)
and `macro_unemp_change_4q` (0.35), the capital and macro-cycle drivers the earlier years also
lean on, with the Texas ratio further down; the published probability is the calibrated 12-month one and the low-confidence flag
travels with it.

Booster parameters at the fixed split (`settings.models.gbdt.params`): `learning_rate = 0.03`,
`num_leaves = 31`, `min_samples_leaf = 50`, `n_estimators = 200`; the walk-forward fits re-select learning
rate, leaves and leaf size per year from a 2x2x2 grid. The linear models are unweighted
because class balancing let the collinear capital ratios overfit in Prototype 1 (PR-AUC 0.20
against 0.39). The optional Cox model was dropped: `lifelines` pins `pandas < 3`.

## 6. Walk-forward metrics

Protocol (spec 8.2): one model per test year Y from 2006 (`walkforward.FIRST_TEST_YEAR`) to
the latest label-complete year (2024 at 4q; 2023 at 8q, where the series still starts in
2008). The two earliest years are thin but usable: the 2006 models tune on eight validation
quarters (2003-2004, 17 failures) and the 2007 models on twelve (2003-2005, 18 failures)
against 34 inner-training failures (13 at the hazard's one-quarter label), the 2006 test
year holds 8 failures and 2007 holds 80, and the weakest test ranking is the 2006 Texas
ratio (ROC-AUC 0.643). Training rows are every earlier bank-quarter whose outcome window
closed before the first prediction date of Y (the 03-31 report plus 60 days); test rows are
the label-complete reports dated in Y; hyper-parameters are re-selected inside each year's
training window. Pooled rows rank every test year as one list. Confidence intervals are
percentile intervals from 200 cluster-bootstrap draws that resample banks (certs) with
replacement so that a bank's consecutive quarters move together; years with fewer than ten
failures are flagged low confidence and a year without failures has undefined ranking
metrics. Scores for every bank-quarter, model and year live in `walkforward_scores`
(4,199,937 rows: `texas, logit, logit_v1, gbdt, gbdt_mono, hazard` at 4q and `logit, gbdt,
hazard` at 8q); the per-year artefacts in `models/walkforward/<Y>/<model>[_8q]/`.

### Pooled, 4-quarter horizon (test years 2006-2024, 496,179 bank-quarters, 2,191 failures)

| model | PR-AUC | 95% CI | recall @ top 2% | 95% CI | ROC-AUC | Brier raw | Brier calibrated |
|---|---|---|---|---|---|---|---|
| hazard | 0.3194 | [0.286, 0.352] | 0.7033 | [0.674, 0.733] | 0.9568 | 0.0037 | 0.0038 |
| gbdt_mono | 0.3022 | [0.275, 0.335] | 0.6992 | [0.671, 0.724] | 0.9057 | 0.0041 | 0.0044 |
| logit | 0.2953 | [0.264, 0.330] | 0.6787 | [0.648, 0.710] | 0.9527 | 0.0037 | 0.0050 |
| logit_v1 | 0.2801 | [0.247, 0.315] | 0.7047 | [0.671, 0.736] | 0.9475 | 0.0037 | 0.0047 |
| texas | 0.2534 | [0.219, 0.297] | 0.7485 | [0.719, 0.776] | 0.9570 | 0.1509 | n/a |
| gbdt | 0.2160 | [0.191, 0.240] | 0.5728 | [0.545, 0.600] | 0.7786 | 0.0042 | 0.0049 |

Restricted to 2008-2024 (426,019 bank-quarters, 2,103 failures) the same models pool to
hazard 0.3268 / 0.7038, gbdt_mono 0.3131 / 0.7066, logit 0.3056 / 0.6833, logit_v1 0.2901 /
0.6933, gbdt 0.2248 / 0.5787 and texas 0.2606 / 0.7437 (PR-AUC / recall@2%).

### Pooled, 8-quarter horizon (test years 2008-2023, 407,621 bank-quarters, 3,768 failures)

| model | PR-AUC | 95% CI | recall @ top 2% | 95% CI | ROC-AUC | Brier raw | Brier calibrated |
|---|---|---|---|---|---|---|---|
| hazard | 0.4113 | [0.382, 0.442] | 0.6598 | [0.634, 0.688] | 0.9416 | 0.0070 | 0.0082 |
| logit | 0.2566 | [0.225, 0.288] | 0.5207 | [0.493, 0.550] | 0.7447 | 0.0079 | 0.0080 |
| gbdt | 0.1059 | [0.086, 0.127] | 0.2585 | [0.232, 0.292] | 0.4560 | 0.0087 | 0.0312 |

Reading: the hazard model pools best at both horizons; at 4q the hazard-logit gap (0.024
PR-AUC) is inside both intervals, so the backtest does not separate them, and every learner
beats the Texas ratio on PR-AUC while the Texas ratio keeps the highest recall in the top 2
percent. The unconstrained booster is the weakest learner on pooled PR-AUC because its early
years are starved of failures (the 2008 model trains on 37 positives) and its raw scale
drifts between years, which the pooled ranking punishes; year by year it is competitive from
2010 on. The monotone booster does not share that weakness: the registry signs hold its raw
scale together across years (pooled ROC-AUC 0.91 against 0.78) and it pools second on
PR-AUC among the learners, with both intervals disjoint from the
unconstrained booster's, which is the walk-forward evidence for Decision Point 2 (section
5). At 8q the unconstrained booster collapses (recall 0.26; its
2008 fit scores every test row identically, ROC-AUC 0.50) while the logit degrades
gracefully and the hazard, whose one-quarter event is converted with
`1 - (1 - h)^8`, is clearly best (PR-AUC 0.41, the interval disjoint from the logit's).

**The Prototype 1 comparison.** The spec asks whether the best model beats P1's regularised
logistic on pooled walk-forward PR-AUC and recall@2%. The fair comparison is `logit_v1`,
the P1 learner walked forward on its 43 `features_v1` columns with its own `C` re-selected
per year: PR-AUC 0.2801 [0.247, 0.315] and recall@2% 0.7047 [0.671, 0.736] over 2006-2024.
Every P2 probability model except the unconstrained booster sits above it on PR-AUC, but
no interval is disjoint from its, and on recall@2% it is level with the hazard and above
the v2 logit, so the criterion is met on PR-AUC only. An earlier stand-in that borrowed
each year's `C` from the v2 logit's selection had pooled to 0.2194 / 0.5625 on 2008-2024
and overstated the gap. The P1 fixed-split logit (PR-AUC 0.3867, recall@2% 0.7143, test
2010-2013) is a different test period and is not comparable with any pooled row.

### Per year, 4-quarter horizon: PR-AUC [95% CI] and recall @ top 2%

| year | n | failures | hazard PR-AUC [95% CI] | gbdt_mono PR-AUC [95% CI] | logit PR-AUC [95% CI] | logit_v1 PR-AUC [95% CI] | gbdt PR-AUC [95% CI] | texas PR-AUC [95% CI] | hazard recall@2% | gbdt_mono recall@2% | logit recall@2% | logit_v1 recall@2% | gbdt recall@2% | texas recall@2% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2006 (low confidence) | 35407 | 8 | 0.0384 [0.001, 0.192] | 0.0303 [0.000, 0.125] | 0.0250 [0.000, 0.139] | 0.0393 [0.000, 0.194] | 0.0248 [0.000, 0.167] | 0.0206 [0.001, 0.144] | 0.3750 | 0.7500 | 0.7500 | 0.3750 | 0.3750 | 0.5000 |
| 2007 | 34753 | 80 | 0.1624 [0.082, 0.281] | 0.0580 [0.029, 0.127] | 0.0796 [0.031, 0.186] | 0.0735 [0.020, 0.174] | 0.0723 [0.027, 0.153] | 0.1336 [0.067, 0.261] | 0.5875 | 0.4125 | 0.4375 | 0.3500 | 0.3750 | 0.6250 |
| 2008 | 33972 | 429 | 0.3221 [0.267, 0.385] | 0.2387 [0.191, 0.297] | 0.3025 [0.248, 0.361] | 0.2600 [0.210, 0.306] | 0.1186 [0.092, 0.157] | 0.3506 [0.288, 0.427] | 0.5198 | 0.4522 | 0.4918 | 0.4639 | 0.2984 | 0.5408 |
| 2009 | 32811 | 679 | 0.4559 [0.413, 0.502] | 0.3197 [0.279, 0.367] | 0.4172 [0.376, 0.464] | 0.4176 [0.373, 0.472] | 0.3474 [0.306, 0.400] | 0.5714 [0.516, 0.626] | 0.4433 | 0.3741 | 0.4242 | 0.4242 | 0.3903 | 0.5567 |
| 2010 | 31431 | 408 | 0.4897 [0.429, 0.554] | 0.4386 [0.372, 0.515] | 0.4590 [0.394, 0.529] | 0.4252 [0.360, 0.492] | 0.2792 [0.227, 0.350] | 0.4840 [0.417, 0.566] | 0.6471 | 0.6446 | 0.6176 | 0.5907 | 0.5245 | 0.6471 |
| 2011 | 30165 | 231 | 0.5506 [0.464, 0.636] | 0.4663 [0.376, 0.538] | 0.5106 [0.430, 0.589] | 0.4230 [0.350, 0.511] | 0.5078 [0.412, 0.602] | 0.4171 [0.329, 0.530] | 0.8701 | 0.7879 | 0.8225 | 0.7359 | 0.8485 | 0.7749 |
| 2012 | 29130 | 121 | 0.5342 [0.421, 0.669] | 0.4402 [0.320, 0.575] | 0.4854 [0.386, 0.608] | 0.4357 [0.308, 0.577] | 0.4404 [0.322, 0.575] | 0.3574 [0.246, 0.527] | 0.9669 | 0.9339 | 0.9421 | 0.9339 | 0.9669 | 0.9091 |
| 2013 | 27970 | 73 | 0.4190 [0.254, 0.610] | 0.3897 [0.230, 0.563] | 0.3684 [0.204, 0.542] | 0.2546 [0.141, 0.386] | 0.4344 [0.271, 0.584] | 0.1659 [0.081, 0.309] | 0.8767 | 0.8630 | 0.8493 | 0.8767 | 0.8904 | 0.7123 |
| 2014 | 26792 | 37 | 0.4237 [0.260, 0.681] | 0.4628 [0.265, 0.667] | 0.4364 [0.264, 0.678] | 0.2828 [0.169, 0.473] | 0.5425 [0.370, 0.723] | 0.1225 [0.044, 0.283] | 0.9730 | 0.9730 | 0.9730 | 0.9730 | 0.9730 | 0.8649 |
| 2015 | 25517 | 24 | 0.3305 [0.135, 0.631] | 0.3638 [0.131, 0.648] | 0.2671 [0.134, 0.524] | 0.2288 [0.103, 0.563] | 0.3295 [0.145, 0.636] | 0.1152 [0.039, 0.252] | 1.0000 | 1.0000 | 1.0000 | 0.9583 | 1.0000 | 0.9167 |
| 2016 | 24351 | 28 | 0.4899 [0.275, 0.786] | 0.4956 [0.275, 0.768] | 0.5337 [0.281, 0.800] | 0.4104 [0.190, 0.741] | 0.4447 [0.255, 0.702] | 0.1038 [0.044, 0.234] | 0.8571 | 0.9286 | 0.9643 | 0.9643 | 0.9286 | 0.8214 |
| 2017 (low confidence) | 23321 | 5 | 0.0446 [0.000, 0.389] | 0.0918 [0.000, 0.617] | 0.0454 [0.000, 0.450] | 0.0782 [0.000, 0.532] | 0.0318 [0.000, 0.243] | 0.0112 [0.000, 0.067] | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| 2018 | 22301 | 10 | 0.0750 [0.010, 0.264] | 0.1084 [0.017, 0.347] | 0.0961 [0.014, 0.331] | 0.0619 [0.007, 0.258] | 0.0954 [0.015, 0.297] | 0.0219 [0.003, 0.078] | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 |
| 2019 | 21362 | 18 | 0.4849 [0.153, 0.859] | 0.5118 [0.216, 0.864] | 0.5079 [0.176, 0.862] | 0.4465 [0.164, 0.836] | 0.5101 [0.220, 0.883] | 0.3994 [0.072, 0.747] | 0.9444 | 0.9444 | 0.9444 | 0.9444 | 0.9444 | 0.9444 |
| 2020 (low confidence) | 20475 | 4 | 0.8269 [0.567, 1.000] | 0.4985 [0.135, 1.000] | 0.8750 [0.667, 1.000] | 0.9500 [0.833, 1.000] | 0.0006 [0.000, 0.003] | 0.6506 [0.098, 1.000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 2021 (low confidence) | 19942 | 0 | n/a n/a | n/a n/a | n/a n/a | n/a n/a | n/a n/a | n/a n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2022 | 19285 | 17 | 0.0048 [0.000, 0.032] | 0.0308 [0.001, 0.350] | 0.0219 [0.001, 0.210] | 0.0222 [0.001, 0.153] | 0.0126 [0.001, 0.136] | 0.0035 [0.000, 0.027] | 0.2353 | 0.1176 | 0.2353 | 0.3529 | 0.1765 | 0.1176 |
| 2023 | 18796 | 10 | 0.0116 [0.000, 0.044] | 0.0254 [0.001, 0.135] | 0.0417 [0.000, 0.278] | 0.0332 [0.000, 0.270] | 0.0390 [0.000, 0.226] | 0.0068 [0.001, 0.035] | 0.5000 | 0.7000 | 0.5000 | 0.5000 | 0.6000 | 0.2000 |
| 2024 (low confidence) | 18398 | 9 | 0.0283 [0.000, 0.336] | 0.1120 [0.000, 0.630] | 0.1117 [0.000, 0.630] | 0.1117 [0.000, 0.630] | 0.1134 [0.001, 0.630] | 0.0031 [0.000, 0.038] | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 |
| pooled | 496179 | 2191 | 0.3194 [0.286, 0.352] | 0.3022 [0.275, 0.335] | 0.2953 [0.264, 0.330] | 0.2801 [0.247, 0.315] | 0.2160 [0.191, 0.240] | 0.2534 [0.219, 0.297] | 0.7033 | 0.6992 | 0.6787 | 0.7047 | 0.5728 | 0.7485 |

### Per year, 8-quarter horizon: PR-AUC [95% CI] and recall @ top 2%

| year | n | failures | hazard PR-AUC [95% CI] | logit PR-AUC [95% CI] | gbdt PR-AUC [95% CI] | hazard recall@2% | logit recall@2% | gbdt recall@2% |
|---|---|---|---|---|---|---|---|---|
| 2008 | 33972 | 1108 | 0.3727 [0.329, 0.417] | 0.0719 [0.057, 0.093] | 0.0326 [0.030, 0.036] | 0.2987 | 0.1011 | 0.0027 |
| 2009 | 32811 | 1087 | 0.5104 [0.469, 0.553] | 0.4321 [0.383, 0.483] | 0.2389 [0.196, 0.284] | 0.3615 | 0.3183 | 0.2282 |
| 2010 | 31431 | 639 | 0.5176 [0.463, 0.581] | 0.4486 [0.394, 0.517] | 0.3703 [0.309, 0.430] | 0.5258 | 0.4883 | 0.3646 |
| 2011 | 30165 | 352 | 0.5488 [0.476, 0.621] | 0.4904 [0.408, 0.554] | 0.4486 [0.371, 0.515] | 0.7415 | 0.6648 | 0.7017 |
| 2012 | 29130 | 194 | 0.4875 [0.393, 0.617] | 0.4235 [0.341, 0.550] | 0.4136 [0.318, 0.530] | 0.7990 | 0.7577 | 0.8144 |
| 2013 | 27970 | 110 | 0.4280 [0.286, 0.579] | 0.3344 [0.207, 0.479] | 0.3048 [0.186, 0.458] | 0.8909 | 0.7909 | 0.7818 |
| 2014 | 26792 | 61 | 0.3788 [0.239, 0.577] | 0.2985 [0.178, 0.476] | 0.3521 [0.198, 0.512] | 0.9836 | 0.9508 | 0.9344 |
| 2015 | 25517 | 52 | 0.3337 [0.163, 0.594] | 0.2936 [0.137, 0.556] | 0.3157 [0.168, 0.602] | 0.8846 | 0.8269 | 0.8269 |
| 2016 | 24351 | 33 | 0.4165 [0.202, 0.712] | 0.3297 [0.161, 0.665] | 0.2936 [0.148, 0.661] | 0.7273 | 0.7273 | 0.7879 |
| 2017 | 23321 | 15 | 0.0254 [0.003, 0.136] | 0.0398 [0.003, 0.160] | 0.0305 [0.006, 0.129] | 0.4000 | 0.3333 | 0.5333 |
| 2018 | 22301 | 28 | 0.2010 [0.048, 0.458] | 0.2388 [0.059, 0.501] | 0.2315 [0.055, 0.486] | 0.6786 | 0.7500 | 0.7143 |
| 2019 | 21362 | 22 | 0.5557 [0.225, 0.898] | 0.5270 [0.203, 0.902] | 0.6082 [0.244, 0.888] | 0.9545 | 0.9545 | 0.9545 |
| 2020 (low confidence) | 20475 | 4 | 0.8750 [0.643, 1.000] | 0.8929 [0.683, 1.000] | 0.6917 [0.267, 1.000] | 1.0000 | 1.0000 | 1.0000 |
| 2021 | 19942 | 17 | 0.0007 [0.000, 0.002] | 0.0016 [0.000, 0.004] | 0.0016 [0.000, 0.004] | 0.0000 | 0.0000 | 0.0000 |
| 2022 | 19285 | 27 | 0.0056 [0.001, 0.028] | 0.0173 [0.001, 0.142] | 0.0028 [0.001, 0.009] | 0.1852 | 0.1481 | 0.0370 |
| 2023 | 18796 | 19 | 0.0065 [0.001, 0.031] | 0.0132 [0.001, 0.101] | 0.0124 [0.001, 0.052] | 0.2105 | 0.2105 | 0.3158 |
| pooled | 407621 | 3768 | 0.4113 [0.382, 0.442] | 0.2566 [0.225, 0.288] | 0.1059 [0.086, 0.127] | 0.6598 | 0.5207 | 0.2585 |

## 7. Calibration

Per walk-forward year a **binned isotonic** map is fitted on out-of-sample scores and applied
to the full-window model's test scores (`calibration.binned_isotonic`, `min_bin` 50). The
calibration rows are the same model's walk-forward scores from the two most recent earlier
test years whose every row's outcome window closed before Y's first prediction date (4q: Y-2
and Y-3; 8q: Y-3 and Y-4; `calibration.trailing_years`, re-checked row by row with
`assert_no_leakage`), so the map is learned out of sample and on the score scale it is
applied to. Those rows are sorted by score and cut into contiguous bins of at least 50 rows
(a short last bin merges into its predecessor), and `IsotonicRegression` is fitted on bin mean
score against bin failure rate weighted by bin size, so every step of the map, the top one
included, is a rate observed on at least 50 banks: a plain isotonic fit had let three failed
banks map to 1.0 and 170 banks of 2009Q2 to exactly 0.5. Where fewer than two trailing years
exist (2006-2008 at 4q, 2008-2011 at 8q) the map falls back to the last complete label year
inside the training window, scored by the year's own model for the logits (`full`) or by an
inner model trained on the earlier years for the boosters (`inner`), through the same binned
fit and recorded as `fallback = true`. No test outcome shapes a map. The map is stored as
`models/walkforward/<Y>/<model>[_8q]/calibration.joblib` with its recipe in
`calibration.json` (`slice_scorer`, `effective_scorer`, `fallback`, `calibration_years`,
`n_rows`, `n_bins`) and its output as `walkforward_scores.score_calibrated` (filled for every
probability model; the Texas ratio is a ranking). The **production map**
(`models/production/<model>/calibration.joblib`, `calibration.production_calibrator`) is the
same recipe on the two most recent test years whose rows are all label-complete, 2023 and
2024 (37,194 rows, 19 failures, 743 bins); its top steps are 0.0319 for `gbdt_mono` and
0.0155 for `hazard`, so no published probability exceeds about 3 percent and the largest
calibrated value anywhere in the backtest is 0.5968 (`gbdt_mono`, 2016Q2-Q3). Pooled 4q
results over 2006-2024 against a failure rate of 0.44 percent:

| model | mean raw score | mean calibrated | Brier raw | Brier calibrated | years where calibration worsens Brier | top decile: mean calibrated / observed |
|---|---|---|---|---|---|---|
| hazard | 0.0018 | 0.0062 | 0.0037 | 0.0038 | 8 of 19 | 0.054 / 0.039 |
| gbdt_mono | 0.0038 | 0.0091 | 0.0041 | 0.0044 | 6 of 19 | 0.067 / 0.037 |
| logit_v1 | 0.0028 | 0.0088 | 0.0037 | 0.0047 | 12 of 19 | 0.080 / 0.040 |
| gbdt | 0.0022 | 0.0144 | 0.0042 | 0.0049 | 8 of 19 | 0.052 / 0.030 |
| logit | 0.0021 | 0.0103 | 0.0037 | 0.0050 | 7 of 19 | 0.093 / 0.040 |

Reading: the raw scores of every learner already sit near the observed rate below the top
decile and under-predict the top decile by about a factor of two; the maps close most of that
gap for the hazard and the monotone booster (top decile 0.054 and 0.067 against observed
0.039 and 0.037) and the pooled Brier score moves by a few ten-thousandths either way. The
years that hurt are the first recovery years: a map learned on 2007-2008 or 2008-2009 scores
carries the crisis failure rate into 2010-2012 (2010 mean calibrated: logit 0.094, gbdt_mono
0.095, hazard 0.053 against a rate of 0.013; Brier 0.0287 / 0.0209 / 0.0124 against raw
0.0091 / 0.0110 / 0.0091), which is where the logits lose their pooled advantage, while 2009
improves for four of the five models (hazard 0.0189 raw to 0.0162, gbdt_mono 0.0206 to
0.0185). After 2016 no trailing pair holds more than 61 failures (2023: four) and the differences are in
the fifth decimal. At 8q the linear maps are close to neutral (logit 0.0079 raw against
0.0080 calibrated; hazard 0.0070 against 0.0082, the converted eight-quarter hazard
over-predicting 2010-2011) but the booster map is worse than raw (0.0087 against 0.0312):
the trailing recipe assumes a model's raw scale is stable from one year to the next, and
the 8q `gbdt` models break it (the 2008-2009 models emit almost nothing above 1e-4 while
the 2011 model's 99th percentile is 0.60), so the 2012 map learned on 2008-2009 scores sends
ordinary 2012 scores to its top bin (mean calibrated 0.427 against a rate of 0.0067) and the
2011 `inner` fallback does the same (0.350 against 0.0117). The 8q booster should not be
read as calibrated until its maps get a scale-aware recipe; the earlier one-year in-window
recipe and its numbers are kept in `docs/DECISIONS.md`. Reliability curves:
`reports/figures/reliability_{logit,logit_v1,gbdt,gbdt_mono,hazard}.png`.

## 8. Lead time

For each failed bank, the number of calendar quarters between the first report at which it
entered the top 2 percent of that quarter's walk-forward ranking and its failure date; banks
never flagged are in every denominator. The 2009-2012 cohort (440 failed banks) is the
headline because 2008 failures can only be flagged inside 2008 and post-2024 failures are
scored only through the 2024 model.

| model | failed banks | flagged before failure | median lead (quarters) | flagged >= 2 quarters ahead |
|---|---|---|---|---|
| hazard | 440 | 403 (91.6%) | 6 | 89.8% |
| logit | 440 | 398 (90.5%) | 5 | 88.4% |
| gbdt_mono | 440 | 387 (88.0%) | 5 | 86.4% |
| logit_v1 | 440 | 387 (88.0%) | 5 | 86.4% |
| gbdt | 440 | 382 (86.8%) | 5 | 85.5% |
| texas | 440 | 372 (84.5%) | 5 | 82.5% |

Over every failure with a scored quarter before it (552 banks, the 2006 and 2007 test years
included) the hazard flags 91.5 percent, median lead 6 quarters, 89.7 percent at least two
quarters ahead (logit 90.4 / 6 / 88.4). The spec's target (most failures flagged at least
two quarters ahead) is met for the crisis cohort by every model; the 2006-2007 scores
lengthen the 2009 cohort's lead times because a 2008 flag no longer has to be the first.
Figures: `reports/figures/lead_time_<model>.png`.

## 9. The 2023 case study

Question: would a model trained on the 2008 crisis have flagged Silicon Valley Bank,
Signature Bank and First Republic Bank? Four models are fitted on one rule-6.2 cut (every
bank-quarter whose 4q window closed before the 2022Q4 prediction date: reports through
2021Q3, 622,341 rows, 2,226 failures) on two feature views, `credit_only` (the 43 Prototype 1
ratios) and `rate_aware` (all 83 features), and score the 2022Q3, 2022Q4 and 2023Q1 reports
of every bank (about 4,800 per quarter). Rank 1 is the riskiest bank of the quarter.

| bank, 2022Q4 report | credit-only logit | credit-only gbdt_mono | rate-aware logit | rate-aware gbdt_mono |
|---|---|---|---|---|
| Silicon Valley Bank | 1677 (65th pct) | 2089 (56th) | 1854 (61st) | 2711 (43rd) |
| Signature Bank | **113 (97.6th)** | 1008 (79th) | 627 (86.9th) | 301 (93.7th) |
| First Republic Bank | 1570 (67th) | 978 (79.5th) | 1750 (63rd) | 1813 (62nd) |

Winsorised-era ranks, for the record: `gbdt_mono` credit-only 2149 / 1279 / 1207 and rate-aware
2368 / 564 / 901 (SVB / Signature / First Republic).

The credit-only view saw nothing at SVB (Texas ratio 0.009). The rate-aware production booster
now sees the bank's own values, `adjusted_tier1_leverage` = -0.33 (Tier 1 capital net of
unrealised securities losses was negative; 116 training bank-quarters had that property and 55
failed) and `unrealized_loss_to_tier1` = -1.04, where the winsorised fit had been shown 3.94 and
-0.19. The leverage is SVB's largest driver at +0.47 log-odds and `uninsured_share` adds +0.16,
but the rest of the balance sheet (total risk-based capital 16 percent, securities at 56 percent
of assets, a Texas ratio near zero, a tiny large-time-deposit share, eight quarters without a
loss) pulls -1.06 the other way and the benign macro cycle another -0.27, so SVB sits at rank
2,711 of 4,773, the 43rd percentile, below its credit-only rank. What the monotone booster does
see about SVB is exactly the mechanism that failed it, and it puts that mechanism at the top of
the explanation; what it does not see is any precedent, because in 2001-2021 negative adjusted
leverage never coincided with a spotless credit book and pristine regulatory capital, and a
monotone fit spreads the effect of one ratio over its whole range rather than carving the
116-row pocket into a leaf. The unconstrained, winsorised booster of an earlier version had done
the latter (a single +2.1 contribution, rank 245, 94.9th percentile); that fit no longer exists,
the backtest rejected its configuration, and the honest reading is that the raw inputs move the
right driver to the top without changing the verdict. The rate-aware logit does not help because
over 2001-2021 `uninsured_share` and `macro_fedfunds_change_4q` enter with *safer* signs (failed
banks averaged 13 percent uninsured deposits against 20 percent for survivors, and SVB's
fed-funds change of 4.49 points lies outside the training range, maximum 2.02), so the linear
model cancels the loss terms. The rate-aware booster does more for the other two: Signature
reaches the 94th percentile (rank 301, on its C&I share and the four-quarter change in unrealised
losses) and First Republic the 62nd, then the 81st on its 2023Q1 report, and over the 14,326
scored reports with a complete label it pools far ahead of the other three fits (PR-AUC 0.159
against 0.015-0.027, 13 failures), mostly on Signature's rank. The honest summary is two
sentences: the production booster puts the right drivers at the top of its explanation for all
three banks and lifts Signature into the top 7 percent, but flags none of them in the top 2
percent before they failed; and no fit in the current pipeline flags SVB. Details:
`notebooks/03_svb_2023_case_study.ipynb`, `reports/svb_2023_case_study.md`.

## 10. Sensitivity summary

Fixed split (train reports 2002Q1-2008Q4, test 2010Q1-2013Q4), one assumption changed per
refit (`reports/sensitivity.md`, `runs/sensitivity/`, `notebooks/05_sensitivity.ipynb`):

| analysis | logit PR-AUC | gbdt PR-AUC | what moves |
|---|---|---|---|
| horizon 4q / 8q | 0.4437 / 0.3764 | 0.3913 / 0.2385 | the booster's recall@2% falls from 0.79 to 0.50 at 8q; the logit loses 0.20 recall |
| censored rows kept / dropped | 0.4437 / 0.4559 | 0.3913 / 0.4265 | dropping censored rows lifts PR-AUC by 0.01-0.04 (the booster's recall@2% slips 0.789 to 0.777); the ranking of models is unchanged |
| availability lag 45 / 60 / 90 days | 0.4692 / 0.4437 / 0.4273 | 0.2005 / 0.3913 / 0.1467 | the logit loses 0.04 PR-AUC from 45 to 90 days; the raw-feature booster loses 0.19 at 45 days and 0.24 at 90 (recall@2% 0.56 / 0.79 / 0.46), the winsorised fit had moved 0.4494 / 0.4324 / 0.3613; 851 / 833 / 809 test failures because the windows shift |

The logit is the less sensitive model in every analysis (largest shift 0.20, the horizon);
the raw-feature booster is the more fragile one at the longer horizon and in both directions
of the availability lag (largest shift 0.33 recall@2%, the 90-day lag), which is the one
finding that argues against the raw inputs and is recorded rather than tuned away.

## 11. False-positive summary

Top-2-percent flags of the walk-forward `gbdt_mono` booster (4q) that did not fail inside the window,
followed eight quarters from `avail_date` (`reports/false_positives.md`,
`notebooks/04_false_positives.ipynb`). Of 8,553 flagged bank-quarters, 13.9 percent failed
within four quarters. Of the 890 flagged non-failing bank-years of 2009-2012, 17.1 percent
failed in quarters five to eight and 9.7 percent later still, 10.0 percent were acquired within
two years (8.5 percent by recorded merger against a 5.1 percent base rate over 31,376
bank-years), 1.5 percent closed voluntarily and 61.8 percent were still open after eight
quarters. Outside the crisis 74.4 percent of flags are still open, because a fixed 2 percent
head must be filled even in years with five failures. The top 2 percent is a watch list, not
a verdict: roughly one flagged bank in seven fails within the year, one in four within a few
years, one in ten is bought, and the rest had the same symptoms and recovered.

## 12. Limitations

- Regime change. Every model learns the 2008-2012 failure mode (construction lending,
  non-performing loans, thin capital). The 2022-2024 test years, with 17, 10 and 9 failures,
  are the worst years of the backtest for every model (PR-AUC below 0.12), and section 9
  shows that the interest-rate and deposit-run features only partly transfer; a linear model
  learns the wrong sign for uninsured deposits from twenty years in which they were a mark
  of size, not of risk. Scores after 2021 should be read with that in mind.
- Data vintages are not modelled. Financials are the FDIC's current values, which include
  restatements the market did not see at the time, and the macro series are current FRED
  vintages, not ALFRED releases. The point-in-time discipline covers *when* a value was
  available, not *what* value was shown then.
- Censoring by acquisition. A troubled bank sold before the FDIC closed it is a zero in the
  label. Section 11 shows the flagged-then-acquired group at twice the base rate, so some of
  the reported false positives are the "rescued" banks the label cannot see, and the true
  precision of the watch list is understated by an unknown amount.
- Small-N years after 2014. From 2014 on no test year holds more than 37 failures and six
  hold fewer than ten, so the per-year confidence intervals span most of [0, 1] and a single
  bank changes a year's PR-AUC. Pooled numbers are dominated by 2008-2012.
- Calibration. The maps are fitted on two trailing out-of-sample years each, so after 2016
  they rest on 4 to 61 failures and move the Brier score in the fifth decimal either way;
  the maps of the first recovery years (2010-2012) inherit the crisis failure rate of
  their trailing years and over-predict, the 8q booster maps break on the drift of that
  model's raw scale between years, and the first years of each series fall back to a
  single in-window slice (section 7). Calibrated probabilities are year-of-fit estimates,
  not a guarantee, and the published production map rests on 19 failures.
- Per-year hyper-parameters. Every walk-forward fit re-selects its hyper-parameters (the
  logit and hazard `C`; the booster's learning rate, leaves and leaf size) on a validation
  slice inside that year's own training period, so no test year informs a choice, but the
  models of different years are not one configuration. A thin slice can pick a degenerate
  setting: the 2020 4q booster (`learning_rate 0.1, num_leaves 63`, PR-AUC 0.0006 on four
  failures) and the 2008 8q booster (training window closed at 2005-12-31, every test row
  scored identically) are the examples, and that variance is part of the reported numbers
  rather than tuned away on the test years.
- Backend caveat. The booster runs on LightGBM 4.7 here; on a machine without an OpenMP
  runtime the same code falls back to scikit-learn's `HistGradientBoostingClassifier`, which
  is the same algorithm with different defaults and tie handling, so refitted scores and SHAP
  values will differ slightly from the committed run records. One run never mixes backends
  (`settings.models.gbdt.backend`).
- No FFIEC raw data. Securities and uninsured-deposit fields come from the FDIC summary, not
  the raw Call Report schedules; the two agree exactly for the three 2023 banks, but the raw
  schedules also carry maturity buckets and pledged amounts the model does not use, and the
  uninsured estimate is self-reported by the bank.
- The hazard conversion `1 - (1 - h)^H` assumes a bank's covariates persist across the
  horizon; the 2010 and 2011 monotone boosters saturate (91 and 97 bank-quarters above 0.999,
  54 and 59 of them failures), so the top of those rankings is a block of near-certain scores
  and any exact ties are ordered by `cert`; the panel starts in 2001, so the 1980s-1990s
  failure wave is not in the training data.

## 13. Ethical considerations and disclaimers

BankCanary is an educational project. Its output is **not a credit rating, not investment
or deposit advice, and not a supervisory assessment**; it has no access to examinations,
enforcement actions or management information, and it is wrong about specific banks in ways
the sections above document. Publishing a ranked list of banks by estimated failure
probability can itself harm a bank if readers act on it, so every presentation of these
scores carries this section, states the base rate (about one flagged bank in seven fails
within a year), shows the confidence interval, and names the drivers so that a reader can see
which published ratio produced the score. No non-public or personal data is used.

Deposits at FDIC-insured banks are insured up to $250,000 per depositor, per insured bank,
per ownership category; the FDIC's standard practice in a failure is to make insured deposits
available within one or two business days. The FDIC's official information is at
[fdic.gov/resources/deposit-insurance](https://www.fdic.gov/resources/deposit-insurance/).

## 14. References

- Cole, R. A. and White, L. J. (2012). "Déjà Vu All Over Again: The Causes of U.S.
  Commercial Bank Failures This Time Around." *Journal of Financial Services Research*
  42(1-2), 5-29. The CAMELS-proxy logistic model behind `logit`.
- Shumway, T. (2001). "Forecasting Bankruptcy More Accurately: A Simple Hazard Model."
  *Journal of Business* 74(1), 101-124. The discrete-time hazard behind `hazard`.
- Board of Governors of the Federal Reserve System (April 2023). *Review of the Federal
  Reserve's Supervision and Regulation of Silicon Valley Bank.*
- Federal Deposit Insurance Corporation (April 2023). *FDIC's Supervision of Signature Bank.*
- Federal Deposit Insurance Corporation (September 2023). *FDIC's Supervision of First
  Republic Bank.*
- FDIC BankFind Suite API documentation, `https://banks.data.fdic.gov/docs/`; FRED,
  `https://fred.stlouisfed.org/`; FFIEC Central Data Repository, `https://cdr.ffiec.gov/public/`.
