# Model card: BankCanary Prototype 2

Version: Prototype 2 ("Depth"), walk-forward models fitted 2026-09-25 on the FDIC pull of the
same date. Companion documents: [`docs/FEATURES.md`](FEATURES.md) (every feature with its
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
- Fold-fitted preprocessing: winsorisation limits (0.5th and 99.5th percentiles), imputation
  medians and scaler statistics are fitted inside the sklearn pipeline on the training rows of
  each fit, so test-year distributions never touch them.
- Identifiers (`cert`, names, dates) never enter a model; the feature list is read from the
  registry and stored beside each pipeline (`features.json`).
- Tests (`tests/test_labels*.py`, `tests/test_time_split*.py`,
  `tests/test_walkforward*.py`) pin the window arithmetic, the rule 6.2 mask and the
  no-leakage assertion on synthetic frames; they never read `data/`.

## 5. Models

All estimators are sklearn pipelines: `Winsorizer(0.005, 0.995)` then, for the linear models,
`SimpleImputer(median, add_indicator)` and `StandardScaler`; trees take missing values
natively, so the booster keeps only the winsoriser.

| name | learner | inputs | notes |
|---|---|---|---|
| `texas` | ranking by the Texas ratio (non-performing assets over tangible equity plus reserves) | one ratio | no fit; the classic single-number warning signal and the baseline everything must beat |
| `logit` | L2 logistic regression, no class weighting, `C` re-selected per test year on the inner slice | 83 `features_v2` columns | the Prototype 1 learner (Cole and White 2012 style) on the richer feature set |
| `gbdt` | gradient-boosted trees, `make_gbdt(backend, monotone)` | 83 features | backend `lightgbm` 4.7 (`settings.models.gbdt.backend`); `HistGradientBoostingClassifier` is the fallback when LightGBM cannot load its OpenMP runtime, same histogram algorithm, native NaN, monotone constraints |
| `gbdt_mono` | the same booster under the registry's monotone signs (+1 risk-increasing, -1 risk-decreasing, 0 free) | 83 features | walked forward beside `gbdt` at 4q with its own per-year tuning (section 6); Decision Point 2 is open and both result sets are presented: the constrained booster pools better (PR-AUC 0.3138 against 0.2647, recall@2% 0.7147 against 0.6367, the recall intervals disjoint) although the inner-validation slice preferred the unconstrained one (0.197 against 0.230), so `settings.models.gbdt.monotone = false` stays until the owner decides and the production and SHAP models remain `gbdt` |
| `hazard` | discrete-time hazard (Shumway 2001): logistic regression on the one-quarter event `y_1q`, converted to H quarters by `1 - (1 - h)^H` | 83 features | assumes covariates persist over the horizon (a documented approximation); trained at its own 1q mask inside each walk-forward year |

Booster parameters at the fixed split: `learning_rate = 0.03`, `num_leaves = 63`,
`min_samples_leaf = 200`, `n_estimators = 200`; the walk-forward fits re-select learning
rate, leaves and leaf size per year from a 2x2x2 grid. The linear models are unweighted
because class balancing let the collinear capital ratios overfit in Prototype 1 (PR-AUC 0.20
against 0.39). The optional Cox model was dropped: `lifelines` pins `pandas < 3`.

## 6. Walk-forward metrics

Protocol (spec 8.2): one model per test year Y from 2008 to the latest label-complete year
(2024 at 4q, 2023 at 8q). Training rows are every earlier bank-quarter whose outcome window
closed before the first prediction date of Y (the 03-31 report plus 60 days); test rows are
the label-complete reports dated in Y; hyper-parameters are re-selected inside each year's
training window. Pooled rows rank every test year as one list. Confidence intervals are
percentile intervals from 200 cluster-bootstrap draws that resample banks (certs) with
replacement so that a bank's consecutive quarters move together; years with fewer than ten
failures are flagged low confidence and a year without failures has undefined ranking
metrics. Scores for every bank-quarter, model and year live in `walkforward_scores`
(3,352,958 rows: `texas, logit, gbdt, gbdt_mono, hazard` at 4q and `logit, gbdt, hazard` at
8q); the per-year artefacts in `models/walkforward/<Y>/<model>[_8q]/`.

### Pooled, 4-quarter horizon (test years 2008-2024, 426,019 bank-quarters, 2,103 failures)

| model | PR-AUC | 95% CI | recall @ top 2% | 95% CI | ROC-AUC | Brier raw | Brier calibrated |
|---|---|---|---|---|---|---|---|
| hazard | 0.3268 | [0.294, 0.357] | 0.7038 | [0.679, 0.730] | 0.9575 | 0.0041 | 0.0038 |
| gbdt_mono | 0.3138 | [0.282, 0.345] | 0.7147 | [0.692, 0.739] | 0.8993 | 0.0045 | 0.0048 |
| logit | 0.3056 | [0.275, 0.334] | 0.6833 | [0.657, 0.713] | 0.9601 | 0.0041 | 0.0038 |
| gbdt | 0.2647 | [0.236, 0.297] | 0.6367 | [0.603, 0.663] | 0.8204 | 0.0043 | 0.0044 |
| texas | 0.2606 | [0.226, 0.298] | 0.7437 | [0.716, 0.771] | 0.9599 | 0.1707 | n/a |

### Pooled, 8-quarter horizon (test years 2008-2023, 407,621 bank-quarters, 3,768 failures)

| model | PR-AUC | 95% CI | recall @ top 2% | 95% CI | ROC-AUC | Brier raw | Brier calibrated |
|---|---|---|---|---|---|---|---|
| hazard | 0.4113 | [0.382, 0.442] | 0.6598 | [0.634, 0.688] | 0.9416 | 0.0070 | 0.0080 |
| logit | 0.2566 | [0.225, 0.288] | 0.5207 | [0.493, 0.550] | 0.7447 | 0.0079 | 0.0077 |
| gbdt | 0.0887 | [0.070, 0.108] | 0.2200 | [0.193, 0.252] | 0.4193 | 0.0088 | 0.0203 |

Reading: the hazard model pools best at both horizons; at 4q the hazard-logit gap (0.021
PR-AUC) is inside both intervals, so the backtest does not separate them, and every learner
beats the Texas ratio on PR-AUC while the Texas ratio keeps the highest recall in the top 2
percent. The unconstrained booster is the weakest learner on pooled PR-AUC because its early
years are starved of failures (the 2008 model trains on 37 positives) and its raw scale
drifts between years, which the pooled ranking punishes; year by year it is competitive from
2010 on. The monotone booster does not share that weakness: the registry signs hold its raw
scale together across years (pooled ROC-AUC 0.90 against 0.82) and it pools second on
PR-AUC and first on recall@2% among the learners, which is the walk-forward evidence for
Decision Point 2 (section 5). At 8q the unconstrained booster collapses (recall 0.22; its
2008 fit scores every test row identically, ROC-AUC 0.50) while the logit degrades
gracefully and the hazard, whose one-quarter event is converted with
`1 - (1 - h)^8`, is clearly best (PR-AUC 0.41, the interval disjoint from the logit's). The
Prototype 1 fixed-split logit (PR-AUC 0.3867, recall@2% 0.7143, test 2010-2013, v1 features)
is a different test period and is not comparable with any pooled row.

### Per year, 4-quarter horizon: PR-AUC [95% CI] and recall @ top 2%

| year | n | failures | hazard PR-AUC [95% CI] | logit PR-AUC [95% CI] | gbdt PR-AUC [95% CI] | gbdt_mono PR-AUC [95% CI] | texas PR-AUC [95% CI] | hazard recall@2% | logit recall@2% | gbdt recall@2% | gbdt_mono recall@2% | texas recall@2% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2008 | 33972 | 429 | 0.3221 [0.267, 0.385] | 0.3025 [0.248, 0.361] | 0.1102 [0.086, 0.142] | 0.2020 [0.159, 0.250] | 0.3506 [0.288, 0.427] | 0.5198 | 0.4918 | 0.2727 | 0.4289 | 0.5408 |
| 2009 | 32811 | 679 | 0.4559 [0.413, 0.502] | 0.4172 [0.376, 0.464] | 0.3863 [0.341, 0.447] | 0.3414 [0.304, 0.390] | 0.5714 [0.516, 0.626] | 0.4433 | 0.4242 | 0.4197 | 0.3741 | 0.5567 |
| 2010 | 31431 | 408 | 0.4897 [0.429, 0.554] | 0.4590 [0.394, 0.529] | 0.4377 [0.378, 0.501] | 0.4544 [0.398, 0.521] | 0.4840 [0.417, 0.566] | 0.6471 | 0.6176 | 0.6324 | 0.6471 | 0.6471 |
| 2011 | 30165 | 231 | 0.5506 [0.464, 0.636] | 0.5106 [0.430, 0.589] | 0.5422 [0.454, 0.629] | 0.4245 [0.348, 0.505] | 0.4171 [0.329, 0.530] | 0.8701 | 0.8225 | 0.8571 | 0.8009 | 0.7749 |
| 2012 | 29130 | 121 | 0.5342 [0.421, 0.669] | 0.4854 [0.386, 0.608] | 0.4626 [0.347, 0.592] | 0.4796 [0.349, 0.598] | 0.3574 [0.246, 0.527] | 0.9669 | 0.9421 | 0.9587 | 0.9421 | 0.9091 |
| 2013 | 27970 | 73 | 0.4190 [0.254, 0.610] | 0.3684 [0.204, 0.542] | 0.4074 [0.240, 0.602] | 0.3906 [0.246, 0.575] | 0.1659 [0.081, 0.309] | 0.8767 | 0.8493 | 0.8767 | 0.8767 | 0.7123 |
| 2014 | 26792 | 37 | 0.4237 [0.260, 0.681] | 0.4364 [0.264, 0.678] | 0.5152 [0.304, 0.690] | 0.4484 [0.249, 0.666] | 0.1225 [0.044, 0.283] | 0.9730 | 0.9730 | 0.9730 | 0.9730 | 0.8649 |
| 2015 | 25517 | 24 | 0.3305 [0.135, 0.631] | 0.2671 [0.134, 0.524] | 0.3090 [0.136, 0.641] | 0.2998 [0.120, 0.581] | 0.1152 [0.039, 0.252] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9167 |
| 2016 | 24351 | 28 | 0.4899 [0.275, 0.786] | 0.5337 [0.281, 0.800] | 0.4090 [0.208, 0.686] | 0.4653 [0.250, 0.765] | 0.1038 [0.044, 0.234] | 0.8571 | 0.9643 | 0.9286 | 0.9286 | 0.8214 |
| 2017 (low confidence) | 23321 | 5 | 0.0446 [0.000, 0.389] | 0.0454 [0.000, 0.450] | 0.0664 [0.000, 0.538] | 0.2402 [0.000, 0.770] | 0.0112 [0.000, 0.067] | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| 2018 | 22301 | 10 | 0.0750 [0.010, 0.264] | 0.0961 [0.014, 0.331] | 0.1131 [0.018, 0.346] | 0.1330 [0.018, 0.427] | 0.0219 [0.003, 0.078] | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.7000 |
| 2019 | 21362 | 18 | 0.4849 [0.153, 0.859] | 0.5079 [0.176, 0.862] | 0.4509 [0.198, 0.802] | 0.5376 [0.274, 0.827] | 0.3994 [0.072, 0.747] | 0.9444 | 0.9444 | 0.9444 | 0.9444 | 0.9444 |
| 2020 (low confidence) | 20475 | 4 | 0.8269 [0.567, 1.000] | 0.8750 [0.667, 1.000] | 0.0074 [0.000, 0.040] | 0.5521 [0.200, 1.000] | 0.6506 [0.098, 1.000] | 1.0000 | 1.0000 | 0.2500 | 1.0000 | 1.0000 |
| 2021 (low confidence) | 19942 | 0 | n/a n/a | n/a n/a | n/a n/a | n/a n/a | n/a n/a | n/a | n/a | n/a | n/a | n/a |
| 2022 | 19285 | 17 | 0.0048 [0.000, 0.032] | 0.0219 [0.001, 0.210] | 0.0092 [0.001, 0.094] | 0.0121 [0.001, 0.128] | 0.0035 [0.000, 0.027] | 0.2353 | 0.2353 | 0.1176 | 0.1176 | 0.1176 |
| 2023 | 18796 | 10 | 0.0116 [0.000, 0.044] | 0.0417 [0.000, 0.278] | 0.1289 [0.000, 0.672] | 0.1761 [0.001, 0.750] | 0.0068 [0.001, 0.035] | 0.5000 | 0.5000 | 0.7000 | 0.6000 | 0.2000 |
| 2024 (low confidence) | 18398 | 9 | 0.0283 [0.000, 0.336] | 0.1117 [0.000, 0.630] | 0.0578 [0.001, 0.506] | 0.1119 [0.000, 0.631] | 0.0031 [0.000, 0.038] | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 |
| pooled | 426019 | 2103 | 0.3268 [0.294, 0.357] | 0.3056 [0.275, 0.334] | 0.2647 [0.236, 0.297] | 0.3138 [0.282, 0.345] | 0.2606 [0.226, 0.298] | 0.7038 | 0.6833 | 0.6367 | 0.7147 | 0.7437 |

### Per year, 8-quarter horizon: PR-AUC [95% CI] and recall @ top 2%

| year | n | failures | hazard PR-AUC [95% CI] | logit PR-AUC [95% CI] | gbdt PR-AUC [95% CI] | hazard recall@2% | logit recall@2% | gbdt recall@2% |
|---|---|---|---|---|---|---|---|---|
| 2008 | 33972 | 1108 | 0.3727 [0.329, 0.417] | 0.0719 [0.057, 0.093] | 0.0326 [0.030, 0.036] | 0.2987 | 0.1011 | 0.0027 |
| 2009 | 32811 | 1087 | 0.5104 [0.469, 0.553] | 0.4321 [0.383, 0.483] | 0.3836 [0.343, 0.433] | 0.3615 | 0.3183 | 0.2962 |
| 2010 | 31431 | 639 | 0.5176 [0.463, 0.581] | 0.4486 [0.394, 0.517] | 0.2975 [0.256, 0.352] | 0.5258 | 0.4883 | 0.3474 |
| 2011 | 30165 | 352 | 0.5488 [0.476, 0.621] | 0.4904 [0.408, 0.554] | 0.3618 [0.289, 0.430] | 0.7415 | 0.6648 | 0.6080 |
| 2012 | 29130 | 194 | 0.4875 [0.393, 0.617] | 0.4235 [0.341, 0.550] | 0.4364 [0.345, 0.555] | 0.7990 | 0.7577 | 0.8299 |
| 2013 | 27970 | 110 | 0.4280 [0.286, 0.579] | 0.3344 [0.207, 0.479] | 0.2917 [0.183, 0.447] | 0.8909 | 0.7909 | 0.7909 |
| 2014 | 26792 | 61 | 0.3788 [0.239, 0.577] | 0.2985 [0.178, 0.476] | 0.3634 [0.221, 0.532] | 0.9836 | 0.9508 | 0.9016 |
| 2015 | 25517 | 52 | 0.3337 [0.163, 0.594] | 0.2936 [0.137, 0.556] | 0.3677 [0.203, 0.627] | 0.8846 | 0.8269 | 0.8269 |
| 2016 | 24351 | 33 | 0.4165 [0.202, 0.712] | 0.3297 [0.161, 0.665] | 0.3605 [0.174, 0.713] | 0.7273 | 0.7273 | 0.7879 |
| 2017 | 23321 | 15 | 0.0254 [0.003, 0.136] | 0.0398 [0.003, 0.160] | 0.0419 [0.006, 0.196] | 0.4000 | 0.3333 | 0.4667 |
| 2018 | 22301 | 28 | 0.2010 [0.048, 0.458] | 0.2388 [0.059, 0.501] | 0.2372 [0.051, 0.557] | 0.6786 | 0.7500 | 0.7143 |
| 2019 | 21362 | 22 | 0.5557 [0.225, 0.898] | 0.5270 [0.203, 0.902] | 0.5549 [0.220, 0.909] | 0.9545 | 0.9545 | 0.9545 |
| 2020 (low confidence) | 20475 | 4 | 0.8750 [0.643, 1.000] | 0.8929 [0.683, 1.000] | 0.6429 [0.211, 1.000] | 1.0000 | 1.0000 | 1.0000 |
| 2021 | 19942 | 17 | 0.0007 [0.000, 0.002] | 0.0016 [0.000, 0.004] | 0.0017 [0.000, 0.004] | 0.0000 | 0.0000 | 0.0000 |
| 2022 | 19285 | 27 | 0.0056 [0.001, 0.028] | 0.0173 [0.001, 0.142] | 0.0049 [0.001, 0.014] | 0.1852 | 0.1481 | 0.0741 |
| 2023 | 18796 | 19 | 0.0065 [0.001, 0.031] | 0.0132 [0.001, 0.101] | 0.0084 [0.001, 0.055] | 0.2105 | 0.2105 | 0.2105 |
| pooled | 407621 | 3768 | 0.4113 [0.382, 0.442] | 0.2566 [0.225, 0.288] | 0.0887 [0.070, 0.108] | 0.6598 | 0.5207 | 0.2200 |

The post-2014 years hold 0 to 37 failures each, so their intervals span most of [0, 1]; the
2022-2024 rows (17, 10 and 9 failures) are the years in which the 2008-shaped models did
worst, which section 9 takes up.

## 7. Calibration

Per walk-forward year an isotonic map is fitted on the last complete label year inside the
training window (widened backwards while it holds fewer than five failures) and applied to
the full-window model. Who scores that slice depends on the learner
(`calibration.SLICE_SCORER_BY_MODEL`): the logit and the hazard score it with the year's own
full-window model, so the map is learned on the score scale it is applied to (in-sample for
those rows, which an L2 logit on several hundred thousand rows tolerates); the boosters
separate their training rows perfectly (in-sample slice Brier 0.0000 with four thresholds),
so an inner model fitted on the years before the slice scores it instead. No test outcome
shapes a map. The map is stored as `models/walkforward/<Y>/<model>[_8q]/calibration.joblib`
and its output as `walkforward_scores.score_calibrated` (filled for logit, gbdt, gbdt_mono
and hazard; the Texas ratio is a ranking, not a probability). Pooled 4q results against a
failure rate of 0.49 percent:

| model | slice scored by | mean raw score | mean calibrated | Brier raw | Brier calibrated | years where calibration worsens Brier | top decile: mean calibrated / observed |
|---|---|---|---|---|---|---|---|
| logit | full-window model | 0.0025 | 0.0044 | 0.0041 | 0.0038 | 8 of 17 | 0.042 / 0.045 |
| hazard | full-window model | 0.0020 | 0.0055 | 0.0041 | 0.0038 | 10 of 17 | 0.053 / 0.043 |
| gbdt | inner model | 0.0021 | 0.0079 | 0.0043 | 0.0044 | 7 of 17 | 0.045 / 0.036 |
| gbdt_mono | inner model | 0.0044 | 0.0141 | 0.0045 | 0.0048 | 8 of 17 | 0.068 / 0.042 |

Reading: for the two linear models the map now lowers the pooled Brier score and puts the
top decile within a few thousandths of the observed rate (the raw scores under-predicted it
by about a factor of two); the years where it still worsens Brier are the post-2013 years
with 0 to 37 failures, where any map fitted on a thin slice is noise, and the differences
there are in the fifth decimal. Under the inner-model recipe that Prototype 2 first used for
every learner the logit and hazard maps were unusable in the crisis years (2009 mean
calibrated 0.19 and 0.47 against a failure rate of 0.021; pooled Brier 0.0189 and 0.0545)
because the inner model, trained before any crisis failure, put the whole 2009 test year
above its top threshold; `docs/DECISIONS.md` keeps those numbers. The boosters' maps are
within tolerance but over-predict the top decile (gbdt_mono by half again), and the 8q maps
follow the same pattern for the linear models (logit 0.0079 raw against 0.0077 calibrated;
hazard 0.0070 against 0.0080, the converted eight-quarter hazard over-predicting 2010-2011),
while the 8q booster map is worse than raw (0.0088 against 0.0203): the retuned 2011-2013
full-window boosters put their test rows on a coarser score scale than the inner model that
scored their slice, so those maps extrapolate. Reliability curves:
`reports/figures/reliability_{logit,gbdt,gbdt_mono,hazard}.png`.

## 8. Lead time

For each failed bank, the number of calendar quarters between the first report at which it
entered the top 2 percent of that quarter's walk-forward ranking and its failure date; banks
never flagged are in every denominator. The 2009-2012 cohort (440 failed banks) is the
headline because 2008 failures can only be flagged inside 2008 and post-2024 failures are
scored only through the 2024 model.

| model | failed banks | flagged before failure | median lead (quarters) | flagged >= 2 quarters ahead |
|---|---|---|---|---|
| hazard | 440 | 398 (90.5%) | 5 | 88.4% |
| logit | 440 | 395 (89.8%) | 5 | 87.7% |
| gbdt_mono | 440 | 384 (87.3%) | 5 | 85.5% |
| gbdt | 440 | 364 (82.7%) | 4.5 | 81.6% |
| texas | 440 | 372 (84.5%) | 4 | 82.5% |

Over every failure with a scored quarter before it (545 banks) the logit flags 89.4 percent,
median lead 5 quarters, 87.5 percent at least two quarters ahead. The spec's target (most
failures flagged at least two quarters ahead) is met for the crisis cohort by every model.
Figures: `reports/figures/lead_time_<model>.png`.

## 9. The 2023 case study

Question: would a model trained on the 2008 crisis have flagged Silicon Valley Bank,
Signature Bank and First Republic Bank? Four models are fitted on one rule-6.2 cut (every
bank-quarter whose 4q window closed before the 2022Q4 prediction date: reports through
2021Q3, 622,341 rows, 2,226 failures) on two feature views, `credit_only` (the 43 Prototype 1
ratios) and `rate_aware` (all 83 features), and score the 2022Q3, 2022Q4 and 2023Q1 reports
of every bank (about 4,800 per quarter). Rank 1 is the riskiest bank of the quarter.

| bank, 2022Q4 report | credit-only logit | credit-only gbdt | rate-aware logit | rate-aware gbdt |
|---|---|---|---|---|
| Silicon Valley Bank | 1677 (65th pct) | 1412 (70th) | 1854 (61st) | **245 (94.9th)** |
| Signature Bank | **113 (97.6th)** | 1648 (65th) | 627 (86.9th) | 2634 (45th) |
| First Republic Bank | 1570 (67th) | 1167 (75.5th) | 1750 (63rd) | 1416 (70th) |

The credit-only view saw nothing at SVB (Texas ratio 0.009). The rate-aware booster lifts it
to the 95th percentile, fourth among the 34 banks above $100B, on `adjusted_tier1_leverage`
= -0.33 (Tier 1 capital net of unrealised securities losses was negative; 116 training
bank-quarters had that property and 55 failed). The rate-aware logit does not help because
over 2001-2021 `uninsured_share` and `macro_fedfunds_change_4q` enter with *safer* signs
(failed banks averaged 13 percent uninsured deposits against 20 percent for survivors, and
SVB's fed-funds change of 4.49 points lies outside the training range, maximum 2.02), so the
linear model cancels the loss terms. Signature is flagged only by the credit-only logit, on a
2008-shaped concentration profile (C&I share, CRE at three times capital, multifamily), and
First Republic by nothing above the 76th percentile before it failed. The honest summary is
two sentences: the rate-aware booster would have put SVB on a watch list one quarter before
it failed, on the right driver, but none of the four models ranked it in the top 2 percent,
and Signature and First Republic were not caught by the rate-aware view at all. Details:
`notebooks/03_svb_2023_case_study.ipynb`, `reports/svb_2023_case_study.md`.

## 10. Sensitivity summary

Fixed split (train reports 2002Q1-2008Q4, test 2010Q1-2013Q4), one assumption changed per
refit (`reports/sensitivity.md`, `runs/sensitivity/`, `notebooks/05_sensitivity.ipynb`):

| analysis | logit PR-AUC | gbdt PR-AUC | what moves |
|---|---|---|---|
| horizon 4q / 8q | 0.4437 / 0.3764 | 0.4324 / 0.1328 | the booster's recall@2% falls from 0.78 to 0.29 at 8q; the logit loses 0.20 recall |
| censored rows kept / dropped | 0.4437 / 0.4559 | 0.4324 / 0.4467 | dropping censored rows flatters every metric by 0.01-0.03; the ranking of models is unchanged |
| availability lag 45 / 60 / 90 days | 0.4692 / 0.4437 / 0.4273 | 0.4494 / 0.4324 / 0.3613 | a longer lag costs the booster 0.07 PR-AUC and the logit 0.04; 851 / 833 / 809 test failures because the windows shift |

The logit is the less sensitive model in every analysis; the booster is the more fragile
one at longer horizons and longer lags.

## 11. False-positive summary

Top-2-percent flags of the walk-forward booster (4q) that did not fail inside the window,
followed eight quarters from `avail_date` (`reports/false_positives.md`,
`notebooks/04_false_positives.ipynb`). Of 8,553 flagged bank-quarters, 13.5 percent failed
within four quarters. Of the 875 flagged non-failing bank-years of 2009-2012, 17.4 percent
failed in quarters five to eight and 9.0 percent later still, 9.9 percent were acquired within
two years (8.9 percent by recorded merger against a 5.1 percent base rate over 31,376
bank-years), 1.8 percent closed voluntarily and 61.8 percent were still open after eight
quarters. Outside the crisis 76.5 percent of flags are still open, because a fixed 2 percent
head must be filled even in years with five failures. The top 2 percent is a watch list, not
a verdict: roughly one flagged bank in seven fails within the year, one in four within a few
years, one in ten is bought, and the rest had the same symptoms and recovered.

## 12. Limitations

- Regime change. Every model learns the 2008-2012 failure mode (construction lending,
  non-performing loans, thin capital). The 2022-2024 test years, with 17, 10 and 9 failures,
  are the worst years of the backtest for every model (PR-AUC below 0.18), and section 9
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
- Calibration. The maps are fitted on one label year each, so after 2013 they rest on 0 to
  37 failures and move the Brier score in the fifth decimal either way; the boosters'
  maps over-predict the top decile, the eight-quarter hazard map over-predicts 2010-2011 and
  the eight-quarter booster map extrapolates in 2011-2013 (section 7). Calibrated probabilities are year-of-fit estimates, not a guarantee.
- Per-year hyper-parameters. Every walk-forward fit re-selects its hyper-parameters (the
  logit and hazard `C`; the booster's learning rate, leaves and leaf size) on a validation
  slice inside that year's own training period, so no test year informs a choice, but the
  models of different years are not one configuration. A thin slice can pick a degenerate
  setting: the 2020 4q booster (`learning_rate 0.1, num_leaves 63`, PR-AUC 0.0074 on four
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
  horizon; the 2010 booster saturates (68 bank-quarters above 0.999), so ties at the very top
  of that year's ranking are ordered by `cert`; the panel starts in 2001, so the 1980s-1990s
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
