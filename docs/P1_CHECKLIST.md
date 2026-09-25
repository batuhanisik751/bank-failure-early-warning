# Prototype 1 acceptance checklist

Spec: `PROJECT_SPEC.md`, "Prototype 1 — Foundation". Verified 2026-09-25 on the cached
FDIC pull of the same date (`data/raw/fdic/failures/all.json`, `fetched_at = 2026-09-25`).
Numbers come from `reports/data_quality.md`, `reports/p1_baselines.md`,
`models/<name>/metrics.json` and `notebooks/01_foundation.ipynb`.

- [x] **Re-running the pipeline from cache produces byte-identical Parquet outputs.**
  Verified by recording `shasum -a 256` of `data/parquet/labels.parquet` and
  `data/parquet/features_v1.parquet`, re-running `bankcanary build-labels` (1.6 s) and
  `bankcanary build-features` (6.5 s) from the cached tables, and checking both sums
  again: unchanged. `write_table` sorts by key, fixes column order and writes with pyarrow
  zstd; labels use the cached `fetched_at` as `as_of_date` rather than the wall clock
  (`docs/DECISIONS.md`). `financials_raw` and `panel` were not re-run for this check
  (the full financials rebuild exceeds the two-minute step budget); they go through the same
  `write_table` path and `tests/test_storage.py` pins the byte-identical round trip.

- [x] **Failure counts per year match the FDIC failed-bank list; >= 95% of 2001+ failures
  are matched to a prior Call Report.** The `failures` table is the full BankFind
  `/failures` pull (4,117 rows: 3,524 FAILURE, 593 ASSISTANCE); counts per year are
  computed straight from it and tabulated in `README.md` and section 2 of the notebook
  (574 failures from 2001 through 2026-09-25: 140 in 2009, 157 in 2010, 92 in 2011).
  Matching, from `reports/data_quality.md` section 4: 573 of 574 (99.8%) have at least
  one panel row before the failure; the single miss is cert 34264, FIRST ALLIANCE BANK AND
  TRUST CO, which failed on 2001-02-02 before the first 2001Q1 report in the panel.

- [x] **All label and leakage unit tests pass; the Rule 6.2 split function is used by every
  training path.** `uv run pytest`: 214 passed. Label tests
  (`tests/test_labels.py`, `tests/test_labels_adversarial.py`): 72 passed. Split, leakage and
  model tests (`tests/test_time_split*.py`, `tests/test_models.py`): 37 passed. Rule 6.2 lives
  only in `bankcanary.splits.time_split.training_mask`; `fixed_split_masks` and
  `walk_forward_folds` both call it, and `bankcanary.models.train` selects rows only
  through `fixed_split_masks` and re-checks them with `assert_no_leakage` (verified by
  `grep -rn training_mask src`: no other caller builds a training mask). The 4q split
  actually used: train reports 2002-03-31..2008-12-31 (252,330 rows, 554 positives), test
  2010-03-31..2013-12-31 (118,696 rows, 833 positives), first test prediction 2010-05-30.

- [ ] **The regularised logistic model beats the Texas ratio on PR-AUC on the fixed test
  split.** Not met. Test PR-AUC (4q): texas 0.3726, logit_small 0.3720, logit 0.1985
  (`reports/p1_baselines.md`, re-scored from the saved pipelines by `bankcanary evaluate`
  and again in the notebook). The all-feature logit is hurt by four collinear capital
  measures with sign-flipped coefficients; stronger L2 (C down to 0.001) lifts it only to
  0.26 (`docs/DECISIONS.md`). Left open for Prototype 2 feature selection and the
  gradient-boosting model rather than patched by tuning on the test split.

- [x] **The notebook runs top-to-bottom on a laptop in < 10 minutes after data is cached.**
  `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_foundation.ipynb`
  finishes in about 9 seconds (15 cells, 0 errors, 212 KB executed). It reads Parquet
  tables and the saved model artefacts and re-scores the test split; it never retrains.

## Feature list check (spec items 1-11)

| item | status | evidence |
|---|---|---|
| 1 scaffold (uv, ruff, pytest, pre-commit) | done | `pyproject.toml`, `.pre-commit-config.yaml`, `Makefile` |
| 2 FDIC client (cache, pagination, retries, rate limit) | done | `bankcanary.sources.fdic`, `tests/test_fdic_client.py` |
| 3 verified field map | done | `config/fields.yaml` (96 financial columns with `first_available`) |
| 4 ingest failures/institutions/history/financials 2001Q1+ | done | panel spans 2001-03-31..2026-06-30 |
| 5 raw JSON -> Parquet -> DuckDB | done | `data/parquet/*.parquet`, `data/warehouse.duckdb` |
| 6 label builder with unit tests | done | `bankcanary.labels`, 72 passed |
| 7 ~20 features with YTD de-accumulation and tests | done | 43 features in `features_v1`, `tests/test_features.py`, `tests/test_ytd*.py` |
| 8 Texas, small logistic, regularised logistic | done | `models/{texas,logit_small,logit}/` |
| 9 fixed out-of-time evaluation | done | `reports/p1_baselines.md` |
| 10 notebook | done | `notebooks/01_foundation.ipynb`, built by `scripts/make_notebook_01.py` |
| 11 CLI | done | `bankcanary --help` lists ingest, build-panel, build-labels, build-features, dq-report, train, evaluate |

Known gotcha: in this checkout the editable install's `.pth` file is not picked up by the
virtualenv, so `uv run bankcanary` fails with `ModuleNotFoundError` unless prefixed with
`PYTHONPATH=src`; the notebook adds `src/` to `sys.path` itself.
