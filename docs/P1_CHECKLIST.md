# Prototype 1 acceptance checklist

Spec: `PROJECT_SPEC.md`, "Prototype 1 — Foundation". Every criterion below was re-measured
on 2026-09-25 against the cached FDIC pull of the same date
(`data/raw/fdic/failures/all.json`, `fetched_at = 2026-09-25`, sha256 `3e3dbdc8…f0a9c6`,
1,123,777 bytes). The numbers come from the commands quoted next to each item, not from
the reports.

- [x] **Re-running the pipeline from cache produces byte-identical Parquet outputs.**
  Recorded `shasum -a 256` of the three derived tables, then ran `bankcanary build-panel`
  (8.5 s), `bankcanary build-labels` (1.4 s) and `bankcanary build-features` (6.1 s) as
  separate commands and hashed again. All three unchanged:
  `panel.parquet fa0199b2…8c62`, `labels.parquet e3d8e836…2a03`,
  `features_v1.parquet 8a4e5874…70cd`. `bankcanary ingest --what failures` re-run
  afterwards finished in 0.7 s and left `data/raw/fdic/failures/all.json` untouched
  (same mtime, size and sha256; no download). `financials_raw` was not rebuilt (the
  financials pull exceeds the two-minute step budget); it goes through the same
  `write_table` path (sort by key, fixed column order, pyarrow zstd) that
  `tests/test_storage.py` pins.

- [x] **Failure counts per year match the FDIC failed-bank list; >= 95% of 2001+ failures
  are matched to a prior Call Report.** Read from `data/warehouse.duckdb`:
  `failures` has 4,117 rows (3,524 FAILURE, 593 ASSISTANCE); FAILURE rows per
  `fail_year` are 2009 = 140, 2010 = 157, 2011 = 92, matching the FDIC list. Of the 574
  FAILURE rows with `fail_year >= 2001`, 573 (99.83%) have at least one `panel` row with
  `repdte < fail_date`. The single miss is cert 34264 (FIRST ALLIANCE BANK AND TRUST CO,
  failed 2001-02-02), which closed before the first 2001Q1 report in the panel
  (panel: 710,691 rows, `repdte` 2001-03-31..2026-06-30).

- [x] **All label and leakage unit tests pass; the Rule 6.2 split function is used by every
  training path.** `uv run pytest`: 214 passed in 4.0 s. Label tests
  (`tests/test_labels.py`, `tests/test_labels_adversarial.py`): 72 passed. Split, leakage and
  model tests (`tests/test_time_split*.py`, `tests/test_models.py`): 37 passed.
  `grep -rn -E "fixed_split_masks|training_mask" src/bankcanary/models` hits only
  `train.py`: `train_model` selects rows with `fixed_split_masks` (line 126) and re-checks
  them with `assert_no_leakage`; `evaluate_model` takes its test rows from the same call
  (line 233). `grep -rn "\.fit(" src/bankcanary/models` finds one call (`train.py:136`),
  on the rows that mask returned. `training_mask` itself is called only from
  `splits/time_split.py` (`fixed_split_masks` and `walk_forward_folds`), so no training
  path selects rows any other way. The 4q split: train 252,330 rows / 554 positives,
  test 118,696 rows / 833 positives.

- [ ] **The regularised logistic model beats the Texas ratio on PR-AUC on the fixed test
  split.** Not met. `models/<name>/metrics.json`, `test.pr_auc` (4q, n = 118,696,
  833 failures): texas 0.3726, logit_small 0.3720, logit 0.1985 (ROC-AUC 0.974 / 0.979 /
  0.964). The all-feature logit is hurt by collinear capital measures with sign-flipped
  coefficients; stronger L2 lifts it only to about 0.26 (`docs/DECISIONS.md`). Left open
  for Prototype 2 feature selection and the gradient-boosting model rather than tuned on
  the test split.

- [x] **The notebook runs top-to-bottom on a laptop in < 10 minutes after data is cached.**
  `uv run jupyter nbconvert --to notebook --execute --output /tmp/nb_check.ipynb
  --ExecutePreprocessor.timeout=110 notebooks/01_foundation.ipynb`: 6.8 s wall clock,
  7 code cells, 0 error outputs, 209 KB executed. It reads the Parquet tables and the
  saved model artefacts and re-scores the test split; it never retrains.

- [x] **Hygiene.** A case-insensitive grep of every commit message for co-author trailers
  and tool or vendor names prints nothing; the same grep over `*.py, *.md, *.toml, *.yaml,
  *.ipynb` (excluding `.venv`, `data`, `PROJECT_SPEC.md`) prints nothing.

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

Known gotcha: in this checkout every file under `.venv/` carries the macOS `hidden` file
flag (`ls -lO`), and Python 3.12's `site` module skips hidden `.pth` files, so the editable
install's `_editable_impl_bankcanary.pth` is never applied. `uv run bankcanary` and
`uv run pytest` then fail with `ModuleNotFoundError` unless prefixed with `PYTHONPATH=src`;
`chflags nohidden .venv/lib/python3.12/site-packages/*.pth` fixes it until the flag is
re-applied. The notebook adds `src/` to `sys.path` itself.
