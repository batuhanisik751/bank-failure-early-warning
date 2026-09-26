# Decision log

Append-only. One dated bullet per non-obvious decision, with the reason. Spec decision
points (⚖️) that were resolved with their stated default are marked as such and remain
open for the owner to revisit.

- 2026-09-25 — **Decision Point 1 (data start year):** Prototype 1 starts at 2001Q1, the
  spec's default. The FDIC API has quarterly financials back to 1984, so extending to
  1992–2000 later is a one-line change to `start_quarter` in `config/settings.yaml`.
- 2026-09-25 — **Dataframe library:** pandas + pyarrow for the modeling layer (scikit-learn,
  SHAP and LightGBM consume pandas natively); DuckDB for the warehouse and the heavy joins.
  The panel is under a million rows, so pandas is comfortably fast enough.
- 2026-09-25 — **History endpoint scope:** only institution-level structure events
  (`CHANGECODE` < 500: new institution, failure, merger, voluntary closing, conservatorship,
  charter/class changes) are ingested. Branch-level events (codes 5xx–8xx, ~490k of the
  ~585k rows) carry no information about a bank's exit and are skipped.
- 2026-09-25 — **API key:** the FDIC BankFind API served every probed endpoint without a
  key (rate-limit window of 20 requests advertised in response headers). `FDIC_API_KEY`
  stays optional; the client sends it only when set.
- 2026-09-25 — **FDIC cache validation:** a cached file is honoured only when its stored
  canonical query (endpoint, filters, sorted fields, sort) matches the request; a name
  collision with a different field list refetches and overwrites instead of silently
  returning stale columns. Retry back-off (`wait`) is injectable so tests run with zero
  waits; HTTP 4xx other than 429 are not retried (the API returns 404 for bad paths).
- 2026-09-25 — **Field map (B1):** `config/fields.yaml` pulls 107 `/financials` codes.
  Dollar Tier 1 capital = `RBCT1J` (PCA definition, allowance-adjusted; the ratio
  `RBC1AAJ` is built from it), total risk-based capital dollars = `RBC` (`RBC-TOTAL-PCA`,
  the only total-RBC dollar item in the dictionary; not yet probed on the live endpoint),
  HTM securities at amortized cost = `SCHA` (`SECURITIES-HA`, paired with `SCHF` at fair
  value). `FED_RSSD` is absent from `risview_properties.yaml` but returned live, so it is
  kept with `in_dictionary: false`. Prototype 2 rate/run-risk items (`SCAA`, `SCAF`,
  `SCHF`, `SCHA`, `DEPINS`, `DEPUNINS`, `ERNAST`, `CD3LES`/`CD3T12`/`CD1T3`/`CDOV3`) are
  in the pull list now so financials never need re-downloading. Spec guesses `SCHTM`,
  `SCAFFV`, `SCHTMFV`, `RBCTOT`, `TD250K`, `ELECTCBLR`, `DEPUNS`, `FRBHLBADV`, `IDNCLNLS`
  do not exist and are recorded under `not_found`.
- 2026-09-25 — **Failures table keeps every FDIC row (B2):** 488 pre-1966 failures have no
  `CERT`, so the contract key `(cert, fail_date)` is not unique for them. All 4,117 rows are
  kept (3,524 `FAILURE`, 593 `ASSISTANCE`); an `fdic_id` column (the API's record `ID`)
  breaks ties so the Parquet file is reproducible. The panel only uses rows with a cert.
- 2026-09-25 — **History key is not unique (B2):** `/history` `TRANSNUM` repeats within a
  cert (10,056 repeated `(cert, transnum)` pairs, mostly `transnum` 0) and the API rejects
  `sort_by=ID`, so the pull stays sorted by `TRANSNUM` and the clean table is sorted
  through `(cert, transnum, effdate, changecode, acq_cert, out_cert)`. Whole-row duplicates
  are dropped (none in the 93,897-row pull). Match events on `(cert, effdate, changecode)`.
- 2026-09-25 — **Dates in Parquet are `datetime64[ns]`:** pandas 3 infers microsecond
  resolution from strings, so `write_table`/`read_table` coerce every datetime column to
  nanoseconds. DuckDB stores them as `TIMESTAMP`; cast with `::DATE` in SQL.
- 2026-09-25 — **Financials pull is one request per quarter, chunked from the CLI:** `ingest
  --what financials --start Q --end Q --no-build` pulls up to 8 quarters per invocation
  (about 5 s per quarter, ~9,600 rows in 2001 down to 4,313 in 2026Q2) into
  `data/raw/fdic/financials/<YYYYMMDD>.json`; a plain `ingest --what financials` then serves
  every quarter from cache, probes the latest published `REPDTE` with one forced request
  (`CERT:3511`, `cache_name=latest_probe`) and rebuilds `financials_raw` in ~45 s. The
  `--build/--no-build` flag exists only so chunked pulls skip the rebuild.
- 2026-09-25 — **`FED_RSSD` is empty on `/financials`:** the endpoint silently drops the
  field (absent from all 710,691 records), so `financials_raw.fed_rssd` is all-null. Take
  the institution RSSD id from `institutions.fed_rssd` when it is needed.
- 2026-09-25 — **`financials_raw` typing:** every financial code (including the `CBLRIND`
  flag and the `DEPSMB` count) is float64; `cb` is bool, `cert`/`rssdhcr`/`fed_rssd` are
  nullable Int64, `estymd` is parsed from the integer `YYYYMMDD` the API sends. No row had
  null or non-positive `ASSET` in the 2001Q1-2026Q2 pull, so the drop rule removed nothing.
- 2026-09-25 — **Exit reason matching:** the institution-level `history` event is matched on
  `subject_cert` (`out_cert` filled with `cert`, because the FDIC records failure events
  with a null `CERT`) within 7 days of `endefymd`; the closest event wins and ties go to the
  most exit-like code (failure > merger family > voluntary closing > charter change > other),
  so a bank whose charter change is booked as a same-day merger of the old cert into the
  new one is a `merger`, not a `charter_change` (no bank ends up as `charter_change`). 691
  banks (nearly all 1970s-90s RTC-era thrifts) exit on a failure-type code with no
  `failures` row at all and are labelled `failure_unmatched`; a failure-type code on a cert
  that has only `ASSISTANCE` rows is `other`. Failed and open banks have `exit_date` and
  `exit_reason` null (`exit_reason` is not `unknown` there; `unknown` means an exit with no
  event in the window, 1,284 banks). 22 inactive banks have no `endefymd` and get no exit.
- 2026-09-25 — **Panel keeps quarter-level attributes:** `bkclass`, `stalp`, `estymd` and
  `cb` from `financials_raw` are point-in-time and stay; only their gaps are filled from
  `institutions`. `rssdhcr` is never filled (a null quarter means no holding company then,
  and the institution value would leak later affiliations); `fed_rssd`, `latitude` and
  `longitude` come from `institutions` only. 162 reporting banks (7,131 rows, none failed,
  56 still filing in 2026Q2) have no `institutions` record and therefore no exit facts.
- 2026-09-25 — **Labels keep dropped rows and use the cached failures date:** `labels` has
  one row per panel row; the 410 rows whose `fail_date <= avail_date` (a failed bank whose
  last reports became usable only after it was closed) stay in the table with
  `dropped_failed_before_avail = True` so the count can be audited, and every consumer
  filters them out. `as_of_date` is the `fetched_at` of `data/raw/fdic/failures/all.json`
  (2026-09-25), not the wall clock, so a rebuild from cache is byte-identical; windows
  ending after it are `label_complete = False` (2025Q1 onward for 4q, 2024Q2 onward for
  8q). `censored_in_window` needs a null `fail_date` and an `exit_date` inside
  `(avail_date, window_end]`; assistance-only banks are plain negatives. `window_end` uses
  `DateOffset(months=3*H)`, which clamps a leap-day `avail_date` (2008-02-29) to
  2009-02-28; the tests pin that.
- 2026-09-25 — **Data-quality report reads the Parquet files, not `warehouse.duckdb`:**
  `bankcanary dq-report` opens an in-memory DuckDB with one view per Parquet table, so the
  report is tied to the canonical store and tests can point it at a `tmp_path` warehouse.
  "Structural" missingness = a null share moving by ≥ 30% between consecutive years (the
  threshold catches the 2007 real-estate breakdown items, the 2014 and 2020 capital-rule
  changes and CBLR filers dropping `rbc1rwaj`/`rbct1cer` from 2020); fields with ≥ 5% nulls
  in the latest full year are listed separately. `first_available` in `config/fields.yaml`
  is only ever written for financial columns present in `financials_raw`; identity and
  attribute fields stay `null`. The report has no timestamp so a rebuild from the same
  tables is byte-identical.
- 2026-09-25 — **YTD de-accumulation matches the previous quarter by exact date, and a NaN
  prior YTD is not a "missing quarter":** `features.ytd` looks up the same cert's row at
  `repdte - QuarterEnd(1)` via a merge, never by row position, so filing gaps and mergers
  cannot difference against the wrong quarter. `ytd_prev_missing` is set only when the cert
  has *no row* at the previous quarter-end (then quarterly = YTD / quarter number); if the
  row exists but that item is NaN the quarterly value stays NaN, preserving structural
  missingness for the missing-indicator columns downstream. Q1 rows are never flagged. On
  the full `financials_raw` table 0.14% of rows carry the flag and the three helpers run in
  well under a second.
- 2026-09-25 — **Split utility (Rule 6.2):** `splits.time_split.training_mask` keeps a row
  only if `window_end_Hq < prediction_date(test_start)`, `label_complete_Hq` and not
  `dropped_failed_before_avail`; `fixed_split_masks` intersects that with the nominal
  `train_start..train_end` from settings. With a 60-day lag and `test_start = 2010-03-31`
  the effective training end is 2008Q4 for H=4 (252,330 rows) but 2007Q4 for H=8
  (218,358 rows): the nominal `train_end` is a ceiling, not the cut. `walk_forward_folds`
  skips years with no usable test row and warns on partial years (e.g. 4q in 2025), so
  callers pick `last_test_year` as the latest complete year. `test_mask` carries
  `__test__ = False` so pytest does not collect it from test modules.
- 2026-09-25 — **`features.ytd._shifted_lookup` matches on the calendar day and aligns by
  position:** both `repdte` and the target dates go through `pd.to_datetime(...).dt.normalize()`
  before the merge, so `datetime.date` objects, second-resolution timestamps and timestamps
  carrying a time-of-day all find their previous quarter (the contract says dates are
  `datetime64[ns]`, but a silent mismatch on a noon timestamp was worse than being lenient).
  The result's index is copied from the input rather than round-tripped through
  `reset_index`/`set_index("index")`, so a named index, a non-unique index or a column called
  `index` no longer break the helpers. Duplicated `(cert, repdte)` keys raise an explicit
  `ValueError` instead of relying on `reindex` refusing duplicate labels.
- 2026-09-25 (splits): `time_split._usable` no longer casts flag columns straight to
  `bool`. `bool(float("nan"))` is `True`, so after a join that introduced `NaN` an unknown
  `label_complete_Hq` was treated as complete and the row entered training/test. A missing
  completeness flag now counts as incomplete and a missing `dropped_failed_before_avail`
  as dropped (both excluded), matching how a missing `window_end` is already handled.
  Nullable `boolean` columns with `pd.NA` are accepted instead of raising.
- 2026-09-25 (C5 evaluation): recall@k uses `k = ceil(frac * n)` (at least 1, capped at
  n) so a small test slice still has a non-empty head; head counts larger than the sample
  are capped at n. Ranking is score descending with NaN scores last and a deterministic
  tie-break (ascending `tie_breaker`, typically `cert`, else stable input order); for the
  AUC metrics NaN scores are replaced by a value below the smallest real score so they
  count as "least risky" rather than raising. `evaluate_by_year` returns `year` as a
  string column so the `pooled` row fits. Lead time counts calendar quarters between the
  first flagged report date and the failure date. Figures strip the `Software` and
  creation-time metadata so re-runs are byte-identical.
- 2026-09-25 (C2 features): `wholesale_funding_ratio = (othbor + bro) / asset`, without
  adding `othbfhlb`: the FDIC defines "other borrowed money" as *including* FHLB advances
  (and in the panel `othbor == othbfhlb` on 83% of rows, never meaningfully below it), so
  adding the advances again would double count them. Concentration ratios divide by `rbc`
  (total risk-based capital, dollars) and fall back to `rbct1j + lnatres` when `rbc <= 0`,
  which covers the 45,601 CBLR-filer rows that report `rbc = 0`. `cre_to_capital` uses the
  non-owner-occupied split `lnrenrot` where reported (2007 onward) and total `lnrenres`
  before that. `texas_ratio` and `reserve_coverage` are capped at 10; a computed Texas
  ratio above 10 is clipped and flagged like a non-positive denominator. Missing `intan`
  counts as zero (0.25% of rows). Management growth proxies live in `features/structure.py`
  (no separate module). Rows with `asset <= 0` (none in the panel) are kept, with NaN
  ratios, so `features_v1` stays one-to-one with `panel`; any exclusion is a modelling step.
- 2026-09-25 (C2 features): `total_rbc_ratio` treats `rbcrwaj == 0` as not reported (NaN,
  `total_rbc_ratio_missing = True`). The API never returns null for the ratio: all 45,423
  CBLR-filer rows (2020+) and roughly 70-120 rows a year before that carry an exact 0, which
  no going-concern bank has. Negative ratios (a real, insolvent bank) are kept as signal.
- 2026-09-25 — **Baselines (C4):** the `Winsorizer` leaves binary (0/1) columns and
  all-missing columns unclipped: a rare flag such as `texas_ratio_capped` (0.15% of rows)
  would otherwise be clipped to all zeros at the 99.5th percentile. The Texas-ratio
  baseline is a one-step pipeline (no winsorise/impute/scale): the ratio is already capped
  at 10 and clipping its top half-percent would tie exactly the banks a ranking metric
  cares about. Missing Texas ratio scores -1, below every real value. Artefacts for the
  primary horizon live in `models/<name>/`; other horizons get `models/<name>_<H>q/` and
  `reports/p1_baselines_<H>q.md`. `bankcanary evaluate` re-scores the saved pipeline on the
  test split instead of reading `metrics.json` back, so the artefact itself is checked.
- 2026-09-25 — **`.gitignore` anchoring (C4):** the unanchored `models/` pattern also hid
  the new `src/bankcanary/models/` package from git, so it is now `/models/` (repo-root
  artefact directory only). `data/` is left as is; no source directory carries that name.
- 2026-09-25 — **Regularised logit below the Texas ratio (C4):** on the fixed 4q split
  the all-feature logit (C=1.0, balanced) scores PR-AUC 0.20 against 0.37 for the Texas
  ratio and 0.37 for the six-feature logit. Coefficients show the four collinear capital
  measures with large, sign-flipped weights. Stronger L2 helps but does not close the gap
  (PR-AUC 0.20 at C=0.1, 0.22 at C=0.01, 0.26 at C=0.001), so C stays at the contract's
  1.0 and the spec acceptance criterion "logit beats Texas on PR-AUC" is open for the
  feature-selection / gradient-boosting steps rather than patched here.
- 2026-09-25 — **Foundation notebook (C6):** `notebooks/01_foundation.ipynb` is generated
  by `scripts/make_notebook_01.py` (nbformat) and executed in place, so the committed
  notebook is reproducible from source and its outputs come from the same cached tables
  as the reports. It never retrains: the baseline table and the PR curve come from
  `evaluate_model`, which re-scores the saved pipelines on the test split in about a
  second. The first cell adds `src/` to `sys.path` because the editable install's `.pth`
  is not honoured in this checkout (`uv run bankcanary` needs `PYTHONPATH=src`); the
  README's Reproduce block records the same workaround.
- 2026-09-25 — **Acceptance evidence re-measured, root cause of the `.pth` gotcha:** the
  five Prototype 1 criteria in `docs/P1_CHECKLIST.md` were re-verified from the commands
  themselves (hashes before/after a full `build-panel`/`build-labels`/`build-features`
  re-run, DuckDB counts, `metrics.json`, a timed `nbconvert --execute`) rather than from
  the reports; the logit-versus-Texas criterion stays unticked. The `ModuleNotFoundError`
  behind the `PYTHONPATH=src` workaround is not an install problem: every file under
  `.venv/` has the macOS `hidden` flag set, and Python 3.12's `site.addpackage` skips
  hidden `.pth` files (`site._trace` prints "Skipping hidden .pth file"). `chflags
  nohidden` on the `.pth` files fixes it, but the flag was re-applied within minutes, so
  the `PYTHONPATH=src` prefix remains the documented way to run the CLI and the tests.
- 2026-09-25 — **All-feature logit re-specified (P1 gap review):** the contract's
  `class_weight="balanced"` was the cause of the 0.20 PR-AUC, not the feature set. With
  554 training failures against 252k rows, balanced weights scale each failure by about
  450, so a handful of failed banks set the capital coefficients (|coef| up to 3.7 with
  opposite signs on collinear measures). `scripts/tune_logit_c.py` fits every candidate
  on an inner split carved from the training period (spec rule 6.7: validation = usable
  reports 2007Q1-2008Q4, 509 positives; inner training = windows closed before
  2007-05-30, 148,198 rows / 37 positives) and never reads the test years. Validation
  PR-AUC: unweighted C=0.003 0.246, C=0.0003 0.246, C=0.001 0.240, C=0.01 0.193,
  C=1 0.028; every balanced variant is below 0.035. The winner (unweighted, C=0.003) is
  `bankcanary.models.baselines.LOGIT_C`; on the untouched test split it scores PR-AUC
  0.3867 against 0.3726 for the Texas ratio, so the spec acceptance criterion is met
  without tuning on the test years. `logit_small` keeps C=1, balanced: its odds ratios
  are the point of that model and it already ties the Texas ratio. Contract section 8
  updated accordingly.
- 2026-09-25 — **Sensitivity and per-event evaluation (spec 5 rules 3 and 6):** every
  training and evaluation run now also reports the test metrics with the
  `censored_in_window_Hq` rows dropped (3,877 of 118,696 test rows for 4q; PR-AUC rises
  by about 0.01 for every model because the dropped rows are all negatives) and per
  failure event: positive rows sharing `(repdte, rssdhcr, fail_date)` collapse into one
  unit scored by the best-ranked sister bank, so a holding company whose subsidiaries
  fail on the same day counts once. In 2010-2013 that is 12 two-bank events (821 events
  for 833 bank-quarters), which moves PR-AUC by less than 0.002. `load_training_frame`
  left-joins `rssdhcr` and `fail_date` from `panel` for this; they are never features.
  Both blocks are stored in `metrics.json` (`sensitivity_censored_dropped`, `per_event`)
  and in `reports/p1_baselines.md`.
- 2026-09-25 — **Figures and the `.pth` workaround:** `bankcanary evaluate` now writes
  `reports/figures/` (PR curve and score histogram per model, one recall@k chart; about
  140 KB, committed). `[tool.pytest.ini_options].pythonpath = ["src"]` makes `uv run
  pytest` independent of the editable-install `.pth`, whose macOS `hidden` flag keeps
  being re-applied to everything under `.venv/` in this checkout (not by `uv`: a plain
  `uv run` leaves a cleared flag alone, but files `uv` writes into `.venv/` come back
  hidden within minutes). `make fix-venv` (`chflags -R nohidden .venv`) restores
  `uv run bankcanary` until it happens again; `PYTHONPATH=src` remains the fallback.
- 2026-09-25 — **Hidden-flag import failure, root cause and fix:** every agent hit
  `ModuleNotFoundError: No module named 'bankcanary'` under `uv run`. Cause: something on this
  Mac recursively sets the `UF_HIDDEN` flag on dot-directories under the working tree (`.venv`
  included, re-applied whenever the top-level `.venv` entry changes, e.g. on a `uv` re-sync), and
  Python 3.12+ deliberately skips hidden `.pth` files (`python -v` prints
  "Skipping hidden .pth file"). `uv` and `git` are not the cause (a plain `uv sync` does not
  re-hide). Fix: `pytest` gets `pythonpath = ["src"]`, and `scripts/fix_venv.py` writes a
  `sitecustomize.py` into the environment (normal imports ignore the flag) and clears the flags;
  verified with the whole `.venv` flagged hidden. `make setup` runs it; note that `make` itself
  needs the Xcode licence accepted on this machine.
- 2026-09-25 — **Owner decisions after Prototype 1 review:** the repository is PUBLIC from now on
  (MIT licence added, data terms stated in the README); experiment tracking is a plain JSON
  `runs/` directory (Decision Point 3, default); the project keeps the name BankCanary
  (Decision Point 6). Decision Point 1 stays at the 2001Q1 default.
- 2026-09-25 — **FFIEC raw Call Reports are not needed for the Prototype 2 features.** The FDIC
  `/financials` endpoint already carries AFS and HTM securities at amortised cost and fair value
  (`SCAA`, `SCAF`, `SCHA`, `SCHF`; 100% populated from 2015, ~90% for the two "cost" items
  before) and the FDIC's estimated uninsured deposits (`DEPUNINS`, populated for ~98.5% of banks
  of every size). Silicon Valley Bank's 2022Q4 values match its 10-K to the million (HTM
  amortised cost $91.3B vs fair value $76.2B; AFS $28.5B vs $26.0B; uninsured deposits $151.6B
  of $175.4B). Following spec §3.2 ("if the FDIC API has them, prefer it"), step D1 becomes a
  cross-check plus the `IDRSSD ↔ CERT` crosswalk; the FFIEC bulk downloader is optional.
- 2026-09-25 — **FRED without a key:** the FRED API requires a key (owner action), but
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>` serves the same observations
  without one. The FRED client uses the API when `FRED_API_KEY` is set and the CSV endpoint
  otherwise, with the same on-disk cache. Data vintages are not modelled (noted limitation).
- 2026-09-25 — **Gradient-boosting backend and dropped lifelines.** LightGBM and XGBoost wheels
  both dynamically link `libomp.dylib`, which on this Mac only Homebrew provides, and Homebrew
  is blocked until the Xcode licence is accepted (`sudo xcodebuild -license accept`, owner
  action). scikit-learn ships its own OpenMP runtime, so `HistGradientBoostingClassifier`
  (the same histogram-based algorithm, with monotone constraints, native missing values and
  SHAP `TreeExplainer` support) is the fallback backend behind `models.gbdt.make_gbdt`;
  LightGBM is used automatically once it imports. `lifelines` was removed because it pins
  `pandas < 3` and had silently downgraded pandas to 2.3 (one dtype test failed); the optional
  Cox model (spec §8.1 item 6) is deferred, the required discrete-time hazard model is not.
- 2026-09-25 — **D1 cross-check and FFIEC downloader.** `crosswalk_rssd` is a projection of
  `institutions` (27,834 certs, 267 with `FED_RSSD` = 0, stored as null). The FFIEC bulk page
  works with plain `httpx`: GET the form, POST the product selection (`__EVENTTARGET` =
  the product list box) to fill the period list, whose option values are opaque ids
  (`12/31/2022` → `135`, not the date), then POST period id + `TSVRadioButton` + the Download
  button; the 2022Q4 bundle is 6.9 MB zipped, not 50-150 MB. Schedules RC-B and RC-O (and
  others) are split into `(1 of 2)` parts joined on `IDRSSD`; cells marked `CONF` are
  confidential and parsed as missing. All fifteen FDIC values for SVB, Signature and First
  Republic at 2022-12-31 equal the RC-B/RC-O items to the dollar, so the FDIC fields are used
  directly (spec §3.2). Note that `RCON1773` (domestic offices only) differs from `RCFD1773`
  for SVB (21.98B vs 25.98B): the consolidated `RCFD` item is the one that matches `SCAF`.
- 2026-09-25 — **macro_state (step D4a).** Publication lags are applied to the period *end*
  (FRED dates are period starts): state unemployment 45 days after month end, state HPI 75
  days after quarter end, FEDFUNDS/T10Y3M/DGS10 one day. Missing observations (`.`) are
  dropped so the latest published value carries; the boundary is inclusive (period end +
  lag == avail_date is usable). Four-quarter changes re-evaluate the series at
  `(repdte − 4 quarter ends) + availability_lag_days`, never at the row four positions
  earlier, so gaps in a state's panel grid cannot shift the comparison. `fedfunds` and
  `dgs10` levels are stored beside the contract's five columns because they are free and
  step D5 may want them. Rows with a null `stalp` (102 panel rows) are excluded from the
  grid; the five territories in the panel (AS, FM, GU, PR, VI) keep the national columns
  only. Only current vintages are used (no ALFRED): a known limitation.
- 2026-09-25 — **D2 rate/run-risk features and the per-module registry.** Every feature module
  now owns its `SPECS` and `build(panel, **deps)`; `registry.py` fixes the module order per
  version (`v1`: capital, asset_quality, management, earnings, liquidity, concentration,
  structure; `v2` appends sensitivity, run_risk) and `build.py` runs them in that order,
  handing each module the columns built so far as `deps["features"]`. The three growth ratios
  got their own `management` module only because the P1 table put them between asset quality
  and earnings: keeping them in `structure` reordered `features_v1`, and the byte-identical
  parquet (sha256 `8a4e5874…f070cd`, checked against a build with the pre-refactor code in the
  same environment) is the reproducibility proof. The primitives (`FeatureSpec`, `safe_ratio`
  ...) moved to `features/spec.py` so the modules can import them while the registry imports
  the modules; `registry` re-exports them. `FeatureSpec.monotone` is stated in words at the end
  of every explanation. Signs that are deliberately 0: growth (both tails are risky), `nim_q`
  (high margins can mean high-yield lending), loan-mix shares other than construction and
  nonfarm nonresidential, the CBLR missing flag and the one-hots. `adjusted_tier1_leverage`
  divides by period-end `asset` (the panel has no quarterly-average assets), deducts losses
  only and leaves gains out. `features_v2` is written with an explicit key (`TABLE_KEYS` in
  `storage/parquet.py` untouched). SVB 2022Q4: `unrealized_loss_to_tier1` = -1.041,
  `uninsured_share` = 0.864; Signature and First Republic sit inside the >$10B peer band on
  unrealised losses but at the top of it on uninsured deposits (reports/figures/svb_unrealized_losses.png).
- 2026-09-25 — **D3 trend, persistence and regional structure features.** `trends` reads the
  eight ratios from the frame built so far (`deps["features"]`), never the panel, so a trend
  is by construction the difference of the registered ratio. Both differences and the
  persistence windows are measured in calendar quarter-ends via `features.ytd.lag` (exact
  date match): a filing gap gives NaN for `d1q`/`d4q`, is skipped (not back-filled) by the
  window counts and breaks `consecutive_loss_quarters`, which is computed by a sorted
  run-length pass and is uncapped (max 102 in the panel). `neg_roa_quarters_last_8` is NaN
  when fewer than four of the eight quarter-ends are reported; `noncurrent_rising_quarters_last_4`
  is NaN only when none of the four one-quarter changes exists. A NaN ROA counts as "not a
  loss". Trend signs inherit the level's sign (`unrealized_loss_to_tier1` is negative for a
  loss, so its trends carry -1); all three persistence counts are +1. Regions are the four
  Census Bureau regions (DC in the South, as Census does) plus `region_other` for AS, FM, GU,
  PR, VI; a null `stalp` leaves every region column False. `is_community_bank` is the FDIC `cb`
  flag (null -> False) with sign 0. The D2 test fixture gained `stalp`/`cb` and its two
  "v2 == v1 + P2 names" assertions became prefix checks so later modules can append.
  Rebuilt `features_v2`: 710,691 rows x 77 features (34 P2) in 11 s. SVB 2022Q4:
  `d4q_unrealized_loss_to_tier1` = -0.964.
- 2026-09-25 — **D4b macro features joined into `features_v2`.** `features/macro.py` is a pure
  join of `macro_state`: the three state series on `(stalp, avail_date)` and the three
  national series (`t10y3m`, `dgs10`, `fedfunds_change_4q`) on `avail_date` alone, so the
  102 null-`stalp` rows and the territories keep the national columns while their state
  columns stay NaN (1,423 rows, 0.20 % of the table; the trees read the gap as its own
  signal). `macro_dgs10` is registered beside the contract's five columns because the level
  is free and step D5 may use it. Signs: unemployment level and change +1, HPI change -1,
  curve slope -1 (inversion precedes recessions and squeezes maturity transformation), fed
  funds change +1 (the 2023 mechanism), `dgs10` 0 (both high and low levels have coincided
  with crises). `build_features` reads `macro_state` from Parquet only for a version whose
  module list includes `macro` and only when no frame is passed, so v1 builds and the unit
  tests (which pass a four-row `MACRO_STATE` fixture) never touch `data/`. The D2 fixture
  gained `avail_date` and the two existing v2 layout tests now pass that fixture; the D3
  suffix assertion became a contiguous-slice check since macro now ends the table. Rebuilt
  `features_v2`: 710,691 rows x 83 features (40 P2) in 12 s; `features_v1` sha256 unchanged.
  SVB 2022Q4: `macro_t10y3m` = -0.96, `macro_fedfunds_change_4q` = 4.49.
- 2026-09-25 — **D5 registry audit and generated feature docs.** `docs/FEATURES.md` is
  written by `scripts/write_feature_docs.py` from the registry (one table per group in
  CAMELS order, then concentration, trends, structure, macro) and
  `tests/test_registry_audit.py` fails when the committed file differs from the current
  render, so the documentation cannot drift from the code. The explanation shown in the
  table is the registry text minus the trailing monotone sentence, which the `sign`
  column already carries. The audit adds a plain-English rule on top of the contract:
  an explanation may not contain a raw Call Report code (any `column` in
  `config/fields.yaml`, except the English words `asset`/`cb`) unless it is also a
  registered feature name; that flagged `wholesale_funding_ratio` (mentioned `othbor`)
  and the eight `bkclass_*` one-hots, which now spell out the charter class
  (`spec.BKCLASS_NAMES`). Only explanation strings changed, so `features_v1` /
  `features_v2` need no rebuild. Identifier independence is tested by relabelling every
  `cert` bijectively, renaming the bank and shuffling rows: `build_features("v2")` must
  return the same frame. `(same cert)` in the trend formulas is accepted as the grouping
  note. `reports/features_v2_summary.md` (`--summary`) reads the Parquet tables directly
  and reuses the `failure_contrast` filter (label-complete, not dropped) for the P2
  medians; it renders in under two seconds.
- 2026-09-25 — **D6 gradient boosting, inner-validation tuning and JSON run tracking.**
  `lightgbm` 4.7 imports and fits on this Mac, so `settings.models.gbdt.backend = lightgbm`;
  `make_gbdt` still carries the `sklearn` `HistGradientBoostingClassifier` fallback (with
  `early_stopping=False`, since its automatic validation split would tune the iteration
  count on rows the tuning script never sees). `scripts/tune_gbdt.py` reuses the
  `tune_logit_c.py` inner split (validation 2007Q1-2008Q4, inner training windows closed
  before 2007-05-30) and scores the full grid, learning rate {0.03, 0.1} x leaves
  {15, 31, 63} x min leaf {50, 200} x trees {200, 400}, for both variants, 49 fits in three
  calls of under two minutes; each configuration is a `runs/tune_gbdt/` record so a re-run
  skips it. The inner slice holds only 37 training positives (2002-2005 had almost no
  failures), which explains two things: learning rate 0.1 collapses to the base rate
  (PR-AUC 0.006-0.03, the trees memorise the 37 rows), and `min_samples_leaf = 200` beats
  50 everywhere. Winner: unconstrained, lr 0.03, 63 leaves, min leaf 200, 200 trees, inner
  PR-AUC 0.2299 (monotone at the same parameters 0.1968; `logit_v2` at the P1 C 0.2589), so
  `settings.models.gbdt.monotone = false` until the owner rules on Decision Point 2. On the
  untouched test split the order flips: `gbdt_mono` 0.4755 PR-AUC / 0.8019 recall@2 %,
  `gbdt` 0.4324 / 0.7791, `logit_v2` 0.4437 / 0.7815, Texas 0.3726 / 0.7611. The
  constraints act as a regulariser that the 37-positive inner slice cannot reward, so the
  inner ranking is weak evidence; the walk-forward backtest (D8) is the right place to
  settle the point. The tuning grid keeps the requested size (nothing was shrunk).
  Run tracking (`bankcanary.tracking`): `run_id = <name>-<horizon>q-<sha1(canonical
  config)[:10]>`, files under `runs/<name>/<run_id>/{config,metrics}.json`, index rows
  deduplicated by id and carrying only scalar metrics; `runs_dir` joined `Settings`.
  `models/texas` is not rewritten by `train-gbdt` (the Texas ranking needs no fit and the
  P1 artefact keeps its v1 config); the v2 logit is saved as `models/logit_v2`.
  Top gain features of `gbdt`: `d4q_texas_ratio`, `d4q_equity_to_assets`,
  `construction_to_capital`, `texas_ratio`, `equity_to_assets` (the four-quarter trend
  features from D3 carry the most split gain, ahead of every level).
- 2026-09-25 — **D7 discrete-time hazard model.** `horizons_quarters` is now `[1, 4, 8]`:
  the labels table gains `y_1q, window_end_1q, censored_in_window_1q, label_complete_1q`
  (window `(avail_date, avail_date + 3 months]`, 572 positives among 705,968 usable
  rows, 0.081 %); the twelve pre-existing columns are byte-identical to the previous
  Parquet (checked column by column), only the file hash changes because of the four
  new columns. The hazard (`bankcanary.models.hazard`) is the P1 pipeline with an
  *unweighted* logit on all 83 `features_v2` features and target `y_1q`; non-failure
  exits are censoring because a bank has no rows after it leaves, so nothing is deleted
  or re-weighted. `C` is tuned like `tune_logit_c.py` but at 1q (`train-hazard --tune`,
  one `runs/tune_hazard/` record per candidate, resumable): the inner slice holds only
  13 one-quarter positives, so the winner `C = 0.0003` (4q validation PR-AUC 0.2844) is
  statistically tied with 0.003 (0.2837); the pre-stated rule (highest 4q PR-AUC, the
  horizon the model is compared on) was applied mechanically. Conversion
  `p_Hq = 1 - (1 - h)^H` assumes the hazard persists at its current level (covariates
  are not projected); ranking metrics are invariant to it, Brier is reported on the
  converted probabilities. Fixed split: 1q test PR-AUC 0.2598; converted 4q PR-AUC
  0.3633 / Brier 0.00643 against `logit_v2` 0.4437 / 0.00506 and `gbdt` 0.4324 / 0.00518
  (75 one-quarter training positives against 554 four-quarter ones explain the gap);
  converted 8q PR-AUC 0.3948 / Brier 0.00959 *beats* `logit_v2` at 8q (0.3764 / 0.00838)
  while `gbdt` collapses at 8q (0.1328, ROC 0.9029; its 8q training set closes at
  2007Q4 under rule 6.2, which is worth a look in the D8 walk-forward). The 8q
  comparators are saved as `models/logit_v2_8q` and `models/gbdt_8q`. Odds ratios come
  from an unpenalised `statsmodels` Logit of `y_1q` on the standardised design of the 15
  interpreted features alone (six small-logit features plus the nine P2 rate/run
  features), standard errors clustered by `cert`, no missing indicators because P2
  ratio pairs sharing a denominator go missing together and their indicators would be
  collinear; the L2-shrunk coefficient of the full hazard is shown next to each one
  (at `C = 0.0003` they sit near 1, which is the shrinkage, not the evidence). The
  hazard artefacts live in `models/hazard/` without a horizon suffix, and successive
  `train-hazard --horizon H` calls merge their converted block into its `metrics.json`
  (a re-fit keeps earlier blocks) so `reports/p2_hazard.md` is re-rendered from disk.
- 2026-09-25 — **D8 walk-forward harness.** `bankcanary walkforward --year Y` fits every
  model for one test year (about 30 s per year for all four, so no gradient-boosting
  iteration cap was needed; `--gbdt-iterations` exists and is recorded in `config.json`
  as `iterations_capped` should a larger panel need it). Rule 6.2 is applied at the label
  a model is *fitted* on: the hazard trains on `training_mask(labels, 1, first quarter
  of Y)` (three more quarters than the 4q mask, leak-free for its 1q event) and is scored
  with `1 - (1 - h)^H` on the horizon's test rows; every other model trains at the
  scoring horizon. The Texas ranking is "fitted" on the same rows so that every year
  directory loads through one `load_year`. Horizon-8 artefacts live in
  `models/walkforward/<Y>/<model>_8q/` (same suffix rule as `models/<name>_8q`). The
  `walkforward_scores` table is rebuilt from `data/walkforward/*.parquet` on every call
  (files read in name order, rows sorted by the table key), never appended to. Pooled
  rows rank the raw scores of seventeen yearly models as one list, as spec 8.2 asks;
  because a booster's score scale drifts from year to year that pooling costs `gbdt`
  more than the logits (pooled 4q ROC-AUC 0.85 against per-year values above 0.95 from
  2010 on), and 2008's booster is weak on its own (PR-AUC 0.18: its training window
  closes at 2006Q4 with 37 positives, the same starvation seen in the D6 inner slice).
  2021 has no 4q failure, so its per-year metrics are undefined. Pooled 4q results:
  logit PR-AUC 0.3843 / recall@2% 0.7713, gbdt 0.3085 / 0.6743, hazard 0.3068 / 0.6881,
  texas 0.2606 / 0.7437; the walk-forward logit is the best model and the P2 candidates
  do not beat it here. The P1 fixed-split logit (0.3867 / 0.7143) is a different test
  period and is only quoted for orientation. At 8q logit 0.2835 / 0.5488 against gbdt
  0.1169 / 0.2710, consistent with the D7 finding that the booster collapses at 8q.
- 2026-09-25 — **Walk-forward hyper-parameters are re-selected per test year (rule 6.7).**
  The D6/D7 constants (`LOGIT_C`, `HAZARD_C`, `settings.models.gbdt.params`) were chosen on
  reports 2007Q1-2008Q4, which is the 2008 test year itself and lies past 2009's training
  cut. `evaluation.walkforward.inner_masks` now carves, for every test year, a validation
  slice from the last 8 report quarters of that year's own `training_mask` at the scoring
  horizon (so every validation outcome was known on the year's first prediction date),
  widened backwards a year at a time while the slice or the inner training rows hold
  fewer than 5 failures; inner models train on windows closed before the slice
  (`assert_no_leakage`). `tune_year` ranks the D6/D7 `C` grid (logit, hazard; the hazard
  selected on its converted 4q PR-AUC) and a 2x2x2 booster grid (learning rate, leaves,
  leaf size; iteration count and the monotone decision stay in settings) by validation
  PR-AUC, caches every candidate under `runs/tune_walkforward/`, and `fit_year` fits the
  winner, recording the slice and choice in `config.json["tuning"]` and `tuning.json`.
  When no width reaches 5 failures the most regularised grid point is used and flagged
  (`fallback`), never a value chosen on later data; real data never needs it. From 2010
  on the slice coincides with the D6 one (2007Q1-2008Q4, inner cut 2005Q4). 2008 tunes on
  2005Q1-2006Q4 (9 failures) with inner training through 2003Q4. Runs are one model per
  CLI call (`--model logit,hazard` then `--model texas,gbdt`), about 30-45 s per model-year
  on an idle machine. Regeneration of the 4q artefacts reached 2019 before this note;
  the remaining years and the 8q horizon are re-fitted in the follow-up step.
- 2026-09-25 — **Sensitivity analyses refit on the P1 fixed split, not the walk-forward.**
  `evaluation.sensitivity` refits `logit` (v2 features, `LOGIT_C`) and `gbdt`
  (`settings.models.gbdt`) once per variant through `fixed_split_masks` +
  `assert_no_leakage`, with the fixed-split constants (chosen inside this split's
  training period) rather than the per-year tuning, so a cell differs from its baseline
  by the one assumption under test only. The baseline cells (4q, censored kept, 60-day
  lag) reproduce the D6 `train` runs exactly (logit 0.4437 / 0.7815, gbdt 0.4324 /
  0.7791). The censored variant drops `censored_in_window_Hq` rows from *both* training
  and test (spec 5 rule 3), unlike the `sensitivity_censored_dropped` block of the
  train runs, which drops them from the test side only. The lag variants relabel in
  memory through `build_labels(..., lag_days=L)` (new optional parameter; the panel's
  `avail_date` is overridden by `repdte + L`, the `labels` table is never rewritten) and
  recompute the split with a settings copy carrying that lag; because the rule 6.2 cut
  and the windows shift together, the training rows are the same reports at every lag
  while the positives (537 / 554 / 606 at 45 / 60 / 90 days) and the test failures
  (851 / 833 / 809) move with the windows. No artefacts go to `models/`; the
  `runs/sensitivity/` record (id from analysis, variant, model, horizon, lag) is the
  artefact and `reports/sensitivity.md` is rebuilt from those records. Findings: 8q
  costs the booster far more than the logit (PR-AUC 0.13 against 0.38), dropping
  censored rows lifts every metric slightly (rescued banks are hard negatives), and a
  90-day lag costs the booster 0.07 PR-AUC while the logit loses 0.02; the logit stays
  the less sensitive model and the PR-AUC ranking never flips.
- 2026-09-25 — **SHAP drivers (D10) are computed per walk-forward year with that year's
  booster, and the quarters past the backtest with the newest one.** `explain.shap_drivers`
  runs `shap.TreeExplainer` on the `gbdt` estimator after the pipeline's own fitted
  `Winsorizer` has clipped the rows (so `feature_value` is what the trees saw); the values
  are log-odds contributions that sum with the expected value to the raw score (checked to
  1e-13 on 2024). The `drivers` key `(cert, repdte, model, rank)` uses ranks 1-5 for the
  largest positive contributions (`raises`) and 6-10 for the most negative (`lowers`); a
  contribution of exactly zero is never a driver, so rows can be fewer than ten. `model` is
  `gbdt` for test-year rows and `gbdt_production` for the rows after the last complete
  test year (2025Q1 onward, scored by the 2024 booster, never evaluated); `model_year`
  records which booster explained the row. Only the 4q boosters are explained: the 8q
  booster collapses (D8) and the table key carries no horizon. Per-year files live in
  `data/drivers/<Y>_gbdt.parquet` and the table is rebuilt from them like
  `walkforward_scores`. Each year logs an `explain` run whose metrics hold the mean |SHAP|
  per feature, and `reports/shap_summary.md` is written from those records (the beeswarm
  recomputes the latest year, about 6 s) rather than from stored SHAP matrices. One year
  takes about 10 s on this machine, so `explain --all` (17 years + production) exceeds the
  two-minute budget; the artefacts were produced with `--year Y --no-rebuild` in batches of
  six and `--latest` last. Rule 6.6 result: no feature exceeds 40 percent of the pooled
  mean |SHAP| (`texas_ratio` leads at 7.2 percent, top three 17.6 percent); the 2008-2010
  boosters lean on `macro_hpi_change_4q` and construction shares, 2022-2024 on
  `macro_fedfunds_change_4q`, `adjusted_tier1_leverage` and `unrealized_loss_to_tier1`.
- 2026-09-25 — **Isotonic calibration (D9) follows the contract's inner-model recipe and is
  reported as it comes out, including where it hurts.** For test year Y and each probability
  model, `evaluation.calibration.calibration_masks` takes the last calendar year whose four
  report quarters all lie in Y's `training_mask` at the scoring horizon as the calibration
  slice (widened backwards a year at a time while it or the inner rows hold fewer than 5
  failures), fits an inner model with Y's tuned hyper-parameters on the rows whose windows
  closed before the slice's first prediction date (`assert_no_leakage` re-checked), scores
  the slice, fits `IsotonicRegression(out_of_bounds="clip")` on those pairs and applies the
  map to the full-window model's year-Y scores (`calibration.joblib` next to the model,
  `score_calibrated` in the per-year file, `runs/calibrate/` keyed by slice bounds and
  params). The hazard's inner model fits at 1q and its slice scores are converted with
  `1 - (1 - h)^4` before the map. Result: the map is learned on the inner model's score
  scale and applied to a model that has seen one more year (for 2010 the 2008 crisis year
  and a re-tuned C), so where the scales differ the calibrated probabilities inherit the
  inner model's plateaus: pooled 4q Brier goes from 0.0041 raw to 0.0189 (logit) and
  0.0545 (hazard), gbdt is unchanged (0.00425 against 0.00435), and mean calibrated
  probabilities overshoot the 0.49 percent failure rate (2.9 percent for the logit). The
  raw outputs remain the better probabilities on Brier; `score_calibrated` is kept as the
  contract specifies and the reliability tables show the mismatch decile by decile. The
  metrics suite lives in `evaluation.metrics_report` (a new module so that `walkforward.py`
  and `calibration.py` stay fit-only): `bankcanary metrics-report` regenerates
  `reports/walkforward.md` in about 35 s from `walkforward_scores` and the panel's
  `fail_date`, superseding `walkforward-report`, and logs a `runs/metrics/` record per
  model and horizon with the pooled headline numbers. Confidence intervals are percentile
  intervals from 200 cluster-bootstrap draws that resample certs (seed 20080101) with row
  weights equal to the draw counts, so PR-AUC is a weighted average precision and recall@2%
  takes the head that holds 2 percent of the resampled weight; years with fewer than 10
  failures are flagged low confidence rather than dropped. Lead time flags the top 2 percent
  of each report quarter's ranking and counts calendar quarters to `fail_date`; the 2009-2012
  cohort (440 failed banks) is the headline because 2008 failures can only be flagged inside
  2008 and post-2024 failures are scored only through 2024. On that cohort the logit flags
  89.8 percent of the failed banks before failure, median lead 5 quarters, 87.7 percent at
  least 2 quarters ahead (hazard 5 / 88.4 percent, gbdt 4.5 / 81.6 percent, Texas 4 / 82.5
  percent). The per-year retune of the walk-forward fits changed the pooled ranking the D8
  report had shown: at 4q the hazard now pools best (PR-AUC 0.326, CI [0.293, 0.357]) ahead
  of the logit (0.307) and the booster (0.281), with overlapping intervals for the top two.
- 2026-09-25 — **Notebooks 02 and 04 are built from `scripts/make_notebook_0{2,4}.py` and
  executed in place; they read tables, `models/` and `runs/` only and never retrain.** 02
  recomputes the walk-forward tables from `walkforward_scores` through
  `metrics_report.ci_table` (about 17 s for four models) rather than parsing
  `reports/walkforward.md`, so the notebook and the report cannot drift; the chart keeps the
  failure counts in a second panel rather than on a second axis. The Decision Point 2
  section reads `settings.models.gbdt.inner_pr_auc` for the inner-validation column and
  states both orderings without recommending one. 04 defines a false positive at the
  bank-year level (a flagged `y = 0` bank kept once per test year at its highest-scored
  quarter) and follows it eight quarters from `avail_date`, mirroring the label windows;
  acquisitions are `exit_reason` in {merger, absorption, affiliated_merger, consolidation}.
  Result for 2009-2012 (875 bank-years): 26 percent failed later (17 percent in quarters
  5-8), 10 percent were acquired within two years (9 percent by recorded merger against a
  5 percent base rate over every bank-year), 62 percent were still open after eight
  quarters. Gotcha for readers: the 2010 booster saturates (68 bank-quarters score above
  0.999, 44 of them failed), so the very top of the 2010 ranking is ordered by `cert`. The
  builders carry a file-level `noqa: E501` because the embedded cell sources are kept as the
  notebook shows them.
- 2026-09-25 — **The 2023 case study (D11b) fits four models on one rule-6.2 cut and
  reports the ranks as they come out.** `evaluation.case_study_2023` fits the P1 logit
  (`LOGIT_C`) and the tuned booster (`settings.models.gbdt`, unconstrained) on two views of
  the same rows, `credit_only` (the 43 `features_v1` columns) and `rate_aware` (all 83
  `features_v2` columns), over every bank-quarter whose 4q window closed before the 2022Q4
  prediction date (`training_mask(labels, 4, "2022-12-31")` + `assert_no_leakage`: reports
  through 2021Q3, 622,341 rows, 2,226 failures; each fit takes 4-8 s, so no subsampling),
  then scores the 2022Q3, 2022Q4 and 2023Q1 reports of every bank, post-failure filings
  included, because a supervisor ranks the reports that exist. Nothing is re-tuned on the
  2022-2023 rows. Run records go to `runs/case_study_2023/` (name, view, model, cut, split
  facts; metrics = the tracked banks' rank/percentile/probability per quarter plus the
  ranking metrics on the 13 complete-label failures among the scored rows, context only).
  Drivers are SHAP for the booster and coefficient x standardised value for the logit, both
  on the pipeline's own transformed inputs so they sum to log-odds minus baseline. Result:
  credit-only models put SVB at the 65th-70th percentile at 2022Q4 (its Texas ratio was
  0.009); the rate-aware booster lifts it to rank 245 of 4,773 (94.9th percentile, fourth
  of the 34 banks above $100B) on `adjusted_tier1_leverage` = -0.33 (+2.1 log-odds; 116
  training bank-quarters had negative capital net of securities losses and 55 failed). The
  rate-aware logit does not help (61st percentile): over 2001-2021 `uninsured_share` and
  `macro_fedfunds_change_4q` enter with *safer* signs (failed banks averaged 13 percent
  uninsured deposits against 20 percent for survivors; SVB's fed-funds change of 4.49 is
  outside the training range, max 2.02) and offset the loss terms. Signature is flagged
  only by the credit-only logit (rank 113, on C&I share, CRE at 3x capital and multifamily,
  a 2008-shaped concentration profile) and First Republic by nothing above the 76th
  percentile before it failed (only its post-failure 2023Q1 filing reaches the 95th).
  Notebook 05 renders `runs/sensitivity/` through `sensitivity.read_runs` /
  `analysis_table` and refits nothing; both notebooks are generated by
  `scripts/make_notebook_0{3,5}.py` and executed in place.
- 2026-09-25 — **D12 model card, README results and the Prototype 2 checklist.** The card's
  per-year tables are compressed from `reports/walkforward.md` (PR-AUC with its interval and
  recall@2% per model, one row per year) rather than copied in full, so the report stays the
  source of record; every other number is quoted from the committed reports and run records
  of the same date. Two acceptance criteria are stated with qualifications instead of a
  plain tick: the "best model beats P1's logit" criterion is met on point estimates (hazard
  0.3261 / 0.7042 against the walk-forward logit 0.3066 / 0.6843 on the same rows) with
  overlapping bootstrap intervals, and the P1 fixed-split figure is named as a different
  test period rather than the comparison; the calibration criterion is left unticked because
  the isotonic map only helps the booster (logit and hazard Brier worsen, 0.0041 -> 0.0189
  and 0.0545) and the fix (calibrate on a slice scored by the same model) is deferred to P3.
  The README's Results section replaces the P1 fixed-split table with the pooled
  walk-forward table, keeps the P1 numbers behind links, and states the 2023 finding in two
  sentences without a verdict. The `runs list` CLI named in the contract was never built;
  the checklist says so and points at `tracking.read_index`.
- 2026-09-25 — **Prototype 2 acceptance pass.** The "beats P1's logit" criterion is now measured
  against the comparison the spec means: the P1 learner walked forward on `features_v1`, the
  same rows and years as the P2 models, with each year's `C` reused from that year's v2 logit
  selection (chosen on the training period only, so no tuning was repeated and nothing from
  the test years enters). The scores stayed in a scratch directory (the `walkforward_scores`
  table keys on `model` and would have overwritten the v2 logit rows); the 17 run records are
  logged under `runs/walkforward/` with `features_version = "v1"`. Result: PR-AUC 0.2194 and
  recall@2% 0.5625 with bootstrap intervals disjoint from the hazard's, which settles the
  criterion that the v2-logit comparison in the model card left open. The 328 run directories
  that `runs/index.jsonl` already named but that were never added to git are committed with
  this pass, so the index and the directories agree. `build-labels` and `build-features-v2`
  reproduce `labels.parquet` and `features_v2.parquet` byte for byte (sha256 unchanged).
- 2026-09-25 — **Calibration slice is scored by the model being calibrated (linear models)
  or by an inner model (boosters).** The completeness review flagged the unticked calibration
  criterion. Cause, read from the saved `calibration.json` diagnostics: for 2009 the inner
  hazard model (trained before any crisis failure) scored the 2007 slice with a maximum of
  0.0015, so 37 percent of the 2009 test rows lay above its top threshold and were clipped
  to a plateau of 1.0 (logit: 0.0057 and 13 percent); the map was learned on a score scale
  the full-window model never used. `calibration.fit_calibrator` now takes `slice_scorer`:
  `"full"` scores the slice with the year's own saved model (no refit), `"inner"` keeps the
  old recipe. The default is per model (`SLICE_SCORER_BY_MODEL`): `full` for `logit` and
  `hazard`, `inner` for `gbdt` and `gbdt_mono`, because the boosters' in-sample slice scores
  separate the slice perfectly (slice Brier 0.0000, four isotonic thresholds), which is
  visible on training-period data alone; the linear models' in-sample and out-of-sample
  scales are close (2009 logit: slice maximum 0.19 against a 2009 test range that no longer
  extrapolates). Pooled 4q Brier moves from 0.0189 to 0.0038 (logit) and 0.0545 to 0.0038
  (hazard), both now below the raw 0.0041; 2009 mean calibrated falls from 0.19 / 0.47 to
  0.022 / 0.025 against a rate of 0.021. The booster maps are unchanged, so their existing
  calibration artefacts and run records were kept (their configs predate the
  `slice_scorer` key; a rerun reproduces the same maps under `inner`). `bankcanary calibrate
  --scorer` overrides the default. The contract's "inner model scores the slice" wording in
  section 13 now describes the boosters' path only.
- 2026-09-25 — **Decision Point 2 walked forward: `gbdt_mono` is a walk-forward model.**
  `evaluation.walkforward.MODELS` gains `gbdt_mono` (`GBDT_MODELS = ("gbdt", "gbdt_mono")`);
  `build_model` sets the constraint from the model name, not from
  `settings.models.gbdt.monotone`, so both configurations are fitted with their own per-year
  tuning (same 2x2x2 grid) and logged. Pooled 4q: `gbdt_mono` PR-AUC 0.3138 [0.282, 0.345],
  recall@2% 0.7147 [0.692, 0.739], ROC-AUC 0.8993 against `gbdt` 0.2813 [0.250, 0.311],
  0.6434 [0.609, 0.669], 0.8206; the recall intervals are disjoint and the constrained
  booster wins 11 of the 16 years with failures on PR-AUC. This reverses the inner-validation
  order (0.197 against 0.230) on which the unconstrained configuration was kept. The setting
  is not flipped here: the contract reserves the choice for the owner, and the production
  booster, the SHAP drivers and the 2023 case study keep `gbdt`. Recommendation for the
  owner: adopt the constraints (better pooled ranking, stable scale, explanations that
  cannot contradict the registry's stated directions).
- 2026-09-25 — **Hazard walked forward at 8q.** `models/walkforward/<Y>/hazard_8q/` for 2008
  to 2023 (the 1q hazard converted with `1 - (1 - h)^8`, `C` re-tuned per year on the 8q
  validation slice). Pooled 8q PR-AUC 0.4113 [0.382, 0.442], recall@2% 0.6598 [0.634, 0.688]
  against the logit's 0.2835 / 0.5488: the persistence approximation holds up better at the
  longer horizon than a logit trained on the 8q label, which has to learn from windows that
  span two years of regime. Its isotonic map (full-model scorer) over-predicts 2010-2011
  (pooled Brier 0.0070 raw, 0.0080 calibrated) and is reported as such.
- 2026-09-25 — **Remaining 4q walk-forward years regenerated with per-year tuning.** The
  per-year tuning fix (f6fec75) had left 13 4q artefacts on the earlier fixed
  hyper-parameters (no `tuning` key in `config.json`): 2020 `gbdt` and 2021-2024 `logit`,
  `gbdt`, `hazard`; `texas` needs no fit and `gbdt_mono` was already consistent. Each pair
  was refitted one model per call with `--no-rebuild` and `walkforward_scores` rebuilt once
  from the per-year files; every 4q config now carries `tuning` whose `validation_end`
  precedes the year's first prediction date. Refitted PR-AUC (4q): 2020 gbdt 0.0074 (was
  0.5048 on 4 failures; the slice picked `learning_rate 0.1, num_leaves 63`), 2022 logit
  0.0219 / gbdt 0.0092 / hazard 0.0048, 2023 logit 0.0417 / gbdt 0.1289 / hazard 0.0116,
  2024 logit 0.1117 / gbdt 0.0578 / hazard 0.0283; 2021 has no failures. All of these years
  hold at most 17 failures, so the changes sit inside the low-confidence intervals already
  reported. The superseded `runs/walkforward/` records (80 directories not matching any
  current `models/walkforward` config, among them the 17 v1-comparison runs kept on
  purpose) are left in place: no pruning helper exists in `scripts/` or `tracking`, and the
  index still names them. `reports/walkforward.md` is not regenerated in this pass.
- 2026-09-25 — **8q walk-forward years regenerated with per-year tuning.** `logit_8q` and
  `gbdt_8q` for 2008-2023 (32 pairs) still carried the pre-fix fixed hyper-parameters (no
  `tuning` key); for 2008 and 2009 that was a genuine rule 6.7 overlap, because those
  parameters had been chosen on 2007Q1-2008Q4 reports inside the test years. `hazard_8q` was
  fitted after the fix and is untouched (all 16 configs already carry `tuning`). Each pair
  was refitted one model per call with `--no-rebuild` and `walkforward_scores` rebuilt once;
  no `--gbdt-iterations` cap was needed (slowest call 2017 gbdt, 1 m 43 s). Every 8q config
  now has `tuning.validation_end` before `first_test_prediction_date`, `sufficient=True`,
  `fallback=None`. Pooled 8q PR-AUC: hazard 0.4113 (unchanged), logit 0.2835 -> 0.2566,
  gbdt 0.1169 -> 0.0887. The fall is concentrated in 2008, whose training window closes at
  2005-12-31 with 116 positives: logit 0.3020 -> 0.0719 and gbdt 0.1773 -> 0.0326 (the tuned
  booster scores every 2008 row identically, ROC-AUC 0.5000); the earlier figures had leaked
  the test year through the shared grid choice. From 2011 on the per-year numbers move
  within the reported intervals and the booster beats the logit in 2012, 2014-2017 and 2019.
  Pooled ROC-AUC for `gbdt_8q` (0.42) mixes yearly score scales and is not a ranking claim;
  the per-year table in the report is the reference. `reports/walkforward.md` is not
  regenerated in this pass; its 8q section still shows the superseded figures.
- 2026-09-25 — **Regenerated years recalibrated, explanations and reports refreshed.** The 13
  4q and 32 8q (year, model) pairs refitted after the per-year tuning fix were recalibrated
  (`calibrate --year Y --model ... --horizon H --no-rebuild`, per-model default scorer, 10-15 s
  a year), `walkforward_scores` rebuilt once, `reports/walkforward.md` regenerated by
  `metrics-report` (1 m 31 s), and the 2020-2024 `gbdt` SHAP drivers, the production drivers
  and `reports/shap_summary.md` recomputed. Pooled 4q: hazard PR-AUC 0.3268 / recall@2%
  0.7038 / Brier 0.0041 -> 0.0038; gbdt_mono 0.3138 / 0.7147 / 0.0045 -> 0.0048 (unchanged);
  logit 0.3056 / 0.6833 / 0.0041 -> 0.0038; gbdt 0.2647 / 0.6367 / 0.0043 -> 0.0044; texas
  0.2606 / 0.7437. Pooled 8q: hazard 0.4113 / 0.6598 / 0.0070 -> 0.0080 (unchanged); logit
  0.2566 / 0.5207 / 0.0079 -> 0.0077; gbdt 0.0887 / 0.2200 / 0.0088 -> 0.0203. The `gbdt_8q`
  calibrated Brier is worse than raw because the inner-scorer maps of 2011-2013 over-predict
  (mean calibrated 0.27 / 0.21 / 0.08 against rates 0.012 / 0.007 / 0.004): the retuned
  full-window boosters put their 2011-2013 test rows on a coarser score scale than the inner
  model that scored the slice, so the map extrapolates. Reported as such, not patched. The
  retuned 2020 `gbdt` (`learning_rate 0.1, num_leaves 63`, 4 failures) yields mean |SHAP|
  of 53 for `large_time_deposit_share` and 45 for `unrealized_loss_to_tier1`; because the
  summary pools the plain average over test years, that one degenerate year now tops the
  pooled table (share 0.24 / 0.21). The 2023 column and the production drivers
  (`adjusted_tier1_leverage`, `neg_roa_quarters_last_8`, `unrealized_loss_to_tier1`) are the
  ones to read; a per-year normalisation of the pooled column is left for the owner.
- 2026-09-25 — **Run index is append-only; `runs rebuild-index` compacts it.** `Run.finish`
  used to rewrite `runs/index.jsonl` whole, so two interleaved commands could drop each
  other's rows. It now appends its single line with one `O_APPEND` write; `read_index`
  keeps the latest line per id (first position); `tracking.rebuild_index` rewrites the file
  from `runs/<name>/<id>/{config,metrics}.json` sorted by run id through a temporary file
  and `os.replace`, so two rebuilds of one tree are byte-identical. Rebuilt once here (1426
  runs, previously in append order). `bankcanary runs list [--name] [--limit] [--runs-dir]`
  reads the index; `--runs-dir` exists so the command is testable on a temporary tree.
- 2026-09-25 — **Tuning consistency audit.** `walkforward.check_tuning_consistency(models_dir,
  lag_days)` flags every `models/walkforward/<Y>/<model>/config.json` that lacks `tuning`
  or whose `tuning.validation_end` is on or after Y's first prediction date (03-31 plus the
  lag); configs recording no fit (`texas`) are skipped. Against the real tree: 133 configs,
  0 violations. Not wired into a command; run it from Python when artefacts change.
- 2026-09-25 — **Notebooks, model card, README and checklist refreshed to the regenerated
  walk-forward.** Notebooks 02 and 04 re-executed in place (02 rebuilt from
  `scripts/make_notebook_02.py` after its three hard-coded booster figures were updated: 4q
  `gbdt` 0.28 / 0.64 -> 0.26 / 0.64, 8q `gbdt` 0.12 -> 0.09 against logit 0.26); 05 does not
  read `walkforward_scores` and is untouched. The false-positive report changed only in its
  all-years row (3,713 -> 3,723 flagged non-failing bank-years; 76.6 -> 76.5 percent still
  open), because the 2009-2012 boosters were not refitted. The model card's per-year tables
  are regenerated from `reports/walkforward.md` by a throwaway parser rather than edited by
  hand; the 2022-2024 limitation now reads "PR-AUC below 0.18" (`gbdt_mono` 2023 is
  0.1761) and a new limitation states that hyper-parameters are re-selected inside each
  year's training period, naming the 2020 4q and 2008 8q boosters as the degenerate picks.
  The v1-logit comparison runs (checklist criterion 2) were not refitted; their per-year `C`
  came from the v2 selection at the time, so the 0.2194 / 0.5625 figure predates the fix.
- 2026-09-26 — **Owner decisions for Prototype 3 (all three taken with the recommended option):**
  Decision Point 2: monotone constraints ON — `gbdt_mono` is the production booster (walk-forward
  PR-AUC 0.314 vs 0.265 unconstrained, recall@2% intervals disjoint). Decision Point 4: Python
  pipeline → Neon Postgres → Next.js (App Router) on Vercel, the web app reads only precomputed
  tables. Decision Point 5: bank head-office coordinates come from the FDIC `institutions` table
  (`latitude`, `longitude`), no geocoding.
- 2026-09-26 — **Production score = the monotone booster's calibrated 12-month probability**, with
  its SHAP drivers; the hazard model (best pooled PR-AUC, overlapping interval) is published as a
  secondary "statistical model" score on profile pages and in the methodology. One model gives the
  leaderboard, the time machine and the drivers a single consistent story.
- 2026-09-26 — **Risk bands are percentile-based within each quarter:** `high` = top 2% of scored
  banks that quarter (the recall@2% framing used throughout the backtest), `elevated` = next 8%
  (top 2–10%), `low` = the rest. The probability itself is always shown next to the band.
- 2026-09-26 — **Local Postgres on port 5433** (`docker compose up -d`, container
  `bankcanary-postgres`) because the F1 project's container already holds 5432. Python owns the
  schema (`src/bankcanary/publish/schema.sql`); the web app mirrors it read-only in Drizzle.
- 2026-09-26 — **Production model artefacts are committed** under `models/production/` (a few MB:
  the latest walk-forward `gbdt_mono` and `hazard` pipelines, their calibrators and feature lists)
  so the scheduled refresh job can score a new quarter without retraining.
- 2026-09-26 — **Decision Point 2 applied: `settings.models.gbdt.monotone = true` and the
  monotone booster promoted.** `scripts/promote_production_models.py` copies the latest
  walk-forward `gbdt_mono` and `hazard` (year 2024, horizon 4: `pipeline.joblib`,
  `calibration.joblib`, `features.json`, `config.json`) into `models/production/<model>/` and
  writes `model_version.json` (`<model>-<train_end_repdte>-<git sha>`: `gbdt_mono-2022-12-31-bbc0230`,
  `hazard-2023-12-31-6740dae`). `train_end_repdte` is the config's `train_repdte_max`; `git_sha`
  and `trained_at` are the commit that first added the model's `runs/walkforward/<run_id>/`
  record (the run id is a hash of the same config), never the clock, with `HEAD` plus the
  artefact's mtime as the fallback for an uncommitted retrain. The `.gitignore` rule `/models/`
  had to become `/models/*` so the `!/models/production/` exception can take effect (git does
  not descend into an ignored directory); the committed artefacts total about 0.4 MB.
- 2026-09-26 — **SHAP drivers are `gbdt_mono` only and the pooled summary is per-year
  normalised.** `explain --model` (default `gbdt_mono`) writes `data/drivers/<Y>_<model>.parquet`
  and `rebuild_drivers_table` folds one model family, because keeping the `gbdt` rows beside
  the `gbdt_mono` rows would have doubled the table to 9.05 million rows (CONTRACT 16 keeps
  `gbdt_mono` only; the old `gbdt` driver files were deleted, their run records stay). The
  latest-quarter rows carry `model = 'gbdt_mono_production'`. `pooled_summary` now divides each
  year's mean |SHAP| by that year's total before averaging: under the old plain average the
  four-failure 2020 `gbdt` fit, on a log-odds scale far larger than any other year's, put
  `large_time_deposit_share` first at 24 percent; normalised, the top feature is `texas_ratio`
  at 7.0 percent and the top three carry 18.3 percent. `mean_abs_shap` (plain average) is still
  reported beside the normalised `share`.
- 2026-09-26 — **The 2023 case study under the monotone booster misses SVB.** With
  `gbdt_mono` as the rate-aware booster (the credit-only view kept), SVB's 2022Q4 rank is 2,368
  of 4,773 (50th percentile) against 245 (94.9th) for the unconstrained fit: the constrained
  trees give `adjusted_tier1_leverage` = -0.33 a +0.51 contribution instead of +2.1, and the
  healthy credit and capital ratios outweigh it. Signature (88th percentile) and First
  Republic (81st, then 94.6th in 2023Q1) rank higher than before and the fit pools best
  (PR-AUC 0.068 on 13 failures). The notebook, the report and the model card say so plainly;
  the unconstrained `gbdt` remains available through `case_study_2023.build_model` for the
  comparison but is not fitted by default. Also recorded: the 2024 `gbdt_mono` fit that is
  now the production model is the weakest of its series (PR-AUC 0.11, ROC-AUC 0.55 on 9
  failures) and its SHAP leans on `share_consumer`, `nim_q` and `share_residential`.
- 2026-09-26 — **Publish job core (`bankcanary publish`, step E1a).** `src/bankcanary/publish/`
  holds the schema (`schema.sql`, every CONTRACT 16 table with a disclaimer comment), pure
  builders (`core.py`: one DataFrame per table from the Parquet warehouse, `runs/` and
  `models/production/`), the loader (`writer.py`: truncate + COPY inside one transaction per
  table, or COPY into a temp table + `ON CONFLICT DO UPDATE` for partial writes) and the
  psycopg 3 connection (`db.py`, URL from `Secrets.database_url`). Scores = the horizon-4
  `walkforward_scores` rows of `gbdt_mono` and `hazard` (each quarter by its own test-year
  model) plus the quarters past the last complete year (2025Q1-2026Q2, 26,557 bank-quarters)
  scored on the fly with the production pipelines and isotonic maps. Rank is by raw score
  (ties by cert) so it stays defined where the calibrated map is flat; `percentile` =
  share of the quarter's banks ranked below; bands `high` = top `ceil(2%)`, `elevated` =
  top `ceil(10%)`. A backtest year's `model_version` is `<model>-<train_repdte_max>-<sha of
  the commit that added its run record>`, the same rule as the production promotion.
- 2026-09-26 — **`ratios` stores the ratio and peer-percentile columns as `real`.** With
  doubles the first full publish was 426 MB (ratios 194 MB, scores 192 MB); `real` for the 24
  ratio/percentile columns, fixed-width-first column order in `scores` and dropping a
  non-required `(model, repdte)` index bring it to 342 MB (scores 180 MB, ratios 127 MB).
  The deviation is written into CONTRACT 16. About 58 MB remain for `drivers` and
  `map_quarters`; the next levers, if needed, are the pre-2008 `ratios` rows (about 40 MB)
  and five drivers per bank-quarter instead of ten.
- 2026-09-26 — **Failures without a `cert` are not published.** 488 pre-1970 rows of the FDIC
  failures list carry no certificate number, so they cannot be keyed by `(cert, fail_date)`;
  3,629 rows remain. `banks.holding_company_name` is null for now: the institutions table
  carries `rssdhcr` but no holding-company name, which a later step can fetch.
- 2026-09-26 — **Page tables published (step E1b).** `src/bankcanary/publish/pages.py` builds
  `drivers` (the warehouse SHAP table of `bankcanary explain`, backtest and production rows
  both under `model = 'gbdt_mono'`, filtered to the CONTRACT subset: latest 4 quarters of
  every bank, every quarter of every failed bank, top 5 percent of every quarter; 57,537
  bank-quarters; `feature_label` = the registry explanation's first clause), `map_quarters`
  (every `gbdt_mono` bank-quarter with head-office coordinates; `failed_this_quarter` marks
  the report a failure follows: `fail_date` within the next quarter, or within four quarters
  of the bank's last report; 558 of the 560 failures since 2008Q2 land on a dot),
  `rate_shock_scores` (latest quarter, 4 shocks x 5 durations = 86,260 rows; the extra loss
  is added to the unrealised total before `unrealized_loss_to_tier1` and
  `adjusted_tier1_leverage` are recomputed with the sensitivity module's formulas, missing
  amortised cost counts as zero, then the production `gbdt_mono` pipeline + isotonic map
  re-score and `rank_scores` bands each scenario) and the 2023 case study (28 rank rows and
  37 series rows by re-running `run_case_study` without a run record, about 30 s). `publish`
  now writes all 13 tables in `TABLE_KEYS` order in about 100 s; `--tables` accepts any
  subset and builds the core frames a page table needs without writing them.
- 2026-09-26 — **The size budget needed the pre-named levers.** With ten drivers per
  bank-quarter and every `ratios` row the database was 538 MB. Applied: five drivers per
  bank-quarter (largest |SHAP|, both signs), `ratios` from 2008Q1, `real` for the display
  columns of `scores`, `drivers` and `map_quarters`, no `cert` index on `map_quarters`:
  399.6 MB (scores 178, ratios 85, drivers 61, map 49, rate shock 11). Recorded in CONTRACT
  16. The next levers, in order: `scores.model_version` as a short code joined to
  `model_versions` (about 26 MB of repeated text), `map_quarters` restricted to banks that
  are `elevated`/`high` or failed (the map draws `low` dots from `banks`), and hazard rows
  after 2015 only. Schema type changes are not applied to an existing table by
  `CREATE TABLE IF NOT EXISTS`: drop the table locally and re-run `publish --tables <t>`.
- 2026-09-26 — **The refresh is incremental; `publish` stays the full rebuild (step E2).**
  The walk-forward scores and boosters are not committed, so a GitHub Actions runner
  cannot rebuild `scores` history. `bankcanary refresh` therefore upserts only what a new
  quarter adds: its `quarters` row (every other row keeps its published `model_year` and
  `model_version`, label counts refreshed), its `scores` (production `gbdt_mono` and
  `hazard`, ranked within the quarter, `delta_prob_prior_q` from a re-score of the prior
  quarter), its SHAP `drivers`, `ratios` and `peer_stats`, and `map_quarters` for the last
  four quarters (so `failed_this_quarter` follows new failures); `banks`, `failures` and
  `rate_shock_scores` are replaced whole. Older `drivers` rows that a full publish would
  drop (the latest-four-quarters rule) linger until the next `publish`. The comparison
  quarter comes from Postgres, or from the warehouse `labels` table when the database is
  unreachable, and the run log names the source. Local timings for a simulated new
  quarter (2026Q2 deleted from `scores` and `quarters`, then refreshed): structure tables
  5 s, financials 28 s, panel 7 s, labels 1 s, FRED re-pull + macro 55 s, features_v2
  11 s, warehouse load + frames + publish 11 s (about 2 min in all); a no-op run takes 1 s and writes only
  a `pipeline_runs` row. The workflows cache `data/raw/fdic` and `data/raw/fred` under
  `raw-<latest REPDTE>`; the first run warms the cache with a full `ingest --no-build`.
  `retrain.yml` opens a pull request and never merges it.
- 2026-09-26 — **Web skeleton (step E3).** `web/` is a Next.js 16 App Router app on port
  3100 with the read-only data layer of CONTRACT 18: `lib/db/schema.ts` mirrors the 14
  published tables with Drizzle (dates as ISO strings, bigints as numbers) and
  `tests/schema.test.ts` parses `schema.sql` to fail on any name or type drift;
  `lib/queries/*.ts` are the only database readers, each wrapped in `cached()`
  (`unstable_cache`, tag `data`, 3600 s) and taking quarters as labels (`2026Q2`).
  `POST /api/revalidate` calls `revalidateTag("data", { expire: 0 })`, so the first request
  after a publish blocks on fresh data instead of serving the old quarter for another
  request. `next.config.ts` copies unset variables from the repository `.env` so the
  database URL lives in one file; deployments set their own environment. Off-localhost
  connections always verify TLS certificates regardless of the URL's `sslmode`. Fonts are
  the system stack (no `next/font/google`, so builds never need the network). The
  theme is `data-theme` on `<html>` with a pre-paint script and `useSyncExternalStore`;
  `prefers-color-scheme` applies until the toggle stores a choice. Risk bands use colour,
  a text label and a distinct glyph, and links inside text are underlined (axe
  `link-in-text-block`). The five sections beyond the leaderboard are placeholder pages
  until their steps land; `/bank/[cert]` is not routed yet so the top-10 names are not
  links. `echarts` is pinned to 5.x (npm resolves 6 by default) and `@types/node` to 22
  for vitest 5. Playwright runs `npm run start` itself against a prior `npm run build`.
- 2026-09-26 — **2023 case study, rate-shock tool and methodology pages (steps E8-E10).**
  The case study reads `case_study_2023` and `case_study_series` as published and draws one
  chart per bank (small multiples) rather than one chart with three peer bands, because each
  bank's peer group (size bucket × Census region) is different; the band runs from the peer
  median to the 5th percentile for unrealised losses and to the 95th for the uninsured share.
  The rate-shock page does no arithmetic in the browser: `rateShockScenarios()` joins every
  `rate_shock_scores` cell with the quarter's published `scores` and `ratios` rows in SQL,
  returns a per-scenario summary (banks crossing into `high`, high counts before and after)
  plus the 30 largest climbers, and the client only picks the cell for the chosen shock and
  duration. Because bands are rank percentiles the high count is fixed per scenario; the
  summary therefore reports who crosses in, not how many are high. The methodology page
  renders `docs/model_card.md` with `marked` (headings shifted one level, every markdown table
  wrapped as `.data-table` with the disclaimer caption, relative links resolved to GitHub);
  `npm run sync-docs` copies the card into `web/content/` and the referenced
  `reports/figures` PNGs into `web/public/figures/`, and the copies are committed so the
  build never reads outside `web/`. The Playwright spec lives in `web/e2e/pages3.spec.ts`
  because `playwright.config.ts` scans `./e2e` (not `tests/e2e`). Absolutely positioned
  `.sr-only` spans (RiskBand) escape an `overflow-x-auto` table container unless that
  container is `position: relative`, which showed up as horizontal page scroll at 375 px.
- 2026-09-26 — **Leaderboard and bank profile pages (steps E4 and E5).** `/` is a server
  component driven by URL search params (`q`, `state`, `size`, `charter`, `band`, `sort`,
  `dir`, `page`; parsed by `components/leaderboard/params.ts`), so every view is a shareable
  link and the filter form is a plain GET form that works without JavaScript; the CSV route
  `/api/download/leaderboard.csv` takes the same query string. Search matches name or city
  (ILIKE) or the exact certificate when the text is all digits; the charter filter is the
  FDIC `bkclass` code. Top-3 driver chips come from one `drivers` query per page over the
  page's certificates. `/bank/[cert]` uses `notFound()` for unknown or malformed
  certificates; because the route streams behind `loading.tsx` the HTTP status is 200 with
  the not-found page rendered, so the smoke asserts the page, not the status. The
  probability timeline and the SHAP waterfall are client components that build their
  ECharts options themselves (a server component cannot hand a formatter function to the
  client `EChart`); the 12 CAMELS panels are server-rendered inline SVG sparklines (bank
  line, peer p50 dashed, p10-p90 shaded) so no chart library loads for them. The peer band
  uses the bank's current size bucket and its Census region (`stateToRegion` mirrors
  `structure_p2.py`); the percentile under each value is the bank's own peer group in that
  quarter. The waterfall shows the five drivers the database keeps per bank-quarter (the
  E1b size deviation), not ten. Driver sentences are built from the registry label and the
  published direction; `featureUnit()` in `components/bank/explain.ts` mirrors
  `FeatureSpec.unit` so raw values print in their unit. The bank CSV route is
  `app/api/download/bank/[cert]/route.ts` and strips a `.csv` suffix from the segment
  because a route segment cannot mix a parameter with a literal suffix. Playwright specs
  live in `web/e2e/` (the config's `testDir`); the leaderboard spec waits for the h1 before
  axe runs because the loading skeleton has none. Concurrent agents share `.next`,
  `test-results` and port 3100, so an e2e run can collide with another agent's; re-run when
  the port frees.
- 2026-09-26 — **Time machine and failure replay map (steps E6 and E7).**
  `/time-machine?quarter=YYYYQn` replays the `gbdt_mono` ranking of any scored quarter
  (2008Q1 to the latest) from `scores`, with hindsight from `banks.fail_date`: "failed N
  months later" counts whole months from the report date, while recall@top-2% uses the label
  window the model was scored against (a failure after the quarter's `avail_date` and within
  twelve months of it, as in `labels/build.py`), ordered by raw score descending with `cert`
  as the tie-breaker and a head of ceil(2% of n), exactly as `evaluation/metrics.py`. Pooled
  over a walk-forward year the query reproduces `walkforward_metrics.recall_at_2pct` to the
  last digit (the page says "identical" and the e2e spec checks 2009). Two gotchas: the join
  to `banks` must be a left join because 111 scored certs have no `banks` row and an inner
  join shifts the cutoff (0.3697 instead of 0.3741 for 2009); and bound date parameters are
  cast (`$1::date`) because `unknown + interval` is ambiguous in Postgres. The map is an
  inline SVG: `us-atlas` states-10m (version 3 ships raw lon/lat, not pre-projected) and every
  head office go through `d3-geo`'s `geoAlbersUsa` on a 975 by 610 frame, which drops Puerto
  Rico and the territories (about 30 offices, noted under the map); each band is one `<path>`
  of repeated symbols (circle, diamond, triangle) so a quarter of 8,000 banks is four
  elements, failures are crosses in the foreground colour, and the legend and the failures
  table carry the same information as the shapes. `/api/map/[quarter]` serves one quarter
  (lat/lon rounded to three decimals; about 0.9 MB raw for a 2009 quarter, so a columnar
  payload is the next lever if playback feels slow on Neon); the client keeps every quarter
  it has seen and prefetches the next during playback, and playback waits for a quarter
  rather than skipping it. `mapTimeline()` (bank and failure counts per quarter) and
  `scoredQuarters()` give the quarter ranges. Playwright's `testDir` is `web/e2e`, so the
  spec is `web/e2e/time_machine_map.spec.ts`. The pages import from
  `@/lib/queries/timeMachine` and `@/lib/queries/map` directly rather than the shared index,
  which another step was editing in the same checkout at the same time.
- 2026-09-26 — **Publish fix: walk-forward provenance no longer depends on untracked
  artefacts.** `walkforward_version()` resolves a backtest year's `model_version` from
  `models/walkforward/<year>/<model>/config.json` when it exists and otherwise from the
  committed run record `runs/walkforward/<run id>/config.json` of that model, horizon 4 and
  test year (the record added to git last wins when a year was run twice; on this machine
  that choice reproduces every artefact-derived version exactly, so no score or quarter
  changed). A year with neither source gets a derived row `<model>-wf<year>-unknown`
  (`train_end_repdte` null, `git_sha` "unknown") and a warning, never the production
  version; `build_all` also takes the walk-forward years from the warehouse frame, not
  only from `labels`, and `build_scores` raises when any row's version is unknown, so a
  publish from a fresh clone fails loudly or names the right version instead of writing
  NULL provenance. `model_versions.notes` now says which source named the version.
- 2026-09-26 — **Fix web_queries (adversarial review).** Three confirmed defects.
  (1) CSV formula injection: `csvCell` now prefixes a text cell that starts with `=`, `+`,
  `-`, `@`, tab or carriage return with an apostrophe and quotes it (numbers and numeric
  strings such as "-0.5" are left alone), so an FDIC bank name like `=HYPERLINK(...)` opens as
  literal text in Excel and Sheets; the leaderboard export's driver chips read
  `raises: feature` / `lowers: feature` instead of `+ feature` / `- feature`, which had made
  every driver cell a formula. The adversarial test's one assertion on the raw name now
  expects the apostrophe. (2) TLS to the database: `pg` merges the parsed connection string
  over the explicit config, so `sslmode=disable`, `sslmode=no-verify`, `ssl=false` or
  `uselibpqcompat=true&sslmode=require` in a remote `DATABASE_URL` silently switched off
  verification. `stripTlsParams()` in `web/lib/db/client.ts` removes `ssl`, `sslmode`,
  `sslcert`, `sslkey`, `sslrootcert`, `sslnegotiation` and `uselibpqcompat` from a
  non-local URL (a warning names the dropped keys, never values) and the pool always gets
  `ssl: { rejectUnauthorized: true }`; the local container stays plain. (3) The CSV routes
  answered with `s-maxage=3600`, which a CDN keeps for an hour after `POST /api/revalidate`
  has already expired the page data; `csvHeaders()` now sends `cache-control: no-store`
  (the routes are `force-dynamic` and read the same `unstable_cache` data as the pages, so
  the download is cheap and always the published quarter). `web/components/leaderboard/csv.ts`
  sits outside the step's nominal file list but is the root cause named by the review.
- 2026-09-26 — **CI, README and deployment runbook (step E12).** `ci.yml`'s web job no
  longer gates on the presence of `web/package.json` (the app exists; a missing directory
  should fail the build, not skip it) and runs `npm ci`, `npm run lint`, `npm run typecheck`
  and `npm test` with `working-directory: web`; the Python job is unchanged and Playwright
  stays local because it needs a built app and a published database. README gained the
  architecture diagram (Mermaid: FDIC/FRED → ingest → warehouse → features → models →
  publish → Neon → Next.js on Vercel, with the weekly refresh and the revalidate call), the
  web setup, a Prototype 3 paragraph and a status line that reserves the place for the live
  URL. `docs/RUNBOOK.md` section 7 is the owner's deployment checklist in the order that
  makes the first refresh succeed (Neon via the Vercel marketplace with a separate reader
  role, Vercel project with root directory `web`, GitHub secrets, one `publish` from a
  machine with the warehouse, the counts query, a manual refresh run), plus the TLS notes:
  psycopg hands the URL to libpq, so the GitHub secret needs
  `sslmode=verify-full&sslrootcert=system`; node-postgres strips every TLS parameter and
  forces verified TLS, so the Vercel value carries `sslmode=verify-full` only. No
  `web/vercel.json`: the root directory is a project setting and Next.js is auto-detected.
  Nothing was deployed and no external account was touched.
- 2026-09-26 — **CI: pytest runs with `_TYPER_FORCE_DISABLE_TERMINAL=1` and without
  `GITHUB_OUTPUT`.** The first CI run after the web app landed failed five Python tests that
  pass locally: Typer forces Rich's terminal mode whenever `GITHUB_ACTIONS` is set, so
  `--help` comes out boxed and coloured with the option's first `-` styled separately
  (`'--variant' in output` is false), and the runner exports `GITHUB_OUTPUT`, so
  `write_github_output` writes a file in the test that expects it to do nothing. Both are
  environment, not code, so the pytest step neutralises them; the tests themselves should
  still become hermetic (`monkeypatch.delenv("GITHUB_OUTPUT")`, help assertions on
  `_TYPER_FORCE_DISABLE_TERMINAL` set in `conftest.py`), which belongs to the steps that own
  those files.
- 2026-09-26 — **E11 quality pass on the web app.** Axe over every route in both themes
  found nothing new; the real accessibility gaps were the three tables (leaderboard,
  time-machine ranking and failures, bank drivers) whose scroll containers were not in the
  tab order, so a keyboard user could not reach the columns that overflow at 375 px. They
  are now `role="region"` + unique `aria-label` + `tabIndex=0` like the others (the
  time-machine table takes a `label` prop because it renders twice per page). Responsive
  checks run at 375, 768 and 1280 px both during the streamed skeleton and once settled;
  the skeletons are `min-h-screen` so the footer starts below the fold while a page streams
  (that footer jump was a CLS of 0.18 on the time machine). SEO: `lib/seo.ts`
  `pageMetadata()` gives every page an absolute title (`title.template` in the root layout
  does not apply to `app/page.tsx`, which is the same segment), description, canonical,
  Open Graph, Twitter card and robots; filtered, sorted or paged leaderboard views and the
  bank not-found page are `noindex, follow`; `app/sitemap.ts` lists the six sections, one
  time-machine URL per scored quarter and one profile per production-scored bank (about
  8,900 URLs, cached queries, and it degrades to the six sections when the database is
  unreachable); `app/robots.ts` disallows `/api/`. States: `generateMetadata` on the
  leaderboard, bank, time-machine and rate-shock pages now catches its own query failure
  and falls back to static metadata, because a metadata error leaves `<main>` empty instead
  of reaching `error.tsx` (found by the new `e2e-dbdown/` suite, whose Playwright config
  starts the built app on port 3101 with `DATABASE_URL` on a closed port after deleting
  `.next/cache/fetch-cache`, without which `unstable_cache` answered from the previous run
  and the outage was invisible). Performance: the leaderboard body moved into a client
  component (`LeaderboardRows`) so its rows travel as JSON rather than one serialised
  element tree per cell: home page HTML 206 KB → 150 KB, server payload 112 KB → 66 KB,
  20 KB gzipped, largest script 229 KB (ECharts' 1 MB chunk loads only where a chart
  renders, through the existing dynamic import in `EChart.tsx`). Lighthouse 13.5 via
  `scripts/lighthouse.mjs` (starts `next start`, audits, stops; Playwright's Chromium,
  headless): desktop preset `/` 100/100, `/bank/14` 100/100, `/time-machine` 100/100,
  `/map` 99/100, `/case-study-2023` 100/100, `/rate-shock` 100/100, `/methodology` 100/100
  (performance/accessibility); mobile preset `/` 98/100 (LCP 2.3 s, TBT 30 ms, CLS 0),
  `/time-machine` 97/100, `/bank/14` 96/100. Suites: 114 vitest, 65 Playwright on the main
  config, 6 on the database-down config. The "destination stream closed early" lines the
  server logs during the Playwright run come from tests that measure the skeleton with
  `waitUntil: "commit"`; they are noise, not failures.
- 2026-09-26 — **Prototype 3 acceptance pass (`docs/P3_CHECKLIST.md`).** The end-to-end
  dry run deleted 2026Q2 from `scores`, `drivers`, `map_quarters` and `quarters` in one
  transaction and let the plain `bankcanary refresh` find the gap from the FDIC probe
  rather than `--force-quarter`, so the decision path is the one the cron job takes;
  identical counts came back in 119 s. The 2009 recall check reads the published raw
  `score` column with `cert` as tie-breaker, exactly as the walk-forward evaluation
  does, and is bit-identical (0.374079528718704) to `walkforward_metrics`. Traceability
  used the rank-1 bank of the latest quarter (cert 8681) because its profile exercises
  both models and the drivers table. Deployment stays "pending owner": the checklist
  copies the seven RUNBOOK steps so the orchestrator can hand them over without the
  runbook. The refresh run record is committed like every other run record.
- 2026-09-26 — **Prototype 3 gap pass.** (1) `model_version` now appears on the three pages
  that lacked it: the bank profile prints it under the rank line (`gbdt_mono`, and the
  `hazard` version when present), in the driver-table caption and in a note under the
  probability timeline that counts the walk-forward versions behind the series (the CSV
  already carried it per row); the map's status line shows `quarters.model_version` of the
  quarter on screen (`mapTimeline` now selects it); the 2023 case study cannot cite a model
  version because `case_study_2023` is refit at every publish and has no such column
  (CONTRACT 16, unchanged), so its rank-table caption cites the `pipeline_runs` row that
  wrote the table (latest `status = 'ok'` run whose `rows_written` lists it) and the
  production model version of the newest quarter at that time. (2) The `Makefile` now covers
  the whole pipeline: `p1`, `p2` (per-year `walkforward` loop, `YEARS` overridable),
  `publish`, `publish-dry`, `refresh`, `db-up`/`db-down`, `web-install`, `web-check`,
  `web-build`, `web-start`, `web-e2e`, `web-lighthouse` and `ci` (= `lint test web-check`,
  the same three legs as `ci.yml`). (3) The methodology page gets `WalkforwardCharts`
  (`components/methodology/chartOptions.ts` + client wrapper): PR-AUC and recall@2% with the
  published 95% bounds as dotted lines, and Brier raw against calibrated, low-confidence
  years as hollow markers; the option builder copies columns and never derives values, so
  CONTRACT 19's no-client-arithmetic rule holds (the bootstrap band is drawn as two bound
  lines rather than a filled `hi − lo` area for that reason). Found on the way: ECharts'
  `aria` feature rewrites the container's `aria-label` once it draws, so every chart's
  spoken `ariaLabel` prop was lost after hydration (the case-study test passed only because
  the generated text happened to contain the chart title); `EChart.tsx` now hands the prop to
  ECharts as `aria.label.description`, which keeps the accessible name fixed on every chart.
  Deployment (acceptance criterion 4) remains the owner's action per `docs/RUNBOOK.md`
  section 7.
- 2026-09-26 — **The boosters take raw features; the winsoriser is gone from the tree
  pipelines.** `make_gbdt` built `Winsorizer(0.005, 0.995) -> estimator`, the logit front end
  minus imputer and scaler. Evidence that this clipped away the signal it was meant to protect:
  for Silicon Valley Bank at 2022-12-31 the `drivers` table showed `feature_value` 3.94 for
  `adjusted_tier1_leverage` where the true value is −0.33 and −0.19 for `unrealized_loss_to_tier1`
  where the true value is −1.04; 283 training rows before 2021Q3 have `unrealized_loss_to_tier1`
  below −0.5 (minimum −7.07) and every one of them reached the trees as about −0.19, so no split
  could isolate the banks whose securities losses exceeded their capital. Trees do not need the
  clip (rank-order splits, leaves bounded by `min_samples_leaf`), so `make_gbdt` now returns the
  bare `Pipeline([("model", estimator)])` for both backends; the logistic and hazard pipelines
  are unchanged. `shap_drivers.shap_matrix` feeds the raw matrix to the explainer through the
  new `model_inputs` helper, which applies `pipeline[:-1]` only when a loaded artefact still has
  transformer steps, so older walk-forward files keep explaining consistently until they are
  refit; `feature_value` is now the bank's own ratio (NaN when the ratio is missing, which the
  trees split on). `scripts/tune_gbdt.py` tags booster configs with `inputs: raw`, so the 48
  winsorised grid runs under `runs/tune_gbdt/` no longer satisfy a re-tune, and a re-tune keeps
  the `monotone` flag already in `config/settings.yaml` (Decision Point 2 is the owner's ruling)
  instead of re-deriving it from the inner slice. Not touched here: the walk-forward, calibration,
  explain and production artefacts on disk, which still carry the winsorised booster until the
  per-year commands are re-run. Consumers of the booster (`walkforward`, `calibration`,
  `case_study_2023`, `sensitivity`, `publish.core.score_with_artefacts`, the rate-shock scorer)
  only call `fit`/`predict_proba` on the pipeline and needed no change.
- 2026-09-26 — **Raw-feature grid and fixed-split refit.** The 48-point grid re-scored on raw
  inputs (inner slice, 37 positives) keeps `gbdt_mono` ahead: winner `learning_rate 0.03,
  num_leaves 31, min_samples_leaf 50, n_estimators 200` with inner PR-AUC 0.2142 (`gbdt` at the
  same point 0.1583; `logit_v2` 0.2589), against 0.1968 / 0.2299 for the winsorised pair at the
  old point (`num_leaves 63, min_samples_leaf 200`). On the untouched fixed-split test years the
  raw boosters score `gbdt_mono` PR-AUC 0.4250 (was 0.4755) and `gbdt` 0.3913 (was 0.4324), with
  recall@2% 0.7959 / 0.7887 (was 0.8019 / 0.7791); every `learning_rate 0.1` point collapses on
  the inner slice (PR-AUC below 0.05 for the unconstrained model), so the 37-positive inner slice
  is a thin guide and the walk-forward years, once refit, are the evidence that counts. The
  2010-2013 test split contains no interest-rate failures, so it cannot show the benefit the
  SVB rows motivate; `reports/p2_gbdt.md` and `config/settings.yaml` carry the new numbers.
- 2026-09-26 — **Booster walk-forward regenerated at 4q on raw features.** Every
  `models/walkforward/<Y>/{gbdt,gbdt_mono}/` artefact for 2008-2024 was refit with the bare
  `("model", estimator)` pipeline (step names now `['model']`, no winsoriser in any per-year
  file), the per-year tuning re-run on each year's own inner slice (`check_tuning_consistency`:
  zero violations, every config carries `tuning`, no iteration cap needed; fits took about 8 s
  a year), and `walkforward_scores` rebuilt once (3,352,958 rows; 426,019 per booster at 4q).
  Pooled 4q over 2008-2024: `gbdt_mono` PR-AUC 0.3131, recall@2% 0.7066 (winsorised: 0.3138 /
  0.7147); `gbdt` 0.2248 / 0.5787 (was 0.2647 / 0.6367). The constrained booster is unchanged
  within noise while the unconstrained one loses ground, mostly in 2008 (0.1186), 2010 (0.2792,
  a `learning_rate 0.1` pick) and 2020 (0.0006 on four failures, again a 0.1 pick), which
  repeats the inner-slice finding that raw inputs punish the aggressive learning rate when the
  signs are free. Per-year `gbdt_mono` PR-AUC: 2008 0.2387, 2009 0.3197, 2010 0.4386, 2011
  0.4663, 2012 0.4402, 2013 0.3897, 2014 0.4628, 2015 0.3638, 2016 0.4956, 2017 0.0918, 2018
  0.1084, 2019 0.5118, 2020 0.4985, 2021 n/a (no failures), 2022 0.0308, 2023 0.0254, 2024
  0.1120. The 2022-2024 years stay weak for both boosters, so the SVB-era benefit the raw
  inputs were meant to unlock does not show in the ranking metrics at 4q; the calibration,
  explain and production artefacts still carry the winsorised booster until their commands are
  re-run, and `reports/walkforward.md` is regenerated by `bankcanary metrics-report`.
