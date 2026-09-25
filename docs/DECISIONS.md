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
