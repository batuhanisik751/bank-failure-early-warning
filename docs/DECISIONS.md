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
