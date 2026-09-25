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
