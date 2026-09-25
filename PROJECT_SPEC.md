# Bank Failure Early Warning System — Project Specification

> **Working name:** BankCanary (placeholder — rename freely)
> **Audience of this document:** an LLM coding assistant that will build the project with the owner.
> **Owner's goal:** learn banking/finance fundamentals by building a rigorous, portfolio-grade system — not a toy notebook.

---

## 0. Instructions to the LLM reading this

1. Read the whole document before writing code. Sections 5 (label definition) and 6 (leakage rules) are the parts most likely to be done wrong — treat them as hard requirements.
2. Build in the order of the three prototypes (Section 9). Do not start a prototype until the previous one meets its acceptance criteria.
3. At every **Decision Point** (marked ⚖️), stop and present the options with a recommendation to the owner. Do not silently choose.
4. The field codes listed for the FDIC API are **best-known mnemonics and must be verified** against the official FDIC financial-variable definitions (Step 1.2). If a code is wrong or missing, find the correct one; do not invent values.
5. Time-box investigations: if a data problem takes more than ~30–45 minutes, stop and report what you found and what you suspect, instead of continuing to guess.
6. Git: commits are authored solely by the owner. **Do not add AI co-author trailers** (no `Co-Authored-By` lines) to commit messages.
7. Never commit secrets (API keys, database URLs). Use `.env` + `.env.example`. Never `source` a file that contains a database connection string (special characters such as `&` break shell parsing and can print the secret).
8. This is an educational project. The UI and README must say it is **not a credit rating, not investment advice, and not an official supervisory assessment**.

---

## 1. Problem definition

### 1.1 The real-world problem

Between 2008 and 2014, roughly 500 US banks failed. In 2023, Silicon Valley Bank, Signature Bank and First Republic — three of the largest failures in US history — collapsed within eight weeks. In nearly every case, warning signs were visible in the banks' own public regulatory filings (quarterly **Call Reports**) months or years in advance: thinning capital, rising bad loans, dangerous loan concentrations, reliance on unstable funding, or large unrealized losses on securities.

Regulators run confidential early-warning models (e.g., the FDIC's internal off-site monitoring). The public — depositors above the $250k insurance limit, community groups, journalists, students — has no transparent, reproducible equivalent.

### 1.2 The prediction task

> **Given only the information publicly available at a point in time, estimate the probability that each FDIC-insured bank will fail within the next 12 months (primary) and 24 months (secondary), and explain which factors drive that estimate.**

- **Unit of analysis:** one bank × one quarter ("bank-quarter").
- **Population:** all FDIC-insured institutions that file Call Reports.
- **Outcome:** failure as recorded by the FDIC (closed and resolved by the FDIC), not mergers, not voluntary closures.

### 1.3 Why this is hard (and why it is a good learning project)

| Challenge | Why it matters |
|---|---|
| **Extreme class imbalance** | In a typical quarter, well under 1% of banks fail within a year; in calm years, almost none. Accuracy is meaningless. |
| **Look-ahead leakage** | It is easy to accidentally use data that was not public yet, or to train on outcomes from the test period. Leaky models look brilliant and are worthless. |
| **Censoring** | Many troubled banks are acquired *before* they fail. They vanish from the data without a failure label. |
| **Regime change** | 2008–2014 failures were mostly *credit* failures (bad real-estate loans). 2023 failures were *interest-rate + deposit-run* failures. A model trained on one regime may be blind to the next. |
| **Messy accounting data** | Income items are reported year-to-date, forms changed over time, small banks file shorter forms, ratios have edge cases (zero or negative denominators). |

### 1.4 What success looks like

- The model, evaluated **out-of-time**, ranks banks that later failed near the top of the risk list several quarters before failure.
- It clearly beats simple baselines (the Texas ratio, a small CAPITAL + asset-quality logistic regression).
- It is **honest**: calibrated probabilities, reported uncertainty, a documented blind spot analysis (especially 2023), and a written model card.
- The owner can explain every feature in banking terms.

---

## 2. Project introduction (README-ready text)

> **BankCanary** is an open, reproducible early-warning system for US bank failures. It ingests every FDIC-insured bank's quarterly regulatory filings, computes the same kinds of financial-health ratios bank examiners look at — capital, asset quality, earnings, liquidity, concentration and interest-rate exposure — and estimates each bank's probability of failing within the next year.
>
> Unlike most bank-failure projects, BankCanary is built around **strict point-in-time discipline**: every prediction uses only data that was public on the date the prediction would have been made, and every model is tested on years it never saw. Its "time machine" mode lets you pick any quarter since 2006 and see exactly what the model would have flagged then — and which of those banks actually failed.
>
> The project also asks a harder question: *would a model trained on the 2008 crisis have seen Silicon Valley Bank coming?* It documents where the classic credit-risk view fails and how interest-rate and deposit-run indicators close the gap.
>
> BankCanary is an educational project. It is not a credit rating, investment advice, or a supervisory assessment. Deposits at FDIC-insured banks are insured up to $250,000 per depositor, per bank, per ownership category.

### 2.1 What it does (functional summary)

1. **Ingests** quarterly bank financials and the full failure history from the FDIC BankFind API (plus, in later prototypes, raw FFIEC Call Report schedules and macroeconomic data from FRED).
2. **Builds** a clean bank-quarter panel with point-in-time availability dates.
3. **Engineers** ~20 (P1) → ~60 (P2+) interpretable financial-health features.
4. **Labels** each bank-quarter with "fails within H quarters" using a precise, leakage-safe definition.
5. **Trains** baseline, statistical and machine-learning models, including a survival (hazard) model.
6. **Backtests** walk-forward: for each year, trains only on the past, predicts the next year, and records how well it did.
7. **Explains** every prediction (feature contributions) and compares each bank to its peers.
8. **Publishes** (P3) a web dashboard with a risk leaderboard, bank profiles, the time machine, a 2023 case study, a rate-shock scenario tool and a full methodology/model-card page, refreshed automatically when new quarterly data appears.

---

## 3. Data sources

### 3.1 FDIC BankFind Suite API (primary)

- **Base URL:** `https://api.fdic.gov/banks/`
- **Docs:** `https://banks.data.fdic.gov/docs/` (includes downloadable YAML definitions for every field — use these as the source of truth).
- **Bulk data page:** `https://banks.data.fdic.gov/bankfind-suite/bulkdata` (alternative to paginating the API).
- **Check** whether an API key is currently required. If it is, the **owner** registers for it (free) — the LLM must not create accounts. Store as `FDIC_API_KEY` in `.env`.

Common query parameters: `filters` (Elasticsearch-style, e.g. `REPDTE:20230331`, `CERT:24735`, `FAILYR:[2008 TO 2014]`), `fields` (comma-separated list), `limit` (max 10,000), `offset`, `sort_by`, `sort_order`, `format` (`json`/`csv`).

| Endpoint | Use | Key fields (verify) |
|---|---|---|
| `/failures` | Failure labels | `CERT`, `NAME`, `FAILDATE` (MM/DD/YYYY), `FAILYR`, `RESTYPE` (`FAILURE` vs `ASSISTANCE`), `RESTYPE1` (resolution method), `COST` (estimated loss to the Deposit Insurance Fund), `QBFASSET`, `QBFDEP`, `CITYST`, `CHCLASS1` |
| `/institutions` | Bank attributes, active/inactive status, RSSD crosswalk | `CERT`, `NAME`, `STALP`, `CITY`, `BKCLASS`, `CHARTER`, `ESTYMD`, `ENDEFYMD`, `ACTIVE`, `FED_RSSD`, `RSSDHCR` (holding company), `ASSET` |
| `/history` | Structure events (mergers, charter changes, closures) — used to classify non-failure exits | `CERT`, `CHANGECODE` / change description, effective date, acquirer |
| `/financials` | Quarterly financials, ~1,100 variables | `CERT`, `REPDTE` (YYYYMMDD) + the variables in Section 7 |

**Download strategy:** request one quarter at a time (`filters=REPDTE:<date>`), only the fields needed (~80–120), `limit=10000`. That is ~100 requests for 2001–2025, not hundreds of thousands. Cache every raw response to `data/raw/fdic/<endpoint>/<REPDTE>.json` and never re-download a cached quarter unless forced. Add polite rate limiting and retries with backoff.

**Units:** dollar amounts are in **thousands of dollars**. Income-statement items are **year-to-date** (see 7.3). Some ratios (e.g., ROA, NIM) are pre-computed by the FDIC; prefer computing your own from components for consistency, and use the FDIC ratios only as cross-checks.

### 3.2 FFIEC Central Data Repository — raw Call Reports (Prototype 2+)

- `https://cdr.ffiec.gov/public/` → Bulk Data Download → "Call Reports — Single Period", tab-delimited, one ZIP per quarter, **2001 Q1 onward**, one file per schedule (e.g., Schedule RC-B securities, RC-O deposit insurance).
- Identifier is `IDRSSD`. Crosswalk to FDIC `CERT` via the "Panel of Reporters" (POR) file in the same ZIP, or the FDIC `/institutions` field `FED_RSSD`.
- Needed for fields the FDIC API may not expose cleanly (verify first — if the FDIC API has them, prefer it):

| MDRM code (verify) | Meaning | Use |
|---|---|---|
| `RCFD1754` / `RCON1754` | Held-to-maturity (HTM) securities, amortized cost | Unrealized loss |
| `RCFD1771` / `RCON1771` | HTM securities, fair value | Unrealized loss |
| `RCFD1772` / `RCON1772` | Available-for-sale (AFS) securities, amortized cost | Unrealized loss |
| `RCFD1773` / `RCON1773` | AFS securities, fair value | Unrealized loss |
| `RCON5597` | Estimated amount of uninsured deposits (Schedule RC-O memorandum; reported only by larger banks — verify threshold) | Run risk |

`RCFD` = consolidated (banks with foreign offices), `RCON` = domestic. Take `RCFD` when present, else `RCON`.

### 3.3 FRED — macro context (Prototype 2+)

`https://fred.stlouisfed.org/` via the free FRED API (owner registers for a key → `FRED_API_KEY`). Suggested series (verify IDs):

- State unemployment rate: `<ST>UR` (e.g., `CAUR`, `GAUR`, `FLUR`)
- State house price index (FHFA): `<ST>STHPI` (e.g., `CASTHPI`)
- Federal funds rate: `FEDFUNDS`; 10-year minus 3-month Treasury spread: `T10Y3M`; 10-year Treasury yield: `DGS10`

Macro features must also obey point-in-time rules (use values published by the prediction date; vintage issues can be noted as a limitation rather than solved with ALFRED in P2, and optionally addressed in P3).

### 3.4 Reference literature (for the methodology page and for the owner's learning)

- Cole, R. & White, L. (2012). "Déjà Vu All Over Again: The Causes of U.S. Commercial Bank Failures This Time Around." *Journal of Financial Services Research.* — CAMELS-proxy logistic model; the standard baseline.
- Shumway, T. (2001). "Forecasting Bankruptcy More Accurately: A Simple Hazard Model." *Journal of Business.* — why discrete-time hazard models beat static classifiers for failure prediction.
- The "Texas ratio" (nonperforming assets ÷ (tangible equity + loan-loss reserves)) — the classic single-number warning signal.
- FDIC / Federal Reserve post-mortems on SVB and Signature (2023) — explain the interest-rate + uninsured-deposit failure mode.

---

## 4. Core concepts the owner should learn along the way

The LLM should add short explanatory notes (docstrings, notebook markdown, and the methodology page) for each of these:

- **Call Report** — the quarterly financial filing every insured bank submits.
- **CAMELS** — examiner rating framework: **C**apital, **A**sset quality, **M**anagement, **E**arnings, **L**iquidity, **S**ensitivity to market risk. Features are grouped this way.
- **Tier 1 leverage ratio / risk-based capital ratios** — how much loss-absorbing equity a bank has.
- **Noncurrent loans, OREO (other real estate owned), net charge-offs, allowance for loan and lease losses (ALLL / ACL after CECL)** — the lifecycle of a bad loan.
- **CRE and construction concentration** — why real-estate lending concentrated so many 2008-era failures (regulatory guidance flags construction loans > 100% of capital and total CRE > 300% of capital).
- **Brokered and uninsured deposits** — "hot money" that leaves quickly.
- **AFS vs HTM securities and unrealized losses** — why rising rates in 2022 created hidden holes in balance sheets.
- **Resolution types** — purchase & assumption, deposit payoff, etc.

---

## 5. Label definition (hard requirement)

Notation:

- `q` — report date (quarter end), e.g. 2009-03-31.
- `avail(q)` — the date the report is assumed publicly usable. **Default: `q + 60 days`** (Call Reports are due ~30–35 days after quarter end; 60 days is a conservative, defensible buffer). Make this a config value.
- `H` — horizon in quarters. **Primary `H = 4` (≈12 months). Secondary `H = 8`.**
- `window(q, H) = (avail(q), avail(q) + 3·H months]`.
- `fail_date(bank)` — `FAILDATE` from `/failures` where `RESTYPE == 'FAILURE'` (exclude open-bank `ASSISTANCE` transactions).
- `exit_date(bank)` — date the bank stopped existing for any non-failure reason (merger, voluntary liquidation, charter conversion that changes CERT — inspect `/history`), else null.

Rules:

1. `y_H(bank, q) = 1` if `fail_date ∈ window(q, H)`, else `0`.
2. **Drop** rows where `fail_date ≤ avail(q)` (the bank failed before this report would have been usable).
3. Rows where the bank **exited without failing inside the window** → keep with `y = 0` but set `censored_in_window = True`. Run a sensitivity analysis with these rows dropped. For survival models, treat non-failure exits as **censoring**.
4. Rows whose window extends beyond the latest known date (recent quarters) → `label_complete = False`. These are **scored but never used for training or evaluation**.
5. Store `window_end(q, H)` on every row; the leakage rules (Section 6) depend on it.
6. **Holding-company failures:** several subsidiaries of one holding company sometimes fail the same day (e.g., FBOP Corporation's banks in October 2009). Record `RSSDHCR` so evaluation can report results both per bank and per failure event.

Unit tests (required, synthetic data): failure on day `avail(q)` → dropped; failure on `avail(q) + 1 day` → `y=1`; failure on `window_end` → `y=1`; failure on `window_end + 1 day` → `y=0`; merged bank → `censored_in_window=True`; recent quarter → `label_complete=False`.

Sanity checks: yearly failure counts from `/failures` should match the FDIC's published failed-bank list (e.g., ~140 in 2009, ~157 in 2010, ~92 in 2011). Print a table of failures per year and the share of failed banks that can be matched to at least one prior Call Report (target: >95% for 2001+).

---

## 6. Leakage rules (hard requirement)

1. **Features at row `(bank, q)` may only use data with report date ≤ `q`** (lags and trends look backward only).
2. **Training/evaluation split by time, never randomly.** For a test set whose earliest prediction date is `T` (= `avail(first test q)`), the training set may only contain rows with `window_end < T`. This is the "you only know outcomes that have already happened" rule. Implement it as a single function used everywhere; unit-test it.
3. **Fit all preprocessing (imputation medians, scaling, winsorization bounds, target encodings) on the training fold only.** Use scikit-learn `Pipeline`s.
4. **Macro features** use values available on `avail(q)`.
5. **No bank identifiers, names, or CERT-derived features** in the model.
6. **Feature-importance smoke test:** if any single feature has suspiciously dominant importance, check it for leakage (e.g., "last report before failure" artifacts, fields only populated for failed banks).
7. **Hyperparameter tuning** happens inside the training period (nested walk-forward or a time-based validation split carved from the training period), never on the test years.

---

## 7. Features

Grouped by CAMELS. Mnemonics in `code font` are FDIC API candidates — **verify each in Step 1.2** and record the verified mapping in `config/fields.yaml`.

### 7.1 Base variables to pull (FDIC `/financials`, verify)

- Size & balance sheet: `ASSET`, `DEP`, `DEPDOM`, `EQ` (total equity), `INTAN` / goodwill (`INTANGW`), `LNLSGR` (gross loans), `LNLSNET`, `LNATRES` (loan-loss allowance), `SC` (total securities), `CHBAL` (cash & balances due), `FREPO` (fed funds sold & reverse repos), `ORE` (other real estate owned), other borrowed money / FHLB advances (find code).
- Loan mix: `LNRECONS` (construction & land development), `LNRENRES` (nonfarm nonresidential; check for owner-occupied vs non-owner-occupied split), `LNREMULT` (multifamily), `LNRERES` (1–4 family residential), `LNCI` (commercial & industrial), `LNCON` (consumer), `LNAG` (agricultural).
- Asset quality: `NCLNLS` (noncurrent loans = 90+ days past due + nonaccrual), `P3ASSET` (30–89 days past due), `P9ASSET` (90+ days past due), `NAASSET` (nonaccrual), `NTLNLS` (net charge-offs, YTD), `ELNATR` (provision for credit losses, YTD).
- Earnings (YTD): `NETINC`, `INTINC`, `EINTEXP`, `NONII` (noninterest income), `NONIX` (noninterest expense).
- Capital: `RBC1AAJ` (Tier 1 leverage ratio), `RBCRWAJ` (total risk-based capital ratio), Tier 1 risk-based ratio (find code), community bank leverage ratio election indicator (2020+; find code).
- Funding: `BRO` (brokered deposits), time deposits > $250k (find code), estimated insured deposits (find code; used to estimate uninsured share where RC-O item unavailable).

### 7.2 Engineered features

| CAMELS group | Feature | Definition |
|---|---|---|
| Capital | `equity_to_assets` | `EQ / ASSET` |
| Capital | `tier1_leverage` | `RBC1AAJ` (%) |
| Capital | `total_rbc_ratio` | `RBCRWAJ` (%) — often missing post-2020 for CBLR banks; add missing-indicator |
| Capital | `tangible_equity_to_assets` | `(EQ − INTAN) / (ASSET − INTAN)` |
| Asset quality | `noncurrent_ratio` | `NCLNLS / LNLSGR` |
| Asset quality | `npa_to_assets` | `(NCLNLS + ORE) / ASSET` |
| Asset quality | `early_delinquency` | `P3ASSET / LNLSGR` |
| Asset quality | `nco_rate` | annualized quarterly net charge-offs / avg gross loans |
| Asset quality | `reserve_coverage` | `LNATRES / NCLNLS` (cap; handle 0 denominator) |
| Asset quality | `texas_ratio` | `(NCLNLS + ORE) / (EQ − INTAN + LNATRES)` (if denominator ≤ 0 → set to a high cap + flag) |
| Management (proxy) | `asset_growth_4q`, `asset_growth_12q` | log change in `ASSET` over 4 and 12 quarters |
| Management (proxy) | `loan_growth_4q` | log change in `LNLSGR` over 4 quarters |
| Earnings | `roa_q` | annualized quarterly net income / avg assets |
| Earnings | `nim_q` | annualized (interest income − interest expense) / avg earning assets (or avg assets) |
| Earnings | `efficiency_ratio` | noninterest expense / (net interest income + noninterest income) |
| Earnings | `provision_rate` | annualized provision / avg loans |
| Liquidity | `brokered_share` | `BRO / DEP` |
| Liquidity | `loans_to_deposits` | `LNLSNET / DEP` |
| Liquidity | `liquid_assets_ratio` | `(CHBAL + FREPO + SC) / ASSET` (note: securities with unrealized losses are not fully liquid — see P2) |
| Liquidity | `wholesale_funding_ratio` | (other borrowed money + brokered deposits) / ASSET |
| Liquidity (P2) | `uninsured_share` | estimated uninsured deposits / DEP |
| Concentration | `construction_to_capital` | `LNRECONS / total risk-based capital (or Tier 1 + ALLL as fallback)` |
| Concentration | `cre_to_capital` | `(LNRECONS + LNREMULT + non-owner-occupied LNRENRES) / capital` |
| Concentration | loan mix shares | each loan category / `LNLSGR` |
| Sensitivity (P2) | `unrealized_loss_to_tier1` | `[(AFS_fv − AFS_ac) + (HTM_fv − HTM_ac)] / Tier1 capital` (negative = loss) |
| Sensitivity (P2) | `securities_to_assets` | `SC / ASSET` |
| Sensitivity (P2) | `adjusted_tier1_leverage` | Tier 1 leverage recomputed after subtracting HTM + AFS unrealized losses |
| Structure | `log_assets`, `bank_age_years`, `charter_class` (one-hot `BKCLASS`), `has_holding_company`, `state` (P2: region one-hot or macro only) |
| Trends (P2) | `Δ1q`, `Δ4q` of key ratios; `quarters_with_negative_roa_last_8` |
| Macro (P2) | state unemployment level & 4q change; state HPI 4q change; `T10Y3M`; 4q change in `FEDFUNDS` |

### 7.3 Data-cleaning rules

- **Year-to-date de-accumulation:** income-statement items (net income, provisions, charge-offs, interest income/expense) are cumulative within the calendar year. Quarterly value = YTD(q) − YTD(previous quarter in the same year); Q1 value = YTD(Q1). Unit-test this. Handle missing prior quarters (new banks, mergers) by flagging and annualizing YTD instead.
- **Averages:** "average assets" = mean of current and previous quarter-end values (fallback: current).
- **Ratios:** guard zero/negative denominators; winsorize engineered ratios at the 0.5th/99.5th percentile *computed on the training fold*.
- **Missingness:** keep explicit missing-indicator columns for structurally missing items (small-bank form 051 from 2017, CBLR banks from 2020, uninsured deposits only for large banks). Gradient boosting handles NaN natively; logistic regression gets median imputation + indicator.
- **Scope:** US-insured commercial banks and savings institutions filing Call Reports. Exclude rows with `ASSET <= 0`. Document any other exclusions.

---

## 8. Modeling & evaluation

### 8.1 Models (in order of introduction)

1. **Texas-ratio baseline** — rank by Texas ratio only.
2. **Small logistic regression** — ~6 CAMELS-proxy features (Cole & White style). Interpretable coefficients; report odds ratios.
3. **Regularized logistic regression** — all features, L2 or elastic net, class weights.
4. **Gradient boosting** — LightGBM (or XGBoost), monotonic constraints where banking logic is clear (e.g., risk non-decreasing in `noncurrent_ratio`, non-increasing in `tier1_leverage`) — ⚖️ monotonic constraints on/off is a Decision Point to discuss after seeing results.
5. **Discrete-time hazard model** — logistic regression on bank-quarter rows where the event is "fails next quarter" with time-varying covariates (Shumway 2001); convert to 4q/8q cumulative failure probability.
6. **Cox proportional hazards / survival forest** (optional, P2) via `lifelines` or `scikit-survival`, with mergers as censoring.

### 8.2 Evaluation protocol

- **Walk-forward backtest (primary):** for each test year `Y` from 2008 to the latest complete year: train on all rows with `window_end < Jan 1 of Y`'s first prediction date (Rule 6.2), predict every bank-quarter in `Y`, store predictions. Aggregate across years.
- **Fixed split (for quick iteration):** train on report quarters 2002Q1–2008Q4, test on 2010Q1–2013Q4 (verify the Rule 6.2 gap holds; adjust boundaries as needed).
- **2023 case study:** train on everything with windows closed before the 2022Q4 prediction date; score 2022Q3, 2022Q4 and 2023Q1 reports. Report the rank and percentile of Silicon Valley Bank (CERT 24735 — verify), Signature Bank and First Republic Bank among all banks, for the credit-only model vs the rate-aware model.

**Metrics** (report per year and pooled, always with the number of failures in that year):

- PR-AUC (primary), ROC-AUC.
- **Recall@k** — share of next-year failures captured in the top 1%, 2%, 5% of banks, and top 50/100 banks.
- **Lead time** — for each failed bank, how many quarters before failure it first entered the top 2%.
- **Calibration** — reliability curve, Brier score; apply isotonic/Platt calibration on a time-based validation slice.
- **Bootstrap confidence intervals** by resampling failure events (years with <10 failures must be flagged as low-confidence).
- False-positive review: list the highest-scored banks that did **not** fail, and check whether they were acquired (possible "rescued" troubled banks) — this is an insightful write-up section.

### 8.3 Explainability

- Logistic: coefficients / odds ratios with CIs.
- Gradient boosting: SHAP values (`TreeExplainer`); store the top-5 positive and negative drivers per bank-quarter.
- Peer comparison: peer group = asset-size bucket (<$100M, $100M–$1B, $1B–$10B, $10B–$100B, >$100B) × region; show the bank's percentile for each key ratio.

---

## 9. Prototypes

### Prototype 1 — "Foundation": data pipeline + labels + baselines

**Goal:** a correct, reproducible bank-quarter dataset with leakage-safe labels, and baseline models that prove the signal exists.

Features:

1. Python project scaffold (see Section 11) with `uv`, `ruff`, `pytest`, `pre-commit`.
2. FDIC API client with caching, pagination, retries and rate limiting.
3. Verified field map (`config/fields.yaml`) generated from the FDIC definitions file, with human-readable descriptions.
4. Ingest `/failures`, `/institutions`, `/history`, and `/financials` for **2001Q1 → latest** (⚖️ Decision Point: also include 1992–2000? Default: no, P1 starts at 2001).
5. Storage: raw JSON → typed Parquet → DuckDB database (`data/warehouse.duckdb`) with tables `failures`, `institutions`, `history`, `financials_raw`, `panel`.
6. Label builder implementing Section 5 exactly, with unit tests.
7. ~20 engineered features (Capital, Asset quality, Earnings, Liquidity, Concentration, basic growth) with YTD de-accumulation and unit tests.
8. Models: Texas-ratio baseline, small logistic, regularized logistic.
9. Fixed out-of-time split evaluation: PR-AUC, ROC-AUC, recall@top-2%, recall@top-100.
10. One analysis notebook (`notebooks/01_foundation.ipynb`) that tells the story: failure counts by year, feature distributions for failed vs surviving banks, baseline results.
11. CLI: `bankcanary ingest`, `bankcanary build-panel`, `bankcanary train --model logit`, `bankcanary evaluate`.

**Acceptance criteria:**

- [ ] Re-running the pipeline from cache produces byte-identical Parquet outputs (deterministic).
- [ ] Failure counts per year match the FDIC failed-bank list; ≥95% of 2001+ failures are matched to a prior Call Report.
- [ ] All label and leakage unit tests pass; the Rule 6.2 split function is used by every training path.
- [ ] The regularized logistic model beats the Texas ratio on PR-AUC on the fixed test split.
- [ ] The notebook runs top-to-bottom on a laptop in < 10 minutes after data is cached.

### Prototype 2 — "Depth": richer features, better models, honest backtests, the 2023 question

**Goal:** a research-grade model with walk-forward evaluation, calibration, explanations and the SVB case study.

Features (everything in P1, plus):

1. Raw FFIEC Call Report ingestion for securities (RC-B) and uninsured deposits (RC-O) with `IDRSSD ↔ CERT` crosswalk.
2. Interest-rate/run-risk features: `unrealized_loss_to_tier1`, `adjusted_tier1_leverage`, `uninsured_share`, `securities_to_assets`.
3. Trend features (Δ1q, Δ4q) and persistence features (e.g., consecutive loss quarters).
4. FRED macro features (state unemployment, state HPI, yield spread, fed funds change).
5. Full feature set (~60) with a feature registry: each feature has name, CAMELS group, formula, unit, and a one-line banking explanation (used later by the UI).
6. Models: LightGBM, discrete-time hazard model, optional Cox/survival forest.
7. Walk-forward backtest harness (Section 8.2) that saves one model artifact per test year (`models/walkforward/<year>/`) — **required by P3's time machine**.
8. Calibration (isotonic) and reliability plots.
9. SHAP explanations stored per bank-quarter.
10. Metrics suite: PR-AUC, ROC-AUC, recall@k, lead time, Brier, bootstrap CIs, per-year tables.
11. **2023 case study** notebook: credit-only vs rate-aware model ranks for SVB, Signature, First Republic; charts of their unrealized losses and uninsured share vs peers over 2020–2023.
12. False-positive analysis: top-scored non-failures and what happened to them (acquired? recovered?).
13. Sensitivity analyses: `H=4` vs `H=8`; censored rows kept vs dropped; `avail` lag 45 vs 60 vs 90 days.
14. Lightweight experiment tracking (a `runs/` directory with config + metrics JSON per run, or MLflow — ⚖️ Decision Point).
15. A written `docs/model_card.md` (intended use, data, label, metrics, limitations, ethical considerations).

**Acceptance criteria:**

- [ ] Walk-forward results exist for every year from 2008 to the latest complete year, with failure counts and CIs.
- [ ] The best model beats P1's regularized logistic on pooled walk-forward PR-AUC and recall@top-2%.
- [ ] Median lead time for 2009–2012 failures is reported (target: the model flags most failures ≥2 quarters ahead — report honestly whatever the result is).
- [ ] The 2023 notebook clearly shows how SVB/Signature/First Republic ranked under both models and explains why.
- [ ] Calibrated probabilities: reliability curve within reasonable tolerance on the test period; Brier score reported.
- [ ] Model card written.

### Prototype 3 (final) — "Product": automated pipeline + public web dashboard

**Goal:** a polished, deployed, automatically refreshed application that presents the research transparently.

⚖️ **Decision Point — stack.** Recommended default (matches the owner's existing F1 analytics project): **Python pipeline (offline scoring) → Neon Postgres → Next.js (App Router) on Vercel**. The web app only *reads* precomputed scores; no Python runs on Vercel. Alternative: Streamlit/Dash single app (faster to build, less polished). Present both before building.

Features:

1. **Scheduled pipeline** (GitHub Actions cron, e.g., weekly): check the FDIC API for a new `REPDTE`; if found, ingest → build features → score with the production model → compute SHAP → write to Postgres. Idempotent; logs a run record. Retraining is a separate, manually triggered workflow with a review step.
2. **Database schema** (Postgres): `banks`, `quarters`, `scores` (bank, quarter, horizon, prob, rank, percentile, model_version), `drivers` (top SHAP drivers), `ratios` (key ratios + peer percentiles), `failures`, `walkforward_scores` (historical predictions from the per-year models), `model_versions`, `pipeline_runs`.
3. **Risk leaderboard** (home page): latest quarter, sortable table — rank, bank, city/state, total assets, 12-month failure probability (with a "low / elevated / high" band), change vs prior quarter, top 3 drivers. Filters: state, size bucket, charter class. Search by name, city or CERT.
4. **Bank profile page**: probability timeline across all quarters (with a failure marker if the bank failed), CAMELS ratio panels with peer-median lines, SHAP waterfall for the latest quarter, plain-English explanation generated from the feature registry, basic facts (established date, charter, holding company).
5. **Time machine**: pick any quarter since 2006; the leaderboard is rebuilt from `walkforward_scores` (models trained only on data available then). Banks that later failed are marked, with "failed N months later". Show recall@top-2% for that quarter.
6. **Failure replay map**: US map (bank head-office locations) animated quarter by quarter 2006 → today; colour = risk band; failures appear as events. Include a play/pause control and a quarter slider.
7. **2023 case study page**: interactive version of the P2 notebook — SVB, Signature and First Republic vs peers; credit-only vs rate-aware ranks over time.
8. **Rate-shock scenario tool**: user picks a parallel rate shock (e.g., +100 to +400 bp) and an assumed securities duration; the tool approximates additional unrealized losses (`−duration × Δrate × securities`), recomputes adjusted capital ratios and re-scores banks, showing who moves up the list. Labelled clearly as a simplified approximation.
9. **Methodology & model card page**: data sources, label definition, leakage protections, walk-forward metrics table and charts, calibration plot, limitations, links to references.
10. **Downloads**: CSV of the current leaderboard and of any bank's history.
11. **Disclaimers** on every page footer: educational project; not a rating, not advice; FDIC insurance explanation with link to FDIC's official resources.
12. Quality: responsive layout, accessible charts (labels, contrast, keyboard-navigable tables), light/dark mode, loading/empty/error states, basic SEO metadata, Lighthouse ≥ 90 performance on the home page.
13. Engineering: CI (lint, type-check, tests) on every PR; `README.md` with architecture diagram, setup, and results summary; `.env.example`; reproducible `make`/`just` commands.

**Acceptance criteria:**

- [ ] A new quarter of data is picked up and published end-to-end without manual steps (demonstrate with a dry run against the latest quarter).
- [ ] The time machine for 2009Q2 reproduces the walk-forward results reported in the model card exactly.
- [ ] Every number shown in the UI can be traced to a database row and a model version.
- [ ] Deployed publicly; README links to the live site and summarizes results honestly, including the 2023 finding.

---

## 10. Step-by-step implementation plan

Each step lists its output. Do them in order; commit after each step.

### Phase A — Setup (P1)

1. **A1. Repo scaffold.** `uv init`, Python 3.12, `src/bankcanary/` package, `ruff`, `pytest`, `pre-commit`, `.gitignore` (ignore `data/`, `.env`, model artifacts), `.env.example`, `config/settings.yaml` (horizons, availability lag, date ranges, peer buckets). → Output: installable package, passing empty test suite.
2. **A2. FDIC API client.** `src/bankcanary/sources/fdic.py`: generic `get(endpoint, filters, fields)` with pagination via `offset`, retries/backoff, rate limit, on-disk cache keyed by (endpoint, filters, fields hash). → Unit tests with recorded fixtures (no live calls in tests).

### Phase B — Data (P1)

3. **B1. Field discovery.** Download the FDIC financial-variable definitions; write `scripts/build_field_map.py` that verifies every candidate code in Section 7.1, searches for the "find code" items by description keywords, and writes `config/fields.yaml` (code, description, unit, first available date). → Report of any codes not found; ask the owner if ambiguous.
4. **B2. Ingest failures, institutions, history.** Parse dates, normalize types, save Parquet + DuckDB tables. → Failure-count-by-year table printed and compared with the FDIC list.
5. **B3. Ingest financials** quarter by quarter 2001Q1 → latest for the verified field list. → `financials_raw` table; row counts per quarter (expect roughly 8,000–9,000 in 2001 declining to ~4,500 recently).
6. **B4. Panel build.** Join financials with institution attributes; compute `avail(q)`; derive each bank's `fail_date` and `exit_date` (from `/failures`, `/institutions` `ACTIVE`/`ENDEFYMD`, and `/history`). → `panel` table keyed by (`CERT`, `REPDTE`).
7. **B5. Labels.** Implement Section 5 + all unit tests. → Columns `y_4q`, `y_8q`, `window_end_4q`, `window_end_8q`, `censored_in_window_*`, `label_complete_*`.
8. **B6. Data-quality report.** Missingness by field × year, matched-failure rate, outlier summary. → `reports/data_quality.md` (auto-generated).

### Phase C — Features & baselines (P1)

9. **C1. YTD de-accumulation + averages** with unit tests.
10. **C2. P1 feature set** (~20 features in 7.2 excluding P2 items) in `src/bankcanary/features/`; each feature registered with metadata. → `features_v1` table.
11. **C3. Split utility** implementing Rule 6.2 with tests (a deliberately leaky split must fail the test).
12. **C4. Baselines**: Texas ratio, small logistic, regularized logistic in sklearn `Pipeline`s. → Model artifacts + metrics JSON.
13. **C5. Evaluation module**: PR-AUC, ROC-AUC, recall@k; plots saved to `reports/figures/`.
14. **C6. Notebook 01 + CLI commands.** → P1 acceptance checklist completed and reviewed with the owner.

### Phase D — Depth (P2)

15. **D1. FFIEC bulk ingestion** (RC-B, RC-O) + RSSD crosswalk; cross-check a few banks' totals against FDIC `SC`/`DEP`.
16. **D2. Rate/run-risk features**; plot SVB's unrealized losses vs peers 2020–2023 as a sanity check (should show a large loss relative to capital by 2022).
17. **D3. Trend, persistence and structure features.**
18. **D4. FRED client + macro features** (point-in-time by `avail(q)`).
19. **D5. Feature registry** with plain-English explanations (feeds the UI).
20. **D6. LightGBM model**; tuning with a time-based validation slice inside the training period.
21. **D7. Discrete-time hazard model** (and optional Cox/survival forest) with merger censoring.
22. **D8. Walk-forward harness**: loop over test years, save per-year models + predictions to `walkforward_scores`.
23. **D9. Calibration + metrics suite** (lead time, Brier, bootstrap CIs, per-year tables).
24. **D10. SHAP explanations** stored per bank-quarter.
25. **D11. Notebooks:** `02_models_and_backtest.ipynb`, `03_svb_2023_case_study.ipynb`, `04_false_positives.ipynb`, `05_sensitivity.ipynb`.
26. **D12. Model card** (`docs/model_card.md`). → P2 acceptance checklist reviewed with the owner.

### Phase E — Product (P3)

27. **E0. ⚖️ Stack decision** with the owner (Section 9, P3).
28. **E1. Postgres schema + export job** (`bankcanary publish`) writing scores, drivers, ratios, walk-forward results, model versions.
29. **E2. Scheduled pipeline** (GitHub Actions): detect new quarter → ingest → features → score → publish; secrets via repository secrets; run log table.
30. **E3. Web app skeleton**: layout, navigation, theme, disclaimer footer, data-access layer (server components reading Postgres).
31. **E4. Leaderboard page** with filters/search/CSV download.
32. **E5. Bank profile page** (timeline, CAMELS panels with peers, SHAP waterfall, plain-English drivers).
33. **E6. Time machine** (quarter picker over `walkforward_scores`, failed-later markers, per-quarter recall).
34. **E7. Failure replay map** (animated quarters).
35. **E8. 2023 case study page.**
36. **E9. Rate-shock scenario tool** (client-side recompute over a precomputed per-bank securities/capital snapshot).
37. **E10. Methodology/model-card page.**
38. **E11. Polish**: accessibility pass, responsive checks, performance, SEO metadata, error/empty states.
39. **E12. CI, README, architecture diagram, deploy.** → P3 acceptance checklist reviewed with the owner.

---

## 11. Suggested repository structure

```
bank-failure-early-warning/
├── PROJECT_SPEC.md            # this file
├── README.md
├── pyproject.toml
├── .env.example               # FDIC_API_KEY, FRED_API_KEY, DATABASE_URL
├── config/
│   ├── settings.yaml          # horizons, availability lag, date ranges, peer buckets
│   └── fields.yaml            # verified FDIC/FFIEC field map (generated)
├── src/bankcanary/
│   ├── sources/               # fdic.py, ffiec.py, fred.py
│   ├── storage/               # parquet + duckdb helpers
│   ├── panel/                 # panel build, exits, availability dates
│   ├── labels/                # Section 5
│   ├── features/              # registry + feature modules by CAMELS group
│   ├── splits/                # Rule 6.2 split utility
│   ├── models/                # baselines, logit, lgbm, hazard, survival
│   ├── evaluation/            # metrics, calibration, backtest, plots
│   ├── explain/               # SHAP, peer percentiles
│   ├── publish/               # Postgres export (P3)
│   └── cli.py
├── tests/                     # unit tests incl. labels, leakage, YTD
├── notebooks/                 # 01..05
├── reports/                   # generated data-quality + figures
├── docs/                      # model_card.md, methodology notes
├── models/                    # gitignored artifacts (walkforward/<year>/...)
├── data/                      # gitignored raw/, parquet/, warehouse.duckdb
├── web/                       # P3 Next.js app (if chosen)
└── .github/workflows/         # ci.yml, quarterly-refresh.yml, retrain.yml
```

**Python libraries:** `httpx` (or `requests`), `tenacity`, `polars` or `pandas`, `pyarrow`, `duckdb`, `scikit-learn`, `lightgbm`, `statsmodels`, `lifelines` (and/or `scikit-survival`), `shap`, `matplotlib`/`plotly`, `pydantic` (config), `typer` (CLI), `pytest`, `ruff`.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| FDIC field codes differ from the candidates here | Step B1 verifies every code against the official definitions; ask the owner when ambiguous |
| API access requires a key / rate limits | Owner registers; cache aggressively; use bulk downloads if paginating is slow |
| Very few failures after 2014 → noisy metrics | Always report failure counts and bootstrap CIs; pool years; frame 2015+ as qualitative |
| Label noise from rescued (acquired) troubled banks | Censoring flag + sensitivity analysis + false-positive write-up |
| Form changes (FFIEC 051 in 2017, CBLR in 2020, CECL adoption) | Missing-indicator features; document in data-quality report; test model stability across these dates |
| Leakage creeping in during iteration | Centralized split function, unit tests, feature-importance smoke test |
| Overclaiming | Model card, disclaimers, honest 2023 analysis |

---

## 13. Decision Points summary (⚖️ — ask the owner)

1. Include 1992–2000 data in P1? (Default: no.)
2. Monotonic constraints in the gradient boosting model? (Decide after seeing unconstrained results.)
3. Experiment tracking: plain JSON runs vs MLflow? (Default: plain JSON.)
4. P3 stack: Next.js + Neon + Vercel (default, matches the owner's F1 project) vs Streamlit/Dash.
5. P3 hosting of the map: bank head-office coordinates source (FDIC `/locations` or institution city → geocode) — confirm before building.
6. Any rename of the project from "BankCanary".
