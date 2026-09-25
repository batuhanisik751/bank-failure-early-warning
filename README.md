# BankCanary — bank failure early-warning system

> **Status:** Prototype 1 ("Foundation") complete; Prototype 2 not started. Nothing here is a credit rating,
> investment advice, or a supervisory assessment. See the disclaimer below.

BankCanary is an open, reproducible early-warning system for US bank failures. It ingests
every FDIC-insured bank's quarterly regulatory filings (Call Reports), computes the same
kinds of financial-health ratios bank examiners look at — capital, asset quality, earnings,
liquidity, concentration and interest-rate exposure — and estimates each bank's probability
of failing within the next year.

Unlike most bank-failure projects, BankCanary is built around **strict point-in-time
discipline**: every prediction uses only data that was public on the date the prediction
would have been made, and every model is tested on years it never saw. Its "time machine"
mode lets you pick any quarter since 2006 and see exactly what the model would have flagged
then — and which of those banks actually failed.

The project also asks a harder question: *would a model trained on the 2008 crisis have seen
Silicon Valley Bank coming?* It documents where the classic credit-risk view fails and how
interest-rate and deposit-run indicators close the gap.

## What it does

1. **Ingests** quarterly bank financials and the full failure history from the FDIC BankFind
   API (later: raw FFIEC Call Report schedules and macro data from FRED).
2. **Builds** a clean bank-quarter panel with point-in-time availability dates.
3. **Engineers** interpretable financial-health features grouped by CAMELS.
4. **Labels** each bank-quarter with "fails within H quarters" using a precise, leakage-safe
   definition.
5. **Trains** baseline, statistical and machine-learning models, including a hazard model.
6. **Backtests** walk-forward: train only on the past, predict the next year, repeat.
7. **Explains** every prediction and compares each bank to its peers.
8. **Publishes** a web dashboard (Prototype 3).

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create .venv and install everything
uv run pre-commit install    # lint + format on every commit
uv run pytest                # unit tests (no network access needed)
uv run bankcanary --help     # pipeline commands
```

Copy `.env.example` to `.env` if you need API keys. The FDIC API currently needs none.

## Project layout

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full specification and
[`docs/CONTRACT.md`](docs/CONTRACT.md) for the build contract (module layout, table
schemas, naming, storage paths) that every component follows.

## Results so far

Prototype 1 is complete: FDIC data from 2001Q1 to 2026Q2 is ingested and cached, the
bank-quarter panel (710,691 rows, 11,243 banks) carries leakage-safe labels, 43 CAMELS
features are computed, and three baselines are evaluated on a fixed out-of-time split.
The guided tour is [`notebooks/01_foundation.ipynb`](notebooks/01_foundation.ipynb); the
acceptance checklist is [`docs/P1_CHECKLIST.md`](docs/P1_CHECKLIST.md).

**Failures per year** (FDIC failed-bank list, `restype = FAILURE`; 574 failures, of which
573 have at least one prior Call Report in the panel):

| year | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| failures | 4 | 11 | 3 | 4 | 0 | 0 | 3 | 25 | 140 | 157 | 92 | 51 | 24 |

| year | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| failures | 18 | 8 | 5 | 8 | 0 | 4 | 4 | 0 | 0 | 5 | 2 | 2 | 4 |

\* through the failures pull of 2026-09-25.

**Baselines, 4-quarter horizon** (train on reports 2002Q1-2008Q4, 252,330 rows / 554
failures; test on 2010Q1-2013Q4, 118,696 rows / 833 failures; from
[`reports/p1_baselines.md`](reports/p1_baselines.md)):

| model | PR-AUC | ROC-AUC | recall @ top 2% | recall @ top 100 |
|---|---|---|---|---|
| texas (rank by Texas ratio) | 0.3726 | 0.9739 | 0.761 | 0.072 |
| logit_small (6 features) | 0.3720 | 0.9791 | 0.738 | 0.079 |
| logit (all 43 features, L2, C = 0.003) | 0.3867 | 0.9755 | 0.714 | 0.080 |

The honest reading: the signal is clearly there (a plain Texas-ratio ranking puts three
quarters of the banks that failed within a year inside its top 2%), the six-feature
logistic only ties the Texas ratio, and the all-feature logistic beats it on PR-AUC by a
small margin. That margin came from a lesson, not from a richer model: with balanced class
weights the same logistic scored 0.20, because weighting 554 training failures up by a
factor of 450 let its collinear capital measures overfit. The regularisation strength is
chosen on a validation slice inside the training years (`scripts/tune_logit_c.py`), never
on the test years. The report also gives every number with censored rows dropped and per
failure event (same-day holding-company failures counted once), and `reports/figures/`
holds the PR curves, score histograms and recall@k chart. Prototype 2 (feature selection,
gradient boosting, walk-forward backtest) has to widen the gap.

### Reproduce

```bash
uv sync
uv run bankcanary ingest                 # failures, institutions, history, financials 2001Q1+
uv run bankcanary build-panel
uv run bankcanary build-labels
uv run bankcanary build-features
uv run bankcanary dq-report              # reports/data_quality.md
uv run bankcanary train --model all      # models/{texas,logit_small,logit}/
uv run bankcanary evaluate               # reports/p1_baselines.md + reports/figures/
uv run python scripts/make_notebook_01.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_foundation.ipynb
```

The first `ingest` downloads about 100 quarters of financials and takes a while; every
later run reads the JSON cache and rebuilds the Parquet tables byte-identically. If the
`bankcanary` console script cannot find the package (macOS can mark the editable
install's `.pth` file hidden, which Python then skips), run `make fix-venv` or prefix the
commands with `PYTHONPATH=src`; `uv run pytest` works either way.

## Disclaimer

BankCanary is an educational project. It is **not a credit rating, not investment advice,
and not an official supervisory assessment**. Deposits at FDIC-insured banks are insured up
to $250,000 per depositor, per bank, per ownership category. See
[fdic.gov](https://www.fdic.gov/resources/deposit-insurance/) for official information.
