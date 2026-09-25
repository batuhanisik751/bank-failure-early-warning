# BankCanary — bank failure early-warning system

> **Status:** Prototype 1 ("Foundation") in progress. Nothing here is a credit rating,
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

## Disclaimer

BankCanary is an educational project. It is **not a credit rating, not investment advice,
and not an official supervisory assessment**. Deposits at FDIC-insured banks are insured up
to $250,000 per depositor, per bank, per ownership category. See
[fdic.gov](https://www.fdic.gov/resources/deposit-insurance/) for official information.
