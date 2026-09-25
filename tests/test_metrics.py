"""Hand-checked cases for the evaluation metrics and plots."""

import hashlib
import logging
import math

import numpy as np
import pandas as pd
import pytest

from bankcanary.evaluation import metrics, plots

# 10 banks ranked by score; the failures sit at ranks 2 and 5.
SCORES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
Y = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]


def test_hand_computed_metrics():
    m = metrics.evaluate(Y, SCORES, k_counts=(3, 5))
    # AP = mean of precision at each positive: (1/2 + 2/5) / 2 = 0.45
    assert m["pr_auc"] == pytest.approx(0.45)
    # 16 positive/negative pairs; 0.8 beats 7 negatives, 0.5 beats 5 -> 12/16
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["recall_at_top3"] == pytest.approx(0.5)
    assert m["recall_at_top5"] == pytest.approx(1.0)
    # k = ceil(0.01 * 10) = 1: the top bank did not fail
    assert m["recall_at_1pct"] == 0.0
    assert m["n"] == 10 and m["n_failures"] == 2


def test_default_keys_and_k_counts_capped_at_n():
    m = metrics.evaluate(Y, SCORES)
    assert list(m) == [
        "pr_auc",
        "roc_auc",
        "recall_at_1pct",
        "recall_at_2pct",
        "recall_at_5pct",
        "recall_at_top50",
        "recall_at_top100",
        "n",
        "n_failures",
    ]
    assert m["recall_at_top50"] == 1.0


def test_tie_breaker_orders_ties_and_default_is_stable():
    y = [0, 1, 0]
    scores = [1.0, 1.0, 1.0]
    # Original order: the positive is second -> not in top 1.
    assert metrics.evaluate(y, scores, k_counts=(1,))["recall_at_top1"] == 0.0
    # Ascending cert tie-break: the positive has the smallest cert -> top 1.
    m = metrics.evaluate(y, scores, k_counts=(1,), tie_breaker=[30, 10, 20])
    assert m["recall_at_top1"] == 1.0
    assert list(metrics.rank_order(scores, [30, 10, 20])) == [1, 2, 0]
    assert list(metrics.rank_order(scores)) == [0, 1, 2]


def test_nan_scores_rank_last():
    y = [1, 0, 1]
    scores = [np.nan, 0.2, 0.9]
    assert list(metrics.rank_order(scores)) == [2, 1, 0]
    m = metrics.evaluate(y, scores, k_counts=(1, 2))
    assert m["recall_at_top1"] == pytest.approx(0.5)
    assert m["recall_at_top2"] == pytest.approx(0.5)
    assert not math.isnan(m["pr_auc"]) and not math.isnan(m["roc_auc"])


def test_no_positives_returns_nan_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="bankcanary.evaluation.metrics"):
        m = metrics.evaluate([0, 0, 0], [0.1, 0.2, 0.3])
    assert m["n_failures"] == 0 and m["n"] == 3
    assert all(math.isnan(m[k]) for k in m if k not in ("n", "n_failures"))
    assert "no failures" in caplog.text


def test_evaluate_by_year_pooled_matches_evaluate():
    df = pd.DataFrame({"y": Y, "s": SCORES, "year": [2009] * 5 + [2010] * 5})
    out = metrics.evaluate_by_year(df, "s", "y", "year")
    assert list(out["year"]) == ["2009", "2010", "pooled"]
    assert list(out.columns[:3]) == ["year", "n", "n_failures"]
    pooled = out[out["year"] == "pooled"].iloc[0]
    whole = metrics.evaluate(df["y"], df["s"])
    for k, v in whole.items():
        assert pooled[k] == pytest.approx(v)
    # 2010 has no failures: n_failures is still reported, metrics are NaN.
    row_2010 = out[out["year"] == "2010"].iloc[0]
    assert row_2010["n_failures"] == 0 and math.isnan(row_2010["pr_auc"])


def test_lead_time_quarters():
    q = pd.to_datetime(["2008-03-31", "2008-06-30", "2008-09-30"])
    df = pd.DataFrame(
        {
            "cert": [1, 2, 3] * 3,
            "repdte": np.repeat(q, 3),
            "s": [0.1, 0.9, 0.5, 0.9, 0.1, 0.5, 0.2, 0.1, 0.9],
            "fail_date": [pd.Timestamp("2009-02-01"), pd.NaT, pd.Timestamp("2008-12-05")] * 3,
        }
    )
    # top_frac=0.34 of 3 banks -> top 2 per quarter (ceil).
    out = metrics.lead_time_quarters(df, "s", "cert", "repdte", "fail_date", top_frac=0.34)
    assert list(out["cert"]) == [1, 3]
    by = out.set_index("cert")
    assert by.loc[1, "first_flag_repdte"] == pd.Timestamp("2008-06-30")
    assert by.loc[1, "lead_time_quarters"] == 3  # 2008Q2 -> 2009Q1
    assert by.loc[3, "lead_time_quarters"] == 3  # 2008Q1 -> 2008Q4
    # A failed bank that is never flagged gets NaN lead time.
    quiet = df.assign(s=np.where(df["cert"] == 1, 0.0, df["s"]))
    never = metrics.lead_time_quarters(quiet, "s", "cert", "repdte", "fail_date", top_frac=0.01)
    assert math.isnan(never.set_index("cert").loc[1, "lead_time_quarters"])
    assert never["n_quarters_scored"].tolist() == [3, 3]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plots_write_deterministic_pngs(tmp_path):
    scores = SCORES[:-1] + [np.nan]
    p1 = plots.pr_curve(Y, scores, tmp_path / "figs" / "pr.png")
    p2 = plots.pr_curve(Y, scores, tmp_path / "pr2.png")
    assert p1.exists() and p1.stat().st_size > 0 and p1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert _sha(p1) == _sha(p2)
    assert b"Software" not in p1.read_bytes()
    by_model = {"texas": metrics.evaluate(Y, SCORES), "logit": metrics.evaluate(Y, SCORES[::-1])}
    bars = plots.recall_at_k_bars(by_model, tmp_path / "bars.png")
    dist = plots.score_distributions(Y, scores, tmp_path / "dist.png")
    assert bars.stat().st_size > 0 and dist.stat().st_size > 0
    # Degenerate input still produces a file.
    assert plots.pr_curve([0, 0], [0.1, 0.2], tmp_path / "empty.png").exists()


def test_evaluate_by_event_collapses_same_day_holding_company_failures():
    # Banks 1 and 2 belong to holding company 7 and fail the same day: one event, scored
    # by the better-ranked sister (0.9). Bank 3 fails alone; bank 4 shares the holding
    # company but survives and must stay its own row.
    df = pd.DataFrame(
        {
            "cert": [1, 2, 3, 4, 5, 6, 7],
            "repdte": ["2009-06-30"] * 7,
            "score": [0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4],
            "y": [1, 1, 1, 0, 0, 0, 0],
            "rssdhcr": [7, 7, None, 7, None, None, None],
            "fail_date": ["2009-10-30", "2009-10-30", "2009-11-06", None, None, None, None],
        }
    )
    per_bank = metrics.evaluate(df["y"], df["score"])
    assert per_bank["n_failures"] == 3 and per_bank["pr_auc"] < 1.0  # bank 2 ranks last
    m = metrics.evaluate_by_event(df, "score", "y")
    assert m["n"] == 6 and m["n_failures"] == 2 and m["n_events"] == 2
    assert m["n_multi_bank_events"] == 1 and m["n_banks_in_multi_events"] == 2
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0
    # the same two banks in a *different* quarter are a separate unit
    two_quarters = pd.concat([df, df.assign(repdte="2009-09-30")], ignore_index=True)
    assert metrics.evaluate_by_event(two_quarters, "score", "y")["n_events"] == 4
    # without holding-company columns the result is the per-bank evaluation
    plain = metrics.evaluate_by_event(df.drop(columns=["rssdhcr", "fail_date"]), "score", "y")
    assert plain["n_events"] == 3 and plain["pr_auc"] == pytest.approx(per_bank["pr_auc"])
