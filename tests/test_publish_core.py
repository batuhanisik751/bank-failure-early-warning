"""Publish builders on synthetic warehouse frames (no database, no model artefacts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankcanary.publish import TABLE_KEYS, core

Q = pd.to_datetime(["2008-03-31", "2008-06-30", "2008-09-30", "2008-12-31"])


def test_size_bucket_and_charter_labels():
    buckets = core.size_bucket(pd.Series([50_000, 100_000, 5_000_000, 2e8, np.nan]))
    assert list(buckets[:4]) == ["under_100m", "100m_1b", "1b_10b", "over_100b"]
    assert pd.isna(buckets.iloc[4])
    labels = core.charter_class_label(pd.Series(["N", "SM", "ZZ", None]))
    assert labels.iloc[0] == "National bank" and labels.iloc[1] == "State bank, Fed member"
    assert labels.iloc[2] == "ZZ" and pd.isna(labels.iloc[3])
    assert list(core.quarter_label(pd.Series(Q[:2]))) == ["2008Q1", "2008Q2"]


def test_rank_scores_bands_percentile_and_tie_break():
    n = 100
    frame = pd.DataFrame(
        {
            "cert": np.arange(n),
            "repdte": Q[0],
            "model": "gbdt_mono",
            "score": np.r_[np.full(2, 0.9), np.linspace(0.8, 0.0, n - 2)],
            "probability": np.r_[np.full(2, 0.5), np.linspace(0.4, 0.0, n - 2)],
        }
    )
    ranked = core.rank_scores(frame)
    assert list(ranked["rank"]) == list(range(1, n + 1))
    assert list(ranked["cert"][:2]) == [0, 1], "equal scores break ties by cert"
    assert (ranked["band"] == "high").sum() == 2
    assert (ranked["band"] == "elevated").sum() == 8
    assert (ranked["band"] == "low").sum() == 90
    assert ranked["percentile"].iloc[0] == pytest.approx(99.0)
    assert ranked["percentile"].iloc[-1] == pytest.approx(0.0)


def test_prior_quarter_delta_uses_previous_quarter_of_same_bank_and_model():
    frame = pd.DataFrame(
        {
            "cert": [1, 1, 1, 2],
            "repdte": [Q[0], Q[1], Q[3], Q[1]],
            "model": ["gbdt_mono"] * 4,
            "probability": [0.10, 0.25, 0.30, 0.05],
        }
    )
    delta = core.prior_quarter_delta(frame)
    assert np.isnan(delta[0]) and delta[1] == pytest.approx(0.15)
    assert np.isnan(delta[2]), "a missing 2008Q3 row breaks the chain"
    assert np.isnan(delta[3])


def _walkforward(models=("gbdt_mono", "hazard", "logit"), horizons=(4, 8)):
    rows = []
    rng = np.random.default_rng(0)
    for model in models:
        for horizon in horizons:
            for q in [pd.Timestamp("2007-12-31"), *Q]:
                for cert in range(1, 6):
                    rows.append(
                        {
                            "cert": cert,
                            "repdte": q,
                            "horizon": horizon,
                            "model": model,
                            "test_year": q.year,
                            "score": float(rng.uniform()),
                            "score_calibrated": float(rng.uniform()),
                            "y": int(cert == 5 and q.year == 2008),
                            "label_complete": True,
                            "censored": False,
                        }
                    )
    return pd.DataFrame(rows)


def test_build_scores_keeps_horizon_4_models_and_2008_onward_and_adds_production_rows():
    production = pd.DataFrame(
        {
            "cert": [1, 2],
            "repdte": pd.Timestamp("2009-03-31"),
            "horizon": 4,
            "model": "gbdt_mono",
            "test_year": pd.array([None, None], dtype="Int64"),
            "score": [0.2, 0.9],
            "score_calibrated": [0.1, 0.4],
        }
    )
    versions = {("gbdt_mono", 2008): "gm-2008", ("hazard", 2008): "hz-2008"}
    versions[("gbdt_mono", "production")] = "gm-prod"
    scores = core.build_scores(_walkforward(), production, versions)
    assert set(scores["model"]) == {"gbdt_mono", "hazard"}
    assert scores["repdte"].min() == pd.Timestamp("2008-03-31").date()
    assert set(scores["horizon"]) == {4}
    prod = scores[scores["repdte"] == pd.Timestamp("2009-03-31").date()]
    assert list(prod.sort_values("rank")["cert"]) == [2, 1]
    assert set(prod["model_version"]) == {"gm-prod"}
    back = scores[(scores["model"] == "hazard")]
    assert set(back["model_version"]) == {"hz-2008"}
    assert list(scores.columns) == [
        "cert",
        "repdte",
        "model",
        "horizon",
        "score",
        "probability",
        "rank",
        "percentile",
        "band",
        "delta_prob_prior_q",
        "model_version",
    ]
    assert not scores.duplicated(list(TABLE_KEYS["scores"])).any()
