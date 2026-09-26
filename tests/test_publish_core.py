"""Publish builders on synthetic warehouse frames (no database, no model artefacts)."""

from __future__ import annotations

import json
from pathlib import Path

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


def _run_record(runs: Path, run_id: str, **config) -> None:
    (runs / "walkforward" / run_id).mkdir(parents=True)
    base = {"model": "gbdt_mono", "horizon": 4, "test_year": 2008, "features_version": "v2"}
    (runs / "walkforward" / run_id / "config.json").write_text(json.dumps({**base, **config}))
    (runs / "walkforward" / run_id / "metrics.json").write_text("{}")


def test_walkforward_version_reads_the_stamped_config_and_never_touches_git(tmp_path):
    """No ``models/walkforward`` artefact: the run record of (model, year, horizon 4) names
    the version, a stamped record before an unstamped one; other models, years and
    horizons are ignored; an artefact beats every record."""
    runs, models = tmp_path / "runs", tmp_path / "models"
    _run_record(runs, "walkforward-4q-zzz", train_repdte_max="2006-12-31")
    _run_record(runs, "walkforward-4q-aaa", train_repdte_max="2006-12-31",
                model_version="gbdt_mono-2006-12-31-1234567", model_hash="1234567" * 4)  # fmt: skip
    _run_record(runs, "walkforward-8q-bbb", horizon=8, train_repdte_max="2005-12-31")
    _run_record(runs, "walkforward-4q-ccc", model="hazard", train_repdte_max="2007-12-31")
    _run_record(runs, "walkforward-4q-ddd", test_year=2009, train_repdte_max="2007-12-31")
    row = core.walkforward_version(2008, "gbdt_mono", models, runs)
    assert row["model_version"] == "gbdt_mono-2006-12-31-1234567", row
    assert row["train_end_repdte"] == "2006-12-31" and "run record" in row["notes"]
    assert row["git_sha"] is None and row["trained_at"] is None
    unstamped = core.walkforward_version(2008, "hazard", models, runs)
    assert unstamped["model_version"] == "hazard-2007-12-31-unknown"
    assert core.walkforward_version(2010, "gbdt_mono", models, runs) is None
    art = models / "walkforward" / "2008" / "gbdt_mono"
    art.mkdir(parents=True)
    config = {"model": "gbdt_mono", "train_repdte_max": "2006-12-31", "features_version": "v2"}
    config.update(model_version="gbdt_mono-2006-12-31-fedcba9", model_hash="fedcba9" * 4)
    (art / "config.json").write_text(json.dumps(config))
    (art / "pipeline.joblib").write_bytes(b"pipe")
    row = core.walkforward_version(2008, "gbdt_mono", models, runs)
    assert row["model_version"] == "gbdt_mono-2006-12-31-fedcba9" and "artefact" in row["notes"]
    assert row["trained_at"] is not None and "fedcba9" in row["notes"]


def test_build_model_versions_never_maps_a_backtest_year_to_production(tmp_path, caplog):
    production = tmp_path / "models" / "production"
    for model in core.MODELS:
        (production / model).mkdir(parents=True)
        meta = {"model_version": f"{model}-2022-12-31-prod", "model": model,
                "train_end_repdte": "2022-12-31", "model_hash": "prod" * 10}  # fmt: skip
        (production / model / "model_version.json").write_text(json.dumps(meta))
    _run_record(tmp_path / "runs", "walkforward-4q-aaa", train_repdte_max="2006-12-31")
    with caplog.at_level("WARNING", logger="bankcanary.publish.core"):
        frame, lookup = core.build_model_versions(
            production, tmp_path / "models", tmp_path / "runs", years=[2008]
        )
    assert lookup[("gbdt_mono", 2008)] == "gbdt_mono-2006-12-31-unknown"
    assert lookup[("hazard", 2008)] == "hazard-wf2008-unknown", "derived, not production"
    assert "hazard 2008" in caplog.text
    derived = frame.set_index("model_version").loc["hazard-wf2008-unknown"]
    assert derived["train_end_repdte"] is None and derived["git_sha"] is None
    assert lookup[("gbdt_mono", "production")] == "gbdt_mono-2022-12-31-prod"
    assert set(frame["model_version"]) == set(lookup.values()) and len(frame) == 4


def test_build_scores_refuses_rows_without_a_known_version():
    versions = {("gbdt_mono", 2008): "gm-2008"}
    with pytest.raises(ValueError, match=r"\('hazard', 2008\)"):
        core.build_scores(_walkforward(), None, versions)


def test_scored_quarters_and_build_quarters_keep_only_quarters_with_scores():
    from bankcanary.publish import core

    quarters = pd.date_range("2007-03-31", "2009-12-31", freq="QE-DEC")
    labels = pd.DataFrame(
        {
            "cert": 1,
            "repdte": quarters,
            "y_4q": 0,
            "label_complete_4q": quarters.year <= 2008,
            "dropped_failed_before_avail": False,
        }
    )
    panel = pd.DataFrame({"cert": 1, "repdte": quarters, "avail_date": quarters})
    walkforward = pd.DataFrame(
        {
            "repdte": list(quarters[quarters.year == 2008]) + [pd.Timestamp("2007-12-31")],
            "horizon": [4, 4, 4, 8, 4],
            "model": ["gbdt_mono", "hazard", "gbdt_mono", "gbdt_mono", "gbdt_mono"],
        }
    )
    scored = core.scored_quarters(walkforward, labels)
    # 2007Q4 is before FIRST_SCORED_YEAR, 2008Q4 has only an 8q row, 2009 is production
    assert scored == set(pd.to_datetime(["2008-03-31", "2008-06-30", "2008-09-30"])) | set(
        quarters[quarters.year == 2009]
    )
    assert core.scored_quarters(pd.DataFrame(), pd.DataFrame()) == set()
    versions = {2008: "gbdt_mono-2006-12-31-abc1234", "production": "gbdt_mono-2008-12-31-def5678"}
    table = core.build_quarters(panel, labels, versions, scored)
    assert [str(d) for d in table["repdte"]] == [
        "2008-03-31",
        "2008-06-30",
        "2008-09-30",
        "2009-03-31",
        "2009-06-30",
        "2009-09-30",
        "2009-12-31",
    ]
    assert table["model_year"].tolist()[:3] == [2008, 2008, 2008]
    assert table["model_year"].isna().tolist()[3:] == [True] * 4
    assert set(table["model_version"][3:]) == {"gbdt_mono-2008-12-31-def5678"}
    assert len(core.build_quarters(panel, labels, versions)) == len(quarters)


def test_walkforward_metrics_carries_logit_v1_but_scores_does_not():
    """P2 checklist criterion 2 is read from the database: ``walkforward_metrics`` has the
    ``logit_v1`` rows (per year and pooled) while ``scores`` keeps :data:`core.MODELS`."""
    wf = _walkforward(models=("gbdt_mono", "hazard", "logit_v1", "logit"), horizons=(4,))
    metrics = core.build_walkforward_metrics(wf)
    assert core.METRICS_MODELS == ("gbdt_mono", "hazard", "logit_v1")
    assert set(metrics["model"]) == set(core.METRICS_MODELS)
    pooled = metrics[metrics["test_year"] == core.POOLED_YEAR].set_index("model")
    per_year = metrics[metrics["test_year"] != core.POOLED_YEAR].groupby("model")
    assert pooled["n"].eq(per_year["n"].sum()).all()
    assert pooled["n_failures"].eq(per_year["n_failures"].sum()).all()
    assert sorted(per_year.get_group("logit_v1")["test_year"]) == [2007, 2008]
    versions = {(m, y): f"{m}-{y}" for m in core.MODELS for y in (2007, 2008)}
    scores = core.build_scores(wf, None, versions)
    assert set(scores["model"]) == set(core.MODELS)
    absent = core.build_walkforward_metrics(wf[wf["model"] != "logit_v1"])
    assert set(absent["model"]) == set(core.MODELS)
