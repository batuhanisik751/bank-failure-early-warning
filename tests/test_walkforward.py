"""Walk-forward harness on the synthetic features_v2 + labels frame: no data files, no network."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bankcanary import tracking
from bankcanary.evaluation import walkforward as w
from bankcanary.labels.build import horizon_columns
from bankcanary.splits import prediction_date
from tests.test_hazard import LAG, make_frame, make_settings

YEAR = 2010


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


def test_latest_complete_year_stops_before_an_incomplete_quarter(frame):
    assert w.latest_complete_year(frame, 4) == 2015
    partial = frame.copy()
    complete_col = horizon_columns(4)[3]
    partial.loc[partial["repdte"] == pd.Timestamp("2015-12-31"), complete_col] = False
    assert w.latest_complete_year(partial, 4) == 2014
    assert w.test_years(partial, 4) == list(range(2008, 2015))
    with pytest.raises(ValueError):
        w.latest_complete_year(partial.assign(**{complete_col: False}), 4)


def test_year_masks_follow_rule_62_at_the_fit_label(frame):
    train, test = w.year_masks(frame, 4, YEAR, lag_days=LAG)
    cutoff = prediction_date(pd.Timestamp(year=YEAR, month=3, day=31), LAG)
    assert (frame.loc[train, horizon_columns(4)[1]] < cutoff).all()
    assert frame.loc[train, "repdte"].max() == pd.Timestamp("2008-12-31")
    assert set(frame.loc[test, "repdte"].dt.year) == {YEAR}
    assert int(test.sum()) == 4 * frame["cert"].nunique()
    hazard_train, hazard_test = w.year_masks(frame, 4, YEAR, fit_horizon=1, lag_days=LAG)
    assert hazard_test.equals(test)
    assert hazard_train.sum() > train.sum()
    assert frame.loc[hazard_train, "repdte"].max() == pd.Timestamp("2009-09-30")
    assert w.fit_horizon_of("hazard", 4) == 1 and w.fit_horizon_of("gbdt", 8) == 8


def test_build_model_rejects_unknown_names(tmp_path):
    with pytest.raises(ValueError, match="unknown walk-forward model"):
        w.build_model("cox", make_settings(tmp_path))


@pytest.mark.parametrize("model", w.MODELS)
def test_fit_year_saves_artefacts_scores_and_a_deterministic_run(tmp_path, frame, model):
    settings = make_settings(tmp_path)
    result = w.fit_year(frame, settings, YEAR, model, 4)
    out = w.model_dir(settings, YEAR, model)
    assert out == tmp_path / "models" / "walkforward" / str(YEAR) / model
    for name in ("pipeline.joblib", "config.json", "features.json", "metrics.json"):
        assert (out / name).exists()
    scores = pd.read_parquet(w.scores_path(settings, YEAR, model, 4))
    assert list(scores.columns) == list(w.SCORE_COLUMNS)
    assert scores["score_calibrated"].isna().all()
    assert (scores["test_year"] == YEAR).all() and (scores["horizon"] == 4).all()
    assert (scores["model"] == model).all() and scores["label_complete"].all()
    assert len(scores) == 4 * frame["cert"].nunique()
    config = json.loads((out / "config.json").read_text())
    assert config["test_year"] == YEAR and config["horizon"] == 4
    assert config["fit_horizon"] == (1 if model == "hazard" else 4)
    assert config["train_repdte_max"] == ("2009-09-30" if model == "hazard" else "2008-12-31")
    run_dir = tmp_path / "runs" / "walkforward" / tracking.run_id("walkforward", result.config)
    assert (run_dir / "metrics.json").exists()
    assert tracking.find_metrics("walkforward", config, settings)["n"] == len(scores)
    pipeline, features, _ = w.load_year(settings, YEAR, model)
    assert features == result.features
    _, test = w.year_masks(frame, 4, YEAR, lag_days=LAG)
    rescored = w.score_rows(pipeline, frame.loc[test], features, model, 4, YEAR)
    np.testing.assert_allclose(rescored["score"], scores["score"])


def test_hazard_scores_are_the_converted_quarterly_hazard(tmp_path, frame):
    settings = make_settings(tmp_path)
    result = w.fit_year(frame, settings, YEAR, "hazard", 4, save=False)
    _, test = w.year_masks(frame, 4, YEAR, fit_horizon=1, lag_days=LAG)
    h = result.pipeline.predict_proba(frame.loc[test, result.features])[:, 1]
    np.testing.assert_allclose(result.scores["score"], 1 - (1 - h) ** 4)
    assert 0.0 < result.metrics["brier"] < 1.0
    assert result.paths == {}


def test_texas_has_no_brier_and_ranks_by_the_ratio(tmp_path, frame):
    result = w.fit_year(frame, make_settings(tmp_path), YEAR, "texas", 4, save=False)
    _, test = w.year_masks(frame, 4, YEAR, lag_days=LAG)
    expected = frame.loc[test, "texas_ratio"].fillna(-1.0).to_numpy()
    np.testing.assert_allclose(result.scores["score"], expected)
    assert result.metrics["brier"] is None


def test_gbdt_iteration_cap_is_recorded(tmp_path, frame):
    settings = make_settings(tmp_path)
    result = w.fit_year(frame, settings, YEAR, "gbdt", 4, n_estimators=5, save=False)
    assert result.config["params"]["n_estimators"] == 5
    assert result.config["iterations_capped"] is True
    assert result.config["backend"] == "lightgbm"
    plain = w.fit_year(frame, settings, YEAR, "gbdt", 4, save=False)
    assert plain.config["iterations_capped"] is False
    assert tracking.run_id("walkforward", plain.config) != tracking.run_id(
        "walkforward", result.config
    )


def test_rebuild_scores_table_is_deterministic_and_the_report_covers_every_horizon(tmp_path, frame):
    from bankcanary.storage.duckdb import connect
    from bankcanary.storage.parquet import table_path

    settings = make_settings(tmp_path)
    for year in (YEAR, YEAR + 1):
        for model in ("texas", "logit"):
            w.fit_year(frame, settings, year, model, 4)
    w.fit_year(frame, settings, YEAR, "logit", 8)
    assert w.model_dir(settings, YEAR, "logit", 8).name == "logit_8q"
    table = w.rebuild_scores_table(settings)
    first = table_path(settings, w.TABLE).read_bytes()
    assert list(table.columns) == list(w.SCORE_COLUMNS)
    assert len(table) == 5 * 4 * frame["cert"].nunique()
    keys = table[list(w.TABLE_KEY)]
    assert not keys.duplicated().any()
    assert keys.sort_values(list(w.TABLE_KEY), kind="mergesort").index.equals(keys.index)
    con = connect(settings, read_only=True)
    try:
        assert con.execute(f'SELECT count(*) FROM "{w.TABLE}"').fetchone()[0] == len(table)
    finally:
        con.close()
    w.rebuild_scores_table(settings)
    assert table_path(settings, w.TABLE).read_bytes() == first
    per_year = w.per_year_tables(w.read_scores_table(settings, 4))
    assert set(per_year) == {"texas", "logit"}
    assert per_year["logit"]["year"].tolist() == [str(YEAR), str(YEAR + 1), "pooled"]
    pooled = w.pooled_table(per_year)
    assert pooled["pr_auc"].is_monotonic_decreasing
    path = w.write_walkforward_report(settings, 4)
    text = path.read_text()
    for needle in ("## Horizon 4q", "## Horizon 8q", "### logit", "### logit (8q)", "| pooled |"):
        assert needle in text
    assert "not available" in text  # no models/logit/metrics.json in the temp settings
    with pytest.raises(ValueError, match="no walk-forward scores at 2q"):
        w.write_walkforward_report(settings, 2)


def test_inner_masks_nest_inside_each_years_training_period(frame):
    inner, validation, bounds = w.inner_masks(frame, 4, 2008, lag_days=LAG)
    assert (bounds["validation_start"], bounds["validation_end"]) == ("2005-03-31", "2006-12-31")
    assert bounds["inner_train_repdte_max"] == "2003-12-31" and bounds["sufficient"]
    assert bounds["validation_quarters"] == w.VALIDATION_QUARTERS
    assert int(validation.sum()) == 8 * frame["cert"].nunique()
    assert not (inner & validation).any()
    # The hazard's inner model fits at 1q, so its inner rows run three quarters later.
    _, _, hazard = w.inner_masks(frame, 4, 2008, fit_horizon=1, lag_days=LAG)
    assert hazard["inner_train_repdte_max"] == "2004-09-30"
    assert (hazard["validation_start"], hazard["validation_end"]) == (
        bounds["validation_start"],
        bounds["validation_end"],
    )
    # From 2010 on the slice is the D6/D7 one (reports 2007Q1-2008Q4, inner cut at 2005Q4).
    _, _, later = w.inner_masks(frame, 4, 2010, lag_days=LAG)
    assert (later["validation_start"], later["validation_end"]) == ("2007-03-31", "2008-12-31")
    assert later["inner_train_repdte_max"] == "2005-12-31"
    with pytest.raises(ValueError, match="no training rows"):
        w.inner_masks(frame[frame["repdte"] >= "2008-01-01"], 4, 2008, lag_days=LAG)


def test_inner_masks_widen_the_slice_and_fall_back_when_failures_are_too_sparse(frame):
    quiet = frame.copy()
    y_col = horizon_columns(4)[0]
    # No failures in 2005-2006: the 2008 slice must widen back until it finds some.
    quiet.loc[quiet["repdte"].dt.year.isin([2005, 2006]), y_col] = 0
    _, validation, bounds = w.inner_masks(quiet, 4, 2008, lag_days=LAG)
    assert bounds["sufficient"] and bounds["validation_start"] == "2004-03-31"
    assert bounds["validation_quarters"] == 12
    assert int(quiet.loc[validation, y_col].sum()) >= w.MIN_VALIDATION_POSITIVES
    none = frame.assign(**{y_col: 0})
    _, _, bounds = w.inner_masks(none, 4, 2008, lag_days=LAG)
    assert not bounds["sufficient"] and bounds["validation_quarters"] == w.VALIDATION_QUARTERS


def test_tune_year_picks_the_best_validation_pr_auc_caches_and_falls_back(tmp_path, frame):
    settings = make_settings(tmp_path)
    tuning = w.tune_year(frame, settings, 2008, "logit", 4)
    grid = w.candidate_grid("logit", settings)
    assert [row["params"] for row in tuning["grid"]] == grid and tuning["fallback"] is None
    best = max(tuning["grid"], key=lambda row: row["pr_auc"])
    assert tuning["selected"] == best["params"] and tuning["metric"] == "pr_auc"
    assert len(list((tmp_path / "runs" / w.TUNING_RUN).iterdir())) == len(grid)
    again = w.tune_year(frame, settings, 2008, "logit", 4)
    assert again == tuning
    assert len(list((tmp_path / "runs" / w.TUNING_RUN).iterdir())) == len(grid)
    assert w.tune_year(frame, settings, 2008, "texas", 4)["selected"] == {}
    quiet = frame.assign(**{horizon_columns(4)[0]: 0})
    fallback = w.tune_year(quiet, settings, 2008, "gbdt", 4)
    assert fallback["selected"] == w.fallback_params("gbdt", settings)
    assert "most regularised" in fallback["fallback"] and fallback["grid"] == []


def test_fit_year_records_and_saves_the_tuning(tmp_path, frame):
    settings = make_settings(tmp_path)
    result = w.fit_year(frame, settings, 2009, "hazard", 4)
    saved = json.loads((result.paths["tuning"]).read_text())
    assert saved["selected"] == result.tuning["selected"] == {"C": result.config["C"]}
    # The slice is taken at the scoring horizon (4q closes at 2007Q4 for 2009), not at 1q.
    assert result.config["tuning"]["validation_end"] == "2007-12-31"
    assert "grid" not in result.config["tuning"] and len(saved["grid"]) == len(w.C_GRID)


def test_gbdt_mono_walks_forward_under_the_registry_constraints(tmp_path, frame):
    from bankcanary.models.gbdt import describe

    settings = make_settings(tmp_path)
    mono = w.fit_year(frame, settings, YEAR, "gbdt_mono", 4, n_estimators=5, save=False)
    plain = w.fit_year(frame, settings, YEAR, "gbdt", 4, n_estimators=5, save=False)
    assert mono.config["monotone"] is True and plain.config["monotone"] is False
    assert describe(mono.pipeline)["monotone"] is True
    assert (mono.scores["model"] == "gbdt_mono").all()
    assert w.model_dir(settings, YEAR, "gbdt_mono").name == "gbdt_mono"
    assert tracking.run_id("walkforward", mono.config) != tracking.run_id(
        "walkforward", plain.config
    )
