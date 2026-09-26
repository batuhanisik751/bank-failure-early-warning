"""Isotonic calibration on the synthetic walk-forward frame: no data files, no network."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bankcanary import tracking
from bankcanary.evaluation import calibration as c
from bankcanary.evaluation import walkforward as w
from bankcanary.labels.build import horizon_columns
from bankcanary.splits import prediction_date, training_mask
from tests.test_hazard import LAG, make_frame, make_settings

YEAR = 2010


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


def test_calibration_slice_is_the_last_complete_year_inside_the_training_window(frame):
    inner, sl, bounds = c.calibration_masks(frame, 4, YEAR, lag_days=LAG)
    outer = training_mask(frame, 4, pd.Timestamp(year=YEAR, month=3, day=31), LAG)
    assert bounds["calibration_years"] == [2008] and bounds["sufficient"] is True
    assert set(frame.loc[sl, "repdte"].dt.year) == {2008}
    assert (outer | ~sl).all()  # the slice sits inside the year's training rows
    assert int(sl.sum()) == 4 * frame["cert"].nunique()
    cutoff = prediction_date(pd.Timestamp("2008-03-31"), LAG)
    assert bounds["calibration_prediction_date"] == str(cutoff.date())
    assert (frame.loc[inner, horizon_columns(4)[1]] < cutoff).all()
    assert bounds["inner_train_repdte_max"] == "2006-12-31"
    assert bounds["n_inner_train"] == int(inner.sum()) and inner.sum() > 0
    assert bounds["positives_calibration"] == int(frame.loc[sl, "y_4q"].sum())
    assert inner.name == f"inner_train_4q_{YEAR}" and sl.name == f"calibration_4q_{YEAR}"
    # the hazard's inner model fits at 1q: more rows, windows still closed before the slice
    inner1, sl1, b1 = c.calibration_masks(frame, 4, YEAR, fit_horizon=1, lag_days=LAG)
    assert sl1.equals(sl) and inner1.sum() > inner.sum()
    assert (frame.loc[inner1, horizon_columns(1)[1]] < cutoff).all()
    closed = frame[frame[horizon_columns(1)[1]] < cutoff]["repdte"].max()
    assert b1["inner_train_repdte_max"] == str(closed.date()) == "2007-12-31"  # leap year


def test_calibration_slice_widens_backwards_when_it_holds_too_few_failures(frame, monkeypatch):
    _, sl, bounds = c.calibration_masks(frame, 4, YEAR, lag_days=LAG)
    one_year = bounds["positives_calibration"]
    monkeypatch.setattr(c, "MIN_CALIBRATION_POSITIVES", one_year + 1)
    _, wide, b2 = c.calibration_masks(frame, 4, YEAR, lag_days=LAG)
    assert b2["calibration_years"] == [2007, 2008] and b2["sufficient"] is True
    assert b2["positives_calibration"] > one_year and wide.sum() > sl.sum()
    assert b2["inner_train_repdte_max"] == "2005-12-31"
    monkeypatch.setattr(c, "MIN_CALIBRATION_POSITIVES", 10**6)
    _, narrow, b3 = c.calibration_masks(frame, 4, YEAR, lag_days=LAG)
    assert b3["sufficient"] is False and b3["calibration_years"] == [2008]
    assert narrow.equals(sl)
    with pytest.raises(ValueError, match="no complete year"):
        c.calibration_masks(frame, 4, 2002, lag_days=LAG)


def test_model_params_and_path_helpers(tmp_path):
    assert c.model_params({"model": "logit", "C": 0.01}) == {"C": 0.01}
    assert c.model_params({"model": "gbdt", "params": {"num_leaves": 7}}) == {"num_leaves": 7}
    assert c.model_params({"model": "texas"}) == {}
    settings = make_settings(tmp_path)
    assert c.calibration_path(settings, 2010, "gbdt", 8) == (
        tmp_path / "models" / "walkforward" / "2010" / "gbdt_8q" / "calibration.joblib"
    )


@pytest.mark.parametrize("model", ["logit", "hazard"])
def test_fit_calibrator_fills_a_monotone_map_and_logs_a_deterministic_run(tmp_path, frame, model):
    settings = make_settings(tmp_path)
    w.fit_year(frame, settings, YEAR, model, 4)
    result = c.fit_calibrator(frame, settings, YEAR, model, 4)
    assert result.year == YEAR and result.model == model and result.horizon == 4
    out = w.model_dir(settings, YEAR, model)
    assert (out / "calibration.joblib").exists() and (out / "calibration.json").exists()
    scores = pd.read_parquet(w.scores_path(settings, YEAR, model, 4))
    assert list(scores.columns) == list(w.SCORE_COLUMNS)
    cal = scores["score_calibrated"].to_numpy()
    assert not np.isnan(cal).any() and ((cal >= 0) & (cal <= 1)).all()
    order = np.argsort(scores["score"].to_numpy(), kind="stable")
    assert (np.diff(cal[order]) >= -1e-12).all()  # isotonic: never decreasing in the score
    np.testing.assert_allclose(cal, result.calibrator.predict(scores["score"].to_numpy()))
    cfg, m = result.config, result.metrics
    assert cfg["model"] == model and cfg["fit_horizon"] == (1 if model == "hazard" else 4)
    assert cfg["method"] == "isotonic" and cfg["calibration_years"] == [2008]
    assert cfg["test_year"] == YEAR and cfg["availability_lag_days"] == LAG
    assert cfg["params"] == c.model_params(json.loads((out / "config.json").read_text()))
    assert m["n"] == len(scores) and m["n_failures"] == int(scores["y"].sum())
    assert m["brier_raw"] >= 0 and m["brier_calibrated"] >= 0 and 0 <= m["mean_calibrated"] <= 1
    assert m["n_thresholds"] >= 1 and 0 <= m["extrapolated_share"] <= 1
    run_dir = tmp_path / "runs" / "calibrate" / tracking.run_id("calibrate", cfg)
    assert (run_dir / "metrics.json").exists()
    assert tracking.find_metrics("calibrate", cfg, settings)["n"] == len(scores)
    calibrator, saved_cfg, saved_m = c.load_calibrator(settings, YEAR, model, 4)
    assert saved_cfg["calibration_start"] == "2008-03-31" and saved_m["n"] == m["n"]
    np.testing.assert_allclose(calibrator.predict(scores["score"].to_numpy()), cal)
    payload = json.loads((out / "calibration.json").read_text())
    assert payload["config"]["inner_train_repdte_max"] == cfg["inner_train_repdte_max"]
    # re-running gives the same run id and the same map
    again = c.fit_calibrator(frame, settings, YEAR, model, 4, save=False)
    assert tracking.run_id("calibrate", again.config) == run_dir.name
    np.testing.assert_allclose(again.scores["score_calibrated"], cal)


def test_fit_calibrator_refuses_rankings_and_missing_fits(tmp_path, frame):
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError, match="not a probability model"):
        c.fit_calibrator(frame, settings, YEAR, "texas", 4)
    with pytest.raises(FileNotFoundError):
        c.fit_calibrator(frame, settings, YEAR, "logit", 4)
    with pytest.raises(FileNotFoundError):
        c.load_calibrator(settings, YEAR, "logit", 4)


def test_calibrate_year_skips_models_without_a_fit_and_rebuilds_the_table(tmp_path, frame, caplog):
    settings = make_settings(tmp_path)
    w.fit_year(frame, settings, YEAR, "logit", 4)
    with pytest.raises(ValueError, match="cannot calibrate"):
        c.calibrate_year(frame, settings, YEAR, ["texas"], 4)
    results = c.calibrate_year(frame, settings, YEAR, ["logit", "gbdt"], 4, rebuild=True)
    assert [r.model for r in results] == ["logit"]
    assert any("no walk-forward gbdt fit" in rec.message for rec in caplog.records)
    table = w.read_scores_table(settings, 4)
    assert (table["model"] == "logit").all() and table["score_calibrated"].notna().all()
    assert len(table) == 4 * frame["cert"].nunique()


def test_calibrate_commands_are_registered():
    from typer.testing import CliRunner

    from bankcanary.cli import app

    result = CliRunner().invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0 and "--all-years" in result.output and "--model" in result.output
    result = CliRunner().invoke(app, ["metrics-report", "--help"])
    assert result.exit_code == 0 and "--horizon" in result.output
    result = CliRunner().invoke(app, ["calibrate", "--year", "2010", "--all-years"])
    assert result.exit_code != 0


def test_slice_scorer_defaults_per_model_and_is_recorded(tmp_path, frame):
    assert [c.default_slice_scorer(m) for m in ("logit", "hazard")] == ["full", "full"]
    assert [c.default_slice_scorer(m) for m in ("gbdt", "gbdt_mono")] == ["inner", "inner"]
    settings = make_settings(tmp_path)
    w.fit_year(frame, settings, YEAR, "logit", 4)
    full = c.fit_calibrator(frame, settings, YEAR, "logit", 4, save=False)
    inner = c.fit_calibrator(frame, settings, YEAR, "logit", 4, save=False, slice_scorer="inner")
    assert full.config["slice_scorer"] == "full" and inner.config["slice_scorer"] == "inner"
    assert tracking.run_id("calibrate", full.config) != tracking.run_id("calibrate", inner.config)
    # the full scorer uses the saved full-window model: its slice scores are that model's
    pipeline, features, _ = w.load_year(settings, YEAR, "logit", 4)
    _, sl, _ = c.calibration_masks(frame, 4, YEAR, lag_days=LAG, require_inner=False)
    from bankcanary.models.baselines import score_pipeline

    own = score_pipeline(pipeline, frame.loc[sl, list(features)])
    assert full.metrics["slice_score_max"] == pytest.approx(float(np.max(own)))
    assert full.metrics["slice_score_max"] != pytest.approx(inner.metrics["slice_score_max"])
    with pytest.raises(ValueError, match="slice_scorer"):
        c.fit_calibrator(frame, settings, YEAR, "logit", 4, save=False, slice_scorer="platt")


def test_calibration_masks_can_ignore_the_inner_rows_when_widening(frame, monkeypatch):
    _, _, strict = c.calibration_masks(frame, 4, YEAR, lag_days=LAG)
    monkeypatch.setattr(c, "MIN_CALIBRATION_POSITIVES", strict["positives_inner_train"] + 1)
    _, _, widened = c.calibration_masks(frame, 4, YEAR, lag_days=LAG)
    _, _, loose = c.calibration_masks(frame, 4, YEAR, lag_days=LAG, require_inner=False)
    # inner positives only shrink as the slice widens: the strict rule never succeeds
    assert widened["sufficient"] is False and widened["calibration_years"] == [2008]
    # the slice-only rule keeps widening and succeeds once the slice holds enough failures
    assert loose["calibration_years"][0] < 2008 and loose["calibration_years"][-1] == 2008
    assert loose["sufficient"] is (loose["positives_calibration"] >= c.MIN_CALIBRATION_POSITIVES)
