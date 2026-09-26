"""Binned isotonic maps on trailing out-of-sample years: synthetic frame, no data files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from bankcanary import tracking
from bankcanary.evaluation import calibration as c
from bankcanary.evaluation import walkforward as w
from bankcanary.labels.build import horizon_columns
from bankcanary.splits import prediction_date
from tests.test_hazard import LAG, make_frame, make_settings


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


def write_scores(settings, frame, year, model, horizon, seed=0, complete=True) -> pd.DataFrame:
    """A synthetic walk-forward score file: informative scores in ``[0, 0.8]``."""
    rng = np.random.default_rng(seed + year)
    rows = frame[frame["repdte"].dt.year == year]
    y = rows[horizon_columns(horizon)[0]].to_numpy(dtype=int)
    scores = pd.DataFrame(
        {
            "cert": rows["cert"].to_numpy(),
            "repdte": rows["repdte"].to_numpy(),
            "horizon": horizon,
            "model": model,
            "test_year": year,
            "score": 0.3 * y + 0.5 * rng.random(len(rows)),
            "score_calibrated": np.nan,
            "y": y,
            "label_complete": complete,
            "censored": False,
        }
    )[list(w.SCORE_COLUMNS)]
    path = w.scores_path(settings, year, model, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)
    return scores


def test_bin_table_never_forms_a_bin_below_min_bin():
    rng = np.random.default_rng(1)
    for n in (1, 49, 50, 99, 100, 101, 250, 1234):
        scores, y = rng.random(n), rng.random(n) < 0.1
        table = c.bin_table(scores, y, min_bin=50)
        assert (table["n"] >= min(50, n)).all() and table["n"].sum() == n
        assert len(table) == max(1, n // 50)
        assert (table["score_min"].diff().dropna() >= 0).all()  # contiguous in the score
        assert (table["score_max"] >= table["score_mean"]).all()
    table = c.bin_table([0.1, np.nan, 0.3], [0, 1, 1], min_bin=1)
    assert table["n"].sum() == 2 and table["positives"].sum() == 1  # missing scores dropped
    with pytest.raises(ValueError, match="no rows"):
        c.bin_table([np.nan], [0], min_bin=1)
    with pytest.raises(ValueError, match="shape"):
        c.bin_table([0.1, 0.2], [0], min_bin=1)


def test_three_failed_banks_at_the_top_do_not_map_to_one():
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.random(497) * 0.5, [0.9, 0.95, 0.99]])
    y = np.concatenate([(rng.random(497) < 0.02).astype(int), [1, 1, 1]])
    plain = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, y)
    assert plain.predict([0.99])[0] == 1.0  # the defect: a step resting on three banks
    binned = c.binned_isotonic(scores, y, min_bin=50)
    top = binned.predict([0.99, 5.0])
    assert top[0] < 1.0 and top[0] == top[1]  # the top step is a rate on 50 banks, clipped
    assert 3 / 50 <= float(binned.y_thresholds_[-1]) <= 0.2
    assert binned.bins_["n"].min() >= 50 and len(binned.bins_) == 10
    assert binned.bins_["n"].sum() == 500


def test_binned_map_is_monotone_and_bounded():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-3 * scores + 4))).astype(int)
    calibrator = c.binned_isotonic(scores, y, min_bin=50)
    grid = np.linspace(-5, 5, 501)
    out = calibrator.predict(grid)
    assert (np.diff(out) >= -1e-12).all() and ((out >= 0) & (out <= 1)).all()
    assert out[0] < out[-1]
    assert (
        c.binned_isotonic(scores[:10], y[:10], min_bin=50).predict([-9, 9]).tolist()
        == [float(y[:10].mean())] * 2
    )  # fewer rows than one bin: a single step at the pooled rate


def test_trailing_years_follow_the_horizon_and_the_first_prediction_date(frame):
    assert c.trailing_years(frame, 4, 2012, LAG) == [2009, 2010]
    assert c.trailing_years(frame, 8, 2012, LAG) == [2008, 2009]
    assert c.trailing_years(frame, 4, 2015, LAG, n_years=3) == [2011, 2012, 2013]
    assert c.trailing_years(frame, 4, 2012, LAG, available=[2008, 2010]) == [2008, 2010]
    cutoff = prediction_date(pd.Timestamp("2012-03-31"), LAG)
    years = frame["repdte"].dt.year
    end4, end8 = frame[horizon_columns(4)[1]], frame[horizon_columns(8)[1]]
    assert (end4[years == 2010] < cutoff).all() and not (end4[years == 2011] < cutoff).all()
    assert (end8[years == 2009] < cutoff).all() and not (end8[years == 2010] < cutoff).all()
    # with the backtest starting in 2006 only the first year cannot muster two closed
    # test years at 4q (2006 alone), and 2009 at 8q; a test year is never its own trailing year
    assert [c.trailing_years(frame, 4, y, LAG) for y in (2008, 2009, 2010, 2011)] == [
        [2006],
        [2006, 2007],
        [2007, 2008],
        [2008, 2009],
    ]
    assert [c.trailing_years(frame, 8, y, LAG) for y in (2009, 2010, 2011, 2012)] == [
        [2006],
        [2006, 2007],
        [2007, 2008],
        [2008, 2009],
    ]
    assert c.trailing_years(frame, 4, 2007, LAG) == [] and w.FIRST_TEST_YEAR == 2006


def test_trailing_rows_join_window_end_and_refuse_leaky_rows(tmp_path, frame):
    settings = make_settings(tmp_path)
    assert c.trailing_rows(frame, settings, 2012, "logit", 4, LAG)[1] == []
    for year in (2008, 2009, 2010, 2011):
        write_scores(settings, frame, year, "logit", 4)
    rows, years = c.trailing_rows(frame, settings, 2012, "logit", 4, LAG)
    assert years == [2009, 2010] and len(rows) == 2 * 4 * frame["cert"].nunique()
    assert set(rows["test_year"]) == {2009, 2010}
    assert (rows[horizon_columns(4)[1]] < prediction_date("2012-03-31", LAG)).all()
    # a 2011 report smuggled into the 2010 file closes after 2012's first prediction date
    leaky = write_scores(settings, frame, 2010, "logit", 4)
    leaky.loc[leaky.index[0], "repdte"] = pd.Timestamp("2011-12-31")
    leaky.to_parquet(w.scores_path(settings, 2010, "logit", 4), index=False)
    with pytest.raises(ValueError, match="Rule 6.2"):
        c.trailing_rows(frame, settings, 2012, "logit", 4, LAG)
    # a row absent from the labels has no window_end: also a violation
    orphan = write_scores(settings, frame, 2010, "logit", 4)
    orphan.loc[orphan.index[0], "cert"] = 10**6
    orphan.to_parquet(w.scores_path(settings, 2010, "logit", 4), index=False)
    with pytest.raises(ValueError, match="Rule 6.2"):
        c.trailing_rows(frame, settings, 2012, "logit", 4, LAG)


def test_fit_calibrator_pools_the_trailing_years_and_falls_back_early(tmp_path, frame):
    settings = make_settings(tmp_path)
    for year in (2008, 2009):
        write_scores(settings, frame, year, "logit", 4)
    w.fit_year(frame, settings, 2011, "logit", 4)
    result = c.fit_calibrator(frame, settings, 2011, "logit", 4)
    cfg, m = result.config, result.metrics
    assert cfg["slice_scorer"] == cfg["effective_scorer"] == "trailing" and not cfg["fallback"]
    assert cfg["calibration_years"] == [2008, 2009] and cfg["min_bin"] == c.MIN_BIN
    assert cfg["n_rows"] == cfg["n_calibration"] == 2 * 4 * frame["cert"].nunique() == 320
    assert cfg["calibration_start"] == "2008-03-31" and cfg["calibration_end"] == "2009-12-31"
    assert cfg["calibration_prediction_date"] == str(prediction_date("2011-03-31", LAG).date())
    assert cfg["inner_train_repdte_max"] is None and cfg["n_inner_train"] == 0
    assert m["n_bins"] == 320 // c.MIN_BIN and m["min_bin_size"] >= c.MIN_BIN
    rows, _ = c.trailing_rows(frame, settings, 2011, "logit", 4, LAG)
    same = c.binned_isotonic(rows["score"], rows["y"], c.MIN_BIN)
    grid = np.linspace(0, 1, 101)
    np.testing.assert_allclose(result.calibrator.predict(grid), same.predict(grid))
    cal = result.scores["score_calibrated"].to_numpy()
    order = np.argsort(result.scores["score"].to_numpy(), kind="stable")
    assert (np.diff(cal[order]) >= -1e-12).all() and cal.max() < 1.0
    saved, saved_cfg, _ = c.load_calibrator(settings, 2011, "logit", 4)
    assert saved_cfg["calibration_years"] == [2008, 2009] and saved_cfg["n_rows"] == 320
    assert tracking.find_metrics("calibrate", cfg, settings)["n_bins"] == m["n_bins"]
    # a smaller bin gives more steps and a different run id
    small = c.fit_calibrator(frame, settings, 2011, "logit", 4, save=False, min_bin=20)
    assert small.metrics["n_bins"] == 16 and small.config["min_bin"] == 20
    assert tracking.run_id("calibrate", small.config) != tracking.run_id("calibrate", cfg)
    # 2010 has one closed year only: the trailing default falls back to the full scorer
    w.fit_year(frame, settings, 2010, "logit", 4)
    early = c.fit_calibrator(frame, settings, 2010, "logit", 4, save=False)
    assert early.config["slice_scorer"] == "trailing" and early.config["fallback"] is True
    assert early.config["effective_scorer"] == "full"
    assert early.config["calibration_years"] == [2008] and early.config["n_rows"] == 160
    assert early.metrics["min_bin_size"] >= c.MIN_BIN


def test_production_calibrator_picks_the_two_most_recent_complete_years(tmp_path, frame):
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError, match="needs 2"):
        c.production_calibrator(settings, "gbdt_mono")
    for year in (2011, 2012, 2013, 2014):
        write_scores(settings, frame, year, "gbdt_mono", 4)
    partial = write_scores(settings, frame, 2015, "gbdt_mono", 4)
    partial.loc[partial.index[-1], "label_complete"] = False
    partial.to_parquet(w.scores_path(settings, 2015, "gbdt_mono", 4), index=False)
    assert c.scored_years(settings, "gbdt_mono", 4) == [2011, 2012, 2013, 2014, 2015]
    calibrator, cfg = c.production_calibrator(settings, "gbdt_mono")
    assert cfg["calibration_years"] == [2013, 2014] and cfg["n_rows"] == 320
    assert cfg["slice_scorer"] == "trailing" and cfg["min_bin"] == c.MIN_BIN
    assert cfg["calibration_start"] == "2013-03-31" and cfg["calibration_end"] == "2014-12-31"
    rows = pd.concat(
        [pd.read_parquet(w.scores_path(settings, y, "gbdt_mono", 4)) for y in (2013, 2014)]
    )
    same = c.binned_isotonic(rows["score"], rows["y"])
    grid = np.linspace(0, 1, 101)
    np.testing.assert_allclose(calibrator.predict(grid), same.predict(grid))
    assert cfg["top_plateau"] == float(same.y_thresholds_[-1]) < 1.0
    # the labels frame can veto a year whose labels are not complete as of the failures date
    stale = frame.copy()
    stale.loc[stale["repdte"].dt.year == 2014, horizon_columns(4)[3]] = False
    _, cfg2 = c.production_calibrator(settings, "gbdt_mono", frame=stale)
    assert cfg2["calibration_years"] == [2012, 2013]
    _, cfg3 = c.production_calibrator(settings, "gbdt_mono", n_years=3)
    assert cfg3["calibration_years"] == [2012, 2013, 2014]
    with pytest.raises(ValueError, match="not a probability model"):
        c.production_calibrator(settings, "texas")


def test_calibrate_command_exposes_the_scorer_and_bin_options():
    from typer.testing import CliRunner

    from bankcanary.cli import app

    result = CliRunner().invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0 and "--min-bin" in result.output and "trailing" in result.output
