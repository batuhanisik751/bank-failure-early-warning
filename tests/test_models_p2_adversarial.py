"""Adversarial checks of the Prototype 2 modelling code: leakage first, then arithmetic.

Every expected value below was derived by hand from PROJECT_SPEC.md section 6 (leakage
rules), section 8 and CONTRACT sections 11-13 before anything was run. Synthetic frames
only: no network, no data/ reads.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from bankcanary import tracking
from bankcanary.config import FixedSplit, GbdtSettings, ModelSettings, Settings
from bankcanary.features import macro as macro_features
from bankcanary.features import trends
from bankcanary.features.registry import feature_names
from bankcanary.ingest import macro as macro_ingest
from bankcanary.labels.build import build_labels, horizon_columns
from bankcanary.models import gbdt, hazard
from bankcanary.models.preprocess import Winsorizer, make_pipeline
from bankcanary.splits import fixed_split_masks

ROOT = Path(__file__).resolve().parents[1]
LAG = 60
SPLIT = FixedSplit(
    train_start="2002-03-31", train_end="2008-12-31", test_start="2010-03-31", test_end="2013-12-31"
)
SMALL = {"learning_rate": 0.1, "num_leaves": 7, "min_samples_leaf": 20, "n_estimators": 15}
V2 = feature_names(version="v2")


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module (it is not a package)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_settings(tmp_path, monotone: bool = False) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
        fixed_split=SPLIT,
        models=ModelSettings(gbdt=GbdtSettings(backend="sklearn", monotone=monotone, params=SMALL)),
    )


def labels_frame(n_certs: int = 3) -> pd.DataFrame:
    """Complete 1q/4q/8q labels for quarter-ends 2001Q1-2015Q4, ``repdte`` deliberately
    shuffled so that nothing below can rely on row order."""
    quarters = pd.date_range("2001-03-31", "2015-12-31", freq="QE-DEC")
    frame = pd.concat(
        [pd.DataFrame({"cert": c, "repdte": quarters}) for c in range(1, n_certs + 1)],
        ignore_index=True,
    )
    frame = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    avail = frame["repdte"] + pd.Timedelta(days=LAG)
    frame["dropped_failed_before_avail"] = False
    for horizon in (1, 4, 8):
        y_col, end_col, cens_col, complete_col = horizon_columns(horizon)
        frame[y_col] = (frame["cert"] + frame["repdte"].dt.quarter) % 5 == 0
        frame[y_col] = frame[y_col].astype(int)
        frame[end_col] = (avail + pd.DateOffset(months=3 * horizon)).astype("datetime64[ns]")
        frame[cens_col] = False
        frame[complete_col] = True
    return frame


# --- rule 6.2 inside the inner split and on the fixed split ------------------------------


def test_gbdt_inner_masks_end_at_2005q4_and_never_touch_the_test_years(tmp_path):
    """2006Q1's 4q window closes exactly on 2007-05-30, the validation prediction date,
    so rule 6.2 (strict ``<``) must exclude it: inner training ends at 2005-12-31."""
    tune = load_script("tune_gbdt")
    frame, settings = labels_frame(), make_settings(tmp_path)
    inner_train, validation = tune.inner_masks(frame, settings)
    _, outer_test = fixed_split_masks(frame, settings, 4)
    assert frame.loc[inner_train, "repdte"].max() == pd.Timestamp("2005-12-31")
    assert frame.loc[inner_train, "repdte"].min() == pd.Timestamp("2002-03-31")
    assert frame.loc[validation, "repdte"].min() == pd.Timestamp("2007-03-31")
    assert frame.loc[validation, "repdte"].max() == pd.Timestamp("2008-12-31")
    assert (frame.loc[inner_train, "window_end_4q"] < pd.Timestamp("2007-05-30")).all()
    assert not (inner_train & validation).any()
    assert not (inner_train & outer_test).any() and not (validation & outer_test).any()
    # the validation slice is the full eight quarters, 2006 is a gap on purpose
    assert frame.loc[validation, "repdte"].nunique() == 8
    assert not frame.loc[inner_train | validation, "repdte"].dt.year.eq(2006).any()


def test_hazard_inner_split_ends_at_2006q3_and_stays_inside_the_training_period(tmp_path):
    """At 1q the window is three months, so 2006Q3 (window end 2007-02-28) is the last
    inner training report; 2006Q4 closes on 2007-06-01, after the validation date."""
    frame, settings = labels_frame(), make_settings(tmp_path)
    inner_train, validation = hazard.inner_split(frame, settings)
    outer_train, outer_test = fixed_split_masks(frame, settings, 1)
    assert frame.loc[inner_train, "repdte"].max() == pd.Timestamp("2006-09-30")
    assert (frame.loc[inner_train, "window_end_1q"] < pd.Timestamp("2007-05-30")).all()
    assert frame.loc[validation, "repdte"].min() == pd.Timestamp("2007-03-31")
    assert frame.loc[validation, "repdte"].max() == pd.Timestamp("2008-12-31")
    assert (inner_train & ~outer_train).sum() == 0 and (validation & ~outer_train).sum() == 0
    assert not (inner_train & outer_test).any() and not (validation & outer_test).any()
    assert not (inner_train & validation).any()
    # the 4q labels the tuner selects on close by 2010-03-01, before the first test date
    assert frame.loc[validation, "window_end_4q"].max() == pd.Timestamp("2010-03-01")
    assert frame.loc[validation, "window_end_4q"].max() < pd.Timestamp("2010-05-30")


@pytest.mark.parametrize(
    ("horizon", "last_train"),
    [(1, "2008-12-31"), (4, "2008-12-31"), (8, "2007-12-31")],
)
def test_fixed_split_training_cut_per_horizon_by_hand(tmp_path, horizon, last_train):
    """8q: 2008Q1 -> avail 2008-05-30 -> window end 2010-05-30 == first test prediction
    date, so it must be excluded and 2007Q4 (end 2010-02-28) is the last training row."""
    frame, settings = labels_frame(), make_settings(tmp_path)
    train, test = fixed_split_masks(frame, settings, horizon)
    assert frame.loc[train, "repdte"].max() == pd.Timestamp(last_train)
    assert frame.loc[train, "repdte"].min() == pd.Timestamp("2002-03-31")
    assert frame.loc[test, "repdte"].min() == pd.Timestamp("2010-03-31")
    assert frame.loc[test, "repdte"].max() == pd.Timestamp("2013-12-31")
    end_col = horizon_columns(horizon)[1]
    assert (frame.loc[train, end_col] < pd.Timestamp("2010-05-30")).all()
    assert not (train & test).any()


# --- which rows actually reach fit / predict ---------------------------------------------


class SpyPipeline:
    """Records the row labels handed to ``fit`` and ``predict_proba``; scores are a ramp."""

    def __init__(self):
        self.fit_index = None
        self.predicted: list[pd.Index] = []

    def fit(self, X, y=None):
        self.fit_index = pd.Index(X.index)
        return self

    def predict_proba(self, X):
        self.predicted.append(pd.Index(X.index))
        p = np.linspace(0.01, 0.99, len(X))
        return np.column_stack([1 - p, p])


def features_frame():
    from tests.test_hazard import make_frame

    return make_frame(n_certs=30, seed=3)


def test_tune_gbdt_score_config_fits_inner_rows_and_scores_validation_only(tmp_path, monkeypatch):
    tune = load_script("tune_gbdt")
    frame, settings = features_frame(), make_settings(tmp_path)
    inner_train, validation = tune.inner_masks(frame, settings)
    spy = SpyPipeline()
    monkeypatch.setattr(tune, "make_gbdt", lambda *a, **k: spy)
    config = tune._config("gbdt", "sklearn", dict(SMALL))
    metrics = tune.score_config(config, frame, inner_train, validation, V2)
    assert spy.fit_index.equals(frame.index[inner_train.to_numpy()])
    assert len(spy.predicted) == 1 and spy.predicted[0].equals(frame.index[validation.to_numpy()])
    assert metrics["n"] == int(validation.sum())
    assert frame.loc[spy.fit_index, "repdte"].max() == pd.Timestamp("2005-12-31")
    assert frame.loc[spy.predicted[0], "repdte"].min() == pd.Timestamp("2007-03-31")


def test_hazard_tune_c_fits_inner_rows_and_scores_validation_only(tmp_path, monkeypatch):
    frame, settings = features_frame(), make_settings(tmp_path)
    inner_train, validation = hazard.inner_split(frame, settings)
    spy = SpyPipeline()
    monkeypatch.setattr(hazard, "make_hazard", lambda *a, **k: spy)
    table = hazard.tune_c(frame, settings, grid=(1.0,))
    assert spy.fit_index.equals(frame.index[inner_train.to_numpy()])
    assert len(spy.predicted) == 1 and spy.predicted[0].equals(frame.index[validation.to_numpy()])
    assert frame.loc[spy.fit_index, "repdte"].max() == pd.Timestamp("2006-09-30")
    assert table.loc[0, "n_1q"] == int(validation.sum())
    # rows scored at 4q are the same validation rows, never anything dated 2009 or later
    assert table.loc[0, "n_4q"] == int(validation.sum())


@pytest.mark.parametrize("horizon", [4, 8])
def test_fit_on_fixed_split_fits_train_mask_rows_and_scores_test_rows_only(tmp_path, horizon):
    frame, settings = features_frame(), make_settings(tmp_path)
    train, test = fixed_split_masks(frame, settings, horizon)
    spy = SpyPipeline()
    result = gbdt.fit_on_fixed_split("logit_v2", spy, V2, frame, settings, horizon, save=False)
    assert spy.fit_index.equals(frame.index[train.to_numpy()])
    assert spy.predicted[0].equals(frame.index[test.to_numpy()])
    last = {4: "2008-12-31", 8: "2007-12-31"}[horizon]
    assert result.config["train_repdte_max"] == last
    assert result.config["test_repdte_min"] == "2010-03-31"
    assert len(result.scores) == int(test.sum())


def test_fit_hazard_fits_the_1q_fixed_split_rows_only(tmp_path, monkeypatch):
    frame, settings = features_frame(), make_settings(tmp_path)
    train, test = fixed_split_masks(frame, settings, 1)
    spy = SpyPipeline()
    monkeypatch.setattr(hazard, "make_hazard", lambda *a, **k: spy)
    result = hazard.fit_hazard(frame, settings, save=False)
    assert spy.fit_index.equals(frame.index[train.to_numpy()])
    assert spy.predicted[0].equals(frame.index[test.to_numpy()])
    assert result.config["label"] == "y_1q" and result.config["train_repdte_max"] == "2008-12-31"


# --- preprocessing fitted on the training fold only, no early stopping -------------------


def test_winsor_bounds_and_medians_come_from_the_training_fold_by_hand():
    """Training column 1..200: linear quantiles give 1 + 0.005*199 = 1.995 and
    1 + 0.995*199 = 199.005; median 100.5. A test fold with 1000 and -50 must be clipped
    to those bounds and its NaN filled with 100.5, whatever the test fold contains."""
    train = pd.DataFrame({"a": np.arange(1, 201, dtype=float), "flag": np.arange(200) % 2})
    y = (np.arange(200) % 3 == 0).astype(int)
    pipe = make_pipeline(LogisticRegression(max_iter=200)).fit(train, y)
    w, imp, sc = (pipe.named_steps[k] for k in ("winsorize", "impute", "scale"))
    assert w.lower_bounds_[0] == pytest.approx(1.995) and w.upper_bounds_[0] == pytest.approx(
        199.005
    )
    assert np.isinf(w.lower_bounds_[1]) and np.isinf(w.upper_bounds_[1])  # binary skipped
    assert imp.statistics_[0] == 100.5
    # clipping 1 -> 1.995 and 200 -> 199.005 leaves the sum (20100) unchanged: mean 100.5
    assert sc.mean_[0] == pytest.approx(100.5)
    test = pd.DataFrame({"a": [1000.0, -50.0, np.nan, 100.5], "flag": [1, 0, 1, 0]})
    clipped = w.transform(test)
    assert clipped[0, 0] == pytest.approx(199.005) and clipped[1, 0] == pytest.approx(1.995)
    assert np.isnan(clipped[2, 0])
    filled = imp.transform(clipped)
    assert filled[2, 0] == 100.5 and filled.shape[1] == 2  # no training NaN: no indicator
    scaled = sc.transform(filled)
    assert scaled[3, 0] == pytest.approx(0.0) and scaled[2, 0] == pytest.approx(0.0)
    # fitting again on the same rows must not move anything (no state from the test fold)
    again = Winsorizer().fit(train)
    assert np.array_equal(again.upper_bounds_, w.upper_bounds_)


def test_gbdt_backends_never_carve_a_validation_split_or_stop_early():
    X = np.random.default_rng(0).normal(size=(300, 3))
    y = (X[:, 0] > 0.5).astype(int)
    sk = gbdt.make_gbdt("sklearn", n_estimators=9).fit(X, y).named_steps["model"]
    assert sk.early_stopping is False
    assert sk.n_iter_ == 9 and len(sk.validation_score_) == 0
    if gbdt.lightgbm_available():
        lgb = gbdt.make_gbdt("lightgbm", n_estimators=9).fit(X, y).named_steps["model"]
        assert lgb.booster_.num_trees() == 9
        assert not [k for k in lgb.get_params() if "early" in k]
        assert not lgb.best_iteration_  # 0/None: no early-stopping callback ran


def test_gbdt_pipeline_is_a_bare_estimator_and_rejects_unknown_params():
    pipe = gbdt.make_gbdt("sklearn")
    assert [name for name, _ in pipe.steps] == ["model"]
    assert not any(isinstance(step, Winsorizer) for _, step in pipe.steps)
    with pytest.raises(ValueError, match="unknown gbdt parameter"):
        gbdt.make_gbdt("sklearn", max_depth=3)
    with pytest.raises(ValueError, match="monotone signs"):
        gbdt.make_gbdt("sklearn", monotone=[1, -1], features=["a", "b", "c"])


# --- trend features: exact quarter-end match, backward only ------------------------------

Q = ["2006-03-31", "2006-06-30", "2006-09-30", "2006-12-31", "2007-03-31", "2007-06-30"]
Q7 = Q + ["2007-12-31"]  # cert 7 never filed 2007Q3
Q8 = Q + ["2007-09-30", "2007-12-31"]
NAN = float("nan")


def trend_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [(7, q) for q in Q7] + [(8, q) for q in Q8]
    panel = pd.DataFrame(rows, columns=["cert", "repdte"])
    panel["repdte"] = pd.to_datetime(panel["repdte"])
    feats = pd.DataFrame(index=panel.index)
    feats["noncurrent_ratio"] = [1, 2, 4, 7, 11, 16, 29] + [100, 90, 80, 70, 60, 50, 40, 30]
    feats["roa_q"] = [-1, -1, NAN, -1, 1, -1, -1] + [1.0] * 8
    for name, _, _ in trends.TREND_RATIOS:
        if name not in feats.columns:
            feats[name] = 0.5
    order = np.random.default_rng(11).permutation(len(panel))
    return panel.iloc[order], feats.iloc[order]


def test_trend_differences_by_hand_on_a_shuffled_panel_with_a_filing_gap():
    panel, feats = trend_panel()
    out = trends.build(panel, features=feats)
    assert out.index.equals(panel.index)
    got = out.assign(cert=panel["cert"], repdte=panel["repdte"]).sort_values(["cert", "repdte"])
    c7, c8 = got[got["cert"] == 7], got[got["cert"] == 8]
    np.testing.assert_allclose(c7["d1q_noncurrent_ratio"], [NAN, 1, 2, 3, 4, 5, NAN])
    np.testing.assert_allclose(c7["d4q_noncurrent_ratio"], [NAN] * 4 + [10, 14, 22])
    np.testing.assert_allclose(c8["d1q_noncurrent_ratio"], [NAN] + [-10] * 7)
    np.testing.assert_allclose(c8["d4q_noncurrent_ratio"], [NAN] * 4 + [-40] * 4)
    # a ratio that never moves has a zero trend wherever the earlier quarter exists
    np.testing.assert_allclose(c8["d1q_texas_ratio"], [NAN] + [0.0] * 7)
    np.testing.assert_allclose(c7["d1q_texas_ratio"], [NAN, 0, 0, 0, 0, 0, NAN])


def test_persistence_counts_by_hand_on_a_shuffled_panel_with_a_filing_gap():
    panel, feats = trend_panel()
    out = trends.build(panel, features=feats)
    got = out.assign(cert=panel["cert"], repdte=panel["repdte"]).sort_values(["cert", "repdte"])
    c7, c8 = got[got["cert"] == 7], got[got["cert"] == 8]
    np.testing.assert_allclose(c7["consecutive_loss_quarters"], [1, 2, 0, 1, 0, 1, 1])
    np.testing.assert_allclose(c7["neg_roa_quarters_last_8"], [NAN, NAN, NAN, NAN, 3, 4, 5])
    np.testing.assert_allclose(c7["noncurrent_rising_quarters_last_4"], [NAN, 1, 2, 3, 4, 4, 2])
    np.testing.assert_allclose(c8["consecutive_loss_quarters"], [0] * 8)
    np.testing.assert_allclose(c8["neg_roa_quarters_last_8"], [NAN, NAN, NAN, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(c8["noncurrent_rising_quarters_last_4"], [NAN] + [0] * 7)


def test_trends_never_read_a_later_quarter():
    """Deleting every row after 2006Q4 must leave the 2006 rows' features unchanged."""
    panel, feats = trend_panel()
    full = trends.build(panel, features=feats)
    keep = (panel["repdte"] <= "2006-12-31").to_numpy()
    part = trends.build(panel[keep], features=feats[keep])
    pd.testing.assert_frame_equal(part, full[keep], check_exact=False)


# --- macro: publication lag at the boundary, exact-date join -----------------------------


def fred(dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})


def test_point_in_time_boundary_by_hand_for_monthly_quarterly_and_daily_series():
    ur = macro_ingest.usable_from(
        fred(["2008-02-01", "2008-03-01", "2008-04-01"], [5, 6, 7]), "M", 45
    )
    assert list(ur["usable_from"].dt.strftime("%Y-%m-%d")) == [
        "2008-04-14",
        "2008-05-15",
        "2008-06-14",
    ]
    grid = pd.DataFrame(
        {
            "avail_date": pd.to_datetime(
                ["2008-04-13", "2008-05-14", "2008-05-15", "2008-05-30", "2008-06-14"]
            )
        }
    )
    np.testing.assert_allclose(macro_ingest.point_in_time(grid, ur), [NAN, 5, 6, 6, 7])
    hpi = macro_ingest.usable_from(fred(["2007-10-01", "2008-01-01"], [200, 210]), "Q", 75)
    assert list(hpi["usable_from"].dt.strftime("%Y-%m-%d")) == ["2008-03-15", "2008-06-14"]
    grid = pd.DataFrame({"avail_date": pd.to_datetime(["2008-05-30", "2008-06-13", "2008-06-14"])})
    np.testing.assert_allclose(macro_ingest.point_in_time(grid, hpi), [200, 200, 210])
    daily = macro_ingest.usable_from(fred(["2008-05-29", "2008-05-30"], [1.5, 9.9]), "D", 1)
    grid = pd.DataFrame({"avail_date": pd.to_datetime(["2008-05-30"])})
    np.testing.assert_allclose(macro_ingest.point_in_time(grid, daily), [1.5])


def test_build_macro_state_four_quarter_change_uses_values_known_a_year_earlier():
    """avail 2008-05-30 (report 2008Q1) against avail 2007-05-30 (report 2007Q1): the
    distractor values (999) are dated so that a look-ahead of one period would pick them."""
    series = {
        "CAUR": fred(
            ["2007-03-01", "2007-04-01", "2008-03-01", "2008-04-01"], [4.0, 999, 6.0, 999]
        ),
        "CASTHPI": fred(
            ["2006-10-01", "2007-01-01", "2007-10-01", "2008-01-01"], [100, 999, 110, 999]
        ),
        "FEDFUNDS": fred(
            ["2007-04-01", "2007-05-01", "2008-04-01", "2008-05-01"], [5.25, 999, 2.0, 999]
        ),
        "T10Y3M": fred(["2008-05-29", "2008-05-30"], [1.5, 999]),
        "DGS10": fred(["2008-05-29", "2008-05-30"], [3.9, 999]),
    }
    grid = pd.DataFrame({"stalp": ["CA"], "avail_date": pd.to_datetime(["2008-05-30"])})
    out = macro_ingest.build_macro_state(grid, series, availability_lag_days=LAG)
    assert len(out) == 1 and out.loc[0, "stalp"] == "CA"
    assert out.loc[0, "unemp_rate"] == 6.0 and out.loc[0, "unemp_change_4q"] == 2.0
    assert out.loc[0, "hpi_change_4q"] == pytest.approx(np.log(1.1))
    assert out.loc[0, "fedfunds"] == 2.0 and out.loc[0, "fedfunds_change_4q"] == -3.25
    assert out.loc[0, "t10y3m"] == 1.5 and out.loc[0, "dgs10"] == 3.9


def test_macro_join_is_exact_on_avail_date_and_never_reaches_forward_or_back():
    macro_state = pd.DataFrame(
        {
            "stalp": ["CA", "CA", "NY"],
            "avail_date": pd.to_datetime(["2008-05-30", "2008-08-29", "2008-05-30"]),
            "unemp_rate": [6.0, 7.0, 5.0],
            "unemp_change_4q": [2.0, 3.0, 1.0],
            "hpi_change_4q": [-0.1, -0.2, 0.0],
            "t10y3m": [1.5, 2.5, 1.5],
            "dgs10": [3.9, 4.0, 3.9],
            "fedfunds_change_4q": [-3.25, -3.0, -3.25],
        }
    )
    panel = pd.DataFrame(
        {
            "cert": [1, 2, 3, 4],
            "stalp": ["ca", "CA", "CA", None],
            "avail_date": pd.to_datetime(["2008-05-30", "2008-05-31", "2008-08-28", "2008-05-30"]),
        },
        index=[10, 20, 30, 40],
    )
    out = macro_features.build(panel, macro_state=macro_state)
    assert out.index.equals(panel.index)
    np.testing.assert_allclose(out["macro_unemp_rate"], [6.0, NAN, NAN, NAN])
    np.testing.assert_allclose(out["macro_hpi_change_4q"], [-0.1, NAN, NAN, NAN])
    np.testing.assert_allclose(out["macro_t10y3m"], [1.5, NAN, NAN, 1.5])
    np.testing.assert_allclose(out["macro_fedfunds_change_4q"], [-3.25, NAN, NAN, -3.25])


# --- hazard: the one-quarter window and the persistence conversion ----------------------


def test_one_quarter_window_boundaries_by_hand():
    """Report 2007-09-30 -> avail 2007-11-29 -> 1q window (2007-11-29, 2008-02-29]."""
    fails = ["2008-02-29", "2008-03-01", "2007-11-29", "2007-11-30", None, None, None, None]
    exits = [None, None, None, None, None, "2008-01-15", "2008-02-29", "2008-03-01"]
    panel = pd.DataFrame(
        {
            "cert": range(1, 9),
            "repdte": pd.Timestamp("2007-09-30"),
            "avail_date": pd.Timestamp("2007-11-29"),
            "fail_date": pd.to_datetime(fails),
            "exit_date": pd.to_datetime(exits),
        }
    )
    out = build_labels(panel, [1, 4], as_of_date=pd.Timestamp("2008-02-29"))
    assert (out["window_end_1q"] == pd.Timestamp("2008-02-29")).all()
    assert out["y_1q"].tolist() == [1, 0, 0, 1, 0, 0, 0, 0]
    assert out["dropped_failed_before_avail"].tolist() == [False, False, True] + [False] * 5
    assert out["censored_in_window_1q"].tolist() == [False] * 5 + [True, True, False]
    assert out["label_complete_1q"].all()  # window end == as_of counts as complete
    assert out["y_4q"].tolist() == [1, 1, 0, 1, 0, 0, 0, 0]  # 4q window runs to 2008-11-29
    earlier = build_labels(panel, [1], as_of_date=pd.Timestamp("2008-02-28"))
    assert not earlier["label_complete_1q"].any()


def test_persistence_conversion_by_hand_and_ranking_invariance():
    from bankcanary.evaluation.metrics import evaluate

    h = np.array([0.0, 0.1, 0.5, 1.0])
    np.testing.assert_allclose(hazard.convert_hazard(h, 1), h)
    np.testing.assert_allclose(hazard.convert_hazard(h, 4), [0.0, 0.3439, 0.9375, 1.0])
    np.testing.assert_allclose(hazard.convert_hazard(h, 8), [0.0, 0.56953279, 0.99609375, 1.0])
    assert hazard.convert_hazard([1.7, -0.2], 4).tolist() == [1.0, 0.0]  # clipped first
    with pytest.raises(ValueError):
        hazard.convert_hazard(h, 0)
    rng = np.random.default_rng(5)
    scores = rng.random(400)
    y = (rng.random(400) < scores**3).astype(int)
    base = evaluate(y, scores)
    for horizon in (4, 8):
        conv = evaluate(y, hazard.convert_hazard(scores, horizon))
        for key in ("pr_auc", "roc_auc", "recall_at_2pct", "recall_at_top100"):
            assert conv[key] == pytest.approx(base[key])
    assert hazard.brier([1, 0, 1], [1.0, 0.0, 0.5]) == pytest.approx(0.25 / 3)
    assert hazard.make_hazard(0.01).named_steps["model"].class_weight is None
    assert hazard.make_hazard(0.01).named_steps["model"].C == 0.01


# --- monotone signs against banking logic, and the tracking id ----------------------------

EXPECTED_SIGNS = {
    "noncurrent_ratio": 1,
    "npa_to_assets": 1,
    "nco_rate": 1,
    "texas_ratio": 1,
    "reserve_coverage": -1,
    "tier1_leverage": -1,
    "equity_to_assets": -1,
    "total_rbc_ratio": -1,
    "roa_q": -1,
    "efficiency_ratio": 1,
    "provision_rate": 1,
    "brokered_share": 1,
    "liquid_assets_ratio": -1,
    "wholesale_funding_ratio": 1,
    "construction_to_capital": 1,
    "cre_to_capital": 1,
    "log_assets": -1,
    "bank_age_years": -1,
    "unrealized_loss_to_tier1": -1,
    "adjusted_tier1_leverage": -1,
    "htm_share_of_securities": 1,
    "uninsured_share": 1,
    "uninsured_to_liquid_assets": 1,
    "large_time_deposit_share": 1,
    "d1q_noncurrent_ratio": 1,
    "d4q_noncurrent_ratio": 1,
    "d4q_texas_ratio": 1,
    "d4q_roa_q": -1,
    "d4q_equity_to_assets": -1,
    "d4q_tier1_leverage": -1,
    "d1q_uninsured_share": 1,
    "d4q_unrealized_loss_to_tier1": -1,
    "neg_roa_quarters_last_8": 1,
    "consecutive_loss_quarters": 1,
    "noncurrent_rising_quarters_last_4": 1,
    "macro_unemp_rate": 1,
    "macro_unemp_change_4q": 1,
    "macro_hpi_change_4q": -1,
    "macro_t10y3m": -1,
    "macro_fedfunds_change_4q": 1,
    "securities_to_assets": 0,
    "is_community_bank": 0,
}


def test_monotone_vector_matches_banking_direction_and_the_explanation_text():
    from bankcanary.features.registry import specs
    from bankcanary.features.spec import MONOTONE_TEXT

    signs = dict(zip(V2, gbdt.constraint_vector(V2, True)))
    for name, sign in EXPECTED_SIGNS.items():
        assert signs[name] == sign, name
    by_name = {s.name: s for s in specs("v2")}
    for name, sign in EXPECTED_SIGNS.items():
        assert by_name[name].explanation.endswith(MONOTONE_TEXT[sign]), name
    for s in by_name.values():
        if s.name.startswith(("bkclass_", "region_")):
            assert s.monotone == 0, s.name
    # the vector follows the column order handed to the pipeline, not the registry
    rev = list(reversed(V2))
    assert gbdt.constraint_vector(rev, True) == list(reversed(gbdt.constraint_vector(V2, True)))
    assert gbdt.constraint_vector(V2, False) is None and gbdt.constraint_vector(V2, None) is None
    # the estimator receives exactly that vector, in that order
    model = gbdt.make_gbdt("sklearn", monotone=True, features=V2).named_steps["model"]
    assert list(model.monotonic_cst) == gbdt.constraint_vector(V2, True)


def test_run_id_is_a_pure_function_of_the_config(tmp_path):
    base = {
        "model": "gbdt",
        "horizon": 4,
        "backend": "lightgbm",
        "monotone": False,
        "params": {"learning_rate": 0.03, "num_leaves": 63, "n_estimators": 200},
        "split": {"validation_start": "2007-03-31", "validation_end": "2008-12-31"},
    }
    rid = tracking.run_id("tune_gbdt", base)
    assert rid.startswith("tune_gbdt-4q-") and rid == tracking.run_id("tune_gbdt", dict(base))
    same = {
        "split": {"validation_end": "2008-12-31", "validation_start": "2007-03-31"},
        "params": {
            "n_estimators": np.int64(200),
            "num_leaves": 63,
            "learning_rate": np.float64(0.03),
        },
        "monotone": np.bool_(False),
        "backend": "lightgbm",
        "horizon": np.int64(4),
        "model": "gbdt",
    }
    assert tracking.run_id("tune_gbdt", same) == rid
    changed = [
        {**base, "monotone": True},
        {**base, "backend": "sklearn"},
        {**base, "params": {**base["params"], "num_leaves": 31}},
        {**base, "params": {**base["params"], "learning_rate": 0.1}},
        {**base, "split": {**base["split"], "validation_end": "2007-12-31"}},
        {**base, "horizon": 8},
        {**base, "model": "gbdt_mono"},
    ]
    ids = {tracking.run_id("tune_gbdt", c) for c in changed}
    assert len(ids) == len(changed) and rid not in ids
    assert tracking.run_id("train", base) != rid  # the name is part of the id
    settings = make_settings(tmp_path)
    run = tracking.start_run("tune_gbdt", base, settings)
    run.log_metrics({"pr_auc": 0.25})
    assert run.finish() == tmp_path / "runs" / "tune_gbdt" / rid
    assert tracking.find_metrics("tune_gbdt", same, settings) == {"pr_auc": 0.25}
    assert all(tracking.find_metrics("tune_gbdt", c, settings) is None for c in changed)


def test_model_feature_lists_carry_no_identifiers_and_agree_across_models():
    """Spec rule 6.5: no bank identifiers, names or CERT-derived columns reach a model."""
    banned = {
        "cert",
        "name",
        "fed_rssd",
        "rssdhcr",
        "ultcert",
        "newcert",
        "stalp",
        "repdte",
        "avail_date",
        "fail_date",
        "exit_date",
        "window_start",
        "procdate",
    }
    for names in (gbdt.gbdt_features(), hazard.hazard_features()):
        assert not banned & set(names)
        assert not [n for n in names if "cert" in n or n.startswith(("y_", "window_end"))]
        assert len(names) == len(set(names))
    assert gbdt.gbdt_features() == hazard.hazard_features() == V2
