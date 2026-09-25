"""Baseline models on a synthetic features + labels frame: no data files, no network."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bankcanary.config import FixedSplit, Settings
from bankcanary.features.registry import feature_names
from bankcanary.labels.build import horizon_columns
from bankcanary.models import baselines, preprocess, train

LAG = 60
SPLIT = FixedSplit(
    train_start="2002-03-31", train_end="2008-12-31", test_start="2010-03-31", test_end="2013-12-31"
)


def make_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        fixed_split=SPLIT,
    )


def make_frame(n_certs: int = 60, seed: int = 0) -> pd.DataFrame:
    """Bank-quarters 2001Q1-2015Q4 with random registry features and complete 4q/8q labels.

    ``texas_ratio`` and ``noncurrent_ratio`` are pushed up for positives so a model has
    something to learn; every other feature is noise. About 5 percent of rows fail.
    """
    rng = np.random.default_rng(seed)
    quarters = pd.date_range("2001-03-31", "2015-12-31", freq="QE-DEC")
    frame = pd.concat(
        [pd.DataFrame({"cert": c, "repdte": quarters}) for c in range(1, n_certs + 1)],
        ignore_index=True,
    )
    n = len(frame)
    cols: dict = {}
    for name in feature_names("P1"):
        if name.startswith("bkclass_") or name.endswith(("_missing", "_capped", "company")):
            cols[name] = rng.random(n) < 0.1
        else:
            values = rng.normal(size=n)
            values[rng.random(n) < 0.05] = np.nan
            cols[name] = values
    y = rng.random(n) < 0.05
    cols["texas_ratio"] = np.abs(cols["texas_ratio"]) + np.where(y, 1.0, 0.0)
    cols["noncurrent_ratio"] = cols["noncurrent_ratio"] + np.where(y, 2.0, 0.0)
    avail = frame["repdte"] + pd.Timedelta(days=LAG)
    cols["dropped_failed_before_avail"] = False
    for horizon in (4, 8):
        y_col, end_col, cens_col, complete_col = horizon_columns(horizon)
        cols[y_col] = y.astype(int)
        cols[end_col] = (avail + pd.DateOffset(months=3 * horizon)).astype("datetime64[ns]")
        cols[cens_col] = False
        cols[complete_col] = True
    return frame.assign(**cols)


def test_winsorizer_clips_to_training_quantiles_and_keeps_nan_and_flags():
    train_df = pd.DataFrame(
        {
            "x": np.r_[np.arange(1.0, 1001.0), np.nan],
            "flag": [True] * 3 + [False] * 998,
            "empty": [np.nan] * 1001,
        }
    )
    w = preprocess.Winsorizer(lower=0.01, upper=0.99).fit(train_df)
    assert list(w.get_feature_names_out()) == ["x", "flag", "empty"]
    assert w.lower_bounds_[0] == pytest.approx(np.quantile(np.arange(1.0, 1001.0), 0.01))
    assert w.upper_bounds_[0] == pytest.approx(np.quantile(np.arange(1.0, 1001.0), 0.99))
    # rare flags and all-missing columns are never clipped
    assert w.skipped_.tolist() == [False, True, True]
    out = w.transform(
        pd.DataFrame({"x": [-5.0, 500.0, 1e6, np.nan], "flag": [1, 1, 0, 0], "empty": [np.nan] * 4})
    )
    assert out[0, 0] == pytest.approx(w.lower_bounds_[0])
    assert out[1, 0] == 500.0
    assert out[2, 0] == pytest.approx(w.upper_bounds_[0])
    assert np.isnan(out[3, 0]) and np.isnan(out[0, 2])
    assert out[:, 1].tolist() == [1, 1, 0, 0]


def test_winsorizer_is_fitted_on_the_training_fold_only():
    w = preprocess.Winsorizer(lower=0.0, upper=0.9).fit(np.arange(11.0).reshape(-1, 1))
    assert w.upper_bounds_[0] == pytest.approx(9.0)
    # a later, wilder test fold does not move the bound
    assert preprocess.Winsorizer.transform(w, np.array([[1e9]]))[0, 0] == pytest.approx(9.0)
    with pytest.raises(ValueError):
        preprocess.Winsorizer(lower=0.9, upper=0.1).fit(np.zeros((3, 1)))


def test_pipeline_fits_on_a_synthetic_frame_with_missing_values():
    frame = make_frame(n_certs=20, seed=1)
    for name in ("logit_small", "logit"):
        features = baselines.model_features(name)
        pipe = baselines.make_model(name).fit(frame[features], frame["y_4q"])
        assert [s for s, _ in pipe.steps] == ["winsorize", "impute", "scale", "model"]
        scores = baselines.score_pipeline(pipe, frame[features])
        assert scores.shape == (len(frame),) and np.all((scores >= 0) & (scores <= 1))
        table = baselines.coefficient_table(pipe)
        assert set(features) <= set(table["feature"])
        assert (table["abs_coef"].diff().dropna() <= 0).all()
        assert table["odds_ratio"].tolist() == pytest.approx(np.exp(table["coef"]).tolist())
    assert baselines.model_features("logit") == feature_names("P1")
    assert "cert" not in baselines.model_features("logit")
    with pytest.raises(ValueError):
        baselines.make_model("gbm")


def test_texas_baseline_ranks_by_the_ratio_with_missing_last():
    X = pd.DataFrame({"texas_ratio": [0.2, np.nan, 10.0, 0.0, 1.5], "other": range(5)})
    pipe = baselines.make_model("texas").fit(X, [0, 0, 1, 0, 0])
    scores = baselines.score_pipeline(pipe, X)
    assert scores.tolist() == [0.2, baselines.TEXAS_MISSING_SCORE, 10.0, 0.0, 1.5]
    assert list(np.argsort(-scores, kind="stable")) == [2, 4, 0, 3, 1]
    assert baselines.score_pipeline(pipe, X[["texas_ratio"]]).tolist() == scores.tolist()


def test_train_refuses_a_leaky_training_mask(tmp_path, monkeypatch):
    frame = make_frame(n_certs=10)
    settings = make_settings(tmp_path)
    good_train, test = train.fixed_split_masks(frame, settings, 4)
    # 2009Q4 -> window_end_4q 2011-03-01, after the first test prediction date 2010-05-30
    leaky = good_train | (frame["repdte"] == "2009-12-31")
    assert leaky.sum() > good_train.sum()
    monkeypatch.setattr(train, "fixed_split_masks", lambda *a, **k: (leaky, test))
    with pytest.raises(ValueError, match="Rule 6.2"):
        train.train_model("logit_small", 4, settings, frame=frame)
    assert not (tmp_path / "models").exists()


def test_train_writes_artifacts_and_report(tmp_path):
    frame = make_frame(n_certs=30)
    settings = make_settings(tmp_path)
    results = {}
    for name in baselines.MODEL_NAMES:
        r = train.train_model(name, 4, settings, frame=frame)
        results[name] = r
        d = tmp_path / "models" / name
        assert r.paths["dir"] == d
        for stem in ("pipeline.joblib", "metrics.json", "config.json", "features.json"):
            assert (d / stem).exists()
        metrics = json.loads((d / "metrics.json").read_text())
        assert list(metrics["test"]) == list(r.metrics)
        assert metrics["by_year"][-1]["year"] == "pooled"
        years = {row["year"] for row in metrics["by_year"]}
        assert years == {"2010", "2011", "2012", "2013", "pooled"}
        config = json.loads((d / "config.json").read_text())
        train_range = (config["train_repdte_min"], config["train_repdte_max"])
        assert train_range == ("2002-03-31", "2008-12-31")
        test_range = (config["test_repdte_min"], config["test_repdte_max"])
        assert test_range == ("2010-03-31", "2013-12-31")
        assert config["first_test_prediction_date"] == "2010-05-30"
        assert config["n_test"] == r.metrics["n"]
        assert config["positives_test"] == r.metrics["n_failures"]
        assert json.loads((d / "features.json").read_text()) == baselines.model_features(name)
    # the learned models see the planted signal and beat the noise-only Texas baseline
    for name in baselines.MODEL_NAMES:  # base rate is 0.05
        assert results[name].metrics["pr_auc"] > 0.15
    assert train.model_dir(settings, "logit", 8) == tmp_path / "models" / "logit_8q"

    reloaded = train.evaluate_model(None, 4, settings, frame=frame)
    assert set(reloaded) == set(baselines.MODEL_NAMES)
    assert reloaded["logit"].metrics == pytest.approx(results["logit"].metrics)
    with pytest.raises(FileNotFoundError):
        train.evaluate_model("logit", 8, settings, frame=frame)

    path = train.write_baselines_report(reloaded, settings, 4, train.report_path(settings, 4))
    text = path.read_text()
    assert path == tmp_path / "reports" / "p1_baselines.md"
    assert "| texas |" in text and "Odds ratios, logit_small" in text
    assert "Top 10 |coefficient| features, logit" in text and "Leakage sanity check" in text
    assert "2002-03-31 to 2008-12-31" in text and "2010-03-31 to 2013-12-31" in text
