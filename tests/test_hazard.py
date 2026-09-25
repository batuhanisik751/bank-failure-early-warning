"""Discrete-time hazard on a synthetic features_v2 + labels frame: no data files, no network."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bankcanary import tracking
from bankcanary.config import FixedSplit, GbdtSettings, ModelSettings, Settings
from bankcanary.features.registry import feature_names
from bankcanary.labels.build import build_labels, horizon_columns
from bankcanary.models import hazard

LAG = 60
SPLIT = FixedSplit(
    train_start="2002-03-31", train_end="2008-12-31", test_start="2010-03-31", test_end="2013-12-31"
)
SMALL = {"learning_rate": 0.1, "num_leaves": 7, "min_samples_leaf": 20, "n_estimators": 15}
V2 = feature_names(version="v2")


def make_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
        fixed_split=SPLIT,
        models=ModelSettings(gbdt=GbdtSettings(backend="lightgbm", monotone=False, params=SMALL)),
    )


def make_frame(n_certs: int = 40, seed: int = 0) -> pd.DataFrame:
    """Bank-quarters 2001Q1-2015Q4 with random v2 features and complete 1q/4q/8q labels.

    A latent risk index drives ``y_1q`` (rare) and the wider windows (more frequent);
    ``noncurrent_ratio`` rises and ``tier1_leverage`` falls with it, everything else is
    noise with 5 percent missing values.
    """
    rng = np.random.default_rng(seed)
    quarters = pd.date_range("2001-03-31", "2015-12-31", freq="QE-DEC")
    frame = pd.concat(
        [pd.DataFrame({"cert": c, "repdte": quarters}) for c in range(1, n_certs + 1)],
        ignore_index=True,
    )
    n = len(frame)
    cols: dict = {}
    for name in V2:
        if name.startswith(("bkclass_", "region_")) or name.endswith(
            ("_missing", "_capped", "company", "community_bank")
        ):
            cols[name] = rng.random(n) < 0.1
        else:
            values = rng.normal(size=n)
            values[rng.random(n) < 0.05] = np.nan
            cols[name] = values
    risk = rng.normal(size=n)
    y1 = risk > 2.0  # about 2 percent
    y4 = risk > 1.3
    y8 = risk > 1.0
    cols["noncurrent_ratio"] = cols["noncurrent_ratio"] + risk
    cols["tier1_leverage"] = cols["tier1_leverage"] - risk
    cols["texas_ratio"] = np.abs(cols["texas_ratio"]) + np.clip(risk, 0, None)
    avail = frame["repdte"] + pd.Timedelta(days=LAG)
    cols["dropped_failed_before_avail"] = False
    cols["rssdhcr"], cols["fail_date"] = 0, pd.NaT
    for horizon, y in ((1, y1), (4, y4), (8, y8)):
        y_col, end_col, cens_col, complete_col = horizon_columns(horizon)
        cols[y_col] = y.astype(int)
        cols[end_col] = (avail + pd.DateOffset(months=3 * horizon)).astype("datetime64[ns]")
        cols[cens_col] = False
        cols[complete_col] = True
    return frame.assign(**cols)


def test_convert_hazard_and_brier_identities():
    h = np.array([0.0, 0.01, 0.1, 0.5, 1.0])
    assert np.allclose(hazard.convert_hazard(h, 1), h)
    p4, p8 = hazard.convert_hazard(h, 4), hazard.convert_hazard(h, 8)
    assert np.allclose(p4, 1 - (1 - h) ** 4) and np.all(p8 >= p4) and np.all(p4 >= h)
    assert p4[0] == 0.0 and p4[-1] == 1.0 and np.all((p8 >= 0) & (p8 <= 1))
    assert hazard.convert_hazard([1.5, -0.2], 4).tolist() == [1.0, 0.0]  # clipped first
    with pytest.raises(ValueError, match="positive"):
        hazard.convert_hazard(h, 0)
    assert hazard.brier([0, 1], [0.0, 1.0]) == 0.0
    assert hazard.brier([0, 1], [0.5, 0.5]) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="shape"):
        hazard.brier([0, 1], [0.5])


def test_make_hazard_is_the_p1_pipeline_without_class_weighting():
    pipe = hazard.make_hazard()
    assert [s for s, _ in pipe.steps] == ["winsorize", "impute", "scale", "model"]
    model = pipe.named_steps["model"]
    assert model.C == hazard.HAZARD_C and model.class_weight is None
    assert hazard.make_hazard(0.5).named_steps["model"].C == 0.5
    assert hazard.hazard_features() == V2 and len(hazard.INTERPRETED_FEATURES) == 15
    assert set(hazard.INTERPRETED_FEATURES) <= set(V2)


def test_labels_build_gives_the_one_quarter_event_its_own_window():
    avail = pd.Timestamp("2008-05-30")
    panel = pd.DataFrame(
        {
            "cert": [1, 2, 3, 4],
            "repdte": pd.Timestamp("2008-03-31"),
            "avail_date": avail,
            # fails inside the quarter, fails in month five, merges next month, open
            "fail_date": [pd.Timestamp("2008-07-15"), pd.Timestamp("2008-10-20"), pd.NaT, pd.NaT],
            "exit_date": [pd.NaT, pd.NaT, pd.Timestamp("2008-06-30"), pd.NaT],
        }
    )
    lab = build_labels(panel, [1, 4, 8], pd.Timestamp("2011-01-01")).set_index("cert")
    assert lab["window_end_1q"].eq(avail + pd.DateOffset(months=3)).all()
    assert lab["y_1q"].tolist() == [1, 0, 0, 0] and lab["y_4q"].tolist() == [1, 1, 0, 0]
    assert lab["censored_in_window_1q"].tolist() == [False, False, True, False]
    assert lab["label_complete_1q"].all() and set(lab.columns) >= set(horizon_columns(1))


def test_inner_split_stays_inside_the_training_period_at_1q(tmp_path):
    frame, settings = make_frame(n_certs=10), make_settings(tmp_path)
    inner_train, validation = hazard.inner_split(frame, settings)
    assert not (inner_train & validation).any()
    assert frame.loc[inner_train, "repdte"].max() == pd.Timestamp("2006-09-30")
    assert frame.loc[validation, "repdte"].min() == pd.Timestamp("2007-03-31")
    assert frame.loc[validation, "repdte"].max() == pd.Timestamp("2008-12-31")
    assert frame.loc[inner_train, "repdte"].min() == pd.Timestamp("2002-03-31")


def test_tune_c_logs_one_run_per_candidate_and_reads_them_back(tmp_path):
    frame, settings = make_frame(n_certs=20, seed=1), make_settings(tmp_path)
    table = hazard.tune_c(frame, settings, grid=(1.0, 0.01))
    assert table["C"].tolist() in ([1.0, 0.01], [0.01, 1.0])
    assert not table["cached"].any() and table["pr_auc_4q"].is_monotonic_decreasing
    for col in ("pr_auc_1q", "brier_4q", "brier_raw_hazard_4q", "brier_climatology_4q"):
        assert col in table.columns
    runs = tracking.read_index(settings)
    assert len(runs) == 2 and {r["name"] for r in runs} == {"tune_hazard"}
    again = hazard.tune_c(frame, settings, grid=(1.0, 0.01))
    assert again["cached"].all()
    pd.testing.assert_frame_equal(again.drop(columns="cached"), table.drop(columns="cached"))
    assert len(tracking.read_index(settings)) == 2
    tuning = hazard.tuning_table(settings)
    assert list(tuning.columns) == list(hazard.TUNING_COLUMNS) and len(tuning) == 2


def test_fit_evaluate_odds_ratios_and_report(tmp_path):
    frame, settings = make_frame(n_certs=40), make_settings(tmp_path)
    result = hazard.fit_hazard(frame, settings, C=0.1)
    d = tmp_path / "models" / "hazard"
    assert {p.name for p in d.iterdir()} == {
        "pipeline.joblib",
        "metrics.json",
        "config.json",
        "features.json",
    }
    config = json.loads((d / "config.json").read_text())
    assert config["C"] == 0.1 and config["horizon"] == 1 and config["label"] == "y_1q"
    assert config["train_repdte_max"] == "2008-12-31" and config["test_repdte_min"] == "2010-03-31"
    assert result.features == V2 and "brier" in result.metrics and result.metrics["roc_auc"] > 0.8
    assert len(tracking.read_index(settings)) == 1
    # converted evaluation: ranking is invariant to the conversion, Brier is not
    block = hazard.evaluate_converted(result.pipeline, frame, settings, 4, 0.1, ("logit_v2",))
    hz = block["hazard"]
    assert set(hz) >= {"pr_auc", "brier", "brier_raw_hazard", "brier_climatology"}
    assert (tmp_path / "models" / "logit_v2" / "pipeline.joblib").exists()
    assert set(block["comparators"]) == {"logit_v2"} and "brier" in block["comparators"]["logit_v2"]
    from bankcanary.evaluation.metrics import evaluate
    from bankcanary.splits import fixed_split_masks

    _, test = fixed_split_masks(frame, settings, 4)
    rows = frame.loc[test]
    raw = result.pipeline.predict_proba(rows[V2])[:, 1]
    same = evaluate(rows["y_4q"].to_numpy(), raw, tie_breaker=rows["cert"].to_numpy())
    assert hz["pr_auc"] == pytest.approx(same["pr_auc"]) and hz["brier"] != hz["brier_raw_hazard"]
    assert hz["brier"] == pytest.approx(hazard.brier(rows["y_4q"], hazard.convert_hazard(raw, 4)))
    hazard.evaluate_converted(result.pipeline, frame, settings, 8, 0.1, ())
    saved = json.loads((d / "metrics.json").read_text())
    assert set(saved["converted"]) == {"4q", "8q"} and saved["test"]["n"] == result.metrics["n"]
    assert saved["converted"]["8q"]["comparators"] == {}
    hazard.fit_hazard(frame, settings, C=0.1)  # a re-fit keeps the converted blocks
    assert set(json.loads((d / "metrics.json").read_text())["converted"]) == {"4q", "8q"}
    names = [r["name"] for r in tracking.read_index(settings)]
    assert names.count("train") == 4  # hazard 1q (twice, same id), logit_v2 4q, ->4q, ->8q
    # odds ratios: unpenalised statsmodels fit with cluster-robust intervals
    train, _ = fixed_split_masks(frame, settings, 1)
    odds = hazard.odds_ratio_table(frame, train, result.pipeline)
    assert odds["feature"].tolist() == list(hazard.INTERPRETED_FEATURES)
    assert (odds["ci_low"] <= odds["odds_ratio"]).all() and (
        odds["odds_ratio"] <= odds["ci_high"]
    ).all()
    assert odds["p_value"].between(0, 1).all() and odds["method"].eq(hazard.ODDS_RATIO_METHOD).all()
    by = odds.set_index("feature")
    assert (
        by.loc["noncurrent_ratio", "odds_ratio"] > 1 > by.loc["tier1_leverage", "odds_ratio"]
        if ("tier1_leverage" in by.index)
        else by.loc["noncurrent_ratio", "odds_ratio"] > 1
    )
    assert by.loc["noncurrent_ratio", "ci_low"] > 1 and by.loc["noncurrent_ratio", "monotone"] == 1
    assert odds["full_model_odds_ratio"].notna().all()
    hazard._write_json(d / "odds_ratios.json", {"rows": odds.to_dict(orient="records")})
    path = hazard.write_hazard_report(settings)
    text = path.read_text()
    assert path == tmp_path / "reports" / "p2_hazard.md"
    for needle in (
        "| hazard (converted) |",
        "| logit_v2 |",
        "### 4q test rows",
        "### 8q test rows",
        "1 - (1 - h)^H",
        "| noncurrent_ratio |",
        "cluster-robust",
        "Per failure event",
    ):
        assert needle in text


def test_fit_refuses_a_leaky_training_mask(tmp_path, monkeypatch):
    import bankcanary.splits as splits

    frame, settings = make_frame(n_certs=10), make_settings(tmp_path)
    good_train, test = splits.fixed_split_masks(frame, settings, 1)
    leaky = good_train | (frame["repdte"] == "2009-12-31")
    monkeypatch.setattr(splits, "fixed_split_masks", lambda *a, **k: (leaky, test))
    with pytest.raises(ValueError, match="Rule 6.2"):
        hazard.fit_hazard(frame, settings, C=0.1)
    assert not (tmp_path / "models").exists() and not (tmp_path / "runs").exists()


def test_train_hazard_command_is_registered():
    from typer.testing import CliRunner

    from bankcanary.cli import app

    result = CliRunner().invoke(app, ["train-hazard", "--help"])
    assert result.exit_code == 0 and "--tune" in result.output and "--horizon" in result.output
