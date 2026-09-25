"""Gradient boosting on a synthetic features_v2 + labels frame: no data files, no network."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bankcanary import tracking
from bankcanary.config import FixedSplit, GbdtSettings, ModelSettings, Settings, load_settings
from bankcanary.features.registry import feature_names, monotone_constraints
from bankcanary.labels.build import horizon_columns
from bankcanary.models import gbdt

LAG = 60
SPLIT = FixedSplit(
    train_start="2002-03-31", train_end="2008-12-31", test_start="2010-03-31", test_end="2013-12-31"
)
SMALL = {"learning_rate": 0.1, "num_leaves": 7, "min_samples_leaf": 20, "n_estimators": 15}
V2 = feature_names(version="v2")


def make_settings(tmp_path, monotone: bool = False, backend: str | None = "lightgbm") -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
        fixed_split=SPLIT,
        models=ModelSettings(
            gbdt=GbdtSettings(
                backend=backend,
                monotone=monotone,
                params=SMALL,
                inner_pr_auc={"gbdt": 0.3, "gbdt_mono": 0.2},
            )
        ),
    )


def make_frame(n_certs: int = 40, seed: int = 0) -> pd.DataFrame:
    """Bank-quarters 2001Q1-2015Q4 with random v2 features and complete 4q/8q labels.

    ``noncurrent_ratio`` (sign +1) and ``tier1_leverage`` (sign -1) carry the signal for
    positives; every other feature is noise, with 5 percent missing values.
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
    y = rng.random(n) < 0.05
    cols["noncurrent_ratio"] = cols["noncurrent_ratio"] + np.where(y, 2.0, 0.0)
    cols["tier1_leverage"] = cols["tier1_leverage"] - np.where(y, 2.0, 0.0)
    cols["texas_ratio"] = np.abs(cols["texas_ratio"]) + np.where(y, 1.0, 0.0)
    avail = frame["repdte"] + pd.Timedelta(days=LAG)
    cols["dropped_failed_before_avail"] = False
    cols["rssdhcr"], cols["fail_date"] = 0, pd.NaT
    for horizon in (4, 8):
        y_col, end_col, cens_col, complete_col = horizon_columns(horizon)
        cols[y_col] = y.astype(int)
        cols[end_col] = (avail + pd.DateOffset(months=3 * horizon)).astype("datetime64[ns]")
        cols[cens_col] = False
        cols[complete_col] = True
    return frame.assign(**cols)


def test_make_gbdt_builds_winsorizer_then_estimator_for_both_backends():
    for backend, cls in (
        ("lightgbm", "LGBMClassifier"),
        ("sklearn", "HistGradientBoostingClassifier"),
    ):
        pipe = gbdt.make_gbdt(backend, monotone=True, **SMALL)
        assert [s for s, _ in pipe.steps] == ["winsorize", "model"]
        model = pipe.named_steps["model"]
        assert type(model).__name__ == cls
        desc = gbdt.describe(pipe)
        assert desc["backend"] == backend and desc["monotone"] is True
        assert desc["params"] == SMALL
        signs = model.monotone_constraints if backend == "lightgbm" else model.monotonic_cst
        assert list(signs) == monotone_constraints(V2)
        assert len(signs) == len(V2) and set(signs) == {-1, 0, 1}
    assert gbdt.describe(gbdt.make_gbdt("lightgbm"))["monotone"] is False
    assert gbdt.make_gbdt("sklearn").named_steps["model"].early_stopping is False
    assert gbdt.resolve_backend(None) in gbdt.BACKENDS
    with pytest.raises(ValueError, match="unknown gbdt backend"):
        gbdt.make_gbdt("xgboost")
    with pytest.raises(ValueError, match="unknown gbdt parameter"):
        gbdt.make_gbdt("lightgbm", max_depth=3)
    with pytest.raises(ValueError, match="monotone signs"):
        gbdt.make_gbdt("lightgbm", monotone=[1, -1], features=["a", "b", "c"])
    assert gbdt.constraint_vector(["a", "b"], [1, 0]) == [1, 0]
    assert gbdt.constraint_vector(["a"], None) is None


def test_monotone_constraints_are_enforced_and_nan_is_native():
    frame = make_frame(n_certs=30, seed=2)
    features = ["noncurrent_ratio", "tier1_leverage", "roa_q", "log_assets"]
    assert monotone_constraints(features)[:2] == [1, -1]
    for backend in ("lightgbm", "sklearn"):
        pipe = gbdt.make_gbdt(backend, monotone=True, features=features, **SMALL)
        pipe.fit(frame[features], frame["y_4q"])
        assert frame[features].isna().any().any()  # NaN reaches the trees untouched
        scores = pipe.predict_proba(frame[features])[:, 1]
        assert scores.shape == (len(frame),) and np.all((scores >= 0) & (scores <= 1))
        base = frame[features].iloc[:50].copy()
        grid = np.linspace(-3, 3, 13)
        up = [pipe.predict_proba(base.assign(noncurrent_ratio=v))[:, 1] for v in grid]
        assert all(np.all(b >= a - 1e-12) for a, b in zip(up, up[1:]))
        down = [pipe.predict_proba(base.assign(tier1_leverage=v))[:, 1] for v in grid]
        assert all(np.all(b <= a + 1e-12) for a, b in zip(down, down[1:]))
    # the unconstrained twin has no such guarantee but scores the same rows fine
    free = gbdt.make_gbdt("lightgbm", features=features, **SMALL).fit(
        frame[features], frame["y_4q"]
    )
    assert free.named_steps["model"].monotone_constraints is None


def test_feature_importance_gain_and_permutation():
    frame = make_frame(n_certs=30, seed=3)
    features = ["noncurrent_ratio", "tier1_leverage", "roa_q", "log_assets"]
    X, y = frame[features], frame["y_4q"].to_numpy()
    lgb = gbdt.make_gbdt("lightgbm", features=features, **SMALL).fit(X, y)
    table = gbdt.feature_importance(lgb, features)
    assert list(table.columns) == ["feature", "importance", "kind"]
    assert table["kind"].eq("gain").all() and table["importance"].is_monotonic_decreasing
    assert table["feature"].iloc[0] in ("noncurrent_ratio", "tier1_leverage")
    skl = gbdt.make_gbdt("sklearn", features=features, **SMALL).fit(X, y)
    with pytest.raises(ValueError, match="permutation"):
        gbdt.feature_importance(skl, features)
    perm = gbdt.feature_importance(skl, features, X, y, n_repeats=2)
    assert perm["kind"].eq("permutation").all() and set(perm["feature"]) == set(features)


def test_fit_on_fixed_split_saves_artifacts_logs_a_run_and_writes_the_report(tmp_path):
    frame = make_frame(n_certs=40)
    settings = make_settings(tmp_path)
    results = {}
    for name in ("texas", "logit_v2", "gbdt", "gbdt_mono"):
        pipe, features = gbdt.make_variant(name, settings)
        results[name] = gbdt.fit_on_fixed_split(
            name, pipe, features, frame, settings, 4, save=name != "texas"
        )
    assert results["gbdt"].features == V2 and results["texas"].features == ["texas_ratio"]
    d = tmp_path / "models" / "gbdt_mono"
    assert {p.name for p in d.iterdir()} == {
        "pipeline.joblib",
        "metrics.json",
        "config.json",
        "features.json",
    }
    assert not (tmp_path / "models" / "texas").exists()
    config = json.loads((d / "config.json").read_text())
    assert config["backend"] == "lightgbm" and config["monotone"] is True
    assert config["params"] == SMALL and config["features_version"] == "v2"
    assert config["train_repdte_max"] == "2008-12-31" and config["test_repdte_min"] == "2010-03-31"
    assert "backend" not in json.loads((tmp_path / "models/logit_v2/config.json").read_text())
    runs = tracking.read_index(settings)
    assert len(runs) == 4 and all(r["name"] == "train" and r["horizon"] == 4 for r in runs)
    assert (
        tracking.find_metrics("train", results["gbdt"].config, settings)["test"]["pr_auc"]
        == (results["gbdt"].metrics["pr_auc"])
    )
    assert results["gbdt"].metrics["roc_auc"] > 0.9  # the signal is easy on purpose
    importance = gbdt.feature_importance(results["gbdt"].pipeline, V2)
    path = gbdt.write_gbdt_report(results, settings, 4, gbdt.report_path(settings), importance)
    text = path.read_text()
    assert path == tmp_path / "reports" / "p2_gbdt.md"
    for needle in (
        "| texas |",
        "| logit_v2 |",
        "| gbdt |",
        "| gbdt_mono |",
        "0.3000",
        "0.2000",
        "Top 15 features of `gbdt` by gain",
        "monotone = false",
        "Per failure event",
    ):
        assert needle in text
    assert (tmp_path / "runs" / "index.jsonl").read_text().count("\n") == 4


def test_fit_refuses_a_leaky_training_mask(tmp_path, monkeypatch):
    import bankcanary.splits as splits

    frame = make_frame(n_certs=10)
    settings = make_settings(tmp_path)
    good_train, test = splits.fixed_split_masks(frame, settings, 4)
    leaky = good_train | (frame["repdte"] == "2009-12-31")
    monkeypatch.setattr(splits, "fixed_split_masks", lambda *a, **k: (leaky, test))
    pipe, features = gbdt.make_variant("gbdt", settings)
    with pytest.raises(ValueError, match="Rule 6.2"):
        gbdt.fit_on_fixed_split("gbdt", pipe, features, frame, settings, 4)
    assert not (tmp_path / "models").exists() and not (tmp_path / "runs").exists()


def test_write_gbdt_settings_round_trips_and_replaces_in_place(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "data_dir: data\n# keep me\nfixed_split:\n  train_start: 2002-03-31\n"
        "  train_end: 2008-12-31\n  test_start: 2010-03-31\n  test_end: 2013-12-31\n"
    )
    gbdt.write_gbdt_settings(path, "sklearn", False, {"num_leaves": 31}, {"gbdt": 0.5})
    gbdt.write_gbdt_settings(path, "lightgbm", True, SMALL, {"gbdt": 0.5, "gbdt_mono": 0.6})
    text = path.read_text()
    assert text.count("models:") == 1 and "# keep me" in text
    cfg = load_settings(path, root=tmp_path).models.gbdt
    assert cfg.backend == "lightgbm" and cfg.monotone is True and cfg.params == SMALL
    assert cfg.inner_pr_auc == {"gbdt": 0.5, "gbdt_mono": 0.6}
    with pytest.raises(ValueError, match="unknown variant"):
        gbdt.make_variant("hazard", make_settings(tmp_path))


def test_train_gbdt_command_is_registered():
    from typer.testing import CliRunner

    from bankcanary.cli import app

    result = CliRunner().invoke(app, ["train-gbdt", "--help"])
    assert result.exit_code == 0 and "--variant" in result.output
