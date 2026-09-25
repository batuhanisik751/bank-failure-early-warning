"""SHAP drivers on the synthetic v2 frame: no data files, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankcanary import tracking
from bankcanary.evaluation import walkforward as w
from bankcanary.explain import shap_drivers as sd
from bankcanary.storage.parquet import read_table
from tests.test_hazard import make_frame, make_settings

YEAR = 2014


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="module")
def fitted(tmp_path_factory, frame):
    """Settings with a fitted walk-forward booster for the last two complete years."""
    settings = make_settings(tmp_path_factory.mktemp("explain"))
    for year in (YEAR, YEAR + 1):
        w.fit_year(frame, settings, year, "gbdt", 4)
    return settings


def test_drivers_table_ranks_positive_then_negative_and_skips_zeros():
    keys = pd.DataFrame({"cert": [1, 2], "repdte": pd.to_datetime(["2014-03-31"] * 2)})
    values = np.array([[3.0, -1.0, 0.0, 2.0, -4.0, 0.5, -0.2], [0.0] * 6 + [1.0]])
    table = sd.drivers_table(keys, values, values * 10, list("abcdefg"), "gbdt", 2014, top_n=3)
    assert list(table.columns) == list(sd.COLUMNS)
    bank1 = table[table["cert"] == 1]
    assert bank1["feature"].tolist() == ["a", "d", "f", "e", "b", "g"]
    assert bank1["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert bank1["direction"].tolist() == ["raises"] * 3 + ["lowers"] * 3
    assert (bank1["feature_value"] == bank1["shap_value"] * 10).all()
    bank2 = table[table["cert"] == 2]
    assert len(bank2) == 1 and bank2["feature"].iloc[0] == "g" and bank2["rank"].iloc[0] == 1
    assert not table.duplicated(list(sd.TABLE_KEY)).any()
    with pytest.raises(ValueError, match="do not match"):
        sd.drivers_table(keys, values[:, :3], values, list("abcdefg"), "gbdt", 2014)


def test_shap_values_reconcile_with_the_raw_score(fitted, frame):
    pipeline, features, _ = w.load_year(fitted, YEAR, "gbdt")
    _, test = w.year_masks(frame, 4, YEAR, lag_days=60)
    X = frame.loc[test, features]
    values, expected, clipped = sd.shap_matrix(pipeline, X)
    assert values.shape == (len(X), len(features))
    raw = pipeline.named_steps["model"].predict(clipped, raw_score=True)
    np.testing.assert_allclose(values.sum(axis=1) + expected, raw, atol=1e-6)
    lo = pipeline.named_steps["winsorize"].lower_bounds_
    assert (np.nan_to_num(clipped, nan=np.inf) >= lo).all()


def test_explain_year_saves_file_logs_run_and_rebuilds_deterministically(fitted, frame):
    drivers, mean_abs = sd.explain_year(frame, fitted, YEAR)
    assert (drivers["model"] == "gbdt").all() and (drivers["model_year"] == YEAR).all()
    assert set(drivers["repdte"].dt.year) == {YEAR}
    assert drivers.groupby(["cert", "repdte"])["rank"].max().le(2 * sd.TOP_N).all()
    assert sd.drivers_path(fitted, YEAR).exists()
    assert list(mean_abs.index[:3]) and mean_abs.is_monotonic_decreasing
    assert mean_abs.index[0] in ("noncurrent_ratio", "tier1_leverage", "texas_ratio")
    config = {
        "model": "gbdt",
        "horizon": 4,
        "year": str(YEAR),
        "model_year": YEAR,
        "features_version": "v2",
    }
    metrics = tracking.find_metrics("explain", config, fitted)
    assert metrics["n_rows"] == 4 * frame["cert"].nunique()
    assert metrics["top_feature"] == mean_abs.index[0]
    assert metrics["mean_abs_shap"][mean_abs.index[0]] == pytest.approx(mean_abs.iloc[0])
    first = sd.rebuild_drivers_table(fitted)
    path = fitted.data_dir / "parquet" / "drivers.parquet"
    digest = path.read_bytes()
    sd.explain_year(frame, fitted, YEAR)
    second = sd.rebuild_drivers_table(fitted)
    assert path.read_bytes() == digest
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns[:4]) == list(sd.TABLE_KEY)
    assert not first.duplicated(list(sd.TABLE_KEY)).any()
    assert read_table("drivers", fitted).equals(first)


def test_production_rows_use_the_latest_booster_past_the_backtest(fitted, frame):
    incomplete = frame.copy()
    complete_col = "label_complete_4q"
    last = pd.Timestamp("2015-12-31")
    incomplete.loc[incomplete["repdte"] == last, complete_col] = False
    incomplete.loc[incomplete["repdte"] == last, "dropped_failed_before_avail"] = [True] + [
        False
    ] * (int((incomplete["repdte"] == last).sum()) - 1)
    mask = sd.production_rows(incomplete, 4)
    assert w.latest_complete_year(incomplete, 4) == YEAR
    assert set(incomplete.loc[mask, "repdte"].dt.year) == {YEAR + 1}
    assert int(mask.sum()) == 4 * frame["cert"].nunique() - 1
    drivers, _ = sd.explain_year(incomplete, fitted, sd.PRODUCTION, save=False)
    assert (drivers["model"] == sd.PRODUCTION_MODEL).all()
    assert (drivers["model_year"] == YEAR).all()
    assert drivers["repdte"].max() == last
    with pytest.raises(ValueError, match="no rows"):
        sd.explain_year(incomplete[incomplete["repdte"].dt.year <= YEAR], fitted, sd.PRODUCTION)


def test_pooled_summary_and_dominance_check():
    by_year = pd.DataFrame(
        {"2009": [5.0, 1.0, 0.0], "2010": [1.0, 1.0, 0.0], "production": [9.0, 0.0, 0.0]},
        index=["a", "b", "c"],
    )
    summary = sd.pooled_summary(by_year)
    assert summary["feature"].tolist() == ["a", "b", "c"]
    assert summary["mean_abs_shap"].tolist() == [3.0, 1.0, 0.0]
    assert summary["share"].sum() == pytest.approx(1.0)
    assert sd.dominance_check(summary) == ["a"]
    assert sd.dominance_check(summary, threshold=0.9) == []
    weighted = sd.pooled_summary(by_year, weights={"2009": 1, "2010": 3})
    assert weighted.loc[0, "mean_abs_shap"] == pytest.approx(2.0)


def test_write_summary_reads_the_run_records_and_writes_the_figure(fitted, frame):
    sd.explain_year(frame, fitted, YEAR)
    sd.explain_year(frame, fitted, YEAR + 1)
    by_year = sd.mean_abs_by_year(fitted)
    assert set(by_year.columns) >= {str(YEAR), str(YEAR + 1)}
    path = sd.write_summary(frame, fitted)
    text = path.read_text(encoding="utf-8")
    assert "Feature-importance smoke test" in text
    # the synthetic risk index drives one feature far past the 40 percent share
    flagged = sd.dominance_check(sd.pooled_summary(by_year))
    assert len(flagged) == 1 and flagged[0] in ("texas_ratio", "tier1_leverage", "noncurrent_ratio")
    assert "**flagged**" in text and f"`{flagged[0]}`" in text
    assert f"| {YEAR} |" in text and f"| {YEAR + 1} |" in text
    figure = sd.figure_path(fitted)
    assert figure.exists() and figure.stat().st_size > 1000
    assert "shap_summary_latest.png" in text
    first = figure.read_bytes()
    sd.write_summary(frame, fitted)
    assert figure.read_bytes() == first
