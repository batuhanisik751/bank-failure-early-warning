"""Adversarial checks of the walk-forward harness on synthetic frames: no data files, no network.

Every expected value below is derived by hand from spec rule 6.2 (training rows need
``window_end_Hq < avail(first test quarter)``), rule 6.3 (preprocessing fitted on the
training fold only), rule 6.7 (tuning inside the training period) and section 8.2, with
the synthetic frame of ``tests.test_hazard.make_frame``: 40 banks, quarters 2001Q1-2015Q4,
a 60-day availability lag and complete 1q/4q/8q labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankcanary.evaluation import walkforward as w
from bankcanary.labels.build import horizon_columns
from bankcanary.models.hazard import brier, convert_hazard
from bankcanary.splits import prediction_date, training_mask
from bankcanary.storage.parquet import table_path, write_table
from tests.test_hazard import LAG, make_frame, make_settings

N_CERTS = 40
YEAR = 2010
T_2010 = pd.Timestamp("2010-05-30")  # avail(2010-03-31) with the 60-day lag

# Hand-derived last usable training quarter for test year 2010: the window of the next
# quarter closes exactly on T (a strict inequality must exclude it).
LAST_TRAIN_REPDTE = {1: "2009-09-30", 4: "2008-12-31", 8: "2007-12-31"}
FIRST_EXCLUDED_REPDTE = {1: "2009-12-31", 4: "2009-03-31", 8: "2008-03-31"}
# Quarters from 2001Q1 up to and including the last usable one, times 40 banks.
N_TRAIN_2010 = {1: 35 * N_CERTS, 4: 32 * N_CERTS, 8: 28 * N_CERTS}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame(n_certs=N_CERTS)


def _year_rows(frame: pd.DataFrame, year: int) -> pd.Series:
    return frame["repdte"].dt.year == year


@pytest.mark.parametrize("model", w.MODELS)
@pytest.mark.parametrize("horizon", (4, 8))
def test_training_windows_close_strictly_before_the_first_test_prediction_date(
    frame, model, horizon
):
    fit_h = w.fit_horizon_of(model, horizon)
    train, test = w.year_masks(frame, horizon, YEAR, fit_h, LAG)
    end_col = horizon_columns(fit_h)[1]
    assert w.year_bounds(YEAR)[0] == pd.Timestamp("2010-03-31")
    assert prediction_date(w.year_bounds(YEAR)[0], LAG) == T_2010
    assert (frame.loc[train, end_col] < T_2010).all()
    assert frame.loc[train, "repdte"].max() == pd.Timestamp(LAST_TRAIN_REPDTE[fit_h])
    boundary = frame["repdte"] == pd.Timestamp(FIRST_EXCLUDED_REPDTE[fit_h])
    assert boundary.sum() == N_CERTS and not (train & boundary).any()
    assert int(train.sum()) == N_TRAIN_2010[fit_h]
    assert not (train & test).any()
    assert set(frame.loc[test, "repdte"].dt.year) == {YEAR}
    assert int(test.sum()) == 4 * N_CERTS


def test_horizon_8_training_stops_at_2005q4_for_the_first_test_year(frame):
    # T(2008) = 2008-05-30; 2006Q1's 8q window closes on 2008-05-30 exactly (excluded).
    train, test = w.year_masks(frame, 8, 2008, lag_days=LAG)
    assert frame.loc[train, "repdte"].max() == pd.Timestamp("2005-12-31")
    assert int(train.sum()) == 20 * N_CERTS
    assert set(frame.loc[test, "repdte"].dt.year) == {2008}
    # Leap year: 2007-12-31 + 60 days = 2008-02-29, whose 1q window closes on 2008-05-29,
    # one day before T, so the hazard may still learn from 2007Q4 for test year 2008.
    hazard_train, _ = w.year_masks(frame, 8, 2008, fit_horizon=1, lag_days=LAG)
    assert frame.loc[hazard_train, "repdte"].max() == pd.Timestamp("2007-12-31")
    assert int(hazard_train.sum()) == 28 * N_CERTS


def test_test_rows_are_exactly_the_usable_reports_of_the_year(tmp_path, frame):
    complete_col = horizon_columns(4)[3]
    tampered = frame.copy()
    incomplete = (tampered["cert"] == 1) & (tampered["repdte"] == pd.Timestamp("2010-06-30"))
    dropped = (tampered["cert"] == 2) & (tampered["repdte"] == pd.Timestamp("2010-09-30"))
    tampered.loc[incomplete, complete_col] = False
    tampered.loc[dropped, "dropped_failed_before_avail"] = True
    result = w.fit_year(tampered, make_settings(tmp_path), YEAR, "texas", 4, save=False)
    scores = result.scores
    assert len(scores) == 4 * N_CERTS - 2
    assert (scores["repdte"].dt.year == YEAR).all()
    assert scores["repdte"].nunique() == 4
    assert not ((scores["cert"] == 1) & (scores["repdte"] == pd.Timestamp("2010-06-30"))).any()
    assert not ((scores["cert"] == 2) & (scores["repdte"] == pd.Timestamp("2010-09-30"))).any()
    assert scores["label_complete"].all()
    assert not scores[["cert", "repdte"]].duplicated().any()
    assert result.config["n_test"] == 4 * N_CERTS - 2


def test_preprocessing_bounds_and_medians_come_from_training_rows_only(tmp_path, frame):
    result = w.fit_year(frame, make_settings(tmp_path), YEAR, "logit", 4, save=False)
    train, test = w.year_masks(frame, 4, YEAR, lag_days=LAG)
    x_train = frame.loc[train, result.features].to_numpy(dtype=float)
    x_both = frame.loc[train | test, result.features].to_numpy(dtype=float)
    winsor = result.pipeline.named_steps["winsorize"]
    continuous = ~winsor.skipped_
    assert continuous.sum() > 10
    lo_train = np.nanquantile(x_train[:, continuous], 0.005, axis=0)
    hi_train = np.nanquantile(x_train[:, continuous], 0.995, axis=0)
    np.testing.assert_array_equal(winsor.lower_bounds_[continuous], lo_train)
    np.testing.assert_array_equal(winsor.upper_bounds_[continuous], hi_train)
    hi_both = np.nanquantile(x_both[:, continuous], 0.995, axis=0)
    assert not np.array_equal(hi_train, hi_both)  # the check can tell the folds apart
    medians = result.pipeline.named_steps["impute"].statistics_
    clipped = np.clip(x_train, winsor.lower_bounds_, winsor.upper_bounds_)
    np.testing.assert_allclose(medians, np.nanmedian(clipped, axis=0))


@pytest.mark.parametrize("model", ("logit", "gbdt", "hazard"))
@pytest.mark.parametrize("horizon", (4, 8))
def test_poisoning_every_row_from_the_test_year_on_leaves_the_fit_unchanged(
    tmp_path, frame, model, horizon
):
    """Rule 6.2 + 6.3: nothing dated in or after the test year may touch the fitted model."""
    settings = make_settings(tmp_path)
    clean = w.fit_year(frame, settings, YEAR, model, horizon, save=False)
    poisoned = frame.copy()
    later = poisoned["repdte"] >= pd.Timestamp(f"{YEAR}-01-01")
    for name in clean.features:
        if poisoned[name].dtype == bool:
            poisoned.loc[later, name] = True
        else:
            poisoned.loc[later, name] = 1e6
    y_cols = [horizon_columns(h)[0] for h in (1, 4, 8)]
    poisoned.loc[later, y_cols] = 1
    dirty = w.fit_year(poisoned, settings, YEAR, model, horizon, save=False)
    assert dirty.config["n_train"] == clean.config["n_train"]
    assert dirty.config["positives_train"] == clean.config["positives_train"]
    _, test = w.year_masks(frame, horizon, YEAR, lag_days=LAG)
    rows = frame.loc[test]
    a = w.score_rows(clean.pipeline, rows, clean.features, model, horizon, YEAR)["score"]
    b = w.score_rows(dirty.pipeline, rows, dirty.features, model, horizon, YEAR)["score"]
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_hazard_conversion_is_applied_at_8q_in_metrics_and_after_reload(tmp_path, frame):
    settings = make_settings(tmp_path)
    result = w.fit_year(frame, settings, YEAR, "hazard", 8)
    assert result.config["fit_horizon"] == 1 and result.config["fit_label"] == "y_1q"
    assert result.config["train_repdte_max"] == LAST_TRAIN_REPDTE[1]
    _, test = w.year_masks(frame, 8, YEAR, fit_horizon=1, lag_days=LAG)
    h = result.pipeline.predict_proba(frame.loc[test, result.features])[:, 1]
    converted = 1.0 - (1.0 - h) ** 8
    np.testing.assert_allclose(result.scores["score"], converted)
    np.testing.assert_allclose(result.scores["score"], convert_hazard(h, 8))
    assert (result.scores["score"] > h).all()  # 8 quarters of hazard exceed one quarter
    y8 = frame.loc[test, horizon_columns(8)[0]].to_numpy()
    np.testing.assert_array_equal(result.scores["y"], y8)
    assert result.metrics["brier"] == pytest.approx(brier(y8, converted))
    assert result.metrics["brier"] != pytest.approx(brier(y8, h))
    pipeline, features, config = w.load_year(settings, YEAR, "hazard", 8)
    assert config["conversion"] == "1 - (1 - h)^H"
    rescored = w.score_rows(pipeline, frame.loc[test], features, "hazard", 8, YEAR)
    np.testing.assert_allclose(rescored["score"], converted)
    saved = pd.read_parquet(w.scores_path(settings, YEAR, "hazard", 8))
    np.testing.assert_allclose(saved["score"], converted)
    table = w.rebuild_scores_table(settings)
    per_year = w.per_year_tables(table[table["horizon"] == 8])["hazard"]
    pooled = per_year[per_year["year"] == "pooled"].iloc[0]
    assert pooled["brier"] == pytest.approx(brier(y8, converted))


def test_texas_missing_ratios_rank_last_and_metrics_survive(tmp_path, frame):
    tampered = frame.copy()
    in_year = _year_rows(tampered, YEAR)
    failing = in_year & (tampered[horizon_columns(4)[0]] == 1)
    assert failing.sum() >= 3
    victims = tampered.index[failing][:2].tolist() + tampered.index[in_year & ~failing][:3].tolist()
    tampered.loc[victims, "texas_ratio"] = np.nan
    n_missing = int(tampered.loc[in_year, "texas_ratio"].isna().sum())  # frame noise + victims
    assert n_missing >= 5
    assert (tampered.loc[in_year, "texas_ratio"].dropna() >= 0).all()
    result = w.fit_year(tampered, make_settings(tmp_path), YEAR, "texas", 4, save=False)
    scores = result.scores
    assert np.isfinite(scores["score"]).all()
    assert int((scores["score"] == -1.0).sum()) == n_missing
    assert (scores.loc[scores["score"] != -1.0, "score"] >= 0).all()
    victim_keys = set(zip(tampered.loc[victims, "cert"], tampered.loc[victims, "repdte"]))
    victim_rows = [k in victim_keys for k in zip(scores["cert"], scores["repdte"])]
    assert sum(victim_rows) == 5 and (scores.loc[victim_rows, "score"] == -1.0).all()
    order = scores.sort_values(["score", "cert"], ascending=[False, True], kind="mergesort")
    assert (order["score"].tail(n_missing) == -1.0).all()
    assert result.metrics["brier"] is None
    for name in ("pr_auc", "roc_auc", "recall_at_2pct", "recall_at_top100"):
        assert np.isfinite(result.metrics[name])
    blind = tampered.copy()
    blind.loc[in_year, "texas_ratio"] = np.nan
    result = w.fit_year(blind, make_settings(tmp_path / "blind"), YEAR, "texas", 4, save=False)
    assert (result.scores["score"] == -1.0).all()
    assert np.isfinite(result.metrics["pr_auc"]) and result.metrics["brier"] is None


def test_rebuild_ignores_input_row_order_and_a_rerun_replaces_rather_than_duplicates(
    tmp_path, frame
):
    settings = make_settings(tmp_path)
    w.fit_year(frame, settings, YEAR, "texas", 4)
    first = w.rebuild_scores_table(settings)
    first_bytes = table_path(settings, w.TABLE).read_bytes()
    assert len(first) == 4 * N_CERTS
    year_file = w.scores_path(settings, YEAR, "texas", 4)
    ordered_certs = pd.read_parquet(year_file)["cert"].tolist()
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert not shuffled["cert"].equals(frame["cert"])
    w.fit_year(shuffled, settings, YEAR, "texas", 4)
    assert pd.read_parquet(year_file)["cert"].tolist() != ordered_certs  # per-year file moved
    w.rebuild_scores_table(settings)
    assert table_path(settings, w.TABLE).read_bytes() == first_bytes
    changed = frame.copy()
    changed.loc[_year_rows(changed, YEAR), "texas_ratio"] = 3.5
    w.fit_year(changed, settings, YEAR, "texas", 4)
    third = w.rebuild_scores_table(settings)
    assert len(third) == 4 * N_CERTS
    assert not third[list(w.TABLE_KEY)].duplicated().any()
    assert (third["score"] == 3.5).all()
    keys = third[list(w.TABLE_KEY)]
    assert keys.sort_values(list(w.TABLE_KEY), kind="mergesort").index.equals(keys.index)


def _scores_table(years: dict[int, int]) -> pd.DataFrame:
    """One 4q table with ``logit`` and ``texas`` rows; ``years`` maps year -> failures."""
    rng = np.random.default_rng(1)
    parts = []
    for year, n_fail in years.items():
        for model in ("logit", "texas"):
            n = 4 * N_CERTS
            y = np.zeros(n, dtype=int)
            y[:n_fail] = 1
            score = rng.random(n) if model == "logit" else rng.random(n) * 4
            parts.append(
                pd.DataFrame(
                    {
                        "cert": np.tile(np.arange(1, N_CERTS + 1), 4),
                        "repdte": np.repeat(
                            pd.date_range(f"{year}-03-31", periods=4, freq="QE"), N_CERTS
                        ),
                        "horizon": 4,
                        "model": model,
                        "test_year": year,
                        "score": score,
                        "score_calibrated": np.nan,
                        "y": y,
                        "label_complete": True,
                        "censored": False,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def test_a_test_year_without_failures_is_reported_as_undefined_not_crashed(tmp_path, frame):
    settings = make_settings(tmp_path)
    table = _scores_table({2010: 6, 2011: 0, 2012: 3})
    write_table(table, w.TABLE, key=w.TABLE_KEY, settings=settings)
    per_year = w.per_year_tables(w.read_scores_table(settings, 4))
    logit = per_year["logit"].set_index("year")
    assert logit.loc["2011", "n_failures"] == 0 and logit.loc["2011", "n"] == 4 * N_CERTS
    assert np.isnan(logit.loc["2011", "pr_auc"]) and np.isnan(logit.loc["2011", "roc_auc"])
    empty = table[(table["model"] == "logit") & (table["test_year"] == 2011)]
    assert logit.loc["2011", "brier"] == pytest.approx(float(np.mean(empty["score"] ** 2)))
    assert logit.loc["pooled", "n_failures"] == 9 and np.isfinite(logit.loc["pooled", "pr_auc"])
    texas = per_year["texas"].set_index("year")
    assert texas.loc["2011", "brier"] is None or np.isnan(texas.loc["2011", "brier"])
    pooled = w.pooled_table(per_year)
    assert set(pooled["model"]) == {"logit", "texas"} and (pooled["n_failures"] == 9).all()
    text = w.write_walkforward_report(settings, 4).read_text()
    assert "Years with no failure (2011)" in text
    assert "| 2011 | 160 | 0 | n/a | n/a |" in text
    quiet = frame.copy()
    quiet.loc[_year_rows(quiet, YEAR), horizon_columns(4)[0]] = 0
    result = w.fit_year(quiet, settings, YEAR, "logit", 4)
    assert result.metrics["n_failures"] == 0 and np.isnan(result.metrics["pr_auc"])
    assert 0.0 <= result.metrics["brier"] < 1.0
    assert (tmp_path / "runs" / "walkforward").exists()


# Spec rule 6.7: the hyper-parameters a test year's model uses must be chosen on rows the
# model may train on. The D6/D7 constants were picked on reports 2007Q1-2008Q4, which is the
# 2008 test year itself and lies past 2009's training cut, so the harness re-selects per
# year on a slice inside that year's own training period.
TUNING_HORIZON = 4


@pytest.mark.parametrize("year", [2008, 2009, 2010, 2015])
@pytest.mark.parametrize("model", ("logit", "gbdt", "hazard"))
def test_hyper_parameters_were_selected_inside_the_years_training_period(
    tmp_path, frame, year, model
):
    settings = make_settings(tmp_path)
    start, _ = w.year_bounds(year)
    fit_h = w.fit_horizon_of(model, TUNING_HORIZON)
    train = training_mask(frame, TUNING_HORIZON, start, LAG)
    fit_train = training_mask(frame, fit_h, start, LAG)
    _, test = w.year_masks(frame, TUNING_HORIZON, year, lag_days=LAG)
    inner, validation, bounds = w.inner_masks(frame, TUNING_HORIZON, year, fit_h, LAG)
    assert bounds["sufficient"] and validation.any() and inner.any()
    assert not (validation & test).any(), "tuned on the test year itself"
    assert int((validation & ~train).sum()) == 0, "tuned on rows outside the training period"
    assert int((inner & ~fit_train).sum()) == 0, "inner model saw rows past the training cut"
    assert not (inner & validation).any()
    cutoff = prediction_date(pd.Timestamp(bounds["validation_start"]), LAG)
    assert (frame.loc[inner, horizon_columns(fit_h)[1]] < cutoff).all()
    assert frame.loc[validation, "repdte"].max() == frame.loc[train, "repdte"].max()
    if year == 2008:
        assert (bounds["validation_start"], bounds["validation_end"]) == (
            "2005-03-31",
            "2006-12-31",
        )
    result = w.fit_year(frame, settings, year, model, TUNING_HORIZON, save=False)
    tuning = result.config["tuning"]
    assert tuning["fallback"] is None
    assert (tuning["validation_start"], tuning["validation_end"]) == (
        bounds["validation_start"],
        bounds["validation_end"],
    )
    assert tuning["selected"] in w.candidate_grid(model, settings)
    if model == "gbdt":
        assert result.config["params"] == tuning["selected"]
    else:
        assert result.config["C"] == tuning["selected"]["C"]
