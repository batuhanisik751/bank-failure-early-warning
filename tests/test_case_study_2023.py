"""The 2023 case-study fits on a synthetic v2 frame: no data files, no network."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bankcanary import tracking
from bankcanary.evaluation import case_study_2023 as cs
from bankcanary.features.registry import feature_names
from bankcanary.labels.build import horizon_columns
from bankcanary.splits import prediction_date
from tests.test_hazard import LAG, make_frame, make_settings

CUT = "2014-12-31"
QUARTERS = ("2014-09-30", "2014-12-31", "2015-03-31")
BANKS = {3: "Bank Three", 8: "Bank Eight"}


def make_case_frame(n_certs: int = 40, seed: int = 0) -> pd.DataFrame:
    """The hazard test frame plus ``name`` and ``asset`` (bank 3 and 8 are the big ones)."""
    frame = make_frame(n_certs, seed)
    asset = np.where(frame["cert"].isin([3, 8]), 50_000_000.0, 500_000.0)
    frame = frame.assign(name="BANK " + frame["cert"].astype(str), asset=asset)
    # Drop bank 3's last quarter so one tracked bank has no report there, like SVB in 2023Q1.
    keep = ~(frame["cert"].eq(3) & frame["repdte"].eq(pd.Timestamp("2015-03-31")))
    return frame.loc[keep].reset_index(drop=True)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_case_frame()


@pytest.fixture(scope="module")
def fits(frame, tmp_path_factory) -> tuple:
    settings = make_settings(tmp_path_factory.mktemp("case"))
    out = [
        cs.fit_view(frame, settings, view, model, CUT, QUARTERS, banks=BANKS)
        for view in cs.VIEWS
        for model in cs.MODELS
    ]
    return settings, out


def test_views_see_the_right_columns_and_reject_unknown_names():
    assert cs.view_features("credit_only") == feature_names(version="v1")
    assert cs.view_features("rate_aware") == feature_names(version="v2")
    assert "unrealized_loss_to_tier1" not in cs.view_features("credit_only")
    assert "unrealized_loss_to_tier1" in cs.view_features("rate_aware")
    with pytest.raises(ValueError, match="unknown view"):
        cs.view_features("v3")
    with pytest.raises(ValueError, match="unknown model"):
        cs.build_model("hazard", "rate_aware", None)


def test_training_rows_close_before_the_cut_and_scoring_rows_are_the_quarters(frame, tmp_path):
    mask = cs.training_rows(frame, make_settings(tmp_path), CUT)
    end = horizon_columns(4)[1]
    assert mask.any()
    assert (frame.loc[mask, end] < prediction_date(CUT, LAG)).all()
    assert frame.loc[mask, "repdte"].max() == pd.Timestamp("2013-09-30")
    scored = cs.scoring_rows(frame, QUARTERS)
    assert set(frame.loc[scored, "repdte"].dt.strftime("%Y-%m-%d")) == set(QUARTERS)
    assert not (mask & scored).any()


def test_rank_scores_puts_the_highest_score_first_within_each_quarter():
    scores = pd.DataFrame(
        {
            "cert": [1, 2, 3, 1, 2, 3],
            "repdte": pd.to_datetime(["2014-09-30"] * 3 + ["2014-12-31"] * 3),
            "score": [0.1, 0.9, 0.5, 0.7, 0.7, 0.2],
        }
    )
    ranked = cs.rank_scores(scores)
    q3 = ranked.loc[ranked["repdte"].eq("2014-09-30")].set_index("cert")
    assert q3.loc[2, "rank"] == 1 and q3.loc[3, "rank"] == 2 and q3.loc[1, "rank"] == 3
    assert q3.loc[2, "percentile"] == pytest.approx(100 * 2 / 3)
    assert (ranked["n_scored"] == 3).all()
    q4 = ranked.loc[ranked["repdte"].eq("2014-12-31")].set_index("cert")
    assert q4.loc[1, "rank"] == q4.loc[2, "rank"] == 1 and q4.loc[3, "rank"] == 3


def test_fits_use_their_view_columns_and_log_deterministic_runs(fits):
    settings, out = fits
    assert [(f.view, f.model) for f in out] == [
        ("credit_only", "logit"),
        ("credit_only", "gbdt_mono"),
        ("rate_aware", "logit"),
        ("rate_aware", "gbdt_mono"),
    ]
    for fit in out:
        assert fit.features == cs.view_features(fit.view)
        assert fit.config["n_features"] == len(fit.features)
        assert fit.config["train_repdte_max"] == "2013-09-30"
        assert fit.config["features_version"] == cs.VIEWS[fit.view]
        assert fit.run_dir is not None and fit.run_dir.name.startswith("case_study_2023-4q-")
        assert fit.run_dir == settings.runs_dir / cs.RUN_NAME / tracking.run_id(
            cs.RUN_NAME, fit.config
        )
        metrics = json.loads((fit.run_dir / "metrics.json").read_text())
        assert metrics["bank_eight_2015Q1_rank"] >= 1
        assert "bank_three_2015Q1_rank" not in metrics  # no report that quarter
        assert set(fit.scores["repdte"].dt.strftime("%Y-%m-%d")) == set(QUARTERS)
        assert fit.scores["score"].between(0, 1).all()
    assert "C" in out[0].config and "params" in out[1].config
    index = tracking.read_index(settings)
    assert len(index) == 4 and len({r["run_id"] for r in index}) == 4


def test_bank_table_has_one_row_per_bank_view_model_quarter(fits):
    _, out = fits
    table = cs.bank_table(out, BANKS)
    assert list(table.columns) == [
        "bank",
        "cert",
        "view",
        "model",
        "quarter",
        "probability",
        "rank",
        "percentile",
        "n_scored",
    ]
    assert len(table) == 4 * (3 + 2)  # bank 8 in three quarters, bank 3 in two
    assert set(table["quarter"]) == {"2014Q3", "2014Q4", "2015Q1"}
    assert table.loc[table["bank"].eq("Bank Three"), "quarter"].ne("2015Q1").all()
    assert (table["rank"] >= 1).all() and (table["rank"] <= table["n_scored"]).all()
    assert table["percentile"].between(0, 100).all()


def test_drivers_sum_to_the_log_odds_for_both_learners(fits, frame):
    _, out = fits
    when = pd.Timestamp("2014-12-31")
    for fit in out:
        table = cs.drivers(fit, frame, 8, when, top=5)
        assert list(table.columns) == ["feature", "value", "contribution", "direction"]
        assert len(table) == 5
        assert table["contribution"].abs().is_monotonic_decreasing
        assert (table.loc[table["contribution"] > 0, "direction"] == "riskier").all()
        row = frame.loc[frame["cert"].eq(8) & frame["repdte"].eq(when), fit.features]
        p = float(fit.pipeline.predict_proba(row)[0, 1])
        log_odds = np.log(p / (1 - p))
        assert table.attrs["baseline"] + table.attrs["total"] == pytest.approx(log_odds, abs=1e-4)
    with pytest.raises(ValueError, match="expected one"):
        cs.drivers(out[0], frame, 3, "2015-03-31")


def test_peer_bands_use_only_large_banks_in_the_window(frame):
    band = cs.peer_bands(frame, "uninsured_share", 10_000_000, ("2014-03-31", "2014-12-31"))
    assert list(band.columns) == [0.05, 0.5, 0.95]
    assert len(band) == 4
    big = frame.loc[frame["cert"].isin([3, 8]) & frame["repdte"].eq("2014-06-30")]
    assert band.loc["2014-06-30", 0.5] == pytest.approx(big["uninsured_share"].median())


def test_run_case_study_writes_the_report(frame, tmp_path):
    settings = make_settings(tmp_path)
    study = cs.run_case_study(
        frame, settings, CUT, QUARTERS, BANKS, driver_cert=8, driver_repdte="2014-12-31"
    )
    assert len(study.fits) == 4 and len(study.drivers) == 4
    path = cs.write_report(study, settings)
    text = path.read_text()
    assert path == settings.reports_dir / "svb_2023_case_study.md"
    assert "Bank Eight" in text and "## rate_aware / gbdt_mono" in text
    assert "## Drivers, credit_only / logit" in text
    assert "2013-09-30" in text
    missing = cs.run_case_study(frame, settings, CUT, QUARTERS, BANKS, driver_cert=999)
    assert missing.drivers == {}
