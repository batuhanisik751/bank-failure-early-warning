"""Page-table builders on synthetic frames: no database, no model artefacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankcanary.publish import PAGE_TABLES, TABLE_KEYS, pages

Q = pd.to_datetime(["2008-03-31", "2008-06-30", "2008-09-30", "2008-12-31"])
Q6 = pd.date_range("2008-03-31", periods=6, freq="QE")


def _scores(n_banks: int = 40, quarters=Q6) -> pd.DataFrame:
    """``gbdt_mono`` (+ ``hazard``) rows: banks 1 and 2 rank first, bank 7 last, every quarter."""
    rows = []
    for q in quarters:
        for model in ("gbdt_mono", "hazard"):
            certs = np.arange(1, n_banks + 1)
            rank = np.arange(1, n_banks + 1)
            rank[6] = n_banks  # bank 7
            rank[n_banks - 1] = 7
            prob = 1.0 - rank / (n_banks + 1)
            rows.append(
                pd.DataFrame(
                    {
                        "cert": certs,
                        "repdte": q.date(),
                        "model": model,
                        "score": prob,
                        "probability": prob,
                        "rank": rank,
                        "band": np.where(rank <= 2, "high", "low"),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_feature_labels_are_capitalised_first_clauses():
    labels = pages.feature_labels()
    assert labels["texas_ratio"] == "Problem assets over tangible equity plus reserves"
    assert labels["unrealized_loss_to_tier1"].startswith("Total unrealised securities")
    assert all(";" not in v and not v.endswith(".") for v in labels.values())
    assert all(not v[:1].islower() for v in labels.values()), "capitalised or a digit"
    assert len(labels) >= 80


def test_driver_keys_follow_the_subset_rule():
    scores = _scores()
    failures = pd.DataFrame({"cert": [7, 999], "fail_date": pd.to_datetime(["2009-01-09"] * 2)})
    keys = pages.driver_keys(scores, failures)
    assert len(keys) == 40 * 4 + 2 + 4, "latest 4 of every bank + bank 7 early + top 2 early"
    assert keys.duplicated().sum() == 0
    early = keys[keys["repdte"] < Q6[2]]
    assert set(early["cert"]) == {1, 2, 7}
    assert (keys.groupby("cert").size() >= 4).all()


def test_build_drivers_keeps_monotone_rows_under_one_model_and_labels_features():
    scores = _scores()
    failures = pd.DataFrame({"cert": [7], "fail_date": pd.to_datetime(["2009-01-09"])})
    rows = []
    for q, model in zip(Q6, ["gbdt_mono"] * 4 + ["gbdt_mono_production"] * 2):
        for cert in (1, 7, 20):
            shaps = {1: 0.5, 2: 0.4, 3: 0.3, 6: -0.6, 7: -0.2, 8: -0.1}
            for rank, shap in shaps.items():
                feature = "texas_ratio" if rank == 1 else f"f{rank}"
                rows.append((cert, q, model, rank, feature, shap, 1.0))
    drivers = pd.DataFrame(
        rows, columns=["cert", "repdte", "model", "rank", "feature", "shap_value", "feature_value"]
    )
    drivers["direction"] = np.where(drivers["shap_value"] > 0, "raises", "lowers")
    drivers["model_year"] = 2008
    stray = drivers.iloc[:1].assign(model="gbdt")
    out = pages.build_drivers(
        pd.concat([drivers, stray]), scores, failures, labels={"texas_ratio": "Texas ratio"}
    )
    assert list(out.columns) == list(pages.DRIVER_COLUMNS)
    assert set(out["model"]) == {"gbdt_mono"}
    assert set(out.loc[out["feature"] == "texas_ratio", "feature_label"]) == {"Texas ratio"}
    assert set(out.loc[out["feature"] == "f6", "feature_label"]) == {"f6"}
    assert (out.groupby(["cert", "repdte"]).size() == 5).all(), "five largest |shap| kept"
    assert "f8" not in set(out["feature"])
    first = out[out["rank"] == 1]
    assert set(first["feature"]) == {"f6"} and set(first["direction"]) == {"lowers"}
    assert list(out[(out["cert"] == 1) & (out["repdte"] == Q6[0].date())]["rank"]) == [
        1,
        2,
        3,
        4,
        5,
    ]
    per_bank = out.groupby("cert")["repdte"].nunique()
    assert per_bank.to_dict() == {1: 6, 7: 6, 20: 4}, "bank 20 keeps its latest 4 only"
    assert out.duplicated(list(TABLE_KEYS["drivers"])).sum() == 0


def test_map_quarters_flags_the_report_a_failure_follows():
    scores = pd.DataFrame(
        {
            "cert": [1, 1, 2, 2, 3, 4, 4, 4, 4],
            "repdte": [*Q[:2].date, *Q[:2].date, Q[0].date(), *Q.date],
            "model": "gbdt_mono",
            "band": "low",
            "probability": 0.01,
        }
    )
    hazard = scores.iloc[:1].assign(model="hazard")
    banks = pd.DataFrame(
        {"cert": [1, 2, 3, 4], "latitude": [40.0, 41.0, np.nan, 43.0], "longitude": [-70.0] * 4}
    )
    failures = pd.DataFrame(
        {"cert": [1, 2, 4], "fail_date": pd.to_datetime(["2008-08-15", "2009-02-01", "2011-01-01"])}
    )
    out = pages.build_map_quarters(pd.concat([scores, hazard]), banks, failures)
    assert list(out.columns) == [
        "repdte",
        "cert",
        "latitude",
        "longitude",
        "band",
        "probability",
        "failed_this_quarter",
    ]
    assert 3 not in set(out["cert"]), "no coordinates, no dot"
    assert len(out) == 8 and out.duplicated(["repdte", "cert"]).sum() == 0
    flagged = out.loc[out["failed_this_quarter"], ["cert", "repdte"]].apply(tuple, axis=1)
    assert set(flagged) == {(1, Q[1].date()), (2, Q[1].date())}


def test_shocked_ratios_follow_the_contract_formula():
    rows = pd.DataFrame(
        {
            "scaa": [1000.0, 1000.0],
            "scha": [500.0, np.nan],
            "scaf": [990.0, 990.0],
            "schf": [480.0, np.nan],
            "rbct1j": [200.0, 200.0],
            "asset": [10_000.0, 10_000.0],
        }
    )
    out = pages.shocked_ratios(rows, 200, 4)
    assert out["extra_loss"].iloc[0] == pytest.approx(-4 * 0.02 * 1500)
    assert out["unrealized_loss_to_tier1"].iloc[0] == pytest.approx(-150 / 200)
    assert out["adjusted_tier1_leverage"].iloc[0] == pytest.approx(50 / 10_000 * 100)
    assert out["extra_loss"].iloc[1] == pytest.approx(-80.0), "missing HTM book counts as zero"
    assert np.isnan(out["unrealized_loss_to_tier1"].iloc[1]), "missing fair value stays NaN"


def _fake_scorer(rows: pd.DataFrame) -> pd.DataFrame:
    score = -rows["unrealized_loss_to_tier1"].fillna(0.0).to_numpy(dtype=float)
    return pd.DataFrame({"cert": rows["cert"], "score": score, "score_calibrated": score / 10})


def test_build_rate_shock_grid_ranks_every_scenario():
    n = 5
    latest = pd.Timestamp("2026-06-30")
    features = pd.DataFrame(
        {
            "cert": range(1, n + 1),
            "repdte": [latest] * n,
            "unrealized_loss_to_tier1": [-0.1, -0.2, 0.0, -0.05, np.nan],
            "adjusted_tier1_leverage": 8.0,
            "other": 1.0,
        }
    )
    panel = pd.DataFrame(
        {
            "cert": range(1, n + 1),
            "repdte": [latest] * n,
            "scaa": [100.0, 200.0, 0.0, 50.0, 10.0],
            "scha": [0.0, 100.0, 0.0, 0.0, 0.0],
            "scaf": [90.0, 180.0, 0.0, 48.0, np.nan],
            "schf": [0.0, 90.0, 0.0, 0.0, 0.0],
            "rbct1j": [100.0] * n,
            "asset": [1000.0] * n,
        }
    )
    keys = features[["cert", "repdte"]].iloc[:4]
    out = pages.build_rate_shock(features, panel, _fake_scorer, keys)
    assert len(out) == 4 * 20 and set(out["cert"]) == {1, 2, 3, 4}
    assert set(out["shock_bp"]) == set(pages.SHOCKS_BP)
    assert set(out["duration_years"]) == set(pages.DURATIONS_YEARS)
    per = out.groupby(["shock_bp", "duration_years"])["rank"].agg(["min", "max"])
    assert (per["min"] == 1).all() and (per["max"] == 4).all()
    worst = out[out["cert"] == 2].set_index(["shock_bp", "duration_years"])
    assert (worst["rank"] == 1).all(), "the bank with the largest book is riskiest everywhere"
    assert worst.loc[(400, 6), "probability"] > worst.loc[(100, 2), "probability"]
    assert out.duplicated(list(TABLE_KEYS["rate_shock_scores"])).sum() == 0


def test_case_study_series_tracks_banks_against_peer_bands():
    from tests.test_case_study_2023 import make_case_frame

    frame = make_case_frame()
    window = ("2014-03-31", "2015-03-31")
    banks = {3: "Bank Three", 8: "Bank Eight"}
    out = pages.build_case_study_series(frame, banks, window, min_assets=10_000_000)
    assert list(out.columns) == list(pages.SERIES_COLUMNS)
    assert set(out["cert"]) == {3, 8}
    assert out.groupby("cert").size().to_dict() == {3: 4, 8: 5}, "bank 3 skips 2015Q1"
    peers = frame[frame["cert"].isin([3, 8]) & frame["repdte"].between(*map(pd.Timestamp, window))]
    median = peers.groupby("repdte")["unrealized_loss_to_tier1"].median()
    q95 = peers.groupby("repdte")["uninsured_share"].quantile(0.95)
    for row in out.itertuples(index=False):
        assert row.peer_p50_unrealized == pytest.approx(median[pd.Timestamp(row.repdte)])
        assert row.peer_p95_uninsured == pytest.approx(q95[pd.Timestamp(row.repdte)])
    assert out.duplicated(list(TABLE_KEYS["case_study_series"])).sum() == 0


def test_build_all_names_missing_core_frames_and_respects_publish_order():
    assert pages.core_dependencies(["drivers", "map_quarters"]) == {"scores", "banks", "failures"}
    assert pages.core_dependencies(["case_study_2023"]) == set()
    assert [t for t in TABLE_KEYS if t in PAGE_TABLES] == list(PAGE_TABLES)
    with pytest.raises(KeyError, match="scores"):
        pages.build_all(None, {}, None, ["drivers"])
