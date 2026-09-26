"""Adversarial tests for ``bankcanary publish`` and ``refresh`` (CONTRACT 15-17, 19).

Every expected value is derived by hand from the CONTRACT text and written into the
test; the builders run on synthetic frames. The two database tests use the same
skip-if-unreachable rule as ``tests/test_publish_db.py`` and a scratch schema.
Educational project, not a credit rating, not investment advice, not a supervisory
assessment; FDIC insurance covers $250,000 per depositor, per bank, per ownership category.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from bankcanary.publish import TABLE_KEYS, core, db, pages, writer

Q = pd.date_range("2008-03-31", periods=8, freq="QE")  # 2008Q1 .. 2009Q4
D = [q.date() for q in Q]


def _scores_frame(certs, repdte, model, score, prob=None):
    n = len(certs)
    return pd.DataFrame(
        {
            "cert": list(certs),
            "repdte": [repdte] * n,
            "model": [model] * n,
            "score": list(score),
            "probability": list(score if prob is None else prob),
        }
    )


# --- rank, percentile, band ----------------------------------------------------------


def test_rank_percentile_band_within_quarter_and_model_ties_by_cert_shuffle_invariant():
    """n = 51: high = ceil(1.02) = 2, elevated = ceil(5.1) = 6, low = 45; n = 4: 1, 1, 3."""
    big = _scores_frame(range(100, 151), D[0], "gbdt_mono", np.linspace(0.9, 0.1, 51))
    big.loc[big["cert"].isin([120, 110]), "score"] = 0.95  # tie above everyone
    small = _scores_frame([1, 2, 3, 4], D[0], "gbdt_mono", [0.5, 0.5, 0.5, 0.5])
    small["repdte"] = D[1]
    hazard = _scores_frame([1, 2, 3], D[0], "hazard", [0.1, 0.9, 0.5])
    frame = pd.concat([big, small, hazard], ignore_index=True)
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a, b = core.rank_scores(frame), core.rank_scores(shuffled)
    pd.testing.assert_frame_equal(a, b)
    big_out = a[(a["repdte"] == D[0]) & (a["model"] == "gbdt_mono")]
    assert list(big_out["cert"][:2]) == [110, 120], "tie at 0.95 breaks by cert ascending"
    assert big_out["band"].value_counts().to_dict() == {"low": 45, "elevated": 4, "high": 2}
    assert big_out.set_index("cert").loc[110, "percentile"] == pytest.approx(100 * 50 / 51)
    assert big_out.set_index("cert").loc[120, "percentile"] == pytest.approx(100 * 49 / 51)
    small_out = a[a["repdte"] == D[1]]
    assert list(small_out["rank"]) == [1, 2, 3, 4] and list(small_out["cert"]) == [1, 2, 3, 4]
    assert list(small_out["band"]) == ["high", "low", "low", "low"]
    assert list(small_out["percentile"]) == pytest.approx([75.0, 50.0, 25.0, 0.0])
    hz = a[a["model"] == "hazard"].set_index("cert")
    assert list(hz["rank"]) == [1, 2, 3] and list(hz.index) == [2, 3, 1]
    assert hz.loc[2, "band"] == "high", "the hazard column is ranked on its own"


def test_prior_quarter_delta_uses_exact_previous_quarter_end_only():
    """Only (cert, model) at repdte - 1 quarter end counts; year ends chain to Q1."""
    frame = pd.DataFrame(
        {
            "cert": [1, 1, 1, 2, 1, 2],
            "repdte": [Q[3], Q[4], Q[6], Q[4], Q[4], Q[5]],
            "model": ["gbdt_mono", "gbdt_mono", "gbdt_mono", "gbdt_mono", "hazard", "hazard"],
            "probability": [0.10, 0.40, 0.70, 0.20, 0.90, 0.95],
        }
    )
    delta = core.prior_quarter_delta(frame)
    assert delta[1] == pytest.approx(0.30), "2008-12-31 -> 2009-03-31 crosses the year end"
    assert np.isnan(delta[0]) and np.isnan(delta[2]), "no row at the exact prior quarter end"
    assert np.isnan(delta[3]), "bank 1's prior row must not leak into bank 2"
    assert np.isnan(delta[4]), "gbdt_mono's 2008Q4 row must not feed the hazard delta"
    assert np.isnan(delta[5]), "cert 1 hazard 2009Q1 is not cert 2's prior"


# --- traceability: model_version and model_year --------------------------------------


def _walkforward(years=(2008, 2009), n=60, seed=0):
    """Horizon-4 ``gbdt_mono`` + ``hazard`` rows: one row per (cert, quarter, model)."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for q in Q[Q.year == year]:
            for model in ("gbdt_mono", "hazard"):
                score = rng.uniform(size=n).round(2)  # rounding forces ties
                for cert in range(1, n + 1):
                    rows.append(
                        {
                            "cert": cert,
                            "repdte": q,
                            "horizon": 4,
                            "model": model,
                            "test_year": year,
                            "score": float(score[cert - 1]),
                            "score_calibrated": float(score[cert - 1] / 2),
                            "y": int(cert % 17 == 0 and q.quarter in (1, 3)),
                        }
                    )
    return pd.DataFrame(rows)


VERSIONS = {
    ("gbdt_mono", 2008): "gbdt_mono-2006-12-31-aaa",
    ("gbdt_mono", 2009): "gbdt_mono-2007-12-31-aaa",
    ("hazard", 2008): "hazard-2007-12-31-bbb",
    ("hazard", 2009): "hazard-2008-09-30-bbb",
    ("gbdt_mono", "production"): "gbdt_mono-2007-12-31-aaa",
    ("hazard", "production"): "hazard-2008-09-30-bbb",
}


def test_every_scores_row_carries_its_own_test_year_version_and_production_past_it():
    wf = _walkforward()
    production = pd.DataFrame(
        {
            "cert": [1, 2, 1, 2],
            "repdte": [pd.Timestamp("2010-03-31")] * 2 + [pd.Timestamp("2010-06-30")] * 2,
            "horizon": 4,
            "model": ["gbdt_mono", "gbdt_mono", "hazard", "hazard"],
            "test_year": pd.array([None] * 4, dtype="Int64"),
            "score": [0.3, 0.6, 0.2, 0.1],
            "score_calibrated": [0.15, 0.3, 0.1, 0.05],
        }
    )
    scores = core.build_scores(wf, production, VERSIONS)
    assert scores["model_version"].notna().all(), "CONTRACT 15: every number carries a version"
    year = pd.to_datetime(scores["repdte"]).dt.year
    for (model, y), version in VERSIONS.items():
        if y == "production":
            continue
        got = set(scores.loc[(scores["model"] == model) & (year == y), "model_version"])
        assert got == {version}, (model, y, got)
    prod = scores[year == 2010]
    assert len(prod) == 4 and set(prod["model_version"]) == {
        VERSIONS[("gbdt_mono", "production")],
        VERSIONS[("hazard", "production")],
    }
    assert prod.groupby(["repdte", "model"])["rank"].apply(lambda r: sorted(r) == [1, 2]).all()
    assert not scores.duplicated(list(TABLE_KEYS["scores"])).any()


def test_quarters_model_year_matches_the_scores_that_quarter_holds():
    panel = pd.DataFrame(
        {
            "cert": [1, 2, 1, 2, 1, 2, 1],
            "repdte": [Q[0], Q[0], Q[4], Q[4], pd.Timestamp("2010-03-31")] * 1
            + [pd.Timestamp("2010-03-31"), pd.Timestamp("2007-12-31")],
            "avail_date": pd.Timestamp("2000-01-01"),
        }
    )
    labels = panel.assign(y_4q=[1, 0, 0, 0, 0, 0, 1], label_complete_4q=[True] * 6 + [False])
    by_year = {y: v for (m, y), v in VERSIONS.items() if m == "gbdt_mono"}
    out = core.build_quarters(panel, labels, by_year).set_index("repdte")
    assert out.loc[D[0], "model_year"] == 2008 and out.loc[D[4], "model_year"] == 2009
    assert out.loc[D[0], "model_version"] == VERSIONS[("gbdt_mono", 2008)]
    assert out.loc[D[4], "model_version"] == VERSIONS[("gbdt_mono", 2009)]
    prod = out.loc[pd.Timestamp("2010-03-31").date()]
    assert pd.isna(prod["model_year"]) and prod["model_version"] == "gbdt_mono-2007-12-31-aaa"
    pre = out.loc[pd.Timestamp("2007-12-31").date()]
    assert pd.isna(pre["model_year"]) and pd.isna(pre["model_version"])
    assert out.loc[D[0], "n_failures_next_4q"] == 1 and out.loc[D[0], "n_banks"] == 2
    assert bool(pre["label_complete_4q"]) is False and out.loc[D[0], "label"] == "2008Q1"


# --- time machine: pooled recall@2% of a year equals walkforward_metrics --------------


def test_time_machine_2009_pooled_recall_equals_walkforward_metrics_by_hand():
    """CONTRACT 19: order the year's ``scores`` rows by raw score desc, cert asc, take
    ceil(2% of n) and count the failures found; must equal ``recall_at_2pct``."""
    wf = _walkforward()
    scores = core.build_scores(wf, None, VERSIONS)
    metrics = core.build_walkforward_metrics(wf).set_index(["model", "test_year"])
    year = pd.to_datetime(scores["repdte"]).dt.year
    rows = scores[(scores["model"] == "gbdt_mono") & (year == 2009)].copy()
    labels = wf[(wf["model"] == "gbdt_mono") & (wf["test_year"] == 2009)][["cert", "repdte", "y"]]
    labels["repdte"] = labels["repdte"].dt.date
    rows = rows.merge(labels, on=["cert", "repdte"], how="left", validate="one_to_one")
    assert rows["y"].notna().all() and len(rows) == 240 == metrics.loc[("gbdt_mono", 2009), "n"]
    pooled = rows.sort_values(["score", "cert"], ascending=[False, True], kind="mergesort")
    head = math.ceil(0.02 * len(pooled))  # 5 of 240
    found = pooled["y"].to_numpy()[:head].sum()
    expected = found / pooled["y"].sum()
    assert head == 5 and pooled["y"].sum() == 6 == metrics.loc[("gbdt_mono", 2009), "n_failures"]
    assert metrics.loc[("gbdt_mono", 2009), "recall_at_2pct"] == pytest.approx(expected, abs=1e-12)
    assert set(rows["model_version"]) == {VERSIONS[("gbdt_mono", 2009)]}
    assert rows.groupby("repdte")["rank"].max().eq(60).all(), "per-quarter ranks stay 1..60"
    pooled_row = metrics.loc[("gbdt_mono", 0)]
    assert pooled_row["n"] == 480 and pooled_row["n_failures"] == 12


def test_publish_names_the_walkforward_version_of_every_backtest_quarter(tmp_path):
    """CONTRACT 15/19: backtest quarters carry the walk-forward model of their test
    year; ``build_all`` must not fall back to the production version silently."""
    from bankcanary.config import load_settings

    models = tmp_path / "models" / "production"
    for model in core.MODELS:
        (models / model).mkdir(parents=True)
        meta = {"model_version": f"{model}-2007-12-31-prod", "model": model,
                "train_end_repdte": "2007-12-31", "git_sha": "prod", "trained_at": None,
                "features_version": "v2", "notes": ""}  # fmt: skip
        (models / model / "model_version.json").write_text(json.dumps(meta))
    settings = load_settings().model_copy(
        update={
            "models_dir": tmp_path / "models",
            "runs_dir": tmp_path / "runs",
            "data_dir": tmp_path / "data",
        }  # fmt: skip
    )
    wf = _walkforward(years=(2008,))
    labels = wf[wf["model"] == "gbdt_mono"][["cert", "repdte", "y"]].rename(columns={"y": "y_4q"})
    labels = labels.assign(label_complete_4q=True, dropped_failed_before_avail=False)
    later = labels[labels["repdte"] == Q[0]].assign(repdte=pd.Timestamp("2009-03-31"))
    labels = pd.concat([labels, later.assign(label_complete_4q=False)], ignore_index=True)
    panel = labels[["cert", "repdte"]].assign(avail_date=pd.Timestamp("2000-01-01"))
    production = wf[(wf["repdte"] == Q[0])].assign(
        repdte=pd.Timestamp("2009-03-31"), test_year=pd.array([None] * 120, dtype="Int64")
    )
    wh = core.Warehouse(pd.DataFrame(), panel, labels, pd.DataFrame(), pd.DataFrame(), wf)
    wh.production = production
    out = core.build_all(wh, settings, ["model_versions", "quarters", "scores"], repo=tmp_path)
    scores, quarters = out["scores"], out["quarters"].set_index("repdte")
    assert pd.to_datetime(scores["repdte"]).dt.year.isin([2008, 2009]).all()
    assert set(quarters.loc[[pd.Timestamp("2009-03-31").date()], "model_version"]) == {
        "gbdt_mono-2007-12-31-prod"
    }
    back = scores[pd.to_datetime(scores["repdte"]).dt.year == 2008]
    assert list(quarters.loc[D[:4], "model_year"]) == [2008] * 4, "2008 is a walk-forward year"
    assert back["model_version"].notna().all(), "backtest rows published without a version"
    assert not quarters.loc[D[:4], "model_version"].eq("gbdt_mono-2007-12-31-prod").any()


# --- drivers: subset rule and the five-largest rule -----------------------------------


def _ranked_scores(n_banks: int, quarters, model="gbdt_mono", certs=None) -> pd.DataFrame:
    """Bank ``c`` ranks ``c`` in every quarter (cert 1 riskiest)."""
    certs = list(range(1, n_banks + 1)) if certs is None else list(certs)
    parts = []
    for q in quarters:
        s = _scores_frame(certs, q.date(), model, [1.0 - c / 1000 for c in certs])
        parts.append(s)
    return core.rank_scores(pd.concat(parts, ignore_index=True))


def test_driver_keys_apply_the_subset_rule_exactly_by_hand():
    """60 banks, 6 quarters: top 5% = ceil(0.05 n) by gbdt_mono rank; bank 61 exited
    after 2008Q4 (its own latest 4 = 2008Q1-Q4); bank 40 failed; hazard ranks ignored."""
    live = _ranked_scores(60, Q[:6])
    exited = _ranked_scores(1, Q[:4], certs=[61])  # rank 1 in a one-bank frame, fixed below
    exited["rank"], exited["band"] = 61, "low"
    hazard = _ranked_scores(60, Q[:6], model="hazard")
    hazard["rank"] = 61 - hazard["rank"]  # bank 60 tops the hazard ranking
    scores = pd.concat([live, exited, hazard], ignore_index=True)
    failures = pd.DataFrame({"cert": [40, 9999], "fail_date": pd.to_datetime(["2009-08-01"] * 2)})
    keys = pages.driver_keys(scores, failures)
    keys["repdte"] = keys["repdte"].dt.date
    got = set(map(tuple, keys.to_numpy()))
    expected = {(c, q.date()) for c in range(1, 61) for q in Q[2:6]}  # latest 4 of each
    expected |= {(61, q.date()) for q in Q[:4]}  # the exited bank's own latest 4
    expected |= {(40, q.date()) for q in Q[:6]}  # every quarter of the failed bank
    expected |= {(c, q.date()) for c in (1, 2, 3) for q in Q[:6]}  # top 5% of 60 = 3
    expected |= {(4, q.date()) for q in Q[:4]}  # 61 banks while 61 files: ceil(3.05) = 4
    assert got == expected
    assert (60, Q[0].date()) not in got, "hazard's top bank is not a driver key"


def test_build_drivers_keeps_five_largest_absolute_contributions_and_reranks():
    scores = _ranked_scores(3, Q[:1])
    shaps = {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.05, 5: 0.01, 6: -0.6, 7: -0.35, 8: -0.2, 9: -0.1}
    rows = [(1, Q[0], "gbdt_mono", r, f"f{r}", v, float(r)) for r, v in shaps.items()]
    rows += [
        (2, Q[0], "gbdt_mono_production", r, f"f{r}", v, 1.0) for r, v in list(shaps.items())[:3]
    ]
    rows += [(3, Q[0], "gbdt", 1, "f1", 9.0, 1.0)]  # a different model: never published
    drivers = pd.DataFrame(
        rows, columns=["cert", "repdte", "model", "rank", "feature", "shap_value", "feature_value"]
    )
    drivers["direction"] = np.where(drivers["shap_value"] > 0, "raises", "lowers")
    out = pages.build_drivers(drivers, scores, pd.DataFrame({"cert": [], "fail_date": []}), {})
    one = out[out["cert"] == 1]
    assert list(one["rank"]) == [1, 2, 3, 4, 5]
    assert list(one["feature"]) == ["f6", "f1", "f2", "f7", "f3"], "by |shap|, sign kept"
    assert list(one["direction"]) == ["lowers", "raises", "raises", "lowers", "raises"]
    assert list(one["feature_value"]) == [6.0, 1.0, 2.0, 7.0, 3.0], "values travel with rows"
    assert list(one["shap_value"]) == [-0.6, 0.5, 0.4, -0.35, 0.3]
    two = out[out["cert"] == 2]
    assert len(two) == 3 and set(two["model"]) == {"gbdt_mono"}, "production rows relabelled"
    assert 3 not in set(out["cert"]) and list(out["feature_label"]) == list(out["feature"])
    assert not out.duplicated(list(TABLE_KEYS["drivers"])).any()


# --- rate shock, map, money units -----------------------------------------------------


def _calibrated_scorer(rows: pd.DataFrame) -> pd.DataFrame:
    """Raw score = -unrealised ratio; calibrated = a monotone map of it (never equal)."""
    raw = -rows["unrealized_loss_to_tier1"].fillna(0.0).to_numpy(dtype=float)
    return pd.DataFrame({"cert": rows["cert"], "score": raw, "score_calibrated": 0.5 + raw / 4})


def test_rate_shock_grid_formula_sign_units_and_per_scenario_ranks_by_hand():
    latest = pd.Timestamp("2026-06-30")
    features = pd.DataFrame(
        {
            "cert": [1, 2, 3],
            "repdte": [latest] * 3,
            "unrealized_loss_to_tier1": [0.0, 0.0, 0.0],
            "adjusted_tier1_leverage": [10.0, 10.0, 10.0],
        }
    )
    panel = pd.DataFrame(  # thousands of dollars, as in the warehouse
        {
            "cert": [1, 2, 3],
            "repdte": [latest] * 3,
            "scaa": [4_000.0, 1_000.0, np.nan],  # AFS amortised cost
            "scha": [1_000.0, np.nan, np.nan],  # HTM amortised cost
            "scaf": [4_100.0, 1_000.0, np.nan],  # AFS fair value: a 100 gain
            "schf": [1_000.0, np.nan, np.nan],
            "rbct1j": [500.0, 500.0, 500.0],
            "asset": [10_000.0, 10_000.0, 10_000.0],
        }
    )
    out = pages.build_rate_shock(features, panel, _calibrated_scorer, features[["cert", "repdte"]])
    assert len(out) == 3 * 20 and not out.duplicated(list(TABLE_KEYS["rate_shock_scores"])).any()
    grid = {(s, d) for s in (100, 200, 300, 400) for d in (2, 3, 4, 5, 6)}
    assert set(map(tuple, out[["shock_bp", "duration_years"]].to_numpy())) == grid
    row = out.set_index(["cert", "shock_bp", "duration_years"])
    # bank 1, 300 bp x 4 y: loss = -4 x 0.03 x 5000 = -600 (thousands); total = +100 - 600
    assert row.loc[(1, 300, 4), "extra_loss"] == pytest.approx(-600.0)
    assert row.loc[(1, 300, 4), "unrealized_loss_to_tier1"] == pytest.approx(-500 / 500)
    assert row.loc[(1, 300, 4), "adjusted_tier1_leverage"] == pytest.approx(0.0)
    # bank 1, 100 bp x 2 y: loss -100 exactly offsets the gain: no leverage haircut
    assert row.loc[(1, 100, 2), "unrealized_loss_to_tier1"] == pytest.approx(0.0)
    assert row.loc[(1, 100, 2), "adjusted_tier1_leverage"] == pytest.approx(5.0)
    # bank 2, missing HTM book counts as zero cost: -2 x 0.01 x 1000 = -20
    assert row.loc[(2, 100, 2), "extra_loss"] == pytest.approx(-20.0)
    assert np.isnan(row.loc[(2, 100, 2), "unrealized_loss_to_tier1"]), "as the feature: NaN book"
    assert (out["extra_loss"] <= 0).all(), "a rate rise never adds value"
    assert row.loc[(3, 400, 6), "extra_loss"] == 0.0 and np.isfinite(
        row.loc[(3, 400, 6), "probability"]
    )
    for s, d in grid:  # rank 1 = riskiest; probability is the calibrated value, not raw
        part = out[(out["shock_bp"] == s) & (out["duration_years"] == d)].set_index("cert")
        assert list(part.sort_values("rank").index) == [1, 2, 3], (s, d)
        assert part.loc[1, "probability"] == pytest.approx(
            0.5 - part.loc[1, "unrealized_loss_to_tier1"] / 4
        )
        assert part.loc[1, "band"] == "high" and set(part["band"]) <= {"high", "elevated", "low"}


def test_map_rows_only_for_banks_with_both_coordinates_and_failure_flag_edges():
    scores = pd.concat(
        [
            _scores_frame([1, 2, 3, 4, 5], q.date(), "gbdt_mono", [0.5] * 5).assign(band="low")
            for q in Q[:4]
        ],
        ignore_index=True,
    )
    scores = pd.concat([scores, scores.iloc[:5].assign(model="hazard")], ignore_index=True)
    banks = pd.DataFrame(
        {
            "cert": [1, 2, 3, 4, 5],
            "latitude": [40.0, np.nan, 42.0, 43.0, 44.0],
            "longitude": [-70.0, -71.0, np.nan, -73.0, -74.0],
        }
    )
    failures = pd.DataFrame(
        {
            "cert": [1, 4, 5, 5],
            "fail_date": pd.to_datetime(["2008-06-30", "2008-09-30", "2009-11-30", "2013-01-01"]),
        }
    )
    out = pages.build_map_quarters(scores, banks, failures)
    assert set(out["cert"]) == {1, 4, 5} and len(out) == 12, "a missing lat or lon drops the bank"
    assert out.duplicated(["repdte", "cert"]).sum() == 0
    flagged = {
        (c, d) for c, d in out.loc[out["failed_this_quarter"], ["cert", "repdte"]].to_numpy()
    }
    # cert 1 fails on the next quarter end itself: 2008Q1 flagged, 2008Q2 (fail == repdte) not
    # cert 4 fails 2008-09-30: 2008Q2 flagged only; cert 5 fails 2009-11-30, within 4 quarters
    # of its last report 2008Q4 (2009-12-31): flagged once, on that last report
    assert flagged == {(1, D[0]), (4, D[1]), (5, D[3])}


def test_money_units_are_thousands_of_dollars_in_every_table():
    """$250 million = 250_000 in the warehouse; the same number lands in ``banks`` and
    ``ratios`` and buckets as ``100m_1b``; the peer thresholds equal the settings."""
    from bankcanary.config import load_settings
    from bankcanary.evaluation.case_study_2023 import PEER_MIN_ASSETS

    settings = load_settings()
    assert list(core.SIZE_THRESHOLDS) == settings.peer_asset_buckets_thousands
    assert PEER_MIN_ASSETS == 10_000_000, "$10 billion in thousands"
    panel = pd.DataFrame(
        {
            "cert": [7, 7, 8],
            "repdte": [Q[0], Q[1], Q[1]],
            "asset": [200_000.0, 250_000.0, 99_999.0],
            "stalp": ["CA", "CA", "CA"],
            "exit_reason": [None, None, None],
            "fail_date": [pd.NaT, pd.NaT, pd.NaT],
        }
    )
    institutions = pd.DataFrame(
        {
            "cert": [7, 8],
            "name": ["Seven", "Eight"],
            "city": ["A", "B"],
            "stalp": ["CA", "CA"],
            "bkclass": ["N", "SM"],
            "estymd": ["1990-01-01", None],
            "endefymd": [None, None],
            "active": [True, True],
            "fed_rssd": [1, 2],
            "rssdhcr": [None, None],
            "latitude": [1.0, 2.0],
            "longitude": [3.0, 4.0],
        }  # fmt: skip
    )
    banks = core.build_banks(institutions, panel).set_index("cert")
    assert banks.loc[7, "latest_assets"] == 250_000.0, "the latest quarter, in thousands"
    assert banks.loc[7, "size_bucket"] == "100m_1b" and banks.loc[8, "size_bucket"] == "under_100m"
    features = panel[["cert", "repdte"]].copy()
    for ratio in core.RATIOS:
        features[ratio] = [0.1, 0.2, 0.3]
    ratios = core.build_ratios(features, panel).set_index(["cert", "repdte"])
    assert ratios.loc[(7, D[1]), "total_assets"] == 250_000.0 == banks.loc[7, "latest_assets"]
    assert (
        ratios.loc[(7, D[1]), "size_bucket"] == "100m_1b"
        and ratios.loc[(7, D[1]), "region"] == "west"
    )
    failures = core.build_failures(
        pd.DataFrame(
            {
                "cert": [7.0],
                "fail_date": ["2010-01-08"],
                "name": ["Seven"],
                "cityst": ["Los Angeles, CA"],
                "pstalp": ["CA"],
                "restype1": ["PA"],
                "cost": [12_345.0],
                "qbfasset": [250_000.0],
                "qbfdep": [200_000.0],
            }
        )  # fmt: skip
    )
    assert failures.loc[0, "qbfasset"] == 250_000.0 and failures.loc[0, "city"] == "Los Angeles"


def test_ratio_peer_percentiles_are_within_size_bucket_region_quarter_by_hand():
    """Four peers with values 1, 2, 3, NaN: percentiles 100(r - 0.5)/3 = 16.7, 50, 83.3,
    NaN; a lone bank in another region sits at 50; ``peer_stats`` counts the non-null."""
    panel = pd.DataFrame(
        {
            "cert": [1, 2, 3, 4, 5, 6],
            "repdte": [Q[0]] * 6,
            "asset": [500_000.0] * 5 + [50_000_000.0],
            "stalp": ["CA", "CA", "CA", "CA", "NY", "CA"],
        }
    )
    features = panel[["cert", "repdte"]].copy()
    for ratio in core.RATIOS:
        features[ratio] = [1.0, 2.0, 3.0, np.nan, 9.0, 9.0]
    ratios = core.build_ratios(features, panel).set_index("cert")
    pct = ratios["texas_ratio_pct"]
    assert list(pct.loc[[1, 2, 3]]) == pytest.approx([100 / 6, 50.0, 250 / 3])
    assert np.isnan(pct.loc[4]) and pct.loc[5] == 50.0 and pct.loc[6] == 50.0
    stats = core.build_peer_stats(ratios.reset_index()).set_index(
        ["size_bucket", "region", "ratio"]
    )
    west = stats.loc[("100m_1b", "west", "texas_ratio")]
    assert west["n"] == 3 and west["p50"] == 2.0 and west["p10"] == pytest.approx(1.2)
    assert stats.loc[("10b_100b", "west", "texas_ratio"), "n"] == 1
    assert len(stats) == 3 * len(core.RATIOS)


# --- idempotency against a scratch schema, refresh no-op ------------------------------


def _url() -> str | None:
    try:
        return db.database_url(os.environ.get("DATABASE_URL"))
    except RuntimeError:
        return None


URL = _url()
needs_db = pytest.mark.skipif(
    URL is None or not db.reachable(URL), reason="DATABASE_URL unset or database not reachable"
)


@pytest.fixture
def scratch():
    schema = f"publish_adv_{os.getpid()}"
    with db.connect(URL, autocommit=True) as conn:
        try:
            yield conn, schema
        finally:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def _checksums(conn, schema: str, tables) -> dict[str, tuple[int, str]]:
    out = {}
    for table in tables:
        key = ", ".join(f'"{k}"' for k in TABLE_KEYS[table])
        row = conn.execute(
            f"SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY {key}), '')) "
            f'FROM "{schema}"."{table}" t'
        ).fetchone()
        out[table] = (int(row[0]), row[1])
    return out


@needs_db
def test_publishing_twice_gives_identical_row_counts_and_checksums(scratch):
    conn, schema = scratch
    writer.apply_schema(conn, schema)
    wf = _walkforward()
    frames = {
        "scores": core.build_scores(wf, None, VERSIONS),
        "walkforward_metrics": core.build_walkforward_metrics(wf),
    }
    frames["scores"].loc[0, "score"] = np.nan  # a NULL survives the round trip
    tables = list(frames)
    first, second, third = {}, {}, {}
    for name, frame in frames.items():
        writer.write_table(conn, name, frame.sample(frac=1.0, random_state=1), "replace", schema)
    first = _checksums(conn, schema, tables)
    for name, frame in frames.items():
        writer.write_table(conn, name, frame, "replace", schema)
    second = _checksums(conn, schema, tables)
    for name, frame in frames.items():
        writer.write_table(conn, name, frame.sample(frac=1.0, random_state=2), "upsert", schema)
    third = _checksums(conn, schema, tables)
    assert first == second == third
    assert first["scores"][0] == 960 and first["walkforward_metrics"][0] == 6
    nulls = conn.execute(f'SELECT count(*) FROM "{schema}".scores WHERE score IS NULL').fetchone()
    assert nulls[0] == 1
    ranked = conn.execute(
        f'SELECT repdte, model, min(rank), max(rank), count(*) FROM "{schema}".scores GROUP BY 1, 2'
    ).fetchall()
    assert all(lo == 1 and hi == n == 60 for _, _, lo, hi, n in ranked)


def test_refresh_without_a_new_quarter_writes_only_a_pipeline_runs_row(tmp_path, monkeypatch):
    from bankcanary import refresh
    from bankcanary.config import load_settings
    from bankcanary.publish import writer as writer_module

    settings = load_settings().model_copy(update={"runs_dir": tmp_path / "runs"})
    today = pd.Timestamp("2026-06-30").date()
    decision = refresh.Decision(today, today, "database", False, None, "API is not newer")
    monkeypatch.setattr(refresh, "plan", lambda *a, **k: decision)
    monkeypatch.setattr(refresh, "ingest_quarter", lambda *a, **k: pytest.fail("ingested"))
    monkeypatch.setattr(refresh, "rebuild_warehouse", lambda *a, **k: pytest.fail("rebuilt"))
    written: list[tuple[str, str, pd.DataFrame]] = []

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(db, "reachable", lambda url=None: True)
    monkeypatch.setattr(db, "connect", lambda url=None, autocommit=False: FakeConn())
    monkeypatch.setattr(db, "database_size_bytes", lambda conn: 4_000)

    def fake_write(conn, table, frame, mode="replace", schema="public"):
        written.append((table, mode, frame))
        return len(frame)

    monkeypatch.setattr(writer_module, "write_table", fake_write)
    result = refresh.run(settings, url="postgresql://ignored")
    assert [(t, m) for t, m, _ in written] == [("pipeline_runs", "upsert")]
    row = written[0][2].iloc[0]
    assert row["status"] == "ok" and bool(row["new_quarter"]) is False
    assert json.loads(row["rows_written"]) == {} and row["latest_repdte"] == today
    assert result.rows_written == {} and result.status == "ok" and result.db_bytes == 4_000
    assert (tmp_path / "runs" / "refresh" / result.run_id / "metrics.json").exists()
    assert "ingest" not in result.log_text and "API is not newer" in result.log_text


def test_refresh_scores_the_new_quarter_only_with_delta_from_the_prior_quarter(monkeypatch):
    """``score_quarter`` returns the new quarter ranked 1..n with ``delta_prob_prior_q``
    = probability minus the re-scored prior quarter; post-failure reports are skipped."""
    from bankcanary import refresh
    from bankcanary.config import load_settings

    quarter, prior = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31")
    labels = pd.DataFrame(
        {
            "cert": [1, 2, 3, 1, 2, 3, 1],
            "repdte": [prior] * 3 + [quarter] * 3 + [pd.Timestamp("2025-12-31")],
            "dropped_failed_before_avail": [False, False, False, False, False, True, False],
        }
    )
    features = labels[["cert", "repdte"]].assign(x=[0.1, 0.2, 0.3, 0.5, 0.3, 0.9, 0.0])
    wh = core.Warehouse(pd.DataFrame(), pd.DataFrame(), labels, features, pd.DataFrame(), None)

    def fake_score(rows, model_dir, model):
        score = rows["x"].to_numpy(dtype=float) * (1.0 if model == "gbdt_mono" else 0.5)
        return pd.DataFrame(
            {
                "cert": rows["cert"].to_numpy(),
                "repdte": pd.to_datetime(rows["repdte"]).to_numpy(),
                "horizon": 4,
                "model": model,
                "test_year": pd.array([None] * len(rows), dtype="Int64"),
                "score": score,
                "score_calibrated": score / 2,
            }  # fmt: skip
        )

    monkeypatch.setattr(core, "score_with_artefacts", fake_score)
    lookup = {("gbdt_mono", "production"): "gm-prod", ("hazard", "production"): "hz-prod"}
    out = refresh.score_quarter(load_settings(), wh, quarter.date(), lookup)
    assert set(out["repdte"]) == {quarter.date()} and set(out["cert"]) == {1, 2}
    g = out[out["model"] == "gbdt_mono"].set_index("cert")
    assert list(g.sort_values("rank").index) == [1, 2] and list(g["band"]) == ["high", "low"]
    assert g.loc[1, "delta_prob_prior_q"] == pytest.approx(0.25 - 0.05)
    assert g.loc[2, "delta_prob_prior_q"] == pytest.approx(0.15 - 0.10)
    h = out[out["model"] == "hazard"].set_index("cert")
    assert h.loc[1, "delta_prob_prior_q"] == pytest.approx((0.25 - 0.05) / 2)
    assert set(out["model_version"]) == {"gm-prod", "hz-prod"} and set(out["horizon"]) == {4}
    assert 3 not in set(out["cert"]), "a post-failure report is never scored"
