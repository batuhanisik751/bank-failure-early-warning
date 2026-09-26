"""Hand-checked cases for the Prototype 2 metrics: Brier, lead-time summary, bootstrap CIs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from bankcanary.evaluation import metrics


def test_brier_hand_cases_and_shape_check():
    assert metrics.brier([0, 1], [0.0, 1.0]) == 0.0
    assert metrics.brier([0, 1], [0.5, 0.5]) == pytest.approx(0.25)
    # NaN labels count as non-failures: (0.2^2 + 0.9^2) / 2
    assert metrics.brier([np.nan, 1], [0.2, 0.1]) == pytest.approx((0.04 + 0.81) / 2)
    assert np.isnan(metrics.brier([], []))
    with pytest.raises(ValueError, match="shape"):
        metrics.brier([0, 1], [0.5])


def _lead_frame():
    return pd.DataFrame(
        {
            "cert": [1, 2, 3, 4, 5],
            "fail_date": pd.to_datetime(
                ["2009-06-01", "2010-03-15", "2011-11-30", "2013-02-01", "2009-09-09"]
            ),
            "first_flag_repdte": pd.to_datetime(
                ["2008-09-30", "2009-12-31", None, "2012-12-31", "2008-12-31"]
            ),
            "lead_time_quarters": [3.0, 1.0, np.nan, 1.0, 3.0],
            "n_quarters_scored": [5, 8, 12, 20, 6],
        }
    )


def test_lead_time_summary_counts_never_flagged_banks_in_the_denominator():
    s = metrics.lead_time_summary(_lead_frame())
    assert s["n_failed"] == 5 and s["n_flagged"] == 4
    assert s["share_flagged"] == pytest.approx(0.8)
    assert s["median_lead_time_quarters"] == pytest.approx(2.0)  # median of 3, 1, 1, 3
    assert s["median_lead_time_all"] == pytest.approx(1.0)  # 3, 1, 0, 1, 3
    assert s["share_flagged_ahead"] == pytest.approx(0.4)  # two banks at 3 quarters
    assert s["ahead_quarters"] == 2 and s["low_confidence"] is True
    assert "fail_years" not in s
    crisis = metrics.lead_time_summary(_lead_frame(), fail_years=(2009, 2012))
    assert crisis["n_failed"] == 4 and crisis["fail_years"] == [2009, 2012]
    assert crisis["share_flagged_ahead"] == pytest.approx(0.5)
    empty = metrics.lead_time_summary(_lead_frame(), fail_years=(2020, 2021))
    assert empty["n_failed"] == 0 and np.isnan(empty["share_flagged_ahead"])
    one_ahead = metrics.lead_time_summary(_lead_frame(), ahead_quarters=1)
    assert one_ahead["share_flagged_ahead"] == pytest.approx(0.8)


def _sample(seed: int = 3, n_banks: int = 300, quarters: int = 4):
    rng = np.random.default_rng(seed)
    cert = np.repeat(np.arange(n_banks), quarters)
    risk = np.repeat(rng.normal(size=n_banks), quarters)
    score = risk + rng.normal(scale=0.5, size=len(cert))
    y = (risk + rng.normal(scale=0.7, size=len(cert)) > 1.6).astype(int)
    return y, score, cert


def test_weighted_average_precision_matches_sklearn_on_a_duplicated_sample():
    y, s, _ = _sample()
    rng = np.random.default_rng(0)
    w = rng.integers(0, 4, size=len(y)).astype(float)
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    group_end = np.flatnonzero(np.append(s_sorted[1:] != s_sorted[:-1], True))
    ap = metrics._weighted_average_precision(y[order], w[order], group_end)
    expanded_y, expanded_s = np.repeat(y, w.astype(int)), np.repeat(s, w.astype(int))
    assert ap == pytest.approx(average_precision_score(expanded_y, expanded_s))
    ones = metrics._weighted_average_precision(y[order], np.ones(len(y)), group_end)
    assert ones == pytest.approx(average_precision_score(y, s))


def test_weighted_recall_at_frac_with_unit_weights_is_recall_at_k():
    y, s, c = _sample()
    ranked = metrics.rank_order(s, c)
    got = metrics._weighted_recall_at_frac(y[ranked], np.ones(len(y)), 0.02)
    assert got == pytest.approx(metrics.evaluate(y, s, tie_breaker=c)["recall_at_2pct"])
    assert np.isnan(metrics._weighted_recall_at_frac(np.zeros(3), np.ones(3), 0.5))


def test_cluster_bootstrap_ci_is_reproducible_brackets_the_point_estimate_and_flags():
    y, s, c = _sample()
    point = metrics.evaluate(y, s, tie_breaker=c)
    ci = metrics.cluster_bootstrap_ci(y, s, c, tie_breaker=c)
    again = metrics.cluster_bootstrap_ci(y, s, c, tie_breaker=c)
    assert ci == again  # fixed seed
    assert ci["n_draws"] == 200 and ci["n_clusters"] == 300 and ci["n"] == len(y)
    assert ci["n_failures"] == int(y.sum()) and ci["low_confidence"] is False
    lo, hi = ci["pr_auc_ci"]
    assert lo < point["pr_auc"] < hi and 0.0 <= lo < hi <= 1.0
    lo, hi = ci["recall_at_2pct_ci"]
    assert lo <= point["recall_at_2pct"] <= hi and hi - lo < 1.0
    assert ci["n_failed_draws"] == 0
    other = metrics.cluster_bootstrap_ci(y, s, c, tie_breaker=c, seed=1)
    assert other["pr_auc_ci"] != ci["pr_auc_ci"]
    narrow = metrics.cluster_bootstrap_ci(y, s, c, tie_breaker=c, k_frac=0.05, n_draws=50)
    assert "recall_at_5pct_ci" in narrow and narrow["n_draws"] == 50


def test_cluster_bootstrap_ci_resamples_banks_not_rows():
    # Two banks only: every draw is either one bank twice or both banks once, so the
    # interval spans the two extremes and the failing bank alone gives a NaN draw.
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8])
    c = np.array([1, 1, 2, 2])
    ci = metrics.cluster_bootstrap_ci(y, s, c, n_draws=100)
    assert ci["n_clusters"] == 2 and ci["low_confidence"] is True
    assert ci["n_failed_draws"] > 0  # draws with no positive bank
    assert ci["pr_auc_ci"] == [1.0, 1.0]  # the failing bank always outranks the other


def test_cluster_bootstrap_ci_degenerate_inputs():
    empty = metrics.cluster_bootstrap_ci([], [], [])
    assert empty["n"] == 0 and np.isnan(empty["pr_auc_ci"]).all()
    none = metrics.cluster_bootstrap_ci([0, 0, 0], [0.1, 0.2, 0.3], [1, 2, 3])
    assert none["n_failures"] == 0 and np.isnan(none["recall_at_2pct_ci"]).all()
    assert none["low_confidence"] is True
    with pytest.raises(ValueError, match="same length"):
        metrics.cluster_bootstrap_ci([0, 1], [0.1, 0.2], [1])
    with_nan = metrics.cluster_bootstrap_ci([0, 1, 0, 1], [np.nan, 0.9, 0.2, 0.8], [1, 2, 3, 4])
    assert with_nan["pr_auc_ci"][1] <= 1.0


# --- the report module over a synthetic walkforward_scores table -------------------------


def _scores_table(seed: int = 5, n_banks: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Two models over 2008-2010 with a rare failure driven by a latent risk index."""
    from bankcanary.evaluation import walkforward as w

    rng = np.random.default_rng(seed)
    quarters = pd.date_range("2008-03-31", "2010-12-31", freq="QE-DEC")
    base = pd.DataFrame(
        {
            "cert": np.repeat(np.arange(1, n_banks + 1), len(quarters)),
            "repdte": np.tile(quarters, n_banks),
        }
    )
    risk = np.repeat(rng.normal(size=n_banks), len(quarters))
    fail = np.repeat(rng.random(n_banks) < 0.12, len(quarters)) & (risk > 0.5)
    fail_dates = pd.Series(pd.NaT, index=np.arange(1, n_banks + 1), dtype="datetime64[ns]")
    for cert in np.unique(base["cert"][fail]):
        fail_dates[cert] = pd.Timestamp("2009-06-15") + pd.Timedelta(days=int(rng.integers(0, 500)))
    fd = base["cert"].map(fail_dates)
    window_end = base["repdte"] + pd.Timedelta(days=60) + pd.DateOffset(months=12)
    y = (fd.notna() & (fd > base["repdte"]) & (fd <= window_end)).astype(int)
    frames = []
    for model, noise in (("logit", 0.6), ("texas", 1.5)):
        score = 1 / (1 + np.exp(-(risk * 2 - 4 + rng.normal(scale=noise, size=len(base)))))
        cal = np.full(len(base), np.nan) if model == "texas" else np.clip(score * 1.3, 0, 1)
        frames.append(
            base.assign(
                horizon=4,
                model=model,
                test_year=base["repdte"].dt.year,
                score=score,
                score_calibrated=cal,
                y=y,
                label_complete=True,
                censored=False,
            )[list(w.SCORE_COLUMNS)]
        )
    return pd.concat(frames, ignore_index=True), fail_dates.dropna()


def test_reliability_table_bins_by_raw_score_decile():
    from bankcanary.evaluation import metrics_report as r

    table, _ = _scores_table()
    rel = r.reliability_table(table[table["model"] == "logit"])
    assert rel["decile"].tolist() == list(range(1, 11))
    assert rel["n"].sum() == (table["model"] == "logit").sum()
    assert rel["n_failures"].sum() == table.loc[table["model"] == "logit", "y"].sum()
    assert (np.diff(rel["mean_raw"]) > 0).all() and (np.diff(rel["mean_calibrated"]) >= 0).all()
    np.testing.assert_allclose(rel["observed_rate"], rel["n_failures"] / rel["n"])
    assert rel["observed_rate"].iloc[-1] > rel["observed_rate"].iloc[0]
    assert r.reliability_table(table.iloc[:0]).empty


def test_lead_time_tables_and_ci_table_on_the_synthetic_table():
    from bankcanary.evaluation import metrics_report as r

    table, fail_dates = _scores_table()
    logit = table[table["model"] == "logit"]
    lead, summaries = r.lead_time_tables(logit, fail_dates)
    assert sorted(lead["cert"]) == sorted(fail_dates.index)  # every failed bank was scored
    assert summaries["crisis"]["fail_years"] == [2009, 2012]
    assert summaries["all"]["n_failed"] == len(lead)
    assert 0 < summaries["all"]["share_flagged"] <= 1
    ci = r.ci_table(logit, n_draws=20)
    assert ci["year"].tolist() == ["2008", "2009", "2010", "pooled"]
    assert (ci["pr_auc_low"] <= ci["pr_auc"] + 1e-9).all()
    assert ci["low_confidence"].dtype == bool
    br = r.brier_table(table[table["model"] == "texas"])
    assert br["brier_calibrated"].isna().all() and br["brier_raw"].notna().all()


def test_write_metrics_report_renders_tables_figures_and_runs_deterministically(tmp_path):
    from bankcanary.evaluation import metrics_report as r
    from bankcanary.evaluation import walkforward as w
    from bankcanary.storage.parquet import table_path, write_table
    from tests.test_hazard import make_settings

    settings = make_settings(tmp_path)
    table, fail_dates = _scores_table()
    write_table(table, w.TABLE, w.TABLE_KEY, settings)
    panel = pd.DataFrame(
        {
            "cert": fail_dates.index,
            "repdte": pd.Timestamp("2008-03-31"),
            "fail_date": fail_dates.values,
        }
    )
    table_path(settings, "panel").parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(table_path(settings, "panel"), index=False)
    assert r.failure_dates(settings).equals(fail_dates.sort_index())

    out = r.write_metrics_report(settings, 4)
    text = out.read_text()
    for needle in (
        "## Horizon 4q (test years 2008-2010)",
        "### Pooled",
        "#### logit (4q)",
        "#### texas (4q)",
        "### Calibration",
        "Reliability by raw-score decile",
        "### Lead time",
        "| logit | 2009-2012 |",
        "![reliability logit](figures/reliability_logit.png)",
        "![lead time texas](figures/lead_time_texas.png)",
        "pr_auc 95% CI",
        "low confidence",
    ):
        assert needle in text, needle
    assert "#### texas (4q)\n\nPooled Brier" not in text  # no calibration section for a ranking
    figures = tmp_path / "reports" / "figures"
    names = ["reliability_logit.png", "lead_time_logit.png", "lead_time_texas.png"]
    assert all((figures / n).exists() for n in names)
    assert not (figures / "reliability_texas.png").exists()
    runs = sorted((tmp_path / "runs" / "metrics").iterdir())
    assert len(runs) == 2
    m = json.loads((runs[0] / "metrics.json").read_text())
    assert {"pr_auc", "pr_auc_low", "brier_raw", "lead_crisis_share_flagged_ahead"} <= set(m)
    before = {n: (figures / n).read_bytes() for n in names}
    r.write_metrics_report(settings, 4)
    assert out.read_text() == text
    assert all((figures / n).read_bytes() == before[n] for n in names)
    assert len(sorted((tmp_path / "runs" / "metrics").iterdir())) == 2
    with pytest.raises(ValueError, match="no walk-forward scores at 8q"):
        r.write_metrics_report(settings, 8)
