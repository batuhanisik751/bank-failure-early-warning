"""Ranking metrics for bank-failure early warning.

Supervisors act on a *ranked list*: the handful of banks at the top of the score get
extra scrutiny. So besides PR-AUC and ROC-AUC we report recall@k, the share of the
banks that failed within the horizon that appear in the top k of the ranking (k as a
fraction of all banks or as a fixed head count).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

log = logging.getLogger(__name__)

METRIC_ORDER = ("pr_auc", "roc_auc")


def rank_order(scores, tie_breaker=None) -> np.ndarray:
    """Return row indices in ranking order (riskiest first).

    Scores sort descending. Ties break on ``tie_breaker`` ascending (e.g. ``cert``)
    when given, else on original row order; both are stable so results are
    reproducible. NaN scores rank last, after every real score.
    """
    s = np.asarray(scores, dtype=float)
    n = s.shape[0]
    nan = np.isnan(s)
    # Ascending sort on -score; NaN sorts last with numpy, which is what we want.
    keys = [np.arange(n), -s]
    if tie_breaker is not None:
        tb = np.asarray(tie_breaker)
        if tb.shape[0] != n:
            raise ValueError("tie_breaker must have the same length as scores")
        keys = [np.arange(n), tb, -s]
    order = np.lexsort(keys)
    # lexsort puts NaN keys last already; make it explicit for older numpy quirks.
    if nan.any():
        order = np.concatenate([order[~nan[order]], order[nan[order]]])
    return order


def _k_from_frac(frac: float, n: int) -> int:
    """Head size for a fraction of the population: ceil, at least 1 when n > 0."""
    if n == 0:
        return 0
    return max(1, min(n, math.ceil(frac * n)))


def _frac_name(frac: float) -> str:
    return f"recall_at_{frac * 100:g}pct"


def recall_at_k(y_sorted: np.ndarray, k: int) -> float:
    """Share of all positives found in the first ``k`` rows of a ranked label array."""
    n_pos = int(y_sorted.sum())
    if n_pos == 0:
        return float("nan")
    return float(y_sorted[:k].sum() / n_pos)


def _nan_result(k_fracs, k_counts, n: int, n_failures: int) -> dict:
    out = {"pr_auc": float("nan"), "roc_auc": float("nan")}
    for frac in k_fracs:
        out[_frac_name(frac)] = float("nan")
    for k in k_counts:
        out[f"recall_at_top{k}"] = float("nan")
    out["n"] = n
    out["n_failures"] = n_failures
    return out


def evaluate(
    y_true,
    scores,
    k_fracs: Sequence[float] = (0.01, 0.02, 0.05),
    k_counts: Sequence[int] = (50, 100),
    tie_breaker=None,
) -> dict:
    """Score a ranking of banks against the realised failure labels.

    Returns ``pr_auc`` (average precision), ``roc_auc``, ``recall_at_<f>pct`` for each
    fraction in ``k_fracs`` (k = ceil(f * n)), ``recall_at_top<k>`` for each count in
    ``k_counts`` (capped at n), plus ``n`` and ``n_failures``. NaN scores rank last
    and, for the AUC metrics, are treated as a score below every real one. When there
    are no positives every metric is NaN and a warning is logged.
    """
    y = np.asarray(y_true).astype(float)
    y = np.nan_to_num(y, nan=0.0).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    n = int(y.shape[0])
    n_failures = int(y.sum())
    if n == 0 or n_failures == 0:
        log.warning("evaluate: no failures among %d rows; metrics are NaN", n)
        return _nan_result(k_fracs, k_counts, n, n_failures)

    order = rank_order(s, tie_breaker)
    y_sorted = y[order]
    out: dict = {}
    nan = np.isnan(s)
    if nan.any():
        floor = np.nanmin(s) - 1.0 if (~nan).any() else 0.0
        s_auc = np.where(nan, floor, s)
    else:
        s_auc = s
    out["pr_auc"] = float(average_precision_score(y, s_auc))
    if n_failures == n:
        log.warning("evaluate: every row is a failure; roc_auc is undefined")
        out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y, s_auc))
    for frac in k_fracs:
        out[_frac_name(frac)] = recall_at_k(y_sorted, _k_from_frac(frac, n))
    for k in k_counts:
        out[f"recall_at_top{k}"] = recall_at_k(y_sorted, min(int(k), n))
    out["n"] = n
    out["n_failures"] = n_failures
    return out


def evaluate_by_year(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    year_col: str,
    tie_col: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Metrics per test year plus a ``pooled`` row over the whole frame.

    A failure wave (2009–2011) and a quiet stretch (2015–2019) look nothing alike, so
    every row carries ``n_failures``: a year with a handful of failures gives noisy
    metrics and should be read as low-confidence. ``year`` is a string column so the
    ``pooled`` row fits; per-year rows are sorted ascending.
    """
    rows: list[dict] = []
    years = sorted(df[year_col].dropna().unique(), key=lambda v: str(v))
    for year in years:
        sub = df[df[year_col] == year]
        tb = sub[tie_col].to_numpy() if tie_col else None
        m = evaluate(sub[label_col].to_numpy(), sub[score_col].to_numpy(), tie_breaker=tb, **kwargs)
        rows.append({"year": str(year), **m})
    tb = df[tie_col].to_numpy() if tie_col else None
    pooled = evaluate(df[label_col].to_numpy(), df[score_col].to_numpy(), tie_breaker=tb, **kwargs)
    rows.append({"year": "pooled", **pooled})
    out = pd.DataFrame(rows)
    lead_cols = ["year", "n", "n_failures"]
    cols = lead_cols + [c for c in out.columns if c not in lead_cols]
    return out[cols]


def _quarter_index(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d.dt.year * 4 + (d.dt.month - 1) // 3


def lead_time_quarters(
    df: pd.DataFrame,
    score_col: str,
    cert_col: str,
    repdte_col: str,
    fail_date_col: str,
    top_frac: float = 0.02,
) -> pd.DataFrame:
    """How early the ranking flagged each bank that failed.

    Within every report quarter the banks are ranked by score (NaN last, ties by cert)
    and the top ``ceil(top_frac * n)`` are "flagged". For each cert with a failure date
    the result holds ``first_flag_repdte`` (earliest flagged quarter before failure) and
    ``lead_time_quarters`` = calendar quarters between that report and the failure
    (NaN when the bank was never flagged before it failed), plus ``n_quarters_scored``.
    """
    d = df[[cert_col, repdte_col, score_col, fail_date_col]].copy()
    d[repdte_col] = pd.to_datetime(d[repdte_col])
    d[fail_date_col] = pd.to_datetime(d[fail_date_col])
    d["_flag"] = False
    for _, idx in d.groupby(repdte_col).indices.items():
        sub = d.iloc[idx]
        order = rank_order(sub[score_col].to_numpy(), sub[cert_col].to_numpy())
        k = _k_from_frac(top_frac, len(sub))
        d.iloc[idx[order[:k]], d.columns.get_loc("_flag")] = True
    failed = d[d[fail_date_col].notna() & (d[repdte_col] < d[fail_date_col])]
    rows: list[dict] = []
    for cert, sub in failed.groupby(cert_col, sort=True):
        fail_date = sub[fail_date_col].iloc[0]
        flagged = sub.loc[sub["_flag"], repdte_col]
        first = flagged.min() if not flagged.empty else pd.NaT
        lead = float("nan")
        if pd.notna(first):
            q_fail = _quarter_index(pd.Series([fail_date])).iloc[0]
            q_first = _quarter_index(pd.Series([first])).iloc[0]
            lead = int(q_fail - q_first)
        rows.append(
            {
                cert_col: cert,
                fail_date_col: fail_date,
                "first_flag_repdte": first,
                "lead_time_quarters": lead,
                "n_quarters_scored": int(len(sub)),
            }
        )
    cols = [cert_col, fail_date_col, "first_flag_repdte", "lead_time_quarters", "n_quarters_scored"]
    return pd.DataFrame(rows, columns=cols)


def evaluate_by_event(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    cert_col: str = "cert",
    repdte_col: str = "repdte",
    hc_col: str = "rssdhcr",
    fail_date_col: str = "fail_date",
    **kwargs,
) -> dict:
    """Metrics per *failure event* rather than per bank (spec section 5, rule 6).

    When several subsidiaries of one holding company fail on the same day (FBOP's nine
    banks on 2009-10-30, for instance) a per-bank ranking counts one supervisory event
    several times. Here every positive row that shares ``(repdte, rssdhcr, fail_date)``
    with another positive row is collapsed into one unit whose score is the *highest*
    score among the sister banks: the event counts as caught when any of them was
    flagged. Non-failing rows and failed banks without a holding company stay one row
    per bank. Returns the usual :func:`evaluate` keys plus ``n_events`` (positives after
    collapsing), ``n_multi_bank_events`` and ``n_banks_in_multi_events``.
    """
    d = df[[cert_col, repdte_col, score_col, label_col]].copy()
    y = np.nan_to_num(d[label_col].astype(float).to_numpy(), nan=0.0).astype(int)
    hc = pd.to_numeric(df[hc_col], errors="coerce") if hc_col in df.columns else None
    fail = pd.to_datetime(df[fail_date_col]) if fail_date_col in df.columns else None
    has_event = (y == 1) & (hc is not None) & (fail is not None)
    if hc is not None and fail is not None:
        has_event &= hc.fillna(0).to_numpy() > 0
        has_event &= fail.notna().to_numpy()
    event = d[cert_col].astype(str)
    if has_event.any():
        tag = "hc" + hc.astype("Int64").astype(str) + "@" + fail.dt.strftime("%Y-%m-%d")
        event = event.where(~has_event, tag.astype(str))
    d["_event"] = event.to_numpy()
    d["_y"] = y
    grouped = (
        d.groupby([repdte_col, "_event"], sort=True)
        .agg(score=(score_col, "max"), y=("_y", "max"), cert=(cert_col, "min"), n=("_y", "size"))
        .reset_index()
    )
    m = evaluate(
        grouped["y"].to_numpy(),
        grouped["score"].to_numpy(),
        tie_breaker=grouped["cert"].to_numpy(),
        **kwargs,
    )
    multi = grouped[(grouped["y"] == 1) & (grouped["n"] > 1)]
    m["n_events"] = int(grouped["y"].sum())
    m["n_multi_bank_events"] = int(len(multi))
    m["n_banks_in_multi_events"] = int(multi["n"].sum())
    return m


def brier(y_true, probabilities) -> float:
    """Mean squared error of the probabilities against the 0/1 outcome (lower is better).

    Only meaningful for a score that is a probability; a ranking such as the Texas ratio
    has no Brier score. NaN labels count as non-failures, as in :func:`evaluate`.
    """
    y = np.nan_to_num(np.asarray(y_true, dtype=float), nan=0.0)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} labels vs {p.shape} probabilities")
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


LOW_CONFIDENCE_FAILURES = 10
LEAD_TIME_AHEAD_QUARTERS = 2


def lead_time_summary(
    lead: pd.DataFrame,
    fail_date_col: str = "fail_date",
    fail_years: tuple[int, int] | None = None,
    ahead_quarters: int = LEAD_TIME_AHEAD_QUARTERS,
) -> dict:
    """Summarise :func:`lead_time_quarters` output: how early the failures were flagged.

    Every failed bank counts once. ``share_flagged_ahead`` is the share of failed banks
    that first entered the top 2 percent at least ``ahead_quarters`` quarters before
    failing (never-flagged banks are in the denominator, so this is a recall-like number,
    not a statistic of the flagged banks only). ``median_lead_time_quarters`` is over the
    flagged banks; ``median_lead_time_all`` treats a never-flagged bank as lead 0 so that
    a model that misses half the failures cannot report a long median. ``fail_years``
    restricts to failures dated in the inclusive year range (spec: 2009-2012).
    """
    d = lead
    if fail_years is not None:
        years = pd.to_datetime(d[fail_date_col]).dt.year
        d = d[(years >= int(fail_years[0])) & (years <= int(fail_years[1]))]
    lt = d["lead_time_quarters"].astype(float)
    flagged = lt.notna()
    n = int(len(d))
    out = {
        "n_failed": n,
        "n_flagged": int(flagged.sum()),
        "share_flagged": float(flagged.mean()) if n else float("nan"),
        "median_lead_time_quarters": float(lt[flagged].median()) if flagged.any() else float("nan"),
        "median_lead_time_all": float(lt.fillna(0.0).median()) if n else float("nan"),
        "share_flagged_ahead": float((lt.fillna(-1) >= ahead_quarters).mean())
        if n
        else float("nan"),
        "ahead_quarters": int(ahead_quarters),
        "low_confidence": n < LOW_CONFIDENCE_FAILURES,
    }
    if fail_years is not None:
        out["fail_years"] = [int(fail_years[0]), int(fail_years[1])]
    return out


BOOTSTRAP_DRAWS = 200
BOOTSTRAP_SEED = 20080101


def _weighted_average_precision(y_sorted, w_sorted, group_end) -> float:
    """sklearn's step-wise average precision with row weights, on rows sorted by score."""
    tp = np.cumsum(w_sorted * y_sorted)[group_end]
    seen = np.cumsum(w_sorted)[group_end]
    total_tp = tp[-1]
    if total_tp <= 0:
        return float("nan")
    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    return float(np.sum(np.diff(tp, prepend=0.0) * precision) / total_tp)


def _weighted_recall_at_frac(y_ranked, w_ranked, frac: float) -> float:
    """Recall in the head that holds the first ``ceil(frac * total weight)`` resampled rows."""
    total = w_ranked.sum()
    positives = float((w_ranked * y_ranked).sum())
    if total <= 0 or positives <= 0:
        return float("nan")
    k = max(1.0, math.ceil(frac * total))
    starts = np.cumsum(w_ranked) - w_ranked
    head = starts < k
    return float((w_ranked[head] * y_ranked[head]).sum() / positives)


def cluster_bootstrap_ci(
    y_true,
    scores,
    clusters,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    k_frac: float = 0.02,
    tie_breaker=None,
    alpha: float = 0.05,
) -> dict:
    """Percentile confidence intervals for PR-AUC and recall@k by resampling banks.

    Bank-quarters of one bank are not independent (a failing bank is a positive in
    several consecutive quarters), so the resampling unit is the ``cluster`` (the
    ``cert``): each draw resamples the banks with replacement and every row carries the
    number of times its bank was drawn as a weight. The seed is fixed so the interval is
    reproducible. Returns ``pr_auc_ci`` and ``recall_at_<k>pct_ci`` as ``[low, high]``,
    the number of draws, clusters and failures, and ``low_confidence`` when the slice
    holds fewer than :data:`LOW_CONFIDENCE_FAILURES` failures (spec: the interval of such
    a year is not to be trusted). With no failures every bound is NaN.
    """
    y = np.nan_to_num(np.asarray(y_true, dtype=float), nan=0.0)
    s = np.asarray(scores, dtype=float)
    c = np.asarray(clusters)
    if not (y.shape[0] == s.shape[0] == c.shape[0]):
        raise ValueError("y_true, scores and clusters must have the same length")
    n, n_failures = int(len(y)), int(y.sum())
    name = f"{_frac_name(k_frac)}_ci"
    _, inverse = np.unique(c, return_inverse=True)
    m = int(inverse.max()) + 1 if n else 0
    out = {
        "n_draws": int(n_draws),
        "n_clusters": m,
        "n": n,
        "n_failures": n_failures,
        "low_confidence": n_failures < LOW_CONFIDENCE_FAILURES,
    }
    if n == 0 or n_failures == 0:
        return {**out, "pr_auc_ci": [float("nan")] * 2, name: [float("nan")] * 2}
    nan = np.isnan(s)
    if nan.any():
        s = np.where(nan, (np.nanmin(s) - 1.0 if (~nan).any() else 0.0), s)
    by_score = np.argsort(-s, kind="stable")
    s_sorted = s[by_score]
    group_end = np.flatnonzero(np.append(s_sorted[1:] != s_sorted[:-1], True))
    ranked = rank_order(s, tie_breaker)
    rng = np.random.default_rng(seed)
    aps, recalls = np.empty(n_draws), np.empty(n_draws)
    for i in range(n_draws):
        counts = np.bincount(rng.integers(0, m, size=m), minlength=m).astype(float)
        w = counts[inverse]
        aps[i] = _weighted_average_precision(y[by_score], w[by_score], group_end)
        recalls[i] = _weighted_recall_at_frac(y[ranked], w[ranked], k_frac)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out["pr_auc_ci"] = [float(v) for v in np.nanpercentile(aps, q)]
    out[name] = [float(v) for v in np.nanpercentile(recalls, q)]
    out["n_failed_draws"] = int(np.isnan(aps).sum())
    return out
