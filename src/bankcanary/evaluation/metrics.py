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
