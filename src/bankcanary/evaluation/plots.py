"""Small, deterministic PNG figures for the evaluation report.

Every function saves to ``path`` with the Agg backend and strips the volatile metadata
(matplotlib version, creation time) so re-running from the same predictions produces a
byte-identical file that git does not see as changed.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

_METADATA = {"Software": None, "CreationDate": None, "Creation Time": None}
_FIGSIZE = (4.5, 3.2)
_DPI = 110


def _save(fig, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, format="png", metadata=_METADATA)
    plt.close(fig)
    return path


def _clean(y, scores):
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0).astype(int)
    s = np.asarray(scores, dtype=float)
    nan = np.isnan(s)
    if nan.any():
        floor = np.nanmin(s) - 1.0 if (~nan).any() else 0.0
        s = np.where(nan, floor, s)
    return y, s


def pr_curve(y, scores, path: str | os.PathLike) -> Path:
    """Precision-recall curve: precision among flagged banks versus failures caught.

    The dashed line is the base rate (share of failures), the score a random ranking
    would achieve; the legend shows average precision (PR-AUC).
    """
    y, s = _clean(y, scores)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if y.sum() > 0:
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.step(rec, prec, where="post", label=f"AP = {ap:.3f}")
        ax.axhline(y.mean(), linestyle="--", linewidth=1, label=f"base rate = {y.mean():.4f}")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no failures in sample", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (failures caught)")
    ax.set_ylabel("Precision (flagged banks that failed)")
    ax.set_title("Precision-recall", fontsize=10)
    return _save(fig, path)


def recall_at_k_bars(metrics_by_model: dict[str, dict], path: str | os.PathLike) -> Path:
    """Grouped bars of recall@k per model: which ranking catches more failures early.

    ``metrics_by_model`` maps a model name to an ``evaluate()`` result; every key that
    starts with ``recall_at_`` becomes a group on the x-axis.
    """
    names = list(metrics_by_model)
    keys: list[str] = []
    for m in metrics_by_model.values():
        for k in m:
            if k.startswith("recall_at_") and k not in keys:
                keys.append(k)
    fig, ax = plt.subplots(figsize=(max(_FIGSIZE[0], 1.2 * len(keys) + 1.5), _FIGSIZE[1]))
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(names))
    for i, name in enumerate(names):
        vals = [float(metrics_by_model[name].get(k, np.nan)) for k in keys]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([k.removeprefix("recall_at_") for k in keys], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (share of failures caught)")
    ax.set_title("Recall at k", fontsize=10)
    if names:
        ax.legend(fontsize=8)
    return _save(fig, path)


def score_distributions(y, scores, path: str | os.PathLike) -> Path:
    """Histograms of the score for banks that failed and banks that did not.

    Densities rather than counts, because failures are ~1% of rows and would be
    invisible on a shared count axis. Separation between the two shapes is what a
    useful early-warning score looks like; NaN scores are dropped.
    """
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0).astype(int)
    s = np.asarray(scores, dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if s.size:
        bins = np.linspace(np.min(s), np.max(s), 31) if np.min(s) < np.max(s) else 10
        n0, n1 = int((y == 0).sum()), int(y.sum())
        ax.hist(s[y == 0], bins=bins, density=True, alpha=0.6, label=f"survived (n={n0})")
        if n1:
            ax.hist(s[y == 1], bins=bins, density=True, alpha=0.6, label=f"failed (n={n1})")
        ax.legend(fontsize=8)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score distributions", fontsize=10)
    return _save(fig, path)
