"""Sensitivity analyses on the fixed out-of-time split (spec 5 rule 3 and step 13).

Three questions, each answered by refitting the same two models (the regularised logit
on ``features_v2`` and the gradient booster from ``settings.models.gbdt``) under a
changed assumption and comparing the test metrics:

* **horizon**: a 4-quarter against an 8-quarter outcome window (both label sets exist in
  the ``labels`` table);
* **censored**: rows where the bank left the industry without failing inside the window
  (``censored_in_window_Hq``) kept as ``y = 0`` against dropped from *both* training and
  test, so a model is neither taught nor judged on rescued or merged banks;
* **lag**: the availability lag of a report, 45 against 60 (the setting) against 90 days,
  with the labels rebuilt in memory for each lag (later windows, more post-failure drops)
  and the rule 6.2 split recomputed at the same lag.

Row selection always goes through :func:`bankcanary.splits.fixed_split_masks` (with a
settings copy carrying the variant's lag) and is re-checked with
:func:`bankcanary.splits.assert_no_leakage` before ``fit``. Every variant logs a
``runs/sensitivity/`` record keyed by its config; the report is rebuilt from those files.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bankcanary.config import Settings
from bankcanary.labels.build import LABEL_COLUMNS, build_labels, horizon_columns

log = logging.getLogger(__name__)

ANALYSES: tuple[str, ...] = ("horizon", "censored", "lag")
MODELS: tuple[str, ...] = ("logit", "gbdt")
FEATURE_VERSION = "v2"
PRIMARY_HORIZON = 4
HORIZONS: tuple[int, ...] = (4, 8)
LAGS: tuple[int, ...] = (45, 60, 90)
KEY = ["cert", "repdte"]
#: Panel columns the in-memory relabelling and the per-event evaluation need.
PANEL_COLUMNS = ["avail_date", "fail_date", "exit_date", "rssdhcr"]


@dataclass(frozen=True)
class Variant:
    """One cell of a sensitivity table: which assumption is changed, and how."""

    analysis: str
    model: str
    horizon: int = PRIMARY_HORIZON
    drop_censored: bool = False
    lag_days: int | None = None  # None = the settings value

    @property
    def name(self) -> str:
        if self.analysis == "horizon":
            return f"{self.horizon}q"
        if self.analysis == "censored":
            return "dropped" if self.drop_censored else "kept"
        return f"{self.lag_days}d"


def variants(which: str, settings: Settings, models=MODELS) -> list[Variant]:
    """Every variant of one analysis (``horizon`` / ``censored`` / ``lag``) or of ``all``."""
    names = list(ANALYSES) if which == "all" else [which]
    unknown = [n for n in names if n not in ANALYSES]
    if unknown:
        raise ValueError(f"unknown analysis {unknown}; choose from {ANALYSES} or 'all'")
    out: list[Variant] = []
    for analysis in names:
        for model in models:
            if analysis == "horizon":
                out += [Variant(analysis, model, horizon=h) for h in HORIZONS]
            elif analysis == "censored":
                out += [Variant(analysis, model, drop_censored=d) for d in (False, True)]
            else:
                lags = sorted(set(LAGS) | {int(settings.availability_lag_days)})
                out += [Variant(analysis, model, lag_days=lag) for lag in lags]
    return out


def load_frame(settings: Settings) -> pd.DataFrame:
    """``features_v2`` joined with ``labels`` and the panel dates needed to relabel.

    Same rows as :func:`bankcanary.models.gbdt.load_training_frame`; the extra
    ``avail_date`` / ``exit_date`` columns let :func:`relabel` rebuild the labels for a
    different availability lag without reading the panel again.
    """
    from bankcanary.storage.parquet import read_table

    features = read_table(f"features_{FEATURE_VERSION}", settings)
    labels = read_table("labels", settings)
    frame = features.merge(labels, on=KEY, how="inner", validate="one_to_one")
    if len(frame) != len(features):
        log.warning("features_%s and labels do not line up: %d joined", FEATURE_VERSION, len(frame))
    facts = read_table("panel", settings)[KEY + PANEL_COLUMNS]
    return frame.merge(facts, on=KEY, how="left", validate="one_to_one")


def label_columns(frame: pd.DataFrame) -> list[str]:
    """The label columns present in ``frame`` (all horizons), so they can be swapped out."""
    stems = ("y_", "window_end_", "censored_in_window_", "label_complete_")
    fixed = [c for c in LABEL_COLUMNS if c not in KEY]
    return [c for c in frame.columns if c in fixed or c.startswith(stems)]


def relabel(
    frame: pd.DataFrame, horizons, as_of_date: dt.date | pd.Timestamp, lag_days: int
) -> pd.DataFrame:
    """``frame`` with its label columns rebuilt in memory for ``lag_days``.

    Uses :func:`bankcanary.labels.build.build_labels` on the frame's own panel dates;
    the stored ``labels`` table is untouched. Row order and count are preserved.
    """
    fresh = build_labels(frame, [int(h) for h in horizons], as_of_date, lag_days=lag_days)
    fresh = fresh.drop(columns=[c for c in ("window_start",) if c not in frame.columns])
    base = frame.drop(columns=label_columns(frame))
    out = base.merge(fresh, on=KEY, how="left", validate="one_to_one")
    if len(out) != len(frame):
        raise ValueError("relabelling changed the row count")
    return out


def _settings_for(settings: Settings, lag_days: int | None) -> Settings:
    """A settings copy whose availability lag is the variant's, so the split follows it."""
    if lag_days is None or int(lag_days) == int(settings.availability_lag_days):
        return settings
    return settings.model_copy(update={"availability_lag_days": int(lag_days)})


def split_masks(frame: pd.DataFrame, settings: Settings, variant: Variant):
    """``(train, test)`` masks for a variant: the fixed split at the variant's lag,
    minus the censored rows on both sides when the variant drops them."""
    from bankcanary.splits import fixed_split_masks

    train, test = fixed_split_masks(
        frame, _settings_for(settings, variant.lag_days), variant.horizon
    )
    if variant.drop_censored:
        censored = frame[horizon_columns(variant.horizon)[2]].fillna(False).astype(bool)
        train, test = train & ~censored, test & ~censored
    return train, test


@dataclass
class VariantResult:
    variant: Variant
    config: dict
    metrics: dict
    scores: pd.DataFrame
    run_dir: Path | None = None


def _fmt(ts) -> str | None:
    return None if pd.isna(ts) else str(pd.Timestamp(ts).date())


def variant_metrics(scores: pd.DataFrame, model: str) -> dict:
    """P1 ranking metrics (ties broken by ``cert``) plus the Brier score of a probability."""
    from bankcanary.evaluation.metrics import evaluate
    from bankcanary.models.hazard import brier

    y, s = scores["y"].to_numpy(), scores["score"].to_numpy(dtype=float)
    out = evaluate(y, s, tie_breaker=scores["cert"].to_numpy())
    is_prob = len(s) > 0 and np.isfinite(s).all() and s.min() >= 0.0 and s.max() <= 1.0
    out["brier"] = brier(y, s) if is_prob and model != "texas" else None
    return out


def fit_variant(
    frame: pd.DataFrame,
    settings: Settings,
    variant: Variant,
    as_of_date: dt.date | pd.Timestamp | None = None,
    log_run: bool = True,
) -> VariantResult:
    """Refit one model under one changed assumption on the fixed split and score the test years.

    A ``lag`` variant relabels ``frame`` in memory first (``as_of_date`` defaults to the
    cached failures-pull date). The model is built by
    :func:`bankcanary.evaluation.walkforward.build_model` with the fixed-split constants
    (``LOGIT_C``, ``settings.models.gbdt``), which were chosen inside the training period
    of this very split (rule 6.7). Nothing is saved under ``models/``; the run record is
    the artefact.
    """
    from bankcanary import tracking
    from bankcanary.evaluation.walkforward import build_model
    from bankcanary.models.train import score_test_split
    from bankcanary.splits import assert_no_leakage, prediction_date

    lag = int(variant.lag_days if variant.lag_days is not None else settings.availability_lag_days)
    if lag != int(settings.availability_lag_days):
        if as_of_date is None:
            from bankcanary.labels.build import failures_as_of_date

            as_of_date = failures_as_of_date(settings)
        frame = relabel(frame, [variant.horizon], as_of_date, lag)
    horizon = int(variant.horizon)
    y_col, _, censored_col, _ = horizon_columns(horizon)
    pipeline, features, config = build_model(variant.model, settings)
    missing = [c for c in list(features) + [y_col, censored_col] if c not in frame.columns]
    if missing:
        raise KeyError(f"training frame is missing column(s) {missing}")
    train, test = split_masks(frame, settings, variant)
    split = settings.fixed_split
    assert_no_leakage(frame.loc[train], horizon, split.test_start, lag)
    if not train.any() or not test.any():
        raise ValueError(f"sensitivity {variant}: no train or no test rows")
    tr = frame.loc[train]
    y_train = tr[y_col].astype(int).to_numpy()
    log.info(
        "sensitivity %s/%s %s: fitting on %d rows (%d positives, last report %s)",
        variant.analysis,
        variant.name,
        variant.model,
        len(tr),
        int(y_train.sum()),
        _fmt(tr["repdte"].max()),
    )
    pipeline.fit(tr[list(features)], y_train)
    scores = score_test_split(pipeline, frame.loc[test], list(features), y_col, horizon)
    metrics = variant_metrics(scores, variant.model)
    config.update(
        {
            "analysis": variant.analysis,
            "variant": variant.name,
            "horizon": horizon,
            "label": y_col,
            "drop_censored": bool(variant.drop_censored),
            "availability_lag_days": lag,
            "first_test_prediction_date": str(prediction_date(split.test_start, lag).date()),
            "train_repdte_min": _fmt(tr["repdte"].min()),
            "train_repdte_max": _fmt(tr["repdte"].max()),
            "n_train": int(len(tr)),
            "positives_train": int(y_train.sum()),
            "n_test": int(len(scores)),
            "positives_test": int(scores["y"].sum()),
            "censored_test": int(scores["censored"].sum()) if "censored" in scores else 0,
            "n_features": len(features),
        }
    )
    log.info(
        "sensitivity %s/%s %s: pr_auc %.4f",
        variant.analysis,
        variant.name,
        variant.model,
        metrics["pr_auc"],
    )
    result = VariantResult(variant, config, metrics, scores)
    if log_run:
        run = tracking.start_run("sensitivity", config, settings)
        run.log_metrics(metrics)
        result.run_dir = run.finish()
    return result


REPORT_METRICS: tuple[str, ...] = (
    "pr_auc",
    "roc_auc",
    "recall_at_2pct",
    "recall_at_top100",
    "brier",
    "n",
    "n_failures",
)
ANALYSIS_TITLES = {
    "horizon": "Horizon: 4 against 8 quarters",
    "censored": "Censored rows: kept against dropped from training and test",
    "lag": "Availability lag: 45 against 60 against 90 days",
}


def read_runs(settings: Settings) -> list[tuple[dict, dict]]:
    """``(config, metrics)`` of every finished ``runs/sensitivity/`` record, in id order."""
    from bankcanary import tracking

    root = tracking.runs_dir(settings) / "sensitivity"
    if not root.exists():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg, met = d / "config.json", d / "metrics.json"
        if cfg.exists() and met.exists():
            out.append((json.loads(cfg.read_text("utf-8")), json.loads(met.read_text("utf-8"))))
    return out


def _variant_sort_key(row: dict):
    order = {"4q": 0, "8q": 1, "kept": 0, "dropped": 1}
    name = str(row["variant"])
    position = order[name] if name in order else int(name.rstrip("d"))
    return (MODELS.index(row["model"]), position)


def analysis_table(runs, analysis: str) -> pd.DataFrame:
    """One row per (model, variant): config facts and the test metrics."""
    rows = []
    for cfg, met in runs:
        if cfg.get("analysis") != analysis or cfg.get("model") not in MODELS:
            continue
        rows.append(
            {
                "model": cfg["model"],
                "variant": cfg["variant"],
                "n_train": cfg["n_train"],
                "positives_train": cfg["positives_train"],
                "train_repdte_max": cfg["train_repdte_max"],
                **{k: met.get(k) for k in REPORT_METRICS},
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(sorted(rows, key=_variant_sort_key)).reset_index(drop=True)


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _baseline_name(analysis: str, settings: Settings) -> str:
    return {"horizon": "4q", "censored": "kept", "lag": f"{settings.availability_lag_days}d"}[
        analysis
    ]


def interpretation(table: pd.DataFrame, analysis: str, settings: Settings) -> list[str]:
    """Two data-driven sentences: what moves against the baseline variant, and what does not.

    "Moves" is the largest absolute change of PR-AUC or recall at 2 percent across the
    variants of either model; "does not" reports the model whose metrics changed least
    and whether the ranking of the two models is the same under every variant.
    """
    if table.empty:
        return ["No run records for this analysis yet."]
    base = _baseline_name(analysis, settings)
    moves, stable, order_same = [], [], True
    ranking_by_variant: dict[str, list[str]] = {}
    for model, grp in table.groupby("model", sort=False):
        ref = grp.loc[grp["variant"] == base]
        if ref.empty:
            continue
        r = ref.iloc[0]
        biggest = 0.0
        for _, row in grp.loc[grp["variant"] != base].iterrows():
            for metric, label in (("pr_auc", "PR-AUC"), ("recall_at_2pct", "recall at 2%")):
                delta = float(row[metric]) - float(r[metric])
                if abs(delta) > abs(biggest):
                    biggest, text = (
                        delta,
                        f"{model} {label} goes from {r[metric]:.4f} ({base}) to "
                        f"{row[metric]:.4f} ({row['variant']}, {delta:+.4f})",
                    )
        if biggest:
            moves.append((abs(biggest), text))
        stable.append((abs(biggest), model))
    for variant, grp in table.groupby("variant", sort=False):
        ranking_by_variant[variant] = list(grp.sort_values("pr_auc", ascending=False)["model"])
    if ranking_by_variant:
        first = next(iter(ranking_by_variant.values()))
        order_same = all(v == first for v in ranking_by_variant.values())
    first_sentence = "What moves: " + (
        max(moves)[1] + ", the largest shift in the table."
        if moves
        else "no variant changes PR-AUC or recall at 2% against the baseline."
    )
    least = min(stable)[1] if stable else None
    second = "What does not: "
    if least is not None:
        second += f"{least} is the less sensitive model (largest shift {min(stable)[0]:.4f}), and "
    second += (
        "the PR-AUC ranking of the models is the same under every variant."
        if order_same
        else "the PR-AUC ranking of the models changes between variants."
    )
    return [first_sentence, second]


def report_path(settings: Settings) -> Path:
    return Path(settings.reports_dir) / "sensitivity.md"


def write_sensitivity_report(settings: Settings, path: Path | None = None) -> Path:
    """Rebuild ``reports/sensitivity.md`` from the ``runs/sensitivity/`` records.

    One table per analysis (models x variants, test metrics on the fixed split) followed
    by the two-sentence :func:`interpretation`. Analyses without runs are listed as
    pending so a partial state is visible rather than silent.
    """
    path = Path(path) if path is not None else report_path(settings)
    runs = read_runs(settings)
    split = settings.fixed_split
    lines = [
        "# Sensitivity analyses",
        "",
        "Fixed out-of-time split (train reports "
        f"{split.train_start}..{split.train_end}, rule 6.2 trimmed; test "
        f"{split.test_start}..{split.test_end}). Models: `logit` (P1 regularised logit on "
        "`features_v2`, `LOGIT_C`) and `gbdt` (`settings.models.gbdt`). Every cell is one "
        "refit on the split with a single assumption changed; the records live in "
        "`runs/sensitivity/`. Metrics are ranking metrics on the test rows (PR-AUC, ROC-AUC, "
        "recall in the top 2 percent and top 100) plus the Brier score of the probability. "
        "The unchanged cell of every analysis (4q, kept, 60d) is the same fit as the model's "
        "fixed-split `train` run, so the tables read against `reports/p2_gbdt.md`; the 8q "
        "training set is trimmed a year earlier by rule 6.2 and a lag change moves both the "
        "outcome windows and the test positives, so `n_failures` differs between cells.",
        "",
    ]
    for analysis in ANALYSES:
        table = analysis_table(runs, analysis)
        lines += [f"## {ANALYSIS_TITLES[analysis]}", ""]
        if table.empty:
            lines += [f"Pending: run `bankcanary sensitivity --which {analysis}`.", ""]
            continue
        lines += [_md_table(table), "", *interpretation(table, analysis, settings), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    return path
