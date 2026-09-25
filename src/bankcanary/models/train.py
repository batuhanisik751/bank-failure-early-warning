"""Fit a baseline on the fixed out-of-time split, score the test years and save artefacts.

Rows are selected *only* through :func:`bankcanary.splits.fixed_split_masks`, and
:func:`bankcanary.splits.assert_no_leakage` is run on the selected training rows right
before fitting, so a model can never learn from an outcome window that was still open
on the first test prediction date (spec rule 6.2).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from bankcanary.config import Settings
from bankcanary.evaluation.metrics import evaluate, evaluate_by_event, evaluate_by_year
from bankcanary.labels.build import horizon_columns
from bankcanary.models.baselines import (
    MODEL_NAMES,
    coefficient_table,
    make_model,
    model_features,
    score_pipeline,
)
from bankcanary.splits import assert_no_leakage, fixed_split_masks, prediction_date
from bankcanary.storage.parquet import read_table

log = logging.getLogger(__name__)

PRIMARY_HORIZON = 4
KEY = ["cert", "repdte"]
#: Panel columns carried into the score frame for the per-failure-event evaluation.
EVENT_COLUMNS = ["rssdhcr", "fail_date"]


@dataclass
class TrainResult:
    """Everything one training run produced, in memory, before or after saving."""

    name: str
    horizon: int
    pipeline: Pipeline
    features: list[str]
    metrics: dict
    by_year: pd.DataFrame
    config: dict
    scores: pd.DataFrame = field(repr=False)
    paths: dict[str, Path] = field(default_factory=dict)
    #: Test metrics with the censored rows (non-failure exits inside the window) dropped.
    sensitivity: dict = field(default_factory=dict)
    #: Test metrics per failure event (same-day holding-company failures collapsed).
    by_event: dict = field(default_factory=dict)


def model_dir(settings: Settings, name: str, horizon: int = PRIMARY_HORIZON) -> Path:
    """``models/<name>/`` for the primary 4-quarter horizon, ``models/<name>_<H>q/`` otherwise."""
    suffix = "" if horizon == PRIMARY_HORIZON else f"_{horizon}q"
    return Path(settings.models_dir) / f"{name}{suffix}"


def load_training_frame(settings: Settings) -> pd.DataFrame:
    """``features_v1`` joined one-to-one with ``labels`` on ``(cert, repdte)``."""
    features = read_table("features_v1", settings)
    labels = read_table("labels", settings)
    frame = features.merge(labels, on=KEY, how="inner", validate="one_to_one")
    if len(frame) != len(features):
        log.warning(
            "features_v1 (%d rows) and labels (%d rows) do not line up; %d rows joined",
            len(features),
            len(labels),
            len(frame),
        )
    # Holding-company id and failure date ride along for the per-event evaluation
    # (spec 5, rule 6); they are never model inputs (features come from the registry).
    facts = read_table("panel", settings)[KEY + EVENT_COLUMNS]
    return frame.merge(facts, on=KEY, how="left", validate="one_to_one")


def _fmt(ts) -> str | None:
    return None if pd.isna(ts) else str(pd.Timestamp(ts).date())


def _split_config(frame, train, test, settings: Settings, horizon: int, y_col: str) -> dict:
    split = settings.fixed_split
    lag = settings.availability_lag_days
    tr, te = frame.loc[train], frame.loc[test]
    return {
        "horizon": horizon,
        "label": y_col,
        "availability_lag_days": lag,
        "train_start_nominal": str(split.train_start),
        "train_end_nominal": str(split.train_end),
        "train_repdte_min": _fmt(tr["repdte"].min()),
        "train_repdte_max": _fmt(tr["repdte"].max()),
        "test_start": str(split.test_start),
        "test_end": str(split.test_end),
        "test_repdte_min": _fmt(te["repdte"].min()),
        "test_repdte_max": _fmt(te["repdte"].max()),
        "first_test_prediction_date": str(prediction_date(split.test_start, lag).date()),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "positives_train": int(tr[y_col].sum()),
        "positives_test": int(te[y_col].sum()),
    }


def train_model(
    name: str,
    horizon: int,
    settings: Settings,
    frame: pd.DataFrame | None = None,
    save: bool = True,
) -> TrainResult:
    """Fit one baseline on the fixed split, score the test years and (optionally) save it.

    The training rows come from :func:`fixed_split_masks` and are re-checked with
    :func:`assert_no_leakage` before ``fit``; a leaky selection raises ``ValueError``
    and nothing is fitted or written.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose one of {MODEL_NAMES}")
    if frame is None:
        frame = load_training_frame(settings)
    y_col = horizon_columns(horizon)[0]
    features = model_features(name)
    missing = [c for c in features + [y_col] if c not in frame.columns]
    if missing:
        raise KeyError(f"training frame is missing column(s) {missing}")
    train, test = fixed_split_masks(frame, settings, horizon)
    assert_no_leakage(
        frame.loc[train], horizon, settings.fixed_split.test_start, settings.availability_lag_days
    )
    if not train.any() or not test.any():
        raise ValueError(f"fixed split for {horizon}q selected no train or no test rows")
    pipeline = make_model(name)
    X_train = frame.loc[train, features]
    y_train = frame.loc[train, y_col].astype(int).to_numpy()
    log.info("fitting %s on %d rows (%d positives)", name, len(X_train), int(y_train.sum()))
    pipeline.fit(X_train, y_train)
    scores = score_test_split(pipeline, frame.loc[test], features, y_col, horizon)
    metrics = _score_metrics(scores)
    by_year = evaluate_by_year(scores, "score", "y", "year", tie_col="cert")
    sensitivity, by_event = _extra_metrics(scores)
    config = {
        "model": name,
        "n_features": len(features),
        **_split_config(frame, train, test, settings, horizon, y_col),
    }
    result = TrainResult(
        name,
        horizon,
        pipeline,
        features,
        metrics,
        by_year,
        config,
        scores,
        sensitivity=sensitivity,
        by_event=by_event,
    )
    log.info(
        "%s %dq test: pr_auc %.4f roc_auc %.4f",
        name,
        horizon,
        metrics["pr_auc"],
        metrics["roc_auc"],
    )
    if save:
        result.paths = save_artifacts(result, model_dir(settings, name, horizon))
    return result


def _score_metrics(scores: pd.DataFrame) -> dict:
    y, s, cert = (scores[c].to_numpy() for c in ("y", "score", "cert"))
    return evaluate(y, s, tie_breaker=cert)


def score_test_split(pipeline, test_rows: pd.DataFrame, features, y_col: str, horizon: int):
    """Frame of ``cert, repdte, year, y, score`` for the rows a model is evaluated on.

    ``censored`` (the horizon's ``censored_in_window`` flag), ``rssdhcr`` and ``fail_date``
    are added when the input carries them; the sensitivity and per-event metrics need them.
    """
    out = pd.DataFrame(
        {
            "cert": test_rows["cert"].to_numpy(),
            "repdte": test_rows["repdte"].to_numpy(),
            "year": pd.to_datetime(test_rows["repdte"]).dt.year.to_numpy(),
            "y": test_rows[y_col].astype(int).to_numpy(),
            "score": score_pipeline(pipeline, test_rows[features]),
        }
    )
    censored_col = horizon_columns(horizon)[2]
    if censored_col in test_rows.columns:
        out["censored"] = test_rows[censored_col].fillna(False).astype(bool).to_numpy()
    for col in EVENT_COLUMNS:
        if col in test_rows.columns:
            out[col] = test_rows[col].to_numpy()
    return out


def _extra_metrics(scores: pd.DataFrame) -> tuple[dict, dict]:
    """Spec 5 rule 3 sensitivity (censored rows dropped) and rule 6 per-event metrics."""
    censored = (
        scores["censored"].to_numpy(dtype=bool)
        if "censored" in scores.columns
        else np.zeros(len(scores), dtype=bool)
    )
    sensitivity = _score_metrics(scores.loc[~censored])
    sensitivity["n_dropped"] = int(censored.sum())
    by_event = evaluate_by_event(scores, "score", "y")
    return sensitivity, by_event


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    return value


def save_artifacts(result: TrainResult, out_dir: Path) -> dict[str, Path]:
    """Write ``pipeline.joblib``, ``metrics.json``, ``config.json`` and ``features.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"dir": out_dir}
    paths["pipeline"] = out_dir / "pipeline.joblib"
    joblib.dump(result.pipeline, paths["pipeline"])
    payload = {
        "test": _json_ready(result.metrics),
        "by_year": _json_ready(result.by_year.to_dict(orient="records")),
        "sensitivity_censored_dropped": _json_ready(result.sensitivity),
        "per_event": _json_ready(result.by_event),
    }
    for stem, content in (
        ("metrics", payload),
        ("config", _json_ready(result.config)),
        ("features", list(result.features)),
    ):
        paths[stem] = out_dir / f"{stem}.json"
        paths[stem].write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    log.info("saved %s artefacts to %s", result.name, out_dir)
    return paths


def load_model(settings: Settings, name: str, horizon: int = PRIMARY_HORIZON):
    """``(pipeline, features, config)`` from ``models/<name>/``; ``FileNotFoundError`` if absent."""
    d = model_dir(settings, name, horizon)
    if not (d / "pipeline.joblib").exists():
        raise FileNotFoundError(f"no trained {name} model for {horizon}q under {d}")
    pipeline = joblib.load(d / "pipeline.joblib")
    features = json.loads((d / "features.json").read_text(encoding="utf-8"))
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return pipeline, features, config


def evaluate_model(
    name: str | None, horizon: int, settings: Settings, frame: pd.DataFrame | None = None
) -> dict[str, TrainResult]:
    """Re-score saved baselines on the fixed test split; ``name=None`` means every one found.

    Metrics are recomputed from the saved pipeline rather than read back from
    ``metrics.json`` so the artefact itself is what gets checked.
    """
    names = list(MODEL_NAMES) if name is None else [name]
    if frame is None:
        frame = load_training_frame(settings)
    y_col = horizon_columns(horizon)[0]
    _, test = fixed_split_masks(frame, settings, horizon)
    results: dict[str, TrainResult] = {}
    for model_name in names:
        try:
            pipeline, features, config = load_model(settings, model_name, horizon)
        except FileNotFoundError as exc:
            if name is not None:
                raise
            log.warning("%s", exc)
            continue
        scores = score_test_split(pipeline, frame.loc[test], features, y_col, horizon)
        metrics = _score_metrics(scores)
        by_year = evaluate_by_year(scores, "score", "y", "year", tie_col="cert")
        sensitivity, by_event = _extra_metrics(scores)
        results[model_name] = TrainResult(
            model_name,
            horizon,
            pipeline,
            features,
            metrics,
            by_year,
            config,
            scores,
            {"dir": model_dir(settings, model_name, horizon)},
            sensitivity=sensitivity,
            by_event=by_event,
        )
    return results


def report_path(settings: Settings, horizon: int = PRIMARY_HORIZON) -> Path:
    suffix = "" if horizon == PRIMARY_HORIZON else f"_{horizon}q"
    return Path(settings.reports_dir) / f"p1_baselines{suffix}.md"


REPORT_METRICS = ("pr_auc", "roc_auc", "recall_at_2pct", "recall_at_top100", "n", "n_failures")
SENSITIVITY_METRICS = REPORT_METRICS + ("n_dropped",)
EVENT_METRICS = (
    "pr_auc",
    "roc_auc",
    "recall_at_2pct",
    "recall_at_top100",
    "n_events",
    "n_multi_bank_events",
    "n_banks_in_multi_events",
)


def figures_dir(settings: Settings) -> Path:
    return Path(settings.reports_dir) / "figures"


def write_figures(
    results: dict[str, TrainResult], settings: Settings, horizon: int
) -> dict[str, Path]:
    """Save the report figures under ``reports/figures/`` and return them by name.

    One precision-recall curve and one score-distribution histogram per model (their
    scores live on different scales: a Texas ratio against a probability), plus one
    recall@k bar chart comparing every model. Non-primary horizons get a ``_<H>q`` suffix.
    """
    from bankcanary.evaluation import plots

    suffix = "" if horizon == PRIMARY_HORIZON else f"_{horizon}q"
    out_dir = figures_dir(settings)
    paths: dict[str, Path] = {}
    for name, r in results.items():
        y, s = r.scores["y"].to_numpy(), r.scores["score"].to_numpy()
        paths[f"pr_curve_{name}"] = plots.pr_curve(y, s, out_dir / f"pr_curve_{name}{suffix}.png")
        paths[f"score_distributions_{name}"] = plots.score_distributions(
            y, s, out_dir / f"score_distributions_{name}{suffix}.png"
        )
    paths["recall_at_k"] = plots.recall_at_k_bars(
        {n: r.metrics for n, r in results.items()}, out_dir / f"recall_at_k{suffix}.png"
    )
    log.info("wrote %d figures to %s", len(paths), out_dir)
    return paths


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _leakage_note(table: pd.DataFrame) -> str:
    """One line on whether any single feature dominates the fitted logit (spec 6.6)."""
    total = float(table["abs_coef"].sum())
    top = table.iloc[0]
    share = top["abs_coef"] / total if total > 0 else 0.0
    verdict = (
        "suspicious: check that feature for look-ahead"
        if share > 0.5
        else "no single feature dominates, consistent with no look-ahead leakage"
    )
    return (
        f"Leakage sanity check: the largest coefficient (`{top['feature']}`) carries "
        f"{share:.0%} of the total absolute coefficient mass across {len(table)} inputs; {verdict}."
    )


def _section_table(results: dict[str, TrainResult], attr: str, keys: tuple[str, ...]) -> str:
    rows = pd.DataFrame(
        [{"model": n, **{k: getattr(r, attr).get(k) for k in keys}} for n, r in results.items()]
    )
    return _md_table(rows)


def _figure_lines(figures: dict[str, Path], report_path: Path) -> list[str]:
    lines = ["", "## Figures", ""]
    for key, fig in figures.items():
        rel = os.path.relpath(Path(fig), Path(report_path).parent)
        lines.append(f"- {key.replace('_', ' ')}: [`{rel}`]({rel})")
    return lines


def write_baselines_report(
    results: dict[str, TrainResult],
    settings: Settings,
    horizon: int,
    path: Path,
    figures: dict[str, Path] | None = None,
) -> Path:
    """Write the Prototype 1 baseline comparison as markdown and return its path.

    Besides the headline test metrics the report carries the spec 5 rule 3 sensitivity
    run (censored rows dropped), the rule 6 per-failure-event view and, when
    ``figures`` is given (see :func:`write_figures`), links to the saved plots.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = next(iter(results.values()))
    c = first.config
    rows = pd.DataFrame(
        [{"model": n, **{k: r.metrics[k] for k in REPORT_METRICS}} for n, r in results.items()]
    )
    out = [
        f"# Prototype 1 baselines, {horizon}-quarter horizon",
        "",
        "Fixed out-of-time split (spec 8.2, rule 6.2 applied through `fixed_split_masks`).",
        "",
        f"- Training reports actually used: {c['train_repdte_min']} to {c['train_repdte_max']} "
        f"(nominal {c['train_start_nominal']} to {c['train_end_nominal']}; "
        f"{c['n_train']:,} rows, {c['positives_train']:,} positives).",
        f"- Test reports: {c['test_repdte_min']} to {c['test_repdte_max']} "
        f"({c['n_test']:,} rows, {c['positives_test']:,} positives); first test prediction "
        f"date {c['first_test_prediction_date']} with a {c['availability_lag_days']}-day lag.",
        "",
        "## Test-split metrics",
        "",
        _md_table(rows),
    ]
    if {"texas", "logit"} <= results.keys():
        gap = results["logit"].metrics["pr_auc"] - results["texas"].metrics["pr_auc"]
        verdict = "beats" if gap > 0 else "does not beat"
        out += [
            "",
            f"Acceptance check (spec 9): the regularised logit {verdict} the Texas ratio on "
            f"PR-AUC ({results['logit'].metrics['pr_auc']:.4f} vs "
            f"{results['texas'].metrics['pr_auc']:.4f}).",
        ]
    out += [
        "",
        "## Sensitivity: censored rows dropped (spec 5, rule 3)",
        "",
        "Test rows where the bank left the industry without failing inside the window "
        f"(`censored_in_window_{horizon}q`: merged, closed voluntarily, ...) are removed "
        "before scoring; `n_dropped` counts them. They carry `y = 0` in the main run.",
        "",
        _section_table(results, "sensitivity", SENSITIVITY_METRICS),
        "",
        "## Per failure event (spec 5, rule 6)",
        "",
        "Sister banks of one holding company (`rssdhcr`) that failed on the same day are "
        "collapsed into one event per report quarter, scored by the best-ranked sister; every "
        "other row stays one per bank. `n_events` is the number of positive units after "
        "collapsing, `n_multi_bank_events` how many of them bundle several banks.",
        "",
        _section_table(results, "by_event", EVENT_METRICS),
    ]
    if figures:
        out += _figure_lines(figures, path)
    for name, r in results.items():
        if name not in ("logit_small", "logit"):
            continue
        table = coefficient_table(r.pipeline)
        out += ["", f"## Per-year metrics, {name}", "", _md_table(r.by_year)]
        if name == "logit_small":
            out += ["", "## Odds ratios, logit_small (per training-fold standard deviation)", ""]
            out.append(_md_table(table[["feature", "coef", "odds_ratio"]]))
        else:
            out += ["", "## Top 10 |coefficient| features, logit", ""]
            out.append(_md_table(table.head(10)[["feature", "coef", "odds_ratio"]]))
            out += ["", _leakage_note(table)]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    return path
