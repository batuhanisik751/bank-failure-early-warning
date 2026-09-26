"""Walk-forward backtest: one model per test year, scored only on that year (spec 8.2).

For every test year ``Y`` from :data:`FIRST_TEST_YEAR` to the latest year whose four
report quarters all carry a complete label, a model is fitted on the rows Rule 6.2 allows
(:func:`bankcanary.splits.training_mask` at the label the model is fitted on, re-checked
with :func:`bankcanary.splits.assert_no_leakage`) and scores every usable bank-quarter
reported in ``Y``. The per-year artefacts land in ``models/walkforward/<Y>/<model>/``
(``<model>_<H>q`` away from the primary 4-quarter horizon), the per-year scores in
``data/walkforward/<Y>_<model>_<H>q.parquet``, and :func:`rebuild_scores_table` folds
those files into the ``walkforward_scores`` table (CONTRACT section 11) so that the
Parquet and DuckDB copies are always a deterministic function of the per-year files.
Hyper-parameters are re-selected for every test year on a validation slice carved from
that year's own training period (:func:`inner_masks`, :func:`tune_year`; spec rule 6.7),
so no test year ever informs the model that scores it. Each (year, model, horizon) fit
logs one run through :mod:`bankcanary.tracking`, and each tuning candidate another.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from bankcanary.config import Settings
from bankcanary.labels.build import horizon_columns
from bankcanary.splits import assert_no_leakage, test_mask, training_mask

log = logging.getLogger(__name__)

#: Walk-forward models: the Texas-ratio ranking (no fit), the regularised logit at the P1
#: ``LOGIT_C`` on every v2 feature, the tuned gradient booster and the converted hazard.
MODELS: tuple[str, ...] = ("texas", "logit", "gbdt", "gbdt_mono", "hazard")
#: The two booster configurations of Decision Point 2: unconstrained and registry-monotone.
GBDT_MODELS: tuple[str, ...] = ("gbdt", "gbdt_mono")
FIRST_TEST_YEAR = 2008
PRIMARY_HORIZON = 4
FEATURE_VERSION = "v2"
TABLE = "walkforward_scores"
TABLE_KEY: tuple[str, ...] = ("cert", "repdte", "horizon", "model", "test_year")
SCORE_COLUMNS: tuple[str, ...] = TABLE_KEY + (
    "score",
    "score_calibrated",
    "y",
    "label_complete",
    "censored",
)
QUARTERS_PER_YEAR = 4
#: Nested tuning (spec rule 6.7): each test year's hyper-parameters are chosen on the last
#: ``VALIDATION_QUARTERS`` report quarters of *its own* training period, widened backwards a
#: year at a time until both the validation slice and the inner training set hold at least
#: ``MIN_VALIDATION_POSITIVES`` failures. Nothing dated on or after the test year is seen.
VALIDATION_QUARTERS = 8
MIN_VALIDATION_POSITIVES = 5
TUNING_RUN = "tune_walkforward"
#: Inverse L2 strengths for ``logit`` and ``hazard`` (the D6/D7 grid, strongest last).
C_GRID: tuple[float, ...] = (1.0, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)
#: Booster grid around ``settings.models.gbdt.params`` (iteration count and monotone
#: decision are taken from the settings, not re-tuned per year).
GBDT_GRID: dict[str, tuple] = {
    "learning_rate": (0.03, 0.1),
    "num_leaves": (15, 63),
    "min_samples_leaf": (50, 200),
}


def latest_complete_year(labels: pd.DataFrame, horizon: int) -> int:
    """Latest calendar year whose four report quarters all have a complete ``H``-quarter label.

    A year with an incomplete quarter (its outcome window runs past the failures-list
    date) cannot be scored in full, and a partial year would mix a quiet first half with
    nothing else, so the backtest stops at the last year that is complete end to end.
    """
    complete_col = horizon_columns(horizon)[3]
    frame = labels[["repdte", complete_col]].assign(year=labels["repdte"].dt.year)
    by_year = frame.groupby("year").agg(
        n_quarters=("repdte", "nunique"), complete=(complete_col, "min")
    )
    ok = by_year[(by_year["n_quarters"] >= QUARTERS_PER_YEAR) & by_year["complete"].astype(bool)]
    if ok.empty:
        raise ValueError(f"no calendar year has a complete {horizon}q label for all four quarters")
    return int(ok.index.max())


def test_years(labels: pd.DataFrame, horizon: int, first: int = FIRST_TEST_YEAR) -> list[int]:
    """``[first, ..., latest_complete_year]`` for the horizon."""
    return list(range(first, latest_complete_year(labels, horizon) + 1))


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last report date of a test year (03-31 and 12-31)."""
    return pd.Timestamp(year=year, month=3, day=31), pd.Timestamp(year=year, month=12, day=31)


def year_masks(
    labels: pd.DataFrame, horizon: int, year: int, fit_horizon: int | None = None, lag_days=None
) -> tuple[pd.Series, pd.Series]:
    """``(train, test)`` masks for test year ``year`` at scoring horizon ``horizon``.

    Training rows come from :func:`training_mask` at ``fit_horizon`` (the label the model
    is fitted on: 1 for the hazard, ``horizon`` otherwise) with the first report quarter
    of ``year`` as the test start, then :func:`assert_no_leakage` re-checks them. Test
    rows are the usable reports dated in ``year`` at ``horizon``.
    """
    from bankcanary.splits.time_split import DEFAULT_LAG_DAYS

    lag = DEFAULT_LAG_DAYS if lag_days is None else int(lag_days)
    fit_h = horizon if fit_horizon is None else int(fit_horizon)
    start, end = year_bounds(year)
    train = training_mask(labels, fit_h, start, lag).rename(f"train_{fit_h}q_{year}")
    assert_no_leakage(labels.loc[train], fit_h, start, lag)
    test = test_mask(labels, horizon, start, end).rename(f"test_{horizon}q_{year}")
    return train, test


def inner_masks(
    frame: pd.DataFrame, horizon: int, year: int, fit_horizon: int | None = None, lag_days=None
) -> tuple[pd.Series, pd.Series, dict]:
    """``(inner_train, validation, bounds)`` nested inside test year ``year``'s training period.

    Spec rule 6.7: hyper-parameters are chosen on data the final model may train on,
    never on the test year. The validation slice is the last :data:`VALIDATION_QUARTERS`
    report quarters of :func:`training_mask` at the *scoring* horizon (so every
    validation outcome was known on the year's first prediction date); the inner training
    rows are the year's training rows at ``fit_horizon`` whose windows closed before the
    slice's first prediction date, re-checked with :func:`assert_no_leakage`. When either
    side holds fewer than :data:`MIN_VALIDATION_POSITIVES` failures the slice is widened
    backwards a year at a time; ``bounds["sufficient"]`` is ``False`` when no width works
    (the caller then falls back to a fixed default, never to a value chosen on later data).
    """
    from bankcanary.splits.time_split import DEFAULT_LAG_DAYS

    lag = DEFAULT_LAG_DAYS if lag_days is None else int(lag_days)
    fit_h = horizon if fit_horizon is None else int(fit_horizon)
    start, _ = year_bounds(year)
    outer = training_mask(frame, horizon, start, lag)
    outer_fit = training_mask(frame, fit_h, start, lag)
    quarters = sorted(pd.Timestamp(q) for q in frame.loc[outer, "repdte"].unique())
    if not quarters:
        raise ValueError(f"walk-forward {horizon}q {year}: no training rows to tune on")
    y_valid, y_fit = horizon_columns(horizon)[0], horizon_columns(fit_h)[0]
    vend = quarters[-1]
    n = min(VALIDATION_QUARTERS, len(quarters))
    first = None
    while True:
        vstart = quarters[-n]
        validation = test_mask(frame, horizon, vstart, vend) & outer
        inner = training_mask(frame, fit_h, vstart, lag) & outer_fit
        bounds = {
            "validation_start": _fmt(vstart),
            "validation_end": _fmt(vend),
            "validation_quarters": int(n),
            "inner_train_repdte_max": _fmt(frame.loc[inner, "repdte"].max()),
            "n_inner_train": int(inner.sum()),
            "positives_inner_train": int(frame.loc[inner, y_fit].sum()),
            "n_validation": int(validation.sum()),
            "positives_validation": int(frame.loc[validation, y_valid].sum()),
        }
        enough = MIN_VALIDATION_POSITIVES
        bounds["sufficient"] = (
            bounds["positives_validation"] >= enough and bounds["positives_inner_train"] >= enough
        )
        if first is None:
            first = (inner, validation, bounds)
        if bounds["sufficient"] or n + QUARTERS_PER_YEAR > len(quarters):
            break
        n += QUARTERS_PER_YEAR
    if not bounds["sufficient"]:
        inner, validation, bounds = first
    if inner.any():
        assert_no_leakage(frame.loc[inner], fit_h, bounds["validation_start"], lag)
    names = (f"inner_train_{fit_h}q_{year}", f"validation_{horizon}q_{year}")
    return inner.rename(names[0]), validation.rename(names[1]), bounds


def fit_horizon_of(model: str, horizon: int) -> int:
    """The label a model is fitted on: the hazard always learns the one-quarter event."""
    from bankcanary.models.hazard import HAZARD_HORIZON

    return HAZARD_HORIZON if model == "hazard" else int(horizon)


def candidate_grid(model: str, settings: Settings) -> list[dict]:
    """The hyper-parameter candidates :func:`tune_year` ranks for ``model`` (empty for texas).

    ``logit`` and ``hazard`` vary ``C`` over :data:`C_GRID`; ``gbdt`` varies the
    :data:`GBDT_GRID` axes with every other parameter from ``settings.models.gbdt``.
    """
    import itertools

    if model in ("logit", "hazard"):
        return [{"C": float(c)} for c in C_GRID]
    if model in GBDT_MODELS:
        base = dict(settings.models.gbdt.params)
        out = []
        for values in itertools.product(*GBDT_GRID.values()):
            out.append({**base, **dict(zip(GBDT_GRID.keys(), values, strict=True))})
        return out
    return []


def fallback_params(model: str, settings: Settings) -> dict:
    """The most regularised grid point: used only when a year has too few failures to tune."""
    grid = candidate_grid(model, settings)
    if model in ("logit", "hazard"):
        return {"C": min(C_GRID)}
    if model in GBDT_MODELS:
        return {
            **grid[0],
            "learning_rate": min(GBDT_GRID["learning_rate"]),
            "num_leaves": min(GBDT_GRID["num_leaves"]),
            "min_samples_leaf": max(GBDT_GRID["min_samples_leaf"]),
        }
    return {}


def build_model(
    model: str, settings: Settings, n_estimators: int | None = None, params: dict | None = None
):
    """``(unfitted pipeline, feature list, config)`` for one walk-forward model.

    ``texas`` ranks by the Texas ratio alone; ``logit`` is the Prototype 1 regularised
    logit (no class weighting) on every v2 feature; ``gbdt`` takes its backend from
    ``settings.models.gbdt`` and is unconstrained, ``gbdt_mono`` the same booster under
    the registry's monotone signs (the two sides of Decision Point 2, both walked
    forward; ``n_estimators`` overrides the iteration count when a year's training set
    needs capping); ``hazard`` is the unweighted 1q
    logit whose output is converted with ``1 - (1 - h)^H``. ``params`` are the year's
    tuned hyper-parameters from :func:`tune_year` (``C`` for the logits, the booster
    parameters for ``gbdt``); without them the fixed-split constants (``LOGIT_C``,
    ``HAZARD_C``, ``settings.models.gbdt.params``) apply, which is only leak-free for
    test years from 2010 on and is therefore never what :func:`fit_year` does.
    """
    params = dict(params or {})
    from bankcanary.features.registry import feature_names
    from bankcanary.models import baselines, gbdt, hazard

    features = feature_names(version=FEATURE_VERSION)
    config: dict = {"model": model, "features_version": FEATURE_VERSION}
    if model == "texas":
        config.update({"features_version": "v1", "fit": "none (ranking by texas_ratio)"})
        return baselines.make_model("texas"), ["texas_ratio"], config
    if model == "logit":
        from sklearn.linear_model import LogisticRegression

        from bankcanary.models.preprocess import make_pipeline

        c = float(params.get("C", baselines.LOGIT_C))
        pipe = make_pipeline(LogisticRegression(C=c, max_iter=2000))
        config.update({"C": c, "class_weight": None})
        return pipe, features, config
    if model in GBDT_MODELS:
        cfg = settings.models.gbdt
        monotone = model == "gbdt_mono"
        params = {**cfg.params, **params}
        if n_estimators is not None:
            params["n_estimators"] = int(n_estimators)
        pipe = gbdt.make_gbdt(cfg.backend, monotone, features, **params)
        config.update(
            {
                "backend": gbdt.resolve_backend(cfg.backend),
                "monotone": monotone,
                "params": params,
                "iterations_capped": n_estimators is not None
                and int(n_estimators) < int(cfg.params.get("n_estimators", n_estimators)),
            }
        )
        return pipe, features, config
    if model == "hazard":
        c = float(params.get("C", hazard.HAZARD_C))
        config.update({"C": c, "class_weight": None, "conversion": "1 - (1 - h)^H"})
        return hazard.make_hazard(c), features, config
    raise ValueError(f"unknown walk-forward model {model!r}; choose one of {MODELS}")


def tune_year(
    frame: pd.DataFrame,
    settings: Settings,
    year: int,
    model: str,
    horizon: int = PRIMARY_HORIZON,
    n_estimators: int | None = None,
) -> dict:
    """Choose ``model``'s hyper-parameters for test year ``year`` on its nested slice.

    Every candidate from :func:`candidate_grid` is fitted on the inner training rows of
    :func:`inner_masks` and scored on the validation rows at the scoring horizon (the
    hazard's 1q output is converted first); the winner is the highest validation PR-AUC,
    the first in grid order on ties. Each candidate logs a ``runs/tune_walkforward/``
    record and a re-run reads the cached metrics instead of refitting. The returned dict
    carries the slice bounds, the scored ``grid`` and the ``selected`` parameters; when the
    slice is too thin (``sufficient`` False) ``selected`` is :func:`fallback_params` and
    ``fallback`` says why. ``texas`` has nothing to tune and returns an empty selection.
    """
    from bankcanary import tracking
    from bankcanary.evaluation.metrics import evaluate
    from bankcanary.models.baselines import score_pipeline
    from bankcanary.models.hazard import convert_hazard

    horizon, year = int(horizon), int(year)
    grid = candidate_grid(model, settings)
    if not grid:
        return {"selected": {}, "grid": [], "metric": None, "fallback": "nothing to tune"}
    fit_h = fit_horizon_of(model, horizon)
    inner, validation, bounds = inner_masks(
        frame, horizon, year, fit_h, settings.availability_lag_days
    )
    if not bounds["sufficient"]:
        reason = (
            f"fewer than {MIN_VALIDATION_POSITIVES} failures in the validation slice or the "
            "inner training rows: most regularised grid point used"
        )
        log.warning("walk-forward %dq %d %s tuning: %s", horizon, year, model, reason)
        selected = fallback_params(model, settings)
        return {**bounds, "selected": selected, "grid": [], "metric": None, "fallback": reason}
    y_fit, y_val = horizon_columns(fit_h)[0], horizon_columns(horizon)[0]
    tr, va = frame.loc[inner], frame.loc[validation]
    base = {
        "model": model,
        "horizon": horizon,
        "fit_horizon": fit_h,
        "test_year": year,
        "features_version": FEATURE_VERSION,
        "availability_lag_days": int(settings.availability_lag_days),
        **{k: bounds[k] for k in ("validation_start", "validation_end", "n_inner_train")},
    }
    rows = []
    for params in grid:
        config = {**base, "params": params}
        metrics = tracking.find_metrics(TUNING_RUN, config, settings)
        if metrics is None:
            pipe, features, _ = build_model(model, settings, n_estimators, params)
            pipe.fit(tr[list(features)], tr[y_fit].astype(int).to_numpy())
            score = score_pipeline(pipe, va[list(features)])
            if model == "hazard":
                score = convert_hazard(score, horizon)
            y = va[y_val].astype(int).to_numpy()
            metrics = evaluate(y, score, tie_breaker=va["cert"].to_numpy())
            run = tracking.start_run(TUNING_RUN, config, settings)
            run.log_metrics(metrics)
            run.finish()
        keep = ("pr_auc", "roc_auc", "recall_at_2pct", "recall_at_top100")
        rows.append({"params": params, **{k: metrics.get(k) for k in keep}})
    ranked = sorted(
        range(len(rows)),
        key=lambda i: (-(rows[i]["pr_auc"] if pd.notna(rows[i]["pr_auc"]) else -np.inf), i),
    )
    selected = dict(rows[ranked[0]]["params"])
    log.info("walk-forward %dq %d %s tuning: selected %s", horizon, year, model, selected)
    return {**bounds, "selected": selected, "grid": rows, "metric": "pr_auc", "fallback": None}


def model_dir(settings: Settings, year: int, model: str, horizon: int = PRIMARY_HORIZON) -> Path:
    """``models/walkforward/<Y>/<model>/`` (``<model>_<H>q`` away from the 4q horizon)."""
    suffix = "" if int(horizon) == PRIMARY_HORIZON else f"_{int(horizon)}q"
    return Path(settings.models_dir) / "walkforward" / str(int(year)) / f"{model}{suffix}"


def scores_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "walkforward"


def scores_path(settings: Settings, year: int, model: str, horizon: int) -> Path:
    """``data/walkforward/<Y>_<model>_<H>q.parquet``: one test year of one model."""
    return scores_dir(settings) / f"{int(year)}_{model}_{int(horizon)}q.parquet"


@dataclass
class YearResult:
    """One (year, model, horizon) fit: the pipeline, its scores and the run record."""

    year: int
    model: str
    horizon: int
    pipeline: Pipeline
    features: list[str]
    config: dict
    metrics: dict
    scores: pd.DataFrame = field(repr=False)
    paths: dict[str, Path] = field(default_factory=dict)
    tuning: dict = field(default_factory=dict, repr=False)


def _fmt(ts) -> str | None:
    return None if pd.isna(ts) else str(pd.Timestamp(ts).date())


def score_rows(pipeline, rows: pd.DataFrame, features, model: str, horizon: int, year: int):
    """The CONTRACT ``walkforward_scores`` columns for the test rows of one year.

    The hazard's quarterly probability is converted to the scoring horizon first;
    ``score_calibrated`` is left ``NaN`` for the calibration step.
    """
    from bankcanary.models.baselines import score_pipeline
    from bankcanary.models.hazard import convert_hazard

    y_col, _, censored_col, complete_col = horizon_columns(horizon)
    score = score_pipeline(pipeline, rows[list(features)])
    if model == "hazard":
        score = convert_hazard(score, horizon)
    return pd.DataFrame(
        {
            "cert": rows["cert"].to_numpy(),
            "repdte": rows["repdte"].to_numpy(),
            "horizon": np.full(len(rows), int(horizon), dtype="int64"),
            "model": np.full(len(rows), model, dtype=object),
            "test_year": np.full(len(rows), int(year), dtype="int64"),
            "score": np.asarray(score, dtype=float),
            "score_calibrated": np.full(len(rows), np.nan),
            "y": rows[y_col].astype(int).to_numpy(),
            "label_complete": rows[complete_col].fillna(False).astype(bool).to_numpy(),
            "censored": rows[censored_col].fillna(False).astype(bool).to_numpy(),
        }
    )


def year_metrics(scores: pd.DataFrame) -> dict:
    """P1 ranking metrics plus the Brier score when the score is a probability.

    The Texas ratio is a ranking, not a probability, so its Brier score is ``None``.
    """
    from bankcanary.evaluation.metrics import evaluate
    from bankcanary.models.hazard import brier

    y, s = scores["y"].to_numpy(), scores["score"].to_numpy(dtype=float)
    out = evaluate(y, s, tie_breaker=scores["cert"].to_numpy())
    is_probability = len(s) > 0 and np.isfinite(s).all() and s.min() >= 0.0 and s.max() <= 1.0
    texas = (scores["model"] == "texas").all() if "model" in scores.columns else False
    out["brier"] = brier(y, s) if is_probability and not texas else None
    return out


def fit_year(
    frame: pd.DataFrame,
    settings: Settings,
    year: int,
    model: str,
    horizon: int = PRIMARY_HORIZON,
    n_estimators: int | None = None,
    save: bool = True,
) -> YearResult:
    """Tune and fit ``model`` for test year ``year``, score the year, save artefacts, log a run.

    ``frame`` is ``features_v2`` joined with ``labels`` (see
    :func:`bankcanary.models.gbdt.load_training_frame`). Hyper-parameters come from
    :func:`tune_year` on the year's own nested slice (rule 6.7) and are recorded under
    ``config["tuning"]``. Training rows are selected by
    :func:`year_masks` at the model's own fit label and re-checked for leakage; the
    Texas ranking has nothing to learn but is "fitted" on the same rows so that every
    model loads and scores the same way. With ``save`` the pipeline, config and feature
    list go to :func:`model_dir` and the scores to :func:`scores_path`.
    """
    from bankcanary import tracking

    horizon, year = int(horizon), int(year)
    fit_h = fit_horizon_of(model, horizon)
    tuning = tune_year(frame, settings, year, model, horizon, n_estimators)
    pipeline, features, config = build_model(model, settings, n_estimators, tuning["selected"])
    y_fit = horizon_columns(fit_h)[0]
    missing = [c for c in list(features) + [y_fit] if c not in frame.columns]
    if missing:
        raise KeyError(f"training frame is missing column(s) {missing}")
    lag = settings.availability_lag_days
    train, test = year_masks(frame, horizon, year, fit_h, lag)
    if not train.any() or not test.any():
        raise ValueError(f"walk-forward {horizon}q {year}: no train or no test rows")
    tr = frame.loc[train]
    y_train = tr[y_fit].astype(int).to_numpy()
    log.info(
        "walk-forward %dq %d: fitting %s on %d rows (%d positives, last report %s)",
        horizon,
        year,
        model,
        len(tr),
        int(y_train.sum()),
        _fmt(tr["repdte"].max()),
    )
    pipeline.fit(tr[list(features)], y_train)
    scores = score_rows(pipeline, frame.loc[test], features, model, horizon, year)
    metrics = year_metrics(scores)
    start, _ = year_bounds(year)
    from bankcanary.splits import prediction_date

    config.update(
        {
            "horizon": horizon,
            "fit_horizon": fit_h,
            "fit_label": y_fit,
            "test_year": year,
            "availability_lag_days": int(lag),
            "first_test_prediction_date": str(prediction_date(start, lag).date()),
            "train_repdte_min": _fmt(tr["repdte"].min()),
            "train_repdte_max": _fmt(tr["repdte"].max()),
            "n_train": int(len(tr)),
            "positives_train": int(y_train.sum()),
            "n_test": int(len(scores)),
            "positives_test": int(scores["y"].sum()),
            "n_features": len(features),
            "tuning": {k: v for k, v in tuning.items() if k != "grid"},
        }
    )
    result = YearResult(
        year, model, horizon, pipeline, list(features), config, metrics, scores, tuning=tuning
    )
    log.info(
        "walk-forward %dq %d %s: pr_auc %s roc_auc %s",
        horizon,
        year,
        model,
        _round(metrics["pr_auc"]),
        _round(metrics["roc_auc"]),
    )
    if save:
        result.paths = save_year(result, settings)
    run = tracking.start_run("walkforward", config, settings)
    run.log_metrics(metrics)
    run.finish()
    return result


def _round(value) -> str:
    return "nan" if value is None or pd.isna(value) else f"{float(value):.4f}"


def save_year(result: YearResult, settings: Settings) -> dict[str, Path]:
    """Write ``pipeline.joblib``, the config/features/metrics/tuning JSON files and the scores."""
    out = model_dir(settings, result.year, result.model, result.horizon)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"dir": out, "pipeline": out / "pipeline.joblib"}
    joblib.dump(result.pipeline, paths["pipeline"])
    for stem, content in (
        ("config", _json_ready(result.config)),
        ("features", list(result.features)),
        ("metrics", _json_ready(result.metrics)),
        ("tuning", _json_ready(result.tuning)),
    ):
        paths[stem] = out / f"{stem}.json"
        paths[stem].write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    paths["scores"] = scores_path(settings, result.year, result.model, result.horizon)
    paths["scores"].parent.mkdir(parents=True, exist_ok=True)
    result.scores.loc[:, list(SCORE_COLUMNS)].to_parquet(paths["scores"], index=False)
    log.info("saved walk-forward %s %d artefacts to %s", result.model, result.year, out)
    return paths


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_year(settings: Settings, year: int, model: str, horizon: int = PRIMARY_HORIZON):
    """``(pipeline, features, config)`` written by :func:`fit_year`; raises when absent."""
    d = model_dir(settings, year, model, horizon)
    if not (d / "pipeline.joblib").exists():
        raise FileNotFoundError(f"no walk-forward {model} model for {year} at {horizon}q under {d}")
    pipeline = joblib.load(d / "pipeline.joblib")
    features = json.loads((d / "features.json").read_text(encoding="utf-8"))
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return pipeline, features, config


def rebuild_scores_table(settings: Settings) -> pd.DataFrame:
    """Fold every ``data/walkforward/*.parquet`` into ``walkforward_scores`` (Parquet + DuckDB).

    The files are read in sorted name order and :func:`write_table` sorts the rows by the
    table key, so the table is a deterministic function of the per-year files. Any
    earlier rows for a (year, model, horizon) are replaced by the new file, never merged.
    """
    from bankcanary.storage.duckdb import replace_table
    from bankcanary.storage.parquet import read_table, table_path, write_table

    files = sorted(scores_dir(settings).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no walk-forward score files under {scores_dir(settings)}")
    parts = [pd.read_parquet(f)[list(SCORE_COLUMNS)] for f in files]
    table = pd.concat(parts, ignore_index=True)
    table = table.astype({"repdte": "datetime64[ns]", "model": "str"})
    write_table(table, TABLE, key=TABLE_KEY, settings=settings)
    replace_table(TABLE, table_path(settings, TABLE), settings)
    log.info("walkforward_scores rebuilt from %d files: %d rows", len(files), len(table))
    return read_table(TABLE, settings)


REPORT_METRICS: tuple[str, ...] = (
    "pr_auc",
    "roc_auc",
    "recall_at_2pct",
    "recall_at_top100",
)
POOLED_METRICS: tuple[str, ...] = REPORT_METRICS + ("brier",)


def read_scores_table(settings: Settings, horizon: int | None = None) -> pd.DataFrame:
    """The ``walkforward_scores`` table, optionally restricted to one horizon."""
    from bankcanary.storage.parquet import read_table

    table = read_table(TABLE, settings)
    if horizon is not None:
        table = table[table["horizon"] == int(horizon)]
    return table.reset_index(drop=True)


def per_year_tables(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{model: per-year metrics + pooled row}`` from a one-horizon slice of the table.

    The pooled row scores all test years as one ranking (spec 8.2 "aggregate across
    years"); a year with no failure shows ``NaN`` metrics and is flagged in the report.
    """
    from bankcanary.evaluation.metrics import evaluate_by_year
    from bankcanary.models.hazard import brier

    out: dict[str, pd.DataFrame] = {}
    for model in [m for m in MODELS if m in set(table["model"])]:
        sub = table[table["model"] == model]
        years = evaluate_by_year(sub, "score", "y", "test_year", tie_col="cert")
        pooled_brier = year_metrics(sub)["brier"]
        years["brier"] = np.where(years["year"] == "pooled", pooled_brier, np.nan)
        if pooled_brier is not None:
            for year in years.loc[years["year"] != "pooled", "year"]:
                rows = sub[sub["test_year"] == int(year)]
                years.loc[years["year"] == year, "brier"] = brier(rows["y"], rows["score"])
        out[model] = years
    return out


def pooled_table(per_year: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per model with its pooled metrics, best PR-AUC first."""
    rows = []
    for model, years in per_year.items():
        pooled = years[years["year"] == "pooled"].iloc[0]
        keys = ("n", "n_failures") + POOLED_METRICS
        rows.append({"model": model, **{k: pooled[k] for k in keys}})
    table = pd.DataFrame(rows)
    return table.sort_values("pr_auc", ascending=False, kind="mergesort").reset_index(drop=True)


def p1_fixed_split_reference(settings: Settings) -> dict | None:
    """Prototype 1's regularised logit test metrics (``models/logit/metrics.json``), if saved."""
    path = Path(settings.models_dir) / "logit" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("test")


def report_path(settings: Settings) -> Path:
    return Path(settings.reports_dir) / "walkforward.md"


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, float | np.floating):
        return f"{float(value):.4f}"
    return str(value)


def _md_table(df: pd.DataFrame, columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
    body = ["| " + " | ".join(_cell(row[c]) for c in columns) + " |" for _, row in df.iterrows()]
    return "\n".join([head, *body])


def _year_section(model: str, years: pd.DataFrame) -> list[str]:
    cols = ["year", "n", "n_failures", *REPORT_METRICS, "brier"]
    lines = [f"### {model}", "", _md_table(years, cols)]
    empty = years.loc[(years["year"] != "pooled") & (years["n_failures"] == 0), "year"].tolist()
    if empty:
        lines.append("")
        lines.append(
            f"Years with no failure ({', '.join(empty)}) have undefined ranking metrics; "
            "they still count in the pooled row through their non-failing banks."
        )
    return lines + [""]


def _comparison_lines(pooled: pd.DataFrame, reference: dict | None, horizon: int) -> list[str]:
    best = pooled.iloc[0]
    lines = [
        "## Pooled comparison",
        "",
        _md_table(pooled, ["model", "n", "n_failures", *POOLED_METRICS]),
        "",
        f"Best pooled PR-AUC at {horizon}q: **{best['model']}** ({_cell(best['pr_auc'])}, "
        f"recall@2% {_cell(best['recall_at_2pct'])}).",
        "",
        "## Against Prototype 1's regularised logit",
        "",
    ]
    wf = pooled[pooled["model"] == "logit"]
    if reference is None:
        lines.append("`models/logit/metrics.json` is not available; fixed-split reference omitted.")
        return lines
    ref_line = (
        f"- Fixed split (P1 `models/logit`, test 2010-2013, v1 features): PR-AUC "
        f"{_cell(reference.get('pr_auc'))}, recall@2% {_cell(reference.get('recall_at_2pct'))}, "
        f"{reference.get('n_failures')} failures among {reference.get('n')} rows. Not "
        "comparable with the pooled rows: a different test period and one training window."
    )
    lines.append(ref_line)
    if not wf.empty:
        row = wf.iloc[0]
        lines.append(
            f"- Walk-forward pooled logit (same C, v2 features, one model per year): PR-AUC "
            f"{_cell(row['pr_auc'])}, recall@2% {_cell(row['recall_at_2pct'])}. This is the "
            "fair comparison for the other walk-forward rows."
        )
        beats = (best["pr_auc"] > row["pr_auc"]) and (
            best["recall_at_2pct"] > row["recall_at_2pct"]
        )
        if best["model"] == "logit":
            lines.append("- No other model beats the walk-forward logit on pooled PR-AUC.")
        else:
            verdict = "beats" if beats else "does not beat"
            lines.append(
                f"- `{best['model']}` {verdict} the walk-forward logit on both pooled PR-AUC "
                "and recall@2%."
            )
    return lines


def write_walkforward_report(
    settings: Settings, horizon: int = PRIMARY_HORIZON, path: Path | None = None
) -> Path:
    """Render ``reports/walkforward.md`` from the ``walkforward_scores`` table.

    The requested horizon gets the full treatment (per-year table per model, pooled
    comparison, Prototype 1 reference); every other horizon present in the table gets
    its pooled comparison and per-year tables in a shorter section after it.
    """
    table = read_scores_table(settings)
    if table.empty:
        raise ValueError("walkforward_scores is empty; run `bankcanary walkforward` first")
    horizons = sorted(int(h) for h in table["horizon"].unique())
    if int(horizon) not in horizons:
        raise ValueError(f"no walk-forward scores at {horizon}q (have {horizons})")
    lag = settings.availability_lag_days
    lines = [
        "# Walk-forward backtest",
        "",
        "One model per test year: for test year Y the training rows are every earlier "
        f"bank-quarter whose outcome window closed before the first prediction date of Y "
        f"(report 03-31 of Y plus the {lag}-day availability lag, spec rule 6.2), and the "
        "test rows are the label-complete reports dated in Y. Scores of all years are pooled "
        "into one ranking for the pooled rows. `texas` ranks by the Texas ratio without a "
        "fit; `logit` is the Prototype 1 regularised logit refitted on the v2 features; "
        "`gbdt` is the unconstrained gradient booster and `gbdt_mono` the same booster "
        "under the registry's monotone signs (backend from `config/settings.yaml`); "
        "`hazard` is the one-quarter hazard converted with "
        "`1 - (1 - h)^H`. Every model's hyper-parameters (`C` for the logits; learning "
        "rate, leaves and leaf size for the booster) are re-selected for each test year on "
        f"a nested validation slice: the last {VALIDATION_QUARTERS} report quarters of that "
        "year's own training period, widened backwards a year at a time while either it or "
        f"the inner training rows hold fewer than {MIN_VALIDATION_POSITIVES} failures, with "
        "inner models trained on windows closed before the slice (spec rule 6.7). The "
        "chosen values are in `models/walkforward/<Y>/<model>/tuning.json`. Metrics per "
        "year use the year's own ranking; `brier` is reported only for probability outputs.",
        "",
    ]
    ordered = [int(horizon)] + [h for h in horizons if h != int(horizon)]
    for h in ordered:
        per_year = per_year_tables(table[table["horizon"] == h])
        pooled = pooled_table(per_year)
        years = sorted(int(y) for y in table.loc[table["horizon"] == h, "test_year"].unique())
        lines += [f"## Horizon {h}q (test years {years[0]}-{years[-1]})", ""]
        if h == int(horizon):
            lines += _comparison_lines(pooled, p1_fixed_split_reference(settings), h)
            lines += ["", "## Per-year results", ""]
        else:
            lines += [
                "### Pooled",
                "",
                _md_table(pooled, ["model", "n", "n_failures", *POOLED_METRICS]),
                "",
            ]
        for model, frame in per_year.items():
            lines += _year_section(model if h == int(horizon) else f"{model} ({h}q)", frame)
    out = path or report_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    log.info("wrote %s", out)
    return out


def check_tuning_consistency(models_dir: Path, lag_days: int | None = None) -> list[dict]:
    """Every ``<models_dir>/<Y>/<model>/config.json`` whose tuning could have seen year Y.

    A fitted model's config must carry ``tuning`` (rule 6.7) whose ``validation_end`` lies
    before the year's first prediction date (``03-31`` of Y plus ``lag_days``). Configs that
    record no fit (``fit`` starting with ``none``, the Texas ranking) are skipped. Returns one
    ``{"year", "model", "path", "reason"}`` per violation; an empty list means consistent.
    """
    from bankcanary.splits import prediction_date
    from bankcanary.splits.time_split import DEFAULT_LAG_DAYS

    lag = DEFAULT_LAG_DAYS if lag_days is None else int(lag_days)
    violations: list[dict] = []
    for path in sorted(Path(models_dir).glob("*/*/config.json")):
        year_dir = path.parent.parent
        if not year_dir.name.isdigit():
            continue
        year, model = int(year_dir.name), path.parent.name
        config = json.loads(path.read_text(encoding="utf-8"))
        if str(config.get("fit", "")).startswith("none"):
            continue
        first = prediction_date(year_bounds(year)[0], lag)
        tuning = config.get("tuning")
        if not isinstance(tuning, dict):
            reason = "no tuning record"
        elif tuning.get("validation_end") is None:
            reason = "tuning has no validation_end"
        elif pd.Timestamp(tuning["validation_end"]) >= first:
            reason = f"validation_end {tuning['validation_end']} >= first prediction {first.date()}"
        else:
            continue
        violations.append({"year": year, "model": model, "path": str(path), "reason": reason})
    return violations
