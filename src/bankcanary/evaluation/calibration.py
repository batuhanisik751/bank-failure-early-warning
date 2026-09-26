"""Isotonic calibration of the walk-forward scores (spec 8.3, CONTRACT section 13).

A ranking model's raw output is not a failure probability a supervisor can quote: the
logit is fitted without class weights on a 0.5 percent base rate and the booster's scale
drifts from year to year. For every walk-forward test year ``Y`` and probability model
the calibration map is learned strictly inside ``Y``'s training window so that no test
outcome shapes it: the *calibration slice* is the last complete label year inside
:func:`bankcanary.splits.training_mask` at the scoring horizon (widened backwards a
year at a time while it holds fewer than :data:`MIN_CALIBRATION_POSITIVES` failures),
an *inner model* with the year's tuned configuration is fitted on the rows whose
outcome windows closed before the slice's first prediction date and scores the slice,
and ``IsotonicRegression(out_of_bounds="clip")`` is fitted on those (score, outcome)
pairs. The map is then applied to the full-window model's test-year scores and stored
as ``score_calibrated`` in ``data/walkforward/<Y>_<model>_<H>q.parquet``, from which
:func:`bankcanary.evaluation.walkforward.rebuild_scores_table` rebuilds the
``walkforward_scores`` table. Isotonic regression is monotone, so the ranking metrics
are unchanged up to ties; what changes is the Brier score and the reliability curve.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from bankcanary.config import Settings
from bankcanary.evaluation import walkforward as w
from bankcanary.labels.build import horizon_columns
from bankcanary.splits import assert_no_leakage, prediction_date, training_mask

log = logging.getLogger(__name__)

#: Models whose output is a probability; the Texas ratio is a ranking and is not calibrated.
MODELS: tuple[str, ...] = ("logit", "gbdt", "hazard")
RUN_NAME = "calibrate"
METHOD = "isotonic"
#: A slice with fewer failures than this cannot pin down a monotone map; widen it.
MIN_CALIBRATION_POSITIVES = 5


def _fmt(ts) -> str | None:
    return None if pd.isna(ts) else str(pd.Timestamp(ts).date())


def calibration_masks(
    frame: pd.DataFrame, horizon: int, year: int, fit_horizon: int | None = None, lag_days=None
) -> tuple[pd.Series, pd.Series, dict]:
    """``(inner_train, calibration_slice, bounds)`` nested inside test year ``year``.

    The slice is the last calendar year whose four report quarters all lie in the
    year's :func:`training_mask` at the scoring horizon, so every slice outcome was
    known on the year's first prediction date. The inner training rows are the year's
    training rows at ``fit_horizon`` whose windows closed before the slice's first
    prediction date (re-checked with :func:`assert_no_leakage`). While the slice or the
    inner rows hold fewer than :data:`MIN_CALIBRATION_POSITIVES` failures the slice is
    widened backwards a complete year at a time; ``bounds["sufficient"]`` is ``False``
    when no width works and the narrowest slice is returned for the caller to flag.
    """
    from bankcanary.splits.time_split import DEFAULT_LAG_DAYS

    lag = DEFAULT_LAG_DAYS if lag_days is None else int(lag_days)
    fit_h = horizon if fit_horizon is None else int(fit_horizon)
    start, _ = w.year_bounds(year)
    outer = training_mask(frame, horizon, start, lag)
    outer_fit = training_mask(frame, fit_h, start, lag)
    years = frame.loc[outer, "repdte"].dt.year
    complete = sorted(
        int(y)
        for y, q in frame.loc[outer, "repdte"].groupby(years).nunique().items()
        if q >= w.QUARTERS_PER_YEAR
    )
    if not complete:
        raise ValueError(f"walk-forward {horizon}q {year}: no complete year to calibrate on")
    y_slice, y_fit = horizon_columns(horizon)[0], horizon_columns(fit_h)[0]
    first = None
    n = 1
    while True:
        chosen = complete[-n:]
        cstart = pd.Timestamp(year=chosen[0], month=3, day=31)
        sl = outer & years.reindex(frame.index).isin(chosen).fillna(False).astype(bool)
        inner = training_mask(frame, fit_h, cstart, lag) & outer_fit
        bounds = {
            "calibration_years": [int(c) for c in chosen],
            "calibration_start": _fmt(cstart),
            "calibration_end": _fmt(frame.loc[sl, "repdte"].max()),
            "calibration_prediction_date": str(prediction_date(cstart, lag).date()),
            "inner_train_repdte_max": _fmt(frame.loc[inner, "repdte"].max()),
            "n_inner_train": int(inner.sum()),
            "positives_inner_train": int(frame.loc[inner, y_fit].sum()),
            "n_calibration": int(sl.sum()),
            "positives_calibration": int(frame.loc[sl, y_slice].sum()),
        }
        enough = MIN_CALIBRATION_POSITIVES
        bounds["sufficient"] = (
            bounds["positives_calibration"] >= enough and bounds["positives_inner_train"] >= enough
        )
        if first is None:
            first = (inner, sl, bounds)
        if bounds["sufficient"] or n >= len(complete):
            break
        n += 1
    if not bounds["sufficient"]:
        inner, sl, bounds = first
    if inner.any():
        assert_no_leakage(frame.loc[inner], fit_h, bounds["calibration_start"], lag)
    names = (f"inner_train_{fit_h}q_{year}", f"calibration_{horizon}q_{year}")
    return inner.rename(names[0]), sl.rename(names[1]), bounds


@dataclass
class CalibrationResult:
    """One (year, model, horizon) calibration: the map, its provenance and the rescored year."""

    year: int
    model: str
    horizon: int
    calibrator: IsotonicRegression
    config: dict
    metrics: dict
    scores: pd.DataFrame = field(repr=False)
    paths: dict[str, Path] = field(default_factory=dict)


def model_params(config: dict) -> dict:
    """The tuned hyper-parameters recorded in a walk-forward ``config.json``.

    ``C`` for the logits (the hazard included); the booster's full parameter set,
    iteration cap included, for ``gbdt``; nothing for the Texas ranking.
    """
    if config.get("model") == "gbdt":
        return dict(config.get("params", {}))
    return {"C": float(config["C"])} if "C" in config else {}


def calibration_path(settings: Settings, year: int, model: str, horizon: int) -> Path:
    """``models/walkforward/<Y>/<model>[_<H>q]/calibration.joblib``."""
    return w.model_dir(settings, year, model, horizon) / "calibration.joblib"


def fit_calibrator(
    frame: pd.DataFrame, settings: Settings, year: int, model: str, horizon: int = 4, save=True
) -> CalibrationResult:
    """Learn the isotonic map for one walk-forward year and apply it to that year's scores.

    Requires the full-window model saved by :func:`bankcanary.evaluation.walkforward.fit_year`
    (its tuned hyper-parameters are reused for the inner model) and its per-year score
    file. With ``save`` the calibrator goes next to the model as ``calibration.joblib``
    plus ``calibration.json`` and the score file's ``score_calibrated`` column is filled
    in place; the caller rebuilds ``walkforward_scores`` afterwards. Logs a ``calibrate``
    run keyed by the slice bounds and the model configuration.
    """
    from bankcanary import tracking
    from bankcanary.evaluation.metrics import brier, evaluate
    from bankcanary.models.baselines import score_pipeline
    from bankcanary.models.hazard import convert_hazard

    horizon, year = int(horizon), int(year)
    if model not in MODELS:
        raise ValueError(f"{model!r} is not a probability model; calibrate one of {MODELS}")
    _, features, full_config = w.load_year(settings, year, model, horizon)
    score_file = w.scores_path(settings, year, model, horizon)
    if not score_file.exists():
        raise FileNotFoundError(f"no walk-forward scores for {model} {year} at {score_file}")
    fit_h = w.fit_horizon_of(model, horizon)
    lag = settings.availability_lag_days
    inner, sl, bounds = calibration_masks(frame, horizon, year, fit_h, lag)
    if not inner.any() or not sl.any():
        raise ValueError(f"calibration {horizon}q {year} {model}: empty inner or slice rows")
    params = model_params(full_config)
    pipeline, _, config = w.build_model(model, settings, None, params)
    y_fit, y_slice = horizon_columns(fit_h)[0], horizon_columns(horizon)[0]
    tr, cal = frame.loc[inner], frame.loc[sl]
    log.info(
        "calibration %dq %d %s: inner fit on %d rows (%d positives), slice %s-%s (%d positives)",
        horizon,
        year,
        model,
        len(tr),
        bounds["positives_inner_train"],
        bounds["calibration_start"],
        bounds["calibration_end"],
        bounds["positives_calibration"],
    )
    pipeline.fit(tr[list(features)], tr[y_fit].astype(int).to_numpy())
    slice_score = score_pipeline(pipeline, cal[list(features)])
    if model == "hazard":
        slice_score = convert_hazard(slice_score, horizon)
    slice_y = cal[y_slice].astype(int).to_numpy()
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(np.asarray(slice_score, dtype=float), slice_y)
    scores = pd.read_parquet(score_file)[list(w.SCORE_COLUMNS)]
    raw = scores["score"].to_numpy(dtype=float)
    calibrated = np.full(raw.shape, np.nan)
    ok = ~np.isnan(raw)
    calibrated[ok] = calibrator.predict(raw[ok])
    scores = scores.assign(score_calibrated=calibrated)
    y, certs = scores["y"].to_numpy(), scores["cert"].to_numpy()
    metrics = {
        "brier_raw": brier(y, raw),
        "brier_calibrated": brier(y, calibrated),
        "brier_slice_raw": brier(slice_y, slice_score),
        "brier_slice_calibrated": brier(slice_y, calibrator.predict(slice_score)),
        "pr_auc": evaluate(y, raw, tie_breaker=certs)["pr_auc"],
        "pr_auc_calibrated": evaluate(y, calibrated, tie_breaker=certs)["pr_auc"],
        "mean_calibrated": float(np.mean(calibrated)) if len(calibrated) else float("nan"),
        "failure_rate": float(np.mean(y)) if len(y) else float("nan"),
        "n_thresholds": int(len(calibrator.X_thresholds_)),
        "slice_score_max": float(np.max(slice_score)) if len(slice_score) else float("nan"),
        "top_plateau": float(calibrator.y_thresholds_[-1]),
        "extrapolated_share": float(np.mean(raw[ok] > np.max(slice_score))) if ok.any() else 0.0,
        "n": int(len(scores)),
        "n_failures": int(y.sum()),
    }
    config.update(
        {
            "horizon": horizon,
            "fit_horizon": fit_h,
            "fit_label": y_fit,
            "test_year": year,
            "method": METHOD,
            "availability_lag_days": int(lag),
            "params": params,
            **bounds,
        }
    )
    result = CalibrationResult(year, model, horizon, calibrator, config, metrics, scores)
    if save:
        result.paths = save_calibration(result, settings)
    run = tracking.start_run(RUN_NAME, config, settings)
    run.log_metrics(metrics)
    run.finish()
    return result


def save_calibration(result: CalibrationResult, settings: Settings) -> dict[str, Path]:
    """Write ``calibration.joblib`` + ``calibration.json`` and fill ``score_calibrated``."""
    out = w.model_dir(settings, result.year, result.model, result.horizon)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"dir": out, "calibrator": out / "calibration.joblib"}
    joblib.dump(result.calibrator, paths["calibrator"])
    paths["config"] = out / "calibration.json"
    payload = {"config": w._json_ready(result.config), "metrics": w._json_ready(result.metrics)}
    paths["config"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    paths["scores"] = w.scores_path(settings, result.year, result.model, result.horizon)
    result.scores.loc[:, list(w.SCORE_COLUMNS)].to_parquet(paths["scores"], index=False)
    log.info("saved calibration for %s %d to %s", result.model, result.year, out)
    return paths


def load_calibrator(settings: Settings, year: int, model: str, horizon: int = 4):
    """``(calibrator, config, metrics)`` written by :func:`fit_calibrator`; raises when absent."""
    path = calibration_path(settings, year, model, horizon)
    if not path.exists():
        raise FileNotFoundError(f"no calibration for {model} {year} at {horizon}q under {path}")
    payload = json.loads((path.parent / "calibration.json").read_text(encoding="utf-8"))
    return joblib.load(path), payload["config"], payload["metrics"]


def calibrate_year(
    frame: pd.DataFrame,
    settings: Settings,
    year: int,
    models=MODELS,
    horizon: int = 4,
    rebuild: bool = True,
) -> list[CalibrationResult]:
    """Calibrate every requested model that has a walk-forward fit for ``year``.

    Models without a saved fit or score file at this horizon are skipped with a warning
    (the 8-quarter horizon has no hazard, for instance). With ``rebuild`` the
    ``walkforward_scores`` table is rebuilt from the per-year files afterwards.
    """
    results = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"cannot calibrate {model!r}; choose from {MODELS}")
        have_fit = (w.model_dir(settings, year, model, horizon) / "pipeline.joblib").exists()
        if not have_fit or not w.scores_path(settings, year, model, horizon).exists():
            log.warning("no walk-forward %s fit for %d at %dq; skipped", model, year, horizon)
            continue
        results.append(fit_calibrator(frame, settings, year, model, horizon))
    if rebuild and results:
        w.rebuild_scores_table(settings)
    return results
