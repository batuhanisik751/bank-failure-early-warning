"""Isotonic calibration of the walk-forward scores (spec 8.3, CONTRACT section 13).

A ranking model's raw output is not a failure probability a supervisor can quote: the
logit is fitted without class weights on a 0.5 percent base rate and the booster's scale
drifts from year to year. For every walk-forward test year ``Y`` and probability model a
monotone map is learned on (score, outcome) pairs that were all known before ``Y``'s
first prediction date, then applied to ``Y``'s scores. Two choices define a recipe:

*Which rows* is the ``slice_scorer`` choice:

* ``"trailing"`` (the default for every model and horizon): the walk-forward scores of
  the same model from the :data:`TRAILING_YEARS` most recent earlier test years whose
  every row's ``window_end_Hq`` lies before ``Y``'s first prediction date (at 4q the
  years ``Y-2`` and ``Y-3``, at 8q ``Y-3`` and ``Y-4``; :func:`trailing_years` computes
  it from the labels). Those scores are genuinely out of sample and on the score scale
  the map is later applied to. :func:`assert_no_leakage` re-checks every row. Where
  fewer than two such years exist (2008-2010 at 4q, 2008-2011 at 8q) the fit falls back
  to the model's own scorer below (:data:`FALLBACK_SCORER_BY_MODEL`) and records it.
* ``"full"``: the last complete label year inside ``Y``'s :func:`training_mask`
  (:func:`calibration_masks`, widened backwards while it holds fewer than
  :data:`MIN_CALIBRATION_POSITIVES` failures) scored by ``Y``'s full-window model
  itself. In-sample for that model; usable for the L2 logits, whose in-sample and
  out-of-sample scales are close, useless for trees.
* ``"inner"``: the same slice scored by an inner model with ``Y``'s tuned configuration
  fitted on the rows whose windows closed before the slice. Out of sample, but on the
  inner model's score scale, which can differ from the full model's by an order of
  magnitude in crisis years (``docs/DECISIONS.md`` 2026-09-26: the 8q booster's 2011 map
  predicted a mean of 0.353 against a failure rate of 0.012).

*How the map is fitted* is :func:`binned_isotonic`: the rows are sorted by score and cut
into contiguous bins of at least :data:`MIN_BIN` rows, and
``IsotonicRegression(out_of_bounds="clip")`` is fitted on bin mean score against bin
failure rate with the bin sizes as weights. A plain isotonic fit lets its top step rest on
a handful of banks (three failures at the top of the scale map to exactly 1.0, a step of
170 banks to 0.5); every step of the binned map is a rate observed on at least
:data:`MIN_BIN` banks.

The map is applied to the full-window model's test-year scores and stored as
``score_calibrated`` in ``data/walkforward/<Y>_<model>_<H>q.parquet``, from which
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
MODELS: tuple[str, ...] = ("logit", "gbdt", "gbdt_mono", "hazard")
RUN_NAME = "calibrate"
METHOD = "isotonic"
#: A slice with fewer failures than this cannot pin down a monotone map; widen it.
MIN_CALIBRATION_POSITIVES = 5
SLICE_SCORERS: tuple[str, ...] = ("full", "inner", "trailing")
#: The recipe every model and horizon gets unless ``--scorer`` says otherwise.
DEFAULT_SLICE_SCORER = "trailing"
#: How many earlier out-of-sample test years the trailing recipe pools.
TRAILING_YEARS = 2
#: Smallest bin of :func:`binned_isotonic`: every step of the map is a rate on this many banks.
MIN_BIN = 50
#: Who scores the slice when too few trailing years exist (module docstring), per model:
#: the L2 logits score their own slice, the boosters need an inner model because their
#: in-sample scores separate the slice perfectly (slice Brier 0, a handful of thresholds).
FALLBACK_SCORER_BY_MODEL: dict[str, str] = {"logit": "full", "hazard": "full", "gbdt": "inner"}
SLICE_SCORER_BY_MODEL = FALLBACK_SCORER_BY_MODEL


def bin_table(scores, y, min_bin: int = MIN_BIN) -> pd.DataFrame:
    """Contiguous score bins of at least ``min_bin`` rows: ``n, score_mean, rate`` per bin.

    Rows are sorted by score (ties kept in input order) and cut every ``min_bin`` rows;
    a short last bin is merged into its predecessor, so with fewer than ``2 * min_bin``
    rows there is one bin. Rows with a missing score are dropped first.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    if scores.shape != y.shape:
        raise ValueError(f"scores {scores.shape} and y {y.shape} differ in shape")
    keep = ~np.isnan(scores)
    scores, y = scores[keep], y[keep]
    n = len(scores)
    if n == 0:
        raise ValueError("no rows with a score to bin")
    min_bin = max(1, int(min_bin))
    order = np.argsort(scores, kind="stable")
    n_bins = max(1, n // min_bin)
    edges = [i * min_bin for i in range(n_bins)] + [n]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = order[lo:hi]
        rows.append(
            {
                "n": int(len(idx)),
                "score_min": float(scores[idx].min()),
                "score_max": float(scores[idx].max()),
                "score_mean": float(scores[idx].mean()),
                "positives": int(y[idx].sum()),
                "rate": float(y[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def binned_isotonic(scores, y, min_bin: int = MIN_BIN) -> IsotonicRegression:
    """Isotonic map fitted on the :func:`bin_table` points, weighted by bin size.

    ``IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)`` on bin mean score
    against bin failure rate: monotone, clipped outside the observed score range, and
    every plateau it can produce, the top one included, is a rate observed on at least
    ``min_bin`` banks. The fitted regression's ``bins_`` attribute keeps the table.
    """
    table = bin_table(scores, y, min_bin)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(
        table["score_mean"].to_numpy(),
        table["rate"].to_numpy(),
        sample_weight=table["n"].to_numpy(dtype=float),
    )
    calibrator.bins_ = table
    return calibrator


def _fmt(ts) -> str | None:
    return None if pd.isna(ts) else str(pd.Timestamp(ts).date())


def calibration_masks(
    frame: pd.DataFrame,
    horizon: int,
    year: int,
    fit_horizon: int | None = None,
    lag_days=None,
    require_inner: bool = True,
) -> tuple[pd.Series, pd.Series, dict]:
    """``(inner_train, calibration_slice, bounds)`` nested inside test year ``year``.

    The slice is the last calendar year whose four report quarters all lie in the
    year's :func:`training_mask` at the scoring horizon, so every slice outcome was
    known on the year's first prediction date. The inner training rows are the year's
    training rows at ``fit_horizon`` whose windows closed before the slice's first
    prediction date (re-checked with :func:`assert_no_leakage`). While the slice (or,
    with ``require_inner``, the inner rows) holds fewer than
    :data:`MIN_CALIBRATION_POSITIVES` failures the slice is widened backwards a complete
    year at a time; ``bounds["sufficient"]`` is ``False`` when no width works and the
    narrowest slice is returned for the caller to flag. The ``"full"`` slice scorer fits
    no inner model and passes ``require_inner=False``.
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
        bounds["sufficient"] = bounds["positives_calibration"] >= enough and (
            not require_inner or bounds["positives_inner_train"] >= enough
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
    iteration cap included, for ``gbdt`` and ``gbdt_mono``; nothing for the Texas ranking.
    """
    if config.get("model") in w.GBDT_MODELS:
        return dict(config.get("params", {}))
    return {"C": float(config["C"])} if "C" in config else {}


def default_slice_scorer(model: str) -> str:
    """:data:`DEFAULT_SLICE_SCORER` for every probability model (``"trailing"``)."""
    if model not in MODELS:
        raise ValueError(f"{model!r} is not a probability model; choose from {MODELS}")
    return DEFAULT_SLICE_SCORER


def fallback_scorer(model: str) -> str:
    """:data:`FALLBACK_SCORER_BY_MODEL` for ``model`` (``gbdt_mono`` follows ``gbdt``)."""
    key = "gbdt" if model in w.GBDT_MODELS else model
    return FALLBACK_SCORER_BY_MODEL[key]


def trailing_years(
    frame: pd.DataFrame,
    horizon: int,
    year: int,
    lag_days=None,
    n_years: int = TRAILING_YEARS,
    available=None,
) -> list[int]:
    """The ``n_years`` most recent earlier test years usable to calibrate test year ``year``.

    A year ``Y'`` qualifies when every one of its rows in ``frame`` has
    ``window_end_Hq`` before ``year``'s first prediction date, so all its outcomes were
    known when ``year`` was scored (at 4q that is ``year - 2`` and earlier, at 8q
    ``year - 3`` and earlier). ``available`` restricts the candidates to the years that
    have a walk-forward score file. Returned ascending; fewer than ``n_years`` when the
    backtest has not run long enough (the caller then falls back).
    """
    from bankcanary.splits.time_split import DEFAULT_LAG_DAYS

    lag = DEFAULT_LAG_DAYS if lag_days is None else int(lag_days)
    end_col = horizon_columns(horizon)[1]
    cutoff = prediction_date(w.year_bounds(int(year))[0], lag)
    years = frame["repdte"].dt.year
    earlier = frame.loc[(years < int(year)) & (years >= w.FIRST_TEST_YEAR), end_col]
    by_year = earlier.groupby(years.loc[earlier.index]).agg(["max", "count", "size"])
    closed = by_year[(by_year["max"] < cutoff) & (by_year["count"] == by_year["size"])]
    chosen = [int(y) for y in closed.index]
    if available is not None:
        chosen = [y for y in chosen if y in set(int(a) for a in available)]
    return sorted(chosen)[-int(n_years) :]


def trailing_rows(
    frame: pd.DataFrame, settings: Settings, year: int, model: str, horizon: int, lag_days=None
) -> tuple[pd.DataFrame, list[int]]:
    """Calibration rows for the ``"trailing"`` scorer and the years they come from.

    Reads the walk-forward score files of ``model`` for :func:`trailing_years` (only
    years with a file count), joins ``window_end_Hq`` from ``frame`` by ``(cert, repdte)``
    and runs :func:`assert_no_leakage` against ``year``'s first report date, so a
    calibration row whose outcome window closed on or after ``year``'s first prediction
    date raises. Rows without a raw score are dropped.
    """
    from bankcanary.splits.time_split import DEFAULT_LAG_DAYS

    lag = DEFAULT_LAG_DAYS if lag_days is None else int(lag_days)
    end_col = horizon_columns(horizon)[1]
    have = [
        y
        for y in range(w.FIRST_TEST_YEAR, int(year))
        if w.scores_path(settings, y, model, horizon).exists()
    ]
    years = trailing_years(frame, horizon, year, lag, available=have)
    if not years:
        return pd.DataFrame(columns=["cert", "repdte", "test_year", "score", "y", end_col]), []
    parts = [pd.read_parquet(w.scores_path(settings, y, model, horizon)) for y in years]
    rows = pd.concat(parts, ignore_index=True)[["cert", "repdte", "test_year", "score", "y"]]
    rows = rows[rows["score"].notna()]
    ends = frame[["cert", "repdte", end_col]].drop_duplicates(["cert", "repdte"])
    rows = rows.merge(ends, on=["cert", "repdte"], how="left")
    assert_no_leakage(rows, horizon, w.year_bounds(int(year))[0], lag)
    return rows.reset_index(drop=True), years


def calibration_path(settings: Settings, year: int, model: str, horizon: int) -> Path:
    """``models/walkforward/<Y>/<model>[_<H>q]/calibration.joblib``."""
    return w.model_dir(settings, year, model, horizon) / "calibration.joblib"


def fit_calibrator(
    frame: pd.DataFrame,
    settings: Settings,
    year: int,
    model: str,
    horizon: int = 4,
    save=True,
    slice_scorer: str | None = None,
    min_bin: int = MIN_BIN,
) -> CalibrationResult:
    """Learn the binned isotonic map for one walk-forward year and apply it to its scores.

    Requires the full-window model saved by :func:`bankcanary.evaluation.walkforward.fit_year`
    and its per-year score file. ``slice_scorer`` (module docstring) is ``"trailing"``
    (the default, :func:`default_slice_scorer`) when the rows are the same model's
    walk-forward scores from the two most recent earlier test years whose outcomes were
    all known before this year, ``"full"`` when the year's own model scores the last
    complete year inside its training window and ``"inner"`` when an inner model
    refitted before that slice scores it. A trailing fit with fewer than two usable
    years falls back to :func:`fallback_scorer` and records ``fallback = True``. The map
    is :func:`binned_isotonic` with ``min_bin``. With ``save`` the calibrator goes next
    to the model as ``calibration.joblib`` plus ``calibration.json`` and the score file's
    ``score_calibrated`` column is filled in place; the caller rebuilds
    ``walkforward_scores`` afterwards. Logs a ``calibrate`` run keyed by the rows, the
    scorer, the bin size and the model configuration.
    """
    from bankcanary import tracking
    from bankcanary.evaluation.metrics import brier, evaluate
    from bankcanary.models.baselines import score_pipeline
    from bankcanary.models.hazard import convert_hazard

    horizon, year = int(horizon), int(year)
    if model not in MODELS:
        raise ValueError(f"{model!r} is not a probability model; calibrate one of {MODELS}")
    slice_scorer = default_slice_scorer(model) if slice_scorer is None else slice_scorer
    if slice_scorer not in SLICE_SCORERS:
        raise ValueError(f"slice_scorer must be one of {SLICE_SCORERS}, not {slice_scorer!r}")
    full_pipeline, features, full_config = w.load_year(settings, year, model, horizon)
    score_file = w.scores_path(settings, year, model, horizon)
    if not score_file.exists():
        raise FileNotFoundError(f"no walk-forward scores for {model} {year} at {score_file}")
    fit_h = w.fit_horizon_of(model, horizon)
    lag = settings.availability_lag_days
    params = model_params(full_config)
    pipeline, _, config = w.build_model(model, settings, None, params)
    y_fit, y_slice = horizon_columns(fit_h)[0], horizon_columns(horizon)[0]
    effective, fallback, bounds = slice_scorer, False, {}
    if slice_scorer == "trailing":
        rows, years = trailing_rows(frame, settings, year, model, horizon, lag)
        if len(years) < TRAILING_YEARS:
            effective, fallback = fallback_scorer(model), True
            log.warning(
                "calibration %dq %d %s: only %s trailing year(s) usable; falling back to %s",
                horizon,
                year,
                model,
                years,
                effective,
            )
        else:
            slice_score = rows["score"].to_numpy(dtype=float)
            slice_y = rows["y"].astype(int).to_numpy()
            bounds = {
                "calibration_years": years,
                "calibration_start": _fmt(rows["repdte"].min()),
                "calibration_end": _fmt(rows["repdte"].max()),
                "calibration_prediction_date": str(
                    prediction_date(w.year_bounds(year)[0], lag).date()
                ),
                "inner_train_repdte_max": None,
                "n_inner_train": 0,
                "positives_inner_train": 0,
                "n_calibration": int(len(rows)),
                "positives_calibration": int(slice_y.sum()),
                "sufficient": int(slice_y.sum()) >= MIN_CALIBRATION_POSITIVES,
            }
    if effective in ("full", "inner"):
        use_inner = effective == "inner"
        inner, sl, bounds = calibration_masks(frame, horizon, year, fit_h, lag, use_inner)
        if (use_inner and not inner.any()) or not sl.any():
            raise ValueError(f"calibration {horizon}q {year} {model}: empty inner or slice rows")
        tr, cal = frame.loc[inner], frame.loc[sl]
        if use_inner:
            pipeline.fit(tr[list(features)], tr[y_fit].astype(int).to_numpy())
        else:
            pipeline = full_pipeline
        slice_score = np.asarray(score_pipeline(pipeline, cal[list(features)]), dtype=float)
        if model == "hazard":
            slice_score = convert_hazard(slice_score, horizon)
        slice_y = cal[y_slice].astype(int).to_numpy()
    log.info(
        "calibration %dq %d %s (%s%s): rows %s-%s, %d rows, %d positives, bins of >= %d",
        horizon,
        year,
        model,
        effective,
        " fallback" if fallback else "",
        bounds["calibration_start"],
        bounds["calibration_end"],
        bounds["n_calibration"],
        bounds["positives_calibration"],
        min_bin,
    )
    calibrator = binned_isotonic(slice_score, slice_y, min_bin)
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
        "n_bins": int(len(calibrator.bins_)),
        "min_bin_size": int(calibrator.bins_["n"].min()),
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
            "slice_scorer": slice_scorer,
            "effective_scorer": effective,
            "fallback": bool(fallback),
            "min_bin": int(min_bin),
            "n_rows": int(bounds["n_calibration"]),
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


def scored_years(settings: Settings, model: str, horizon: int) -> list[int]:
    """Test years with a walk-forward score file for ``model`` at ``horizon``, ascending."""
    suffix = f"_{model}_{int(horizon)}q.parquet"
    years = []
    for f in w.scores_dir(settings).glob(f"*{suffix}"):
        stem = f.name[: -len(suffix)]
        if stem.isdigit():
            years.append(int(stem))
    return sorted(years)


def production_calibrator(
    settings: Settings,
    model: str,
    horizon: int = 4,
    n_years: int = TRAILING_YEARS,
    min_bin: int = MIN_BIN,
    frame: pd.DataFrame | None = None,
) -> tuple[IsotonicRegression, dict]:
    """The map for ``models/production/<model>`` (scores label-incomplete quarters).

    The rows are the walk-forward scores of ``model`` from the ``n_years`` most recent
    test years whose rows are all label-complete as of the failures list date (the
    score files carry ``label_complete``; with ``frame`` the labels of those years are
    re-checked too). Every one of those scores is out of sample for the model that
    produced it, and the most recent walk-forward model, promoted to production, has
    the last of those years as its own test year. Returns the :func:`binned_isotonic`
    map and a config with ``calibration_years`` and ``n_rows``; raises when fewer than
    ``n_years`` complete years exist.
    """
    if model not in MODELS:
        raise ValueError(f"{model!r} is not a probability model; choose from {MODELS}")
    horizon = int(horizon)
    complete_col = horizon_columns(horizon)[3]
    complete, parts = [], {}
    for y in scored_years(settings, model, horizon):
        rows = pd.read_parquet(w.scores_path(settings, y, model, horizon))
        ok = bool(rows["label_complete"].fillna(False).astype(bool).all()) and len(rows) > 0
        if ok and frame is not None:
            in_year = frame["repdte"].dt.year == y
            ok = bool(frame.loc[in_year, complete_col].fillna(False).astype(bool).all())
        if ok:
            complete.append(y)
            parts[y] = rows
    years = complete[-int(n_years) :]
    if len(years) < int(n_years):
        raise ValueError(
            f"production calibration for {model} at {horizon}q needs {n_years} label-complete "
            f"test years with scores; found {complete}"
        )
    rows = pd.concat([parts[y] for y in years], ignore_index=True)
    rows = rows[rows["score"].notna()]
    calibrator = binned_isotonic(rows["score"].to_numpy(dtype=float), rows["y"].to_numpy(), min_bin)
    config = {
        "model": model,
        "horizon": horizon,
        "method": METHOD,
        "slice_scorer": "trailing",
        "effective_scorer": "trailing",
        "fallback": False,
        "min_bin": int(min_bin),
        "calibration_years": [int(y) for y in years],
        "calibration_start": _fmt(rows["repdte"].min()),
        "calibration_end": _fmt(rows["repdte"].max()),
        "n_rows": int(len(rows)),
        "positives_calibration": int(rows["y"].sum()),
        "n_bins": int(len(calibrator.bins_)),
        "top_plateau": float(calibrator.y_thresholds_[-1]),
    }
    return calibrator, config


def calibrate_year(
    frame: pd.DataFrame,
    settings: Settings,
    year: int,
    models=MODELS,
    horizon: int = 4,
    rebuild: bool = True,
    slice_scorer: str | None = None,
    min_bin: int = MIN_BIN,
) -> list[CalibrationResult]:
    """Calibrate every requested model that has a walk-forward fit for ``year``.

    Models without a saved fit or score file at this horizon are skipped with a warning.
    ``slice_scorer`` and ``min_bin`` are passed to :func:`fit_calibrator` (``None``: the
    ``"trailing"`` default). With ``rebuild`` the ``walkforward_scores`` table is rebuilt
    from the per-year files afterwards.
    """
    results = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"cannot calibrate {model!r}; choose from {MODELS}")
        have_fit = (w.model_dir(settings, year, model, horizon) / "pipeline.joblib").exists()
        if not have_fit or not w.scores_path(settings, year, model, horizon).exists():
            log.warning("no walk-forward %s fit for %d at %dq; skipped", model, year, horizon)
            continue
        results.append(
            fit_calibrator(frame, settings, year, model, horizon, True, slice_scorer, min_bin)
        )
    if rebuild and results:
        w.rebuild_scores_table(settings)
    return results
