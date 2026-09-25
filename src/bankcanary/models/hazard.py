"""Discrete-time hazard model (Shumway 2001) on bank-quarter rows (CONTRACT section 13).

A static classifier asks "will this bank fail within the next H quarters?" and treats
every bank-quarter as an independent draw. The hazard formulation instead asks the
one-period question "does the bank fail inside the next quarter?" (``y_1q``: the failure
date falls in ``(avail_date, avail_date + 3 months]``) and uses every quarter a bank is
observed as one row with that quarter's covariates. Because the panel carries a bank
until its last report, a bank that leaves by merger or voluntary closing simply has no
rows after it exits: that is right-censoring, handled without any deletion or
re-weighting. The estimator is the Prototype 1 logistic pipeline (winsorise, impute,
scale, L2 logit), *unweighted* so that ``predict_proba`` is the quarterly hazard ``h``
itself, with ``C`` chosen on the inner validation slice like ``scripts/tune_logit_c.py``.

Multi-quarter probabilities come from the survival identity
``P(fail within H quarters) = 1 - (1 - h)^H``, which assumes the hazard estimated from
today's covariates persists at the same level for the next H quarters (covariates are
not projected forward). Ranking metrics are invariant to this monotone map, so the
hazard's PR-AUC at 4q or 8q is the PR-AUC of ``h`` itself; the Brier score is not, and
is reported on the converted probabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bankcanary.features.registry import feature_names
from bankcanary.models.baselines import SMALL_FEATURES
from bankcanary.models.preprocess import make_pipeline

log = logging.getLogger(__name__)

MODEL_NAME = "hazard"
#: The hazard event is "fails inside the next quarter": the 1-quarter label.
HAZARD_HORIZON = 1
FEATURE_VERSION = "v2"
#: Inverse L2 strength chosen by ``bankcanary train-hazard --tune`` on the inner validation
#: slice (reports 2007Q1-2008Q4, inner training windows closed before 2007-05-30); the grid
#: and its scores are in ``reports/p2_hazard.md`` and ``runs/tune_hazard/``.
HAZARD_C = 0.0003
GRID: tuple[float, ...] = (1.0, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)
VALIDATION_START = "2007-03-31"
VALIDATION_END = "2008-12-31"

#: Prototype 2 interest-rate sensitivity and deposit-run features (CONTRACT section 12).
RATE_RUN_FEATURES: tuple[str, ...] = (
    "afs_unrealized_to_tier1",
    "htm_unrealized_to_tier1",
    "unrealized_loss_to_tier1",
    "adjusted_tier1_leverage",
    "securities_to_assets",
    "htm_share_of_securities",
    "uninsured_share",
    "uninsured_to_liquid_assets",
    "large_time_deposit_share",
)
#: Features whose odds ratios the report interprets one by one.
INTERPRETED_FEATURES: tuple[str, ...] = tuple(SMALL_FEATURES) + RATE_RUN_FEATURES


def hazard_features(version: str = FEATURE_VERSION) -> list[str]:
    """Every registered feature of the table version, in registry order (identifiers never)."""
    return feature_names(version=version)


def make_hazard(C: float | None = None, max_iter: int = 2000) -> Pipeline:
    """``Winsorizer -> SimpleImputer -> StandardScaler -> LogisticRegression`` for the hazard.

    No class weighting: the fitted probability must be the quarterly hazard on the
    natural scale, otherwise ``1 - (1 - h)^H`` would convert a distorted number.
    """
    c = HAZARD_C if C is None else float(C)
    return make_pipeline(LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs"))


def convert_hazard(h, horizon: int) -> np.ndarray:
    """``1 - (1 - h)^H``: cumulative failure probability over ``H`` quarters.

    Assumes the quarterly hazard stays at ``h`` for every one of the next ``H`` quarters
    (the covariates that produced it are not updated). ``H = 1`` returns ``h`` unchanged.
    """
    if int(horizon) < 1:
        raise ValueError(f"horizon must be a positive number of quarters, got {horizon}")
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    return 1.0 - np.power(1.0 - h, int(horizon))


def brier(y_true, probabilities) -> float:
    """Mean squared error of the probabilities against the 0/1 outcome (lower is better)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} labels vs {p.shape} probabilities")
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def inner_split(frame: pd.DataFrame, settings) -> tuple[pd.Series, pd.Series]:
    """``(inner_train, validation)`` masks inside the fixed training period (spec rule 6.7).

    Mirrors ``scripts/tune_logit_c.py`` at the hazard's own horizon: validation = usable
    1q rows reported 2007Q1-2008Q4; inner training = fixed-split training rows whose
    1q window closed before the validation prediction date (2007-05-30, so reports up to
    2006Q3). Both are subsets of the fixed split's training rows; the test years are
    never touched.
    """
    from bankcanary.splits import fixed_split_masks, test_mask, training_mask

    lag = settings.availability_lag_days
    outer_train, _ = fixed_split_masks(frame, settings, HAZARD_HORIZON)
    inner_train = training_mask(frame, HAZARD_HORIZON, VALIDATION_START, lag) & outer_train
    validation = test_mask(frame, HAZARD_HORIZON, VALIDATION_START, VALIDATION_END) & outer_train
    return inner_train.rename("inner_train_1q"), validation.rename("validation_1q")


def _horizon_metrics(rows: pd.DataFrame, h: np.ndarray, horizon: int) -> dict:
    """Ranking metrics plus Brier of the converted probability against ``y_<H>q``.

    Rows whose label at ``horizon`` is incomplete or dropped are skipped; the raw
    hazard's Brier and the climatology Brier (a constant at the realised rate) are added
    as references. Keys carry the ``_<H>q`` suffix.
    """
    from bankcanary.evaluation.metrics import evaluate
    from bankcanary.labels.build import horizon_columns

    y_col, _, _, complete_col = horizon_columns(horizon)
    usable = rows[complete_col].fillna(False).astype(bool) & ~rows[
        "dropped_failed_before_avail"
    ].fillna(False).astype(bool)
    usable = usable.to_numpy()
    y = rows.loc[usable, y_col].astype(int).to_numpy()
    p = convert_hazard(h[usable], horizon)
    out = evaluate(y, p, tie_breaker=rows.loc[usable, "cert"].to_numpy())
    out["brier"] = brier(y, p)
    out["brier_raw_hazard"] = brier(y, h[usable])
    out["brier_climatology"] = brier(y, np.full(len(y), y.mean() if len(y) else 0.0))
    return {f"{k}_{horizon}q": v for k, v in out.items()}


def tune_c(frame: pd.DataFrame, settings, grid=GRID, select_horizon: int = 4) -> pd.DataFrame:
    """Score every ``C`` in ``grid`` on the inner validation slice; one run record each.

    Each candidate is fitted on the inner training rows (target ``y_1q``) and scored on
    the validation rows at 1q and, converted, at ``select_horizon``; the winner is the
    highest ``pr_auc_<select_horizon>q`` (the primary horizon the model is compared on).
    Configurations already scored under ``runs/tune_hazard/`` are read back instead of
    refitted (``cached = True``), so an interrupted grid resumes where it stopped.
    """
    from bankcanary import tracking
    from bankcanary.labels.build import horizon_columns

    features, y_col = hazard_features(), horizon_columns(HAZARD_HORIZON)[0]
    inner_train, validation = inner_split(frame, settings)
    base = {
        "model": MODEL_NAME,
        "horizon": HAZARD_HORIZON,
        "features_version": FEATURE_VERSION,
        "n_features": len(features),
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "n_inner_train": int(inner_train.sum()),
        "positives_inner_train": int(frame.loc[inner_train, y_col].sum()),
        "n_validation": int(validation.sum()),
    }
    log.info(
        "hazard tuning: inner train %d rows (%d positives, last report %s), validation %d rows",
        base["n_inner_train"],
        base["positives_inner_train"],
        frame.loc[inner_train, "repdte"].max().date(),
        base["n_validation"],
    )
    rows = []
    for c in grid:
        config = {**base, "C": float(c)}
        cached = tracking.find_metrics("tune_hazard", config, settings)
        if cached is None:
            pipe = make_hazard(c).fit(
                frame.loc[inner_train, features], frame.loc[inner_train, y_col].astype(int)
            )
            valid = frame.loc[validation]
            h = pipe.predict_proba(valid[features])[:, 1]
            metrics = {**_horizon_metrics(valid, h, HAZARD_HORIZON)}
            metrics.update(_horizon_metrics(valid, h, select_horizon))
            run = tracking.start_run("tune_hazard", config, settings)
            run.log_metrics(metrics)
            run.finish()
        rows.append({"C": float(c), "cached": cached is not None, **(cached or metrics)})
    table = pd.DataFrame(rows)
    # Cached metrics come back key-sorted from JSON: fix one column order for both paths.
    table = table[["C", "cached"] + sorted(c for c in table.columns if c not in ("C", "cached"))]
    order = table.sort_values(f"pr_auc_{select_horizon}q", ascending=False, kind="mergesort")
    return order.reset_index(drop=True)


def read_runs(name: str, settings) -> list[tuple[dict, dict]]:
    """``[(config, metrics), ...]`` of every finished run under ``runs/<name>/``, by run id."""
    from bankcanary.tracking import runs_dir

    root = runs_dir(settings) / name
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        cfg, met = d / "config.json", d / "metrics.json"
        if cfg.exists() and met.exists():
            out.append((json.loads(cfg.read_text("utf-8")), json.loads(met.read_text("utf-8"))))
    return out


def hazard_dir(settings) -> Path:
    """``models/hazard/``: the hazard has one intrinsic horizon, so no ``_<H>q`` suffix."""
    return Path(settings.models_dir) / MODEL_NAME


def fit_hazard(frame: pd.DataFrame, settings, C: float | None = None, save: bool = True):
    """Fit the hazard on the fixed split's training rows at 1q, score the test rows, log a run.

    Rows come only from :func:`bankcanary.splits.fixed_split_masks` at horizon 1 (every
    2002Q1-2008Q4 report with a complete 1q label, not a dropped post-failure report) and
    are re-checked with :func:`assert_no_leakage` before ``fit`` (rule 6.2). The test
    metrics here are the *one-quarter* metrics; :func:`evaluate_converted` handles 4q/8q.
    Returns a :class:`~bankcanary.models.train.TrainResult` named ``hazard``.
    """
    from bankcanary import tracking
    from bankcanary.evaluation.metrics import evaluate_by_year
    from bankcanary.labels.build import horizon_columns
    from bankcanary.models import train as t
    from bankcanary.splits import assert_no_leakage, fixed_split_masks

    features, horizon = hazard_features(), HAZARD_HORIZON
    y_col = horizon_columns(horizon)[0]
    missing = [c for c in features + [y_col] if c not in frame.columns]
    if missing:
        raise KeyError(f"training frame is missing column(s) {missing}")
    train, test = fixed_split_masks(frame, settings, horizon)
    split = settings.fixed_split
    assert_no_leakage(frame.loc[train], horizon, split.test_start, settings.availability_lag_days)
    if not train.any() or not test.any():
        raise ValueError("fixed split for 1q selected no train or no test rows")
    c = HAZARD_C if C is None else float(C)
    pipeline = make_hazard(c)
    y_train = frame.loc[train, y_col].astype(int).to_numpy()
    log.info("fitting hazard (C=%g) on %d rows (%d positives)", c, int(train.sum()), y_train.sum())
    pipeline.fit(frame.loc[train, features], y_train)
    scores = t.score_test_split(pipeline, frame.loc[test], features, y_col, horizon)
    metrics = t._score_metrics(scores)
    metrics["brier"] = brier(scores["y"].to_numpy(), scores["score"].to_numpy())
    by_year = evaluate_by_year(scores, "score", "y", "year", tie_col="cert")
    sensitivity, by_event = t._extra_metrics(scores)
    config = {
        "model": MODEL_NAME,
        "features_version": FEATURE_VERSION,
        "n_features": len(features),
        "C": c,
        "class_weight": None,
        "conversion": "1 - (1 - h)^H",
    }
    config.update(t._split_config(frame, train, test, settings, horizon, y_col))
    result = t.TrainResult(
        MODEL_NAME,
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
    log.info("hazard 1q test: pr_auc %.4f roc_auc %.4f", metrics["pr_auc"], metrics["roc_auc"])
    if save:
        # Re-fitting rewrites metrics.json; keep the converted-horizon blocks that earlier
        # ``evaluate_converted`` calls merged into it (the fit itself is deterministic).
        path = hazard_dir(settings) / "metrics.json"
        converted = _read_json(path).get("converted")
        result.paths = t.save_artifacts(result, hazard_dir(settings))
        if converted:
            _write_json(path, {**_read_json(path), "converted": converted})
    run = tracking.start_run("train", config, settings)
    run.log_metrics(
        {
            "test": metrics,
            "sensitivity_censored_dropped": sensitivity,
            "per_event": by_event,
            "by_year": by_year.to_dict(orient="records"),
        }
    )
    run.finish()
    return result


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload) -> None:
    from bankcanary.tracking import _json_ready

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


COMPARATORS: tuple[str, ...] = ("logit_v2", "gbdt")


def _comparator(name: str, frame, settings, horizon: int):
    """``(pipeline, features)`` of a comparison model at ``horizon``: saved, else fitted now.

    ``models/<name>/`` (4q) or ``models/<name>_<H>q/`` is reused when present; otherwise
    the variant is fitted through :func:`bankcanary.models.gbdt.fit_on_fixed_split`,
    which saves it and logs its own run, so the next call finds it on disk.
    """
    from bankcanary.models import gbdt
    from bankcanary.models.train import load_model

    try:
        pipeline, features, _ = load_model(settings, name, horizon)
        log.info("comparator %s %dq loaded from disk", name, horizon)
        return pipeline, features
    except FileNotFoundError:
        pipeline, features = gbdt.make_variant(name, settings)
        result = gbdt.fit_on_fixed_split(name, pipeline, features, frame, settings, horizon)
        return result.pipeline, result.features


def evaluate_converted(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    settings,
    horizon: int,
    C: float,
    comparators: tuple[str, ...] = COMPARATORS,
    save: bool = True,
) -> dict:
    """Score the fixed split's ``horizon``-quarter test rows with ``1 - (1 - h)^H``.

    Returns ``{"hazard": metrics, "comparators": {name: metrics}, ...}`` where every
    metrics dict is the ranking suite plus ``brier`` (comparators are scored with their
    own ``predict_proba``; the hazard also reports ``brier_raw_hazard`` for the
    unconverted ``h`` and ``brier_climatology`` for a constant at the realised rate).
    Logs one ``train`` run with ``converted_from = 1`` and, when ``save``, merges the
    block into ``models/hazard/metrics.json`` under ``converted.<H>q`` so successive
    horizons accumulate in one file for the report.
    """
    from bankcanary import tracking
    from bankcanary.evaluation.metrics import evaluate
    from bankcanary.labels.build import horizon_columns
    from bankcanary.models.baselines import score_pipeline
    from bankcanary.splits import fixed_split_masks

    features, y_col = hazard_features(), horizon_columns(horizon)[0]
    _, test = fixed_split_masks(frame, settings, horizon)
    rows = frame.loc[test]
    if rows.empty:
        raise ValueError(f"fixed split for {horizon}q selected no test rows")
    h = score_pipeline(pipeline, rows[features])
    block = _horizon_metrics(rows, h, horizon)
    hazard_metrics = {k.removesuffix(f"_{horizon}q"): v for k, v in block.items()}
    y, cert = rows[y_col].astype(int).to_numpy(), rows["cert"].to_numpy()
    compared = {}
    for name in comparators:
        pipe, feats = _comparator(name, frame, settings, horizon)
        p = score_pipeline(pipe, rows[feats])
        compared[name] = {**evaluate(y, p, tie_breaker=cert), "brier": brier(y, p)}
    config = {
        "model": MODEL_NAME,
        "horizon": int(horizon),
        "converted_from": HAZARD_HORIZON,
        "conversion": "1 - (1 - h)^H",
        "C": float(C),
        "features_version": FEATURE_VERSION,
        "n_features": len(features),
        "test_start": str(settings.fixed_split.test_start),
        "test_end": str(settings.fixed_split.test_end),
        "n_test": int(len(rows)),
        "positives_test": int(y.sum()),
        "comparators": list(comparators),
    }
    out = {"hazard": hazard_metrics, "comparators": compared, "config": config}
    log.info(
        "hazard converted to %dq: pr_auc %.4f brier %.5f (raw hazard brier %.5f)",
        horizon,
        hazard_metrics["pr_auc"],
        hazard_metrics["brier"],
        hazard_metrics["brier_raw_hazard"],
    )
    run = tracking.start_run("train", config, settings)
    run.log_metrics({"test": hazard_metrics, "comparators": compared})
    run.finish()
    if save:
        path = hazard_dir(settings) / "metrics.json"
        saved = _read_json(path)
        saved.setdefault("converted", {})[f"{horizon}q"] = out
        _write_json(path, saved)
    return out


ODDS_RATIO_METHOD = "statsmodels Logit, unpenalised, cluster-robust standard errors by cert"


def odds_ratio_table(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    full_pipeline: Pipeline | None = None,
    features: tuple[str, ...] = INTERPRETED_FEATURES,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Odds ratios with confidence intervals for the interpreted hazard features.

    Method: an *unpenalised* ``statsmodels`` Logit of ``y_1q`` on the standardised design
    of ``features`` alone (winsorised, median-imputed, scaled on the same training rows,
    without missing indicators: pairs of Prototype 2 ratios share a denominator and go
    missing together, so their indicators would be collinear), with standard errors
    clustered by ``cert`` because a bank's quarters are not independent draws. An odds
    ratio is the multiplicative change in the quarterly failure odds per one training
    standard deviation of the feature. When ``full_pipeline`` (the fitted 83-feature
    regularised hazard) is given, its L2-shrunk odds ratio for the same feature is
    joined as ``full_model_odds_ratio`` so the two views can be compared. Columns:
    ``feature, camels_group, monotone, coef, odds_ratio, ci_low, ci_high, p_value,
    full_model_odds_ratio, method``; rows in ``features`` order.
    """
    import statsmodels.api as sm
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from bankcanary.features.registry import specs
    from bankcanary.labels.build import horizon_columns
    from bankcanary.models.baselines import coefficient_table
    from bankcanary.models.preprocess import Winsorizer

    names = list(features)
    y_col = horizon_columns(HAZARD_HORIZON)[0]
    rows = frame.loc[train_mask]
    prep = Pipeline(
        [
            ("winsorize", Winsorizer()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    design = prep.fit_transform(rows[names])
    y = rows[y_col].astype(int).to_numpy()
    groups = pd.factorize(rows["cert"].to_numpy())[0]
    result = sm.Logit(y, sm.add_constant(design, has_constant="add")).fit(
        disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": groups}
    )
    if not result.mle_retvals.get("converged", True):
        log.warning("odds-ratio logit did not converge in %d iterations", 500)
    params = np.asarray(result.params)[1:]
    ci = np.asarray(result.conf_int(alpha=alpha))[1:]
    pvalues = np.asarray(result.pvalues)[1:]
    by_name = {s.name: s for s in specs(FEATURE_VERSION)}
    table = pd.DataFrame(
        {
            "feature": names,
            "camels_group": [by_name[n].camels_group if n in by_name else "" for n in names],
            "monotone": [by_name[n].monotone if n in by_name else 0 for n in names],
            "coef": params,
            "odds_ratio": np.exp(params),
            "ci_low": np.exp(ci[:, 0]),
            "ci_high": np.exp(ci[:, 1]),
            "p_value": pvalues,
        }
    )
    full = pd.Series(np.nan, index=names, dtype=float)
    if full_pipeline is not None:
        coefs = coefficient_table(full_pipeline).set_index("feature")["odds_ratio"]
        full = coefs.reindex(names)
    table["full_model_odds_ratio"] = full.to_numpy()
    table["method"] = ODDS_RATIO_METHOD
    return table


REPORT_METRICS = (
    "pr_auc",
    "roc_auc",
    "recall_at_2pct",
    "recall_at_top100",
    "brier",
    "n",
    "n_failures",
)
TUNING_COLUMNS = ("C", "pr_auc_1q", "roc_auc_1q", "pr_auc_4q", "recall_at_2pct_4q", "brier_4q")


def report_path(settings) -> Path:
    return Path(settings.reports_dir) / "p2_hazard.md"


def tuning_table(settings) -> pd.DataFrame:
    """The ``runs/tune_hazard/`` records as one table sorted by 4q validation PR-AUC."""
    rows = [{"C": c["C"], **m} for c, m in read_runs("tune_hazard", settings)]
    if not rows:
        return pd.DataFrame(columns=list(TUNING_COLUMNS))
    table = pd.DataFrame(rows)
    cols = [c for c in TUNING_COLUMNS if c in table.columns]
    return table[cols].sort_values("pr_auc_4q", ascending=False, kind="mergesort")


def _converted_section(horizon: str, block: dict) -> list[str]:
    from bankcanary.models.train import _md_table

    rows = [{"model": "hazard (converted)", **block["hazard"]}]
    rows += [{"model": n, **m} for n, m in block["comparators"].items()]
    table = pd.DataFrame(rows)[["model", *REPORT_METRICS]]
    hz = block["hazard"]
    return [
        f"### {horizon} test rows: hazard converted with `1 - (1 - h)^{horizon[:-1]}` "
        "against the models fitted on that label",
        "",
        _md_table(table),
        "",
        f"Brier references at {horizon}: raw unconverted hazard {hz['brier_raw_hazard']:.5f}, "
        f"constant at the realised failure rate {hz['brier_climatology']:.5f}. Ranking "
        "metrics of the hazard are identical before and after conversion; only Brier moves.",
    ]


def write_hazard_report(settings, path: Path | None = None) -> Path:
    """Render ``reports/p2_hazard.md`` from ``models/hazard/`` and ``runs/tune_hazard/``."""
    from bankcanary.models.train import _md_table

    d = hazard_dir(settings)
    config, metrics = _read_json(d / "config.json"), _read_json(d / "metrics.json")
    if not config or not metrics:
        raise FileNotFoundError(f"no fitted hazard under {d}; run 'bankcanary train-hazard'")
    odds = pd.DataFrame(_read_json(d / "odds_ratios.json").get("rows", []))
    out = path or report_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    t1 = {k: metrics["test"][k] for k in REPORT_METRICS}
    lines = [
        "# Prototype 2 discrete-time hazard model",
        "",
        "Logistic regression on bank-quarter rows whose event is *fails inside the next "
        "quarter* (`y_1q`: failure date in `(avail_date, avail_date + 3 months]`), with the "
        "Prototype 1 preprocessing pipeline (winsorise, median-impute with missing flags, "
        f"scale), no class weighting and L2 at `C = {config['C']}` on all "
        f"{config['n_features']} `features_v2` registry features. Every quarter a bank is "
        "observed is one row with that quarter's covariates (Shumway 2001); a bank that "
        "leaves by merger or voluntary closing has no rows after it exits, which is "
        "right-censoring and needs no deletion. Rows selected through `fixed_split_masks` "
        "at 1q and checked with `assert_no_leakage`: training reports "
        f"{config['train_repdte_min']} to {config['train_repdte_max']} "
        f"({config['n_train']:,} rows, "
        f"{config['positives_train']:,} one-quarter failures); test reports "
        f"{config['test_repdte_min']} to {config['test_repdte_max']} ({config['n_test']:,} rows, "
        f"{config['positives_test']:,}).",
        "",
        "## Choice of C (inner validation slice, rule 6.7)",
        "",
        "Validation = usable reports 2007Q1-2008Q4; inner training = fixed-split training rows "
        "whose 1q window closed before 2007-05-30. Each candidate is scored on the validation "
        "rows at 1q and, converted, at 4q; the winner is the highest 4q PR-AUC, the horizon "
        "the model is compared on.",
        "",
        _md_table(tuning_table(settings)),
        "",
        "## One-quarter test metrics (the hazard's own event)",
        "",
        _md_table(pd.DataFrame([{"model": "hazard", **t1}])),
        "",
        "## Conversion to 4 and 8 quarters",
        "",
        "`p_Hq = 1 - (1 - h)^H` assumes the quarterly hazard estimated from today's covariates "
        "persists at the current level for the next H quarters; covariates are not projected "
        "forward, so a bank whose condition is deteriorating is under-predicted at long horizons "
        "and one that is recovering over-predicted.",
        "",
    ]
    for horizon, block in sorted(metrics.get("converted", {}).items()):
        lines += _converted_section(horizon, block) + [""]
    if not odds.empty:
        cols = ["feature", "camels_group", "odds_ratio", "ci_low", "ci_high", "p_value"]
        cols += ["full_model_odds_ratio"]
        lines += [
            "## Odds ratios per one training standard deviation (95% confidence intervals)",
            "",
            f"Method: {odds['method'].iloc[0]}, on the standardised design of these "
            f"{len(odds)} features alone (no missing indicators). `full_model_odds_ratio` is "
            "the L2-shrunk coefficient of the same feature in the 83-feature hazard.",
            "",
            _md_table(odds[cols]),
            "",
        ]
    lines += [
        "## Sensitivity: censored rows dropped (spec 5, rule 3), 1q",
        "",
        _md_table(pd.DataFrame([metrics["sensitivity_censored_dropped"]])),
        "",
        "## Per failure event (spec 5, rule 6), 1q",
        "",
        _md_table(pd.DataFrame([metrics["per_event"]])),
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", out)
    return out
