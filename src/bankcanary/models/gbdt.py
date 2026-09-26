"""Gradient-boosted trees on the Prototype 2 features (CONTRACT section 13).

One factory, :func:`make_gbdt`, builds a one-step ``Pipeline([("model", estimator)])``
on the raw registry features: trees take missing values natively and split on rank
order, so none of the logit front end (winsorise, impute, scale) applies. In
particular there is no winsoriser: clipping the 0.5% tails erased exactly the values
that carry the interest-rate signal (Silicon Valley Bank at 2022-12-31 had
``unrealized_loss_to_tier1 = -1.04`` and ``adjusted_tier1_leverage = -0.33``; the
winsorised pipeline showed the trees about -0.19 and 3.94), and a booster cannot be hurt
by an outlier the way a linear model can, because a leaf's value is bounded by
``min_samples_leaf`` rows. The estimator is LightGBM when it imports, otherwise
scikit-learn's ``HistGradientBoostingClassifier`` (same histogram algorithm, native NaN,
monotone constraints); the parameters are named once here and translated per backend.

``monotone=True`` hands the estimator the registry's ``monotone`` sign of every feature
in feature order: failure risk may only rise with ``noncurrent_ratio`` and only fall with
``tier1_leverage``, and so on. Whether the production model uses those constraints is
Decision Point 2 (``settings.models.gbdt.monotone``).
"""

from __future__ import annotations

import importlib.util
import logging

from sklearn.pipeline import Pipeline

from bankcanary.features.registry import feature_names, monotone_constraints

log = logging.getLogger(__name__)

BACKENDS: tuple[str, ...] = ("lightgbm", "sklearn")
FEATURE_VERSION = "v2"

#: Backend-neutral hyper-parameter names and their defaults (the tuning grid's centre).
DEFAULT_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_samples_leaf": 50,
    "n_estimators": 200,
}
_LIGHTGBM_NAMES = {
    "learning_rate": "learning_rate",
    "num_leaves": "num_leaves",
    "min_samples_leaf": "min_child_samples",
    "n_estimators": "n_estimators",
}
_SKLEARN_NAMES = {
    "learning_rate": "learning_rate",
    "num_leaves": "max_leaf_nodes",
    "min_samples_leaf": "min_samples_leaf",
    "n_estimators": "max_iter",
}


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def resolve_backend(backend: str | None = None) -> str:
    """``backend`` as given, else ``lightgbm`` when it imports, else ``sklearn``."""
    if backend is None:
        return "lightgbm" if lightgbm_available() else "sklearn"
    if backend not in BACKENDS:
        raise ValueError(f"unknown gbdt backend {backend!r}; choose one of {BACKENDS}")
    return backend


def gbdt_features(version: str = FEATURE_VERSION) -> list[str]:
    """Every registered feature of the table version, in registry order (identifiers never)."""
    return feature_names(version=version)


def constraint_vector(features: list[str], monotone: bool | list[int] | None) -> list[int] | None:
    """Per-feature monotone signs for the estimator, or ``None`` when unconstrained.

    ``True`` takes the registry signs in feature order; an explicit list is used as
    given (it must match the feature count); ``False``/``None`` means no constraint.
    """
    if monotone is None or monotone is False:
        return None
    if monotone is True:
        return monotone_constraints(list(features))
    signs = [int(v) for v in monotone]
    if len(signs) != len(features):
        raise ValueError(f"{len(signs)} monotone signs for {len(features)} features")
    return signs


def _estimator(backend: str, params: dict, signs: list[int] | None, random_state: int):
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        kwargs = {_LIGHTGBM_NAMES[k]: v for k, v in params.items()}
        # ``verbose=-1`` silences the per-fit warnings about tiny leaves.
        return LGBMClassifier(
            monotone_constraints=signs, random_state=random_state, verbose=-1, **kwargs
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    kwargs = {_SKLEARN_NAMES[k]: v for k, v in params.items()}
    # Early stopping is switched off: its automatic validation split would tune the
    # iteration count on rows the tuning script never sees (rule 6.7).
    return HistGradientBoostingClassifier(
        monotonic_cst=signs, early_stopping=False, random_state=random_state, **kwargs
    )


def make_gbdt(
    backend: str | None = None,
    monotone: bool | list[int] | None = None,
    features: list[str] | None = None,
    random_state: int = 0,
    **params,
) -> Pipeline:
    """One-step pipeline around a gradient-boosting classifier on raw features.

    The pipeline object is kept (``named_steps["model"]``, ``.steps``) so every caller
    treats a booster like the other estimators, but it holds no transformer.

    ``params`` use the backend-neutral names ``learning_rate, num_leaves,
    min_samples_leaf, n_estimators`` (see :data:`DEFAULT_PARAMS`); anything else raises so
    a typo cannot silently fall back to a library default. ``features`` is the column
    order the pipeline will be fitted on (default: every v2 registry feature); it fixes
    the position of each monotone sign.
    """
    backend = resolve_backend(backend)
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"unknown gbdt parameter(s) {unknown}; known: {sorted(DEFAULT_PARAMS)}")
    merged = {**DEFAULT_PARAMS, **params}
    names = list(features) if features is not None else gbdt_features()
    signs = constraint_vector(names, monotone)
    estimator = _estimator(backend, merged, signs, random_state)
    return Pipeline([("model", estimator)])


def describe(pipeline: Pipeline) -> dict:
    """``{backend, monotone, params, n_features}`` of a pipeline built by :func:`make_gbdt`.

    Read back from the estimator itself, so a cloned or re-loaded pipeline describes
    itself the same way as the one the factory returned.
    """
    model = pipeline.named_steps["model"]
    kind = type(model).__name__
    if kind == "LGBMClassifier":
        backend, names, signs = "lightgbm", _LIGHTGBM_NAMES, model.monotone_constraints
    elif kind == "HistGradientBoostingClassifier":
        backend, names, signs = "sklearn", _SKLEARN_NAMES, model.monotonic_cst
    else:
        raise TypeError(f"pipeline model step is a {kind}, not a gbdt estimator")
    params = {k: getattr(model, v) for k, v in names.items()}
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None and signs is not None:
        n_features = len(signs)
    return {
        "backend": backend,
        "monotone": signs is not None,
        "params": params,
        "n_features": n_features,
    }


def load_training_frame(settings, version: str = FEATURE_VERSION):
    """``features_<version>`` joined with ``labels`` plus the per-event panel columns.

    Same shape as :func:`bankcanary.models.train.load_training_frame` (which is pinned to
    ``features_v1`` for Prototype 1 reproducibility), with the table version chosen here.
    """
    from bankcanary.models.train import EVENT_COLUMNS, KEY
    from bankcanary.storage.parquet import read_table

    features = read_table(f"features_{version}", settings)
    labels = read_table("labels", settings)
    frame = features.merge(labels, on=KEY, how="inner", validate="one_to_one")
    if len(frame) != len(features):
        log.warning("features_%s and labels do not line up: %d joined", version, len(frame))
    facts = read_table("panel", settings)[KEY + EVENT_COLUMNS]
    return frame.merge(facts, on=KEY, how="left", validate="one_to_one")


def feature_importance(pipeline: Pipeline, features: list[str], X=None, y=None, n_repeats=3):
    """``DataFrame[feature, importance, kind]`` sorted descending.

    LightGBM reports split *gain* (total loss reduction from splits on the feature).
    scikit-learn's histogram booster has no built-in importance, so permutation
    importance (drop in average precision when the column is shuffled, ``n_repeats``
    draws) on the rows passed as ``X, y`` is used instead.
    """
    import pandas as pd

    model = pipeline.named_steps["model"]
    if type(model).__name__ == "LGBMClassifier":
        gain = model.booster_.feature_importance(importance_type="gain")
        table = pd.DataFrame({"feature": features, "importance": gain, "kind": "gain"})
    else:
        from sklearn.inspection import permutation_importance

        if X is None or y is None:
            raise ValueError("permutation importance needs X and y for the sklearn backend")
        res = permutation_importance(
            pipeline, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=0
        )
        table = pd.DataFrame(
            {"feature": features, "importance": res.importances_mean, "kind": "permutation"}
        )
    order = table.sort_values("importance", ascending=False, kind="mergesort")
    return order.reset_index(drop=True)


SETTINGS_COMMENT = (
    "# Gradient boosting (step D6): written by scripts/tune_gbdt.py from the inner validation\n"
    "# slice. backend = the library every gbdt step must use; monotone = Decision Point 2.\n"
)


def write_gbdt_settings(path, backend: str, monotone: bool, params: dict, inner_pr_auc: dict):
    """Replace (or append) the ``models:`` block of ``config/settings.yaml`` in place.

    Only that block is rewritten, as text, so the hand-written comments elsewhere in the
    file survive; the result still round-trips through :func:`bankcanary.config.load_settings`.
    """
    import re
    from pathlib import Path

    import yaml

    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = {
        "models": {
            "gbdt": {
                "backend": backend,
                "monotone": bool(monotone),
                "params": {k: params[k] for k in sorted(params)},
                "inner_pr_auc": {k: round(float(v), 6) for k, v in sorted(inner_pr_auc.items())},
            }
        }
    }
    rendered = SETTINGS_COMMENT + yaml.safe_dump(block, sort_keys=False, default_flow_style=False)
    pattern = re.compile(r"(?:^#[^\n]*\n)*^models:\n(?:(?:[ \t]+[^\n]*)?\n)*", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(lambda _: rendered, text, count=1)
    else:
        text = text.rstrip("\n") + "\n\n" + rendered
    path.write_text(text, encoding="utf-8")
    return path


VARIANTS: tuple[str, ...] = ("gbdt", "gbdt_mono", "logit_v2", "texas")


def make_variant(name: str, settings) -> tuple[Pipeline, list[str]]:
    """``(unfitted pipeline, feature list)`` for one Prototype 2 comparison model.

    ``gbdt`` / ``gbdt_mono`` take the backend and tuned parameters from
    ``settings.models.gbdt`` (constraints on for ``gbdt_mono`` only); ``logit_v2`` is the
    Prototype 1 regularised logit refitted on every v2 feature at the P1 ``LOGIT_C``;
    ``texas`` ranks by the Texas ratio alone.
    """
    from sklearn.linear_model import LogisticRegression

    from bankcanary.models.baselines import LOGIT_C, make_model
    from bankcanary.models.preprocess import make_pipeline

    features = gbdt_features()
    if name in ("gbdt", "gbdt_mono"):
        cfg = settings.models.gbdt
        pipe = make_gbdt(cfg.backend, name == "gbdt_mono", features, **cfg.params)
        return pipe, features
    if name == "logit_v2":
        return make_pipeline(LogisticRegression(C=LOGIT_C, max_iter=2000)), features
    if name == "texas":
        return make_model("texas"), ["texas_ratio"]
    raise ValueError(f"unknown variant {name!r}; choose one of {VARIANTS}")


def fit_on_fixed_split(name, pipeline, features, frame, settings, horizon=4, save=True):
    """Fit on the P1 fixed split, score the test years, save artefacts and log a run.

    Rows come only from :func:`bankcanary.splits.fixed_split_masks` and are re-checked
    with :func:`assert_no_leakage` before ``fit`` (rule 6.2). Returns the
    :class:`~bankcanary.models.train.TrainResult`; ``result.config`` carries the gbdt
    backend/constraints/params when the pipeline is a gradient booster.
    """
    from bankcanary import tracking
    from bankcanary.evaluation.metrics import evaluate_by_year
    from bankcanary.labels.build import horizon_columns
    from bankcanary.models import train as t
    from bankcanary.splits import assert_no_leakage, fixed_split_masks

    y_col = horizon_columns(horizon)[0]
    missing = [c for c in list(features) + [y_col] if c not in frame.columns]
    if missing:
        raise KeyError(f"training frame is missing column(s) {missing}")
    train, test = fixed_split_masks(frame, settings, horizon)
    split = settings.fixed_split
    assert_no_leakage(frame.loc[train], horizon, split.test_start, settings.availability_lag_days)
    if not train.any() or not test.any():
        raise ValueError(f"fixed split for {horizon}q selected no train or no test rows")
    y_train = frame.loc[train, y_col].astype(int).to_numpy()
    log.info("fitting %s on %d rows (%d positives)", name, int(train.sum()), int(y_train.sum()))
    pipeline.fit(frame.loc[train, list(features)], y_train)
    scores = t.score_test_split(pipeline, frame.loc[test], list(features), y_col, horizon)
    metrics = t._score_metrics(scores)
    by_year = evaluate_by_year(scores, "score", "y", "year", tie_col="cert")
    sensitivity, by_event = t._extra_metrics(scores)
    config = {"model": name, "features_version": FEATURE_VERSION, "n_features": len(features)}
    if name.startswith("gbdt"):
        config.update(describe(pipeline))
    config.update(t._split_config(frame, train, test, settings, horizon, y_col))
    result = t.TrainResult(
        name,
        horizon,
        pipeline,
        list(features),
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
        result.paths = t.save_artifacts(result, t.model_dir(settings, name, horizon))
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


REPORT_METRICS = ("pr_auc", "roc_auc", "recall_at_2pct", "recall_at_top100", "n", "n_failures")


def report_path(settings, horizon: int = 4):
    """``reports/p2_gbdt.md`` (``_<H>q`` suffix for a non-primary horizon)."""
    from pathlib import Path

    suffix = "" if horizon == 4 else f"_{horizon}q"
    return Path(settings.reports_dir) / f"p2_gbdt{suffix}.md"


def write_gbdt_report(results: dict, settings, horizon: int, path, importance=None):
    """Write ``reports/p2_gbdt.md``: inner-validation and test metrics side by side.

    ``results`` maps model name to its :class:`TrainResult` (texas, logit_v2, gbdt,
    gbdt_mono in that order when present); the inner-validation PR-AUCs come from
    ``settings.models.gbdt.inner_pr_auc`` (written by ``scripts/tune_gbdt.py``);
    ``importance`` is the :func:`feature_importance` table of the unconstrained booster.
    """
    from pathlib import Path

    import pandas as pd

    from bankcanary.models.train import _md_table, _section_table

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = settings.models.gbdt
    first = next(iter(results.values())).config
    n_features = max(r.config["n_features"] for r in results.values())
    rows = pd.DataFrame(
        [
            {
                "model": n,
                "inner_pr_auc": cfg.inner_pr_auc.get(n, float("nan")),
                **{k: r.metrics[k] for k in REPORT_METRICS},
            }
            for n, r in results.items()
        ]
    )
    params = ", ".join(f"{k}={v}" for k, v in cfg.params.items())
    out = [
        f"# Prototype 2 gradient boosting, {horizon}-quarter horizon",
        "",
        "Fixed out-of-time split (rule 6.2 through `fixed_split_masks`), `features_v2` "
        f"({n_features} registry features). Training reports "
        f"{first['train_repdte_min']} to {first['train_repdte_max']} ({first['n_train']:,} rows, "
        f"{first['positives_train']:,} positives); test reports {first['test_repdte_min']} to "
        f"{first['test_repdte_max']} ({first['n_test']:,} rows, {first['positives_test']:,} "
        "positives).",
        "",
        f"Backend `{cfg.backend}`; parameters chosen by `scripts/tune_gbdt.py` on the inner "
        "validation slice (reports 2007Q1-2008Q4, inner training windows closed before "
        f"2007-05-30): {params}. `inner_pr_auc` is that slice's PR-AUC; every other column "
        "is the untouched test split. `gbdt_mono` applies the registry's monotone signs.",
        "",
        "## Inner-validation and test metrics",
        "",
        _md_table(rows),
        "",
        f"Decision Point 2: `settings.models.gbdt.monotone = {str(cfg.monotone).lower()}` "
        "(the variant with the higher inner-validation PR-AUC); the owner decides after "
        "comparing both rows above.",
        "",
        "## Sensitivity: censored rows dropped (spec 5, rule 3)",
        "",
        _section_table(results, "sensitivity", REPORT_METRICS + ("n_dropped",)),
        "",
        "## Per failure event (spec 5, rule 6)",
        "",
        _section_table(results, "by_event", ("pr_auc", "roc_auc", "recall_at_2pct", "n_events")),
    ]
    if importance is not None:
        kind = importance["kind"].iloc[0]
        out += ["", f"## Top 15 features of `gbdt` by {kind}", ""]
        out.append(_md_table(importance.head(15)[["feature", "importance"]], "{:.2f}"))
    for name, r in results.items():
        if name.startswith("gbdt"):
            out += ["", f"## Per-year metrics, {name}", "", _md_table(r.by_year)]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    return path
