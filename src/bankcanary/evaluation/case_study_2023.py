"""The 2023 case study: would a model trained through 2022Q4 have flagged SVB, Signature
and First Republic?

Two *views* of the same banks are fitted with the same two learners (the P1 regularised
logit and the tuned gradient booster): ``credit_only`` sees the ``features_v1`` columns
(capital, asset quality, earnings, funding, concentrations) and ``rate_aware`` sees every
``features_v2`` column, which adds unrealised securities losses, uninsured deposits, the
trend and macro blocks. Training rows are every bank-quarter whose 4-quarter outcome window
had closed before the 2022Q4 prediction date (2023-03-01, rule 6.2 through
:func:`bankcanary.splits.training_mask`); the fitted models then score the 2022Q3, 2022Q4
and 2023Q1 reports of every bank so the three March-2023 failures can be ranked among
their peers. Nothing here reads the panel beyond the columns the notebook needs; the
functions take a frame so tests can run them on synthetic rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from bankcanary.config import Settings
from bankcanary.labels.build import horizon_columns

log = logging.getLogger(__name__)

BANKS: dict[int, str] = {
    24735: "Silicon Valley Bank",
    57053: "Signature Bank",
    59017: "First Republic Bank",
}
VIEWS: dict[str, str] = {"credit_only": "v1", "rate_aware": "v2"}
MODELS: tuple[str, ...] = ("logit", "gbdt_mono")
BOOSTERS: tuple[str, ...] = ("gbdt", "gbdt_mono")
HORIZON = 4
CUT_REPDTE = "2022-12-31"
SCORE_QUARTERS: tuple[str, ...] = ("2022-09-30", "2022-12-31", "2023-03-31")
PEER_MIN_ASSETS = 10_000_000  # thousands of dollars: banks above $10 billion
CHART_WINDOW = ("2020-03-31", "2023-03-31")
CHART_FEATURES: dict[str, str] = {
    "unrealized_loss_to_tier1": "Unrealised securities loss / Tier 1 capital",
    "uninsured_share": "Uninsured deposits / total deposits",
}
RUN_NAME = "case_study_2023"
KEY = ["cert", "repdte"]
PANEL_EXTRA = ["name", "asset"]
TOP_DRIVERS = 10


def load_frame(settings: Settings) -> pd.DataFrame:
    """``features_v2`` + labels + panel dates (as the sensitivity frame) + ``name, asset``.

    ``asset`` picks the peer group for the charts and ``name`` labels the tables; neither
    enters a model.
    """
    from bankcanary.evaluation.sensitivity import load_frame as sensitivity_frame
    from bankcanary.storage.parquet import read_table

    frame = sensitivity_frame(settings)
    facts = read_table("panel", settings)[KEY + PANEL_EXTRA]
    return frame.merge(facts, on=KEY, how="left", validate="one_to_one")


def view_features(view: str) -> list[str]:
    """Registry columns a view may see: ``features_v1`` names or every ``features_v2`` name."""
    from bankcanary.features.registry import feature_names

    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; choose one of {sorted(VIEWS)}")
    return feature_names(version=VIEWS[view])


def build_model(model: str, view: str, settings: Settings) -> tuple[Pipeline, list[str], dict]:
    """``(unfitted pipeline, features, config)`` for one learner on one view.

    ``logit`` is the Prototype 1 regularised logit at ``LOGIT_C`` (no class weighting);
    ``gbdt_mono`` (the production booster since Decision Point 2) takes backend and tuned
    parameters from ``settings.models.gbdt`` under the registry's monotone signs, and
    ``gbdt`` is the same booster unconstrained (kept for comparison, not fitted by
    default). The parameters were chosen inside the 2002-2008 training period (rule
    6.7); nothing is re-tuned on the 2022-2023 rows.
    """
    from sklearn.linear_model import LogisticRegression

    from bankcanary.models import gbdt
    from bankcanary.models.baselines import LOGIT_C
    from bankcanary.models.preprocess import make_pipeline

    features = view_features(view)
    config: dict = {"analysis": RUN_NAME, "view": view, "model": model}
    config["features_version"] = VIEWS[view]
    if model == "logit":
        pipe = make_pipeline(LogisticRegression(C=LOGIT_C, max_iter=2000))
        config.update({"C": LOGIT_C, "class_weight": None})
        return pipe, features, config
    if model in BOOSTERS:
        cfg = settings.models.gbdt
        monotone = model == "gbdt_mono"
        pipe = gbdt.make_gbdt(cfg.backend, monotone, features, **cfg.params)
        config.update(
            {
                "backend": gbdt.resolve_backend(cfg.backend),
                "monotone": monotone,
                "params": dict(cfg.params),
            }
        )
        return pipe, features, config
    raise ValueError(f"unknown model {model!r}; choose one of {MODELS + BOOSTERS}")


def training_rows(
    frame: pd.DataFrame, settings: Settings, cut_repdte=CUT_REPDTE, horizon: int = HORIZON
) -> pd.Series:
    """Mask of rows whose ``horizon`` window closed before ``cut_repdte``'s prediction date.

    Rule 6.2 via :func:`bankcanary.splits.training_mask` at the settings' availability
    lag, re-checked with :func:`bankcanary.splits.assert_no_leakage` on the selected rows.
    With the 60-day lag and a 2022Q4 cut the last usable report is 2021Q3.
    """
    from bankcanary.splits import assert_no_leakage, training_mask

    lag = int(settings.availability_lag_days)
    mask = training_mask(frame, horizon, cut_repdte, lag)
    assert_no_leakage(frame.loc[mask], horizon, cut_repdte, lag)
    return mask


def scoring_rows(frame: pd.DataFrame, quarters=SCORE_QUARTERS) -> pd.Series:
    """Mask of every report dated in ``quarters``: all banks, whatever their label status.

    The scored reports are ranked as a supervisor would have ranked them on the
    prediction date, so post-failure reports are kept (a failed bank's last report is
    still a report) and the labels are used afterwards only to say who actually failed.
    """
    dates = pd.to_datetime(list(quarters))
    return frame["repdte"].isin(dates).rename("scored")


@dataclass
class CaseStudyFit:
    """One learner on one view: the fitted pipeline, its run config and the ranked scores.

    ``scores`` has ``cert, repdte, name, score, rank, percentile, n_scored, y, fail_date``;
    ``rank`` 1 is the riskiest bank of that quarter and ``percentile`` is the share of the
    quarter's scored banks ranked below it (100 = riskiest).
    """

    view: str
    model: str
    features: list[str]
    pipeline: Pipeline
    config: dict
    scores: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    run_dir: Path | None = None


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Add ``rank`` (1 = highest score), ``percentile`` and ``n_scored`` within each quarter."""
    out = scores.copy()
    grouped = out.groupby("repdte")["score"]
    out["rank"] = grouped.rank(ascending=False, method="min").astype(int)
    out["n_scored"] = grouped.transform("size").astype(int)
    out["percentile"] = 100.0 * (out["n_scored"] - out["rank"]) / out["n_scored"]
    return out.sort_values(["repdte", "rank"], kind="mergesort").reset_index(drop=True)


def _fmt(ts) -> str | None:
    return None if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def _quarter_label(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def _bank_metrics(scores: pd.DataFrame, banks: dict[int, str]) -> dict:
    """Flat ``<slug>_<quarter>_{rank,percentile,probability}`` entries for the run record."""
    out: dict = {}
    for cert, name in banks.items():
        slug = name.lower().replace(" ", "_")
        for row in scores.loc[scores["cert"].eq(cert)].itertuples(index=False):
            stem = f"{slug}_{_quarter_label(row.repdte)}"
            out[f"{stem}_rank"] = int(row.rank)
            out[f"{stem}_percentile"] = round(float(row.percentile), 2)
            out[f"{stem}_probability"] = float(row.score)
    return out


def fit_view(
    frame: pd.DataFrame,
    settings: Settings,
    view: str,
    model: str,
    cut_repdte=CUT_REPDTE,
    quarters=SCORE_QUARTERS,
    horizon: int = HORIZON,
    banks: dict[int, str] | None = None,
    log_run: bool = True,
) -> CaseStudyFit:
    """Fit ``model`` on ``view``'s columns over the rule-6.2 training rows and rank the
    reports dated in ``quarters``.

    The run record (``runs/case_study_2023/``) carries the split facts plus the tracked
    banks' rank, percentile and probability per quarter; ``metrics`` also holds the
    ranking metrics over the scored reports whose 4q label is complete (a handful of
    failures, so they are context rather than evidence).
    """
    from bankcanary import tracking
    from bankcanary.evaluation.metrics import evaluate
    from bankcanary.splits import prediction_date

    banks = BANKS if banks is None else banks
    y_col, _, _, complete_col = horizon_columns(horizon)
    pipeline, features, config = build_model(model, view, settings)
    missing = [c for c in features + [y_col] if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing column(s) {missing}")
    train = training_rows(frame, settings, cut_repdte, horizon)
    scored = scoring_rows(frame, quarters)
    if not train.any() or not scored.any():
        raise ValueError(f"case study {view}/{model}: no training or no scored rows")
    tr = frame.loc[train]
    y_train = tr[y_col].astype(int).to_numpy()
    log.info(
        "case study %s/%s: fitting on %d rows (%d positives, last report %s)",
        view,
        model,
        len(tr),
        int(y_train.sum()),
        _fmt(tr["repdte"].max()),
    )
    pipeline.fit(tr[features], y_train)
    rows = frame.loc[scored]
    scores = pd.DataFrame(
        {
            "cert": rows["cert"].to_numpy(),
            "repdte": rows["repdte"].to_numpy(),
            "name": rows["name"].to_numpy() if "name" in rows else None,
            "score": np.asarray(pipeline.predict_proba(rows[features]))[:, 1],
            "y": rows[y_col].fillna(0).astype(int).to_numpy(),
            "label_complete": rows[complete_col].fillna(False).astype(bool).to_numpy(),
            "fail_date": rows["fail_date"].to_numpy() if "fail_date" in rows else pd.NaT,
        }
    )
    scores = rank_scores(scores)
    lag = int(settings.availability_lag_days)
    config.update(
        {
            "horizon": horizon,
            "label": y_col,
            "availability_lag_days": lag,
            "cut_repdte": _fmt(cut_repdte),
            "first_test_prediction_date": str(prediction_date(cut_repdte, lag).date()),
            "train_repdte_min": _fmt(tr["repdte"].min()),
            "train_repdte_max": _fmt(tr["repdte"].max()),
            "n_train": int(len(tr)),
            "positives_train": int(y_train.sum()),
            "score_quarters": [_fmt(q) for q in quarters],
            "n_scored": int(len(scores)),
            "n_features": len(features),
        }
    )
    complete = scores.loc[scores["label_complete"]]
    metrics = evaluate(complete["y"], complete["score"]) if complete["y"].sum() else {}
    metrics.update(_bank_metrics(scores, banks))
    result = CaseStudyFit(view, model, features, pipeline, config, scores, metrics)
    if log_run:
        run = tracking.start_run(RUN_NAME, config, settings)
        run.log_metrics(metrics)
        result.run_dir = run.finish()
    return result


def bank_table(fits: list[CaseStudyFit], banks: dict[int, str] | None = None) -> pd.DataFrame:
    """Long table ``bank, cert, view, model, quarter, probability, rank, percentile, n_scored``
    for the tracked banks; a bank with no report in a quarter (SVB and Signature failed
    before filing 2023Q1) simply has no row there.
    """
    banks = BANKS if banks is None else banks
    rows = []
    for fit in fits:
        hits = fit.scores.loc[fit.scores["cert"].isin(list(banks))]
        for r in hits.itertuples(index=False):
            rows.append(
                {
                    "bank": banks[int(r.cert)],
                    "cert": int(r.cert),
                    "view": fit.view,
                    "model": fit.model,
                    "quarter": _quarter_label(r.repdte),
                    "probability": float(r.score),
                    "rank": int(r.rank),
                    "percentile": float(r.percentile),
                    "n_scored": int(r.n_scored),
                }
            )
    cols = ["bank", "cert", "view", "model", "quarter", "probability", "rank", "percentile"]
    return pd.DataFrame(rows, columns=cols + ["n_scored"])


def drivers(fit: CaseStudyFit, frame: pd.DataFrame, cert: int, repdte, top: int = TOP_DRIVERS):
    """Per-feature contributions to one bank-quarter's log-odds under ``fit``.

    For the boosters these are SHAP values from the tree explainer (winsorised inputs); for
    ``logit`` they are coefficient x standardised value on the pipeline's own
    winsorise-impute-scale output, so both sum to ``log-odds - baseline`` and read the same
    way: positive pushes the bank towards failure. Columns ``feature, value,
    contribution, direction``; the ``top`` largest absolute contributions, plus the
    ``baseline`` (expected log-odds / intercept) in ``attrs``.
    """
    row = frame.loc[frame["cert"].eq(cert) & frame["repdte"].eq(pd.Timestamp(repdte))]
    if len(row) != 1:
        raise ValueError(f"cert {cert} at {_fmt(repdte)}: {len(row)} row(s), expected one")
    X = row[fit.features]
    if fit.model in BOOSTERS:
        from bankcanary.explain.shap_drivers import shap_matrix

        values, baseline, _ = shap_matrix(fit.pipeline, X)
        names, contrib = list(fit.features), values[0]
    else:
        front, model = fit.pipeline[:-1], fit.pipeline.named_steps["model"]
        z = np.asarray(front.transform(X), dtype=float)[0]
        names = list(front.get_feature_names_out())
        contrib = np.asarray(model.coef_).ravel() * z
        baseline = float(np.ravel(model.intercept_)[0])
    raw = X.iloc[0]
    table = pd.DataFrame(
        {
            "feature": names,
            "value": [raw.get(n, np.nan) for n in names],
            "contribution": np.asarray(contrib, dtype=float),
        }
    )
    table["direction"] = np.where(table["contribution"] > 0, "riskier", "safer")
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
    out = table.head(top).reset_index(drop=True)
    out.attrs["baseline"] = float(baseline)
    out.attrs["total"] = float(np.sum(contrib))
    return out


def peer_bands(
    frame: pd.DataFrame, column: str, min_assets: float = PEER_MIN_ASSETS, window=CHART_WINDOW
) -> pd.DataFrame:
    """Median and 5th/95th percentile of ``column`` per quarter among banks above
    ``min_assets`` (thousands of dollars) inside ``window``; columns ``0.05, 0.5, 0.95``."""
    lo, hi = (pd.Timestamp(w) for w in window)
    peers = frame.loc[frame["asset"].gt(min_assets) & frame["repdte"].between(lo, hi)]
    return peers.groupby("repdte")[column].quantile([0.05, 0.5, 0.95]).unstack()


@dataclass
class CaseStudy:
    """Every fit of the study plus the tracked-bank table and SVB's 2022Q4 drivers."""

    fits: list[CaseStudyFit]
    table: pd.DataFrame
    drivers: dict[tuple[str, str], pd.DataFrame]


def run_case_study(
    frame: pd.DataFrame,
    settings: Settings,
    cut_repdte=CUT_REPDTE,
    quarters=SCORE_QUARTERS,
    banks: dict[int, str] | None = None,
    driver_cert: int = 24735,
    driver_repdte=CUT_REPDTE,
    log_run: bool = True,
) -> CaseStudy:
    """Fit both learners on both views, rank the tracked banks and explain one bank-quarter."""
    banks = BANKS if banks is None else banks
    fits = [
        fit_view(frame, settings, view, model, cut_repdte, quarters, banks=banks, log_run=log_run)
        for view in VIEWS
        for model in MODELS
    ]
    explained = {}
    has_row = (
        frame["cert"].eq(driver_cert) & frame["repdte"].eq(pd.Timestamp(driver_repdte))
    ).any()
    if has_row:
        for fit in fits:
            explained[(fit.view, fit.model)] = drivers(fit, frame, driver_cert, driver_repdte)
    return CaseStudy(fits, bank_table(fits, banks), explained)


def report_path(settings: Settings) -> Path:
    return Path(settings.reports_dir) / "svb_2023_case_study.md"


def _md_table(df: pd.DataFrame) -> str:
    cells = df.map(lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(r) + " |" for r in cells.astype(str).to_numpy().tolist()]
    return "\n".join(lines)


def write_report(study: CaseStudy, settings: Settings, path: Path | None = None) -> Path:
    """Write ``reports/svb_2023_case_study.md``: the split facts, the rank table per view
    and model, and the top drivers of the explained bank-quarter."""
    path = report_path(settings) if path is None else Path(path)
    first = study.fits[0].config
    lines = [
        "# The 2023 case study: credit-only against rate-aware",
        "",
        f"Training rows: every bank-quarter whose {first['horizon']}q outcome window closed "
        f"before the {first['cut_repdte']} prediction date "
        f"({first['first_test_prediction_date']}), "
        f"reports {first['train_repdte_min']}..{first['train_repdte_max']}, "
        f"{first['n_train']:,} rows and {first['positives_train']:,} failures. Scored reports: "
        f"{', '.join(first['score_quarters'])} (every bank). Views: `credit_only` = the "
        f"{len(view_features('credit_only'))} `features_v1` columns, `rate_aware` = the "
        f"{len(view_features('rate_aware'))} `features_v2` columns. Learners: `logit` at the P1 "
        "`LOGIT_C`, `gbdt_mono` at the `settings.models.gbdt` parameters under the registry's "
        "monotone signs (the production booster). Rank 1 is the riskiest bank of the "
        "quarter; percentile is the share of scored banks ranked below it. Run records: "
        f"`runs/{RUN_NAME}/`.",
        "",
    ]
    for (view, model), fit in {(f.view, f.model): f for f in study.fits}.items():
        sub = study.table.loc[study.table["view"].eq(view) & study.table["model"].eq(model)]
        lines += [f"## {view} / {model}", ""]
        lines.append(_md_table(sub.drop(columns=["view", "model"])))
        pooled = {k: fit.metrics[k] for k in ("pr_auc", "recall_at_top100", "n", "n_failures")}
        lines += [
            "",
            "Scored reports with a complete label: "
            + ", ".join(
                f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in pooled.items()
            ),
            "",
        ]
    for (view, model), table in study.drivers.items():
        lines += [
            f"## Drivers, {view} / {model} (baseline log-odds {table.attrs['baseline']:.3f}, "
            f"contributions sum {table.attrs['total']:.3f})",
            "",
            _md_table(table),
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    return path
