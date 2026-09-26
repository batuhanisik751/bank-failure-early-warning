"""Builders: one DataFrame per published table (CONTRACT section 16).

Every builder is a pure function of warehouse frames (``institutions``, ``panel``,
``labels``, ``features_v2``, ``failures``, ``walkforward_scores``) and of the model
artefacts, so the tests can feed synthetic frames. :func:`load_warehouse` reads the real
Parquet tables; :func:`build_all` runs every builder in publish order.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bankcanary.publish import CORE_TABLES

log = logging.getLogger(__name__)

HORIZON = 4
MODELS: tuple[str, ...] = ("gbdt_mono", "hazard")
PRODUCTION_LABEL = "gbdt_mono_production"
FIRST_SCORED_YEAR = 2008
POOLED_YEAR = 0
BAND_HIGH_FRAC = 0.02
BAND_ELEVATED_FRAC = 0.10

#: The 12 key ratios (CONTRACT 16 ``ratios``), all columns of ``features_v2``.
RATIOS: tuple[str, ...] = (
    "equity_to_assets",
    "tier1_leverage",
    "noncurrent_ratio",
    "npa_to_assets",
    "texas_ratio",
    "roa_q",
    "nim_q",
    "efficiency_ratio",
    "brokered_share",
    "uninsured_share",
    "unrealized_loss_to_tier1",
    "construction_to_capital",
)

#: Asset thresholds in thousands of dollars (``settings.peer_asset_buckets_thousands``).
SIZE_THRESHOLDS: tuple[int, ...] = (100_000, 1_000_000, 10_000_000, 100_000_000)
SIZE_LABELS: tuple[str, ...] = ("under_100m", "100m_1b", "1b_10b", "10b_100b", "over_100b")

CHARTER_LABELS: dict[str, str] = {
    "N": "National bank",
    "NM": "State bank, not Fed member",
    "SM": "State bank, Fed member",
    "SB": "Savings bank",
    "SI": "Stock savings institution",
    "SL": "Savings and loan",
    "NC": "Non-insured commercial bank",
    "NS": "Non-insured savings bank",
    "OI": "Insured US branch of a foreign bank",
}


def size_bucket(assets: pd.Series, thresholds=SIZE_THRESHOLDS, labels=SIZE_LABELS) -> pd.Series:
    """Peer size bucket of a total-assets column (thousands of dollars); NaN stays NaN."""
    values = pd.to_numeric(assets, errors="coerce")
    idx = np.searchsorted(
        np.asarray(thresholds, dtype=float), values.to_numpy(dtype=float), side="right"
    )
    out = pd.Series(np.asarray(labels, dtype=object)[np.clip(idx, 0, len(labels) - 1)])
    out.index = values.index
    return out.where(values.notna())


def charter_class_label(bkclass: pd.Series) -> pd.Series:
    """Human label of an FDIC ``BKCLASS`` code (the code itself when unknown)."""
    codes = bkclass.astype("string")
    return codes.map(CHARTER_LABELS).fillna(codes).astype(object).where(codes.notna())


def quarter_label(repdte: pd.Series) -> pd.Series:
    """``2023Q1`` style labels of quarter-end dates."""
    ts = pd.to_datetime(repdte)
    return ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str)


def build_banks(institutions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per ``cert``: institution facts plus the bank's latest panel quarter."""
    latest = (
        panel.sort_values(["cert", "repdte"], kind="mergesort")
        .groupby("cert", as_index=False)
        .tail(1)[["cert", "asset", "exit_reason", "fail_date"]]
        .rename(columns={"asset": "latest_assets"})
    )
    inst = institutions.rename(columns={"stalp": "state"})
    out = inst[
        [
            c
            for c in (
                "cert",
                "name",
                "city",
                "state",
                "bkclass",
                "estymd",
                "endefymd",
                "active",
                "fed_rssd",
                "rssdhcr",
                "latitude",
                "longitude",
            )
            if c in inst.columns
        ]
    ].merge(latest, on="cert", how="left", validate="one_to_one")
    out["charter_class_label"] = charter_class_label(out["bkclass"])
    out["holding_company_name"] = pd.Series([None] * len(out), dtype=object)
    out["size_bucket"] = size_bucket(out["latest_assets"])
    for col in ("estymd", "endefymd", "fail_date"):
        out[col] = pd.to_datetime(out[col]).dt.date.where(out[col].notna(), None)
    columns = [
        "cert",
        "name",
        "city",
        "state",
        "bkclass",
        "charter_class_label",
        "estymd",
        "endefymd",
        "active",
        "fed_rssd",
        "rssdhcr",
        "holding_company_name",
        "latitude",
        "longitude",
        "latest_assets",
        "size_bucket",
        "exit_reason",
        "fail_date",
    ]
    return out[columns].sort_values("cert").reset_index(drop=True)


def build_quarters(
    panel: pd.DataFrame, labels: pd.DataFrame, versions: dict[int | str, str]
) -> pd.DataFrame:
    """One row per report quarter with its label state and the ``gbdt_mono`` that scored it.

    ``versions`` maps a walk-forward test year (int) or ``'production'`` to a
    ``model_version``; quarters before :data:`FIRST_SCORED_YEAR` have neither.
    """
    counts = panel.groupby("repdte").agg(n_banks=("cert", "size"), avail_date=("avail_date", "min"))
    lab = labels.groupby("repdte").agg(
        n_failures_next_4q=("y_4q", "sum"), label_complete_4q=("label_complete_4q", "min")
    )
    out = counts.join(lab, how="left").reset_index()
    out["label"] = quarter_label(out["repdte"])
    years = out["repdte"].dt.year
    backtest = {y for y in versions if isinstance(y, int)}
    out["model_year"] = years.where(years.isin(backtest)).astype("Int64")
    out["model_version"] = [
        versions.get(int(y))
        if int(y) in backtest
        else (versions.get("production") if int(y) >= FIRST_SCORED_YEAR else None)
        for y in years
    ]
    out["n_failures_next_4q"] = out["n_failures_next_4q"].fillna(0).astype(int)
    out["label_complete_4q"] = out["label_complete_4q"].fillna(False).astype(bool)
    out["repdte"] = pd.to_datetime(out["repdte"]).dt.date
    out["avail_date"] = pd.to_datetime(out["avail_date"]).dt.date
    columns = [
        "repdte",
        "label",
        "avail_date",
        "n_banks",
        "n_failures_next_4q",
        "label_complete_4q",
        "model_year",
        "model_version",
    ]
    return out[columns].sort_values("repdte").reset_index(drop=True)


def build_failures(failures: pd.DataFrame) -> pd.DataFrame:
    """The FDIC failures list with ``city``/``state`` split out of ``cityst``.

    Rows without a ``cert`` (a few pre-1970 entries) cannot be keyed and are dropped.
    """
    out = failures[failures["cert"].notna()].copy()
    out["cert"] = out["cert"].astype("int64")
    cityst = out["cityst"].astype("string").fillna("")
    out["city"] = cityst.str.rsplit(",", n=1).str[0].str.strip().astype(object)
    out["state"] = out["pstalp"] if "pstalp" in out.columns else cityst.str[-2:]
    out["fail_date"] = pd.to_datetime(out["fail_date"]).dt.date
    columns = ["cert", "fail_date", "name", "city", "state", "restype1"]
    columns += ["cost", "qbfasset", "qbfdep"]
    out = out[columns].drop_duplicates(["cert", "fail_date"])
    return out.sort_values(["fail_date", "cert"]).reset_index(drop=True)


def _head_size(frac: float, n: int) -> int:
    """``ceil(frac * n)``, at least 1 when ``n > 0`` (the metrics module's top-k rule)."""
    return max(1, min(n, int(np.ceil(frac * n)))) if n > 0 else 0


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Add ``rank`` (1 = riskiest), ``percentile`` and ``band`` within each (repdte, model).

    Ranking uses the raw ``score`` with ``cert`` as tie-breaker (the isotonic map is
    monotone, so the calibrated order agrees wherever it is defined); ``percentile`` is
    the share of the quarter's scored banks ranked below the bank; bands follow CONTRACT
    15: ``high`` = top 2 percent, ``elevated`` = top 2-10 percent, ``low`` otherwise.
    """
    out = scores.sort_values(
        ["repdte", "model", "score", "cert"], ascending=[True, True, False, True], kind="mergesort"
    ).reset_index(drop=True)
    grp = out.groupby(["repdte", "model"], sort=False)
    out["rank"] = (grp.cumcount() + 1).astype(int)
    n = grp["cert"].transform("size").astype(int)
    out["percentile"] = 100.0 * (n - out["rank"]) / n
    high = n.map(lambda k: _head_size(BAND_HIGH_FRAC, int(k)))
    elevated = n.map(lambda k: _head_size(BAND_ELEVATED_FRAC, int(k)))
    out["band"] = np.select(
        [out["rank"] <= high, out["rank"] <= elevated], ["high", "elevated"], default="low"
    )
    return out


def prior_quarter_delta(scores: pd.DataFrame) -> pd.Series:
    """``probability`` minus the same bank's probability one report quarter earlier."""
    prior = scores[["cert", "repdte", "model", "probability"]].copy()
    prior["repdte"] = pd.to_datetime(prior["repdte"]) + pd.offsets.QuarterEnd(1)
    prior = prior.rename(columns={"probability": "_prior"})
    merged = scores[["cert", "repdte", "model"]].merge(
        prior, on=["cert", "repdte", "model"], how="left", validate="one_to_one"
    )
    return (scores["probability"].to_numpy() - merged["_prior"].to_numpy()).astype(float)


def production_mask(labels: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Rows past the last complete test year, minus post-failure reports (as ``explain``)."""
    from bankcanary.evaluation.walkforward import latest_complete_year

    last = latest_complete_year(labels, horizon)
    dropped = labels["dropped_failed_before_avail"].fillna(True).astype(bool)
    return (labels["repdte"].dt.year > last) & ~dropped


def score_with_artefacts(rows: pd.DataFrame, model_dir: Path, model: str) -> pd.DataFrame:
    """Score ``rows`` with ``models/production/<model>/``: raw score and calibrated probability.

    The hazard's quarterly probability is converted to :data:`HORIZON` quarters before
    the isotonic map, exactly as the walk-forward scoring did.
    """
    import joblib

    from bankcanary.models.baselines import score_pipeline
    from bankcanary.models.hazard import convert_hazard

    pipeline = joblib.load(model_dir / "pipeline.joblib")
    calibrator = joblib.load(model_dir / "calibration.joblib")
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    score = np.asarray(score_pipeline(pipeline, rows[list(features)]), dtype=float)
    if model == "hazard":
        score = convert_hazard(score, HORIZON)
    calibrated = np.full(len(score), np.nan)
    ok = np.isfinite(score)
    if ok.any():
        calibrated[ok] = calibrator.predict(score[ok])
    return pd.DataFrame(
        {
            "cert": rows["cert"].to_numpy(),
            "repdte": pd.to_datetime(rows["repdte"]).to_numpy(),
            "horizon": HORIZON,
            "model": model,
            "test_year": pd.array([None] * len(rows), dtype="Int64"),
            "score": score,
            "score_calibrated": calibrated,
        }
    )


def build_scores(
    walkforward: pd.DataFrame, production: pd.DataFrame | None, versions: dict
) -> pd.DataFrame:
    """The ``scores`` table: backtest rows (own test-year model) plus production rows.

    ``walkforward`` is the ``walkforward_scores`` warehouse table (any horizon or model:
    only :data:`HORIZON` and :data:`MODELS` are kept); ``production`` holds rows in the
    same shape with ``test_year`` null; ``versions`` maps ``(model, test_year)`` and
    ``(model, 'production')`` to a ``model_version``.
    """
    wf = walkforward[(walkforward["horizon"] == HORIZON) & walkforward["model"].isin(MODELS)].copy()
    wf["test_year"] = wf["test_year"].astype("Int64")
    parts = [wf] if production is None else [wf, production[production["model"].isin(MODELS)]]
    cols = ["cert", "repdte", "horizon", "model", "test_year", "score", "score_calibrated"]
    table = pd.concat([p[cols] for p in parts], ignore_index=True)
    table["repdte"] = pd.to_datetime(table["repdte"])
    table = table[table["repdte"].dt.year >= FIRST_SCORED_YEAR]
    table = table.drop_duplicates(["cert", "repdte", "model"], keep="last")
    table = rank_scores(table.rename(columns={"score_calibrated": "probability"}))
    table["delta_prob_prior_q"] = prior_quarter_delta(table)
    table["model_version"] = [
        versions.get((m, int(y)) if pd.notna(y) else (m, "production"))
        for m, y in zip(table["model"], table["test_year"])
    ]
    table["repdte"] = table["repdte"].dt.date
    columns = [
        "cert",
        "repdte",
        "model",
        "horizon",
        "score",
        "probability",
        "rank",
        "percentile",
        "band",
        "delta_prob_prior_q",
        "model_version",
    ]
    return table[columns].reset_index(drop=True)


def build_ratios(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """``total_assets`` + the 12 ratios with peer percentiles (size bucket x region, quarter).

    The percentile is the share of the bank's peers with a lower value (0-100), NaN when
    the ratio itself is missing; ``size_bucket``/``region`` are kept for the peer stats
    and dropped by :func:`build_all` before publishing.
    """
    from bankcanary.features.structure_p2 import region_of

    facts = panel[["cert", "repdte", "asset", "stalp"]].rename(columns={"asset": "total_assets"})
    out = features[["cert", "repdte", *RATIOS]].merge(
        facts, on=["cert", "repdte"], how="left", validate="one_to_one"
    )
    out["size_bucket"] = size_bucket(out["total_assets"]).fillna("unknown")
    out["region"] = region_of(out["stalp"]).fillna("other")
    grp = out.groupby(["repdte", "size_bucket", "region"], sort=False)
    for ratio in RATIOS:
        out[f"{ratio}_pct"] = 100.0 * grp[ratio].rank(method="average", pct=True) - 50.0 / (
            grp[ratio].transform("count").clip(lower=1)
        )
    out["repdte"] = pd.to_datetime(out["repdte"]).dt.date
    columns = ["cert", "repdte", "total_assets"]
    for ratio in RATIOS:
        columns += [ratio, f"{ratio}_pct"]
    return (
        out[[*columns, "size_bucket", "region"]]
        .sort_values(["cert", "repdte"])
        .reset_index(drop=True)
    )


def build_peer_stats(ratios: pd.DataFrame) -> pd.DataFrame:
    """``p10, p50, p90, n`` per (repdte, size_bucket, region, ratio) from :func:`build_ratios`."""
    long = ratios.melt(
        id_vars=["repdte", "size_bucket", "region"],
        value_vars=list(RATIOS),
        var_name="ratio",
        value_name="value",
    ).dropna(subset=["value"])
    grp = long.groupby(["repdte", "size_bucket", "region", "ratio"], sort=True)["value"]
    out = grp.quantile([0.1, 0.5, 0.9]).unstack()
    out.columns = ["p10", "p50", "p90"]
    out["n"] = grp.size().astype(int)
    return out.reset_index()


def build_walkforward_metrics(walkforward: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    """Per test year and pooled (``test_year = 0``) metrics with bootstrap intervals."""
    from bankcanary.evaluation.metrics import cluster_bootstrap_ci, evaluate
    from bankcanary.models.hazard import brier

    wf = walkforward[(walkforward["horizon"] == HORIZON) & walkforward["model"].isin(models)]
    rows = []
    for model in [m for m in models if m in set(wf["model"])]:
        sub = wf[wf["model"] == model]
        slices = [(POOLED_YEAR, sub)] + [(int(y), g) for y, g in sub.groupby("test_year")]
        for year, r in slices:
            y, s, c = r["y"].to_numpy(dtype=float), r["score"].to_numpy(dtype=float), r["cert"]
            point = evaluate(y, s, tie_breaker=c)
            ci = cluster_bootstrap_ci(y, s, c, tie_breaker=c, k_frac=BAND_HIGH_FRAC)
            cal = r["score_calibrated"].to_numpy(dtype=float)
            have = ~np.isnan(cal)
            rows.append(
                {
                    "model": model,
                    "horizon": HORIZON,
                    "test_year": year,
                    "n": int(point["n"]),
                    "n_failures": int(point["n_failures"]),
                    "pr_auc": point["pr_auc"],
                    "pr_auc_lo": ci["pr_auc_ci"][0],
                    "pr_auc_hi": ci["pr_auc_ci"][1],
                    "recall_at_2pct": point["recall_at_2pct"],
                    "recall_lo": ci["recall_at_2pct_ci"][0],
                    "recall_hi": ci["recall_at_2pct_ci"][1],
                    "roc_auc": point["roc_auc"],
                    "brier_raw": brier(y, np.nan_to_num(s, nan=0.0)),
                    "brier_calibrated": brier(y[have], cal[have]) if have.any() else np.nan,
                    "low_confidence": bool(ci["low_confidence"]),
                }
            )
    return pd.DataFrame(rows)


def _git_first_commit(path: Path, repo: Path) -> tuple[str, str] | None:
    """``(short sha, committer date)`` of the commit that added ``path``, or ``None``."""
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%h %cI", "--", str(path)],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.strip().splitlines()
    if not lines:
        return None
    sha, when = lines[-1].split(" ", 1)
    return sha, when


def walkforward_version(
    year: int, model: str, models_dir: Path, runs_dir: Path, repo: Path
) -> dict | None:
    """A ``model_versions`` row for one walk-forward fit, or ``None`` when it is not saved."""
    from bankcanary.tracking import run_id

    config_path = models_dir / "walkforward" / str(year) / model / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    record = runs_dir / "walkforward" / run_id("walkforward", config) / "metrics.json"
    found = _git_first_commit(record, repo) or ("unknown", None)
    train_end = str(config.get("train_repdte_max"))
    return {
        "model_version": f"{model}-{train_end}-{found[0]}",
        "model": model,
        "train_end_repdte": train_end,
        "git_sha": found[0],
        "trained_at": found[1],
        "features_version": config.get("features_version"),
        "notes": f"Walk-forward {model} for test year {year}, trained through {train_end}.",
    }


def build_model_versions(
    production_dir: Path, models_dir: Path, runs_dir: Path, repo: Path, years=()
) -> tuple[pd.DataFrame, dict]:
    """``model_versions`` (production artefacts + every saved walk-forward fit) and the
    ``{(model, year|'production'): model_version}`` lookup the other builders use."""
    rows, lookup = [], {}
    for model in MODELS:
        path = production_dir / model / "model_version.json"
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            rows.append({k: meta.get(k) for k in MODEL_VERSION_COLUMNS})
            lookup[(model, "production")] = meta["model_version"]
        for year in years:
            row = walkforward_version(int(year), model, models_dir, runs_dir, repo)
            if row is not None:
                rows.append(row)
                lookup[(model, int(year))] = row["model_version"]
    out = pd.DataFrame(rows, columns=MODEL_VERSION_COLUMNS).drop_duplicates("model_version")
    out["train_end_repdte"] = pd.to_datetime(out["train_end_repdte"]).dt.date
    out["trained_at"] = pd.to_datetime(out["trained_at"], utc=True, errors="coerce")
    return out.reset_index(drop=True), lookup


MODEL_VERSION_COLUMNS: tuple[str, ...] = (
    "model_version",
    "model",
    "train_end_repdte",
    "git_sha",
    "trained_at",
    "features_version",
    "notes",
)


@dataclass
class Warehouse:
    """The warehouse frames every builder reads (loaded once by :func:`load_warehouse`)."""

    institutions: pd.DataFrame
    panel: pd.DataFrame
    labels: pd.DataFrame
    features: pd.DataFrame
    failures: pd.DataFrame
    walkforward: pd.DataFrame
    production: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


def load_warehouse(settings, tables: set[str] | None = None) -> Warehouse:
    """Read the Parquet tables the requested ``tables`` need (the core tables by default).

    Page tables (:data:`bankcanary.publish.PAGE_TABLES`) pull in the core frames they
    are built from; the warehouse ``drivers`` table lands in ``extra['drivers']``.
    """
    from bankcanary.publish.pages import core_dependencies
    from bankcanary.storage.parquet import read_table

    need = set(tables or CORE_TABLES)
    need |= core_dependencies(need)
    empty = pd.DataFrame()
    wants_scores = bool(need & {"scores", "quarters", "walkforward_metrics"})
    wants_panel = bool(need & {"banks", "quarters", "ratios", "peer_stats", "rate_shock_scores"})
    wants_features = bool(need & {"ratios", "peer_stats", "scores", "rate_shock_scores"})
    return Warehouse(
        institutions=read_table("institutions", settings) if "banks" in need else empty,
        panel=read_table("panel", settings) if wants_panel else empty,
        labels=read_table("labels", settings),
        features=read_table("features_v2", settings) if wants_features else empty,
        failures=read_table("failures", settings) if "failures" in need else empty,
        walkforward=read_table("walkforward_scores", settings) if wants_scores else empty,
        extra={"drivers": read_table("drivers", settings)} if "drivers" in need else {},
    )


def build_all(
    wh: Warehouse, settings, tables=None, repo: Path | None = None
) -> dict[str, pd.DataFrame]:
    """Every requested table (publish order) as a ``{name: frame}`` dict.

    Production scores are computed here from ``features_v2`` and ``models/production/``
    when ``wh.production`` is not given, for every quarter past the last complete test
    year. ``ratios`` and ``peer_stats`` share one :func:`build_ratios` pass; the
    percentiles and peer stats use every quarter, but only the scored quarters
    (:data:`FIRST_SCORED_YEAR` on) are published in ``ratios`` (size budget, CONTRACT 16).
    """
    from bankcanary.evaluation.walkforward import test_years

    want = [t for t in CORE_TABLES if tables is None or t in set(tables)]
    production_dir = Path(settings.models_dir) / "production"
    repo = repo or Path(settings.data_dir).resolve().parent
    years = test_years(wh.labels, HORIZON) if len(wh.labels) else []
    versions, lookup = build_model_versions(
        production_dir, Path(settings.models_dir), Path(settings.runs_dir), repo, years
    )
    out: dict[str, pd.DataFrame] = {}
    if "model_versions" in want:
        out["model_versions"] = versions
    if "banks" in want:
        out["banks"] = build_banks(wh.institutions, wh.panel)
    if "quarters" in want:
        by_year = {y: v for (m, y), v in lookup.items() if m == "gbdt_mono"}
        out["quarters"] = build_quarters(wh.panel, wh.labels, by_year)
    if "scores" in want:
        production = wh.production
        if production is None and len(wh.features):
            mask = production_mask(wh.labels, HORIZON)
            keys = wh.labels.loc[mask, ["cert", "repdte"]]
            rows = wh.features.merge(keys, on=["cert", "repdte"], how="inner")
            log.info("scoring %d production rows with %s", len(rows), production_dir)
            production = pd.concat(
                [score_with_artefacts(rows, production_dir / m, m) for m in MODELS],
                ignore_index=True,
            )
        out["scores"] = build_scores(wh.walkforward, production, lookup)
    if {"ratios", "peer_stats"} & set(want):
        ratios = build_ratios(wh.features, wh.panel)
        if "peer_stats" in want:
            out["peer_stats"] = build_peer_stats(ratios)
        if "ratios" in want:
            scored = pd.to_datetime(ratios["repdte"]).dt.year >= FIRST_SCORED_YEAR
            out["ratios"] = ratios.loc[scored].drop(columns=["size_bucket", "region"])
            out["ratios"] = out["ratios"].reset_index(drop=True)
    if "failures" in want:
        out["failures"] = build_failures(wh.failures)
    if "walkforward_metrics" in want:
        out["walkforward_metrics"] = build_walkforward_metrics(wh.walkforward)
    return {t: out[t] for t in want if t in out}


def pipeline_run_row(
    run_id: str,
    started_at,
    finished_at,
    status: str,
    rows_written: dict,
    log_text: str,
    latest_repdte=None,
    new_quarter: bool = False,
) -> pd.DataFrame:
    """One ``pipeline_runs`` row (``rows_written`` is stored as JSON)."""
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "status": status,
                "latest_repdte": latest_repdte,
                "new_quarter": bool(new_quarter),
                "rows_written": json.dumps(rows_written),
                "log": log_text,
            }
        ]
    )
