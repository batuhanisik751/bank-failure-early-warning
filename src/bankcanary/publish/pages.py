"""Page-specific tables (CONTRACT 16): drivers, the 2023 case study, rate shocks, map quarters.

Every builder is a pure function of the core frames :mod:`bankcanary.publish.core` already
built (``scores``, ``banks``, ``failures``), of warehouse frames and, for the rate-shock
grid, of a scoring callable, so the tests feed synthetic frames and a fake scorer.
:func:`build_all` runs the builders in publish order; :func:`core_dependencies` names the
core frames a set of page tables needs, so ``bankcanary publish --tables drivers`` still
builds ``scores`` and ``failures`` in memory without writing them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from bankcanary.publish import PAGE_TABLES
from bankcanary.publish.core import HORIZON, _head_size, rank_scores

log = logging.getLogger(__name__)

DRIVER_MODEL = "gbdt_mono"
#: ``model`` label of the warehouse driver rows the production booster explained.
DRIVER_PRODUCTION_LABEL = f"{DRIVER_MODEL}_production"
DRIVER_LATEST_QUARTERS = 4
#: Rows kept per bank-quarter: the largest absolute contributions, either sign.
DRIVER_ROWS = 5
DRIVER_TOP_FRAC = 0.05
SHOCKS_BP: tuple[int, ...] = (100, 200, 300, 400)
DURATIONS_YEARS: tuple[int, ...] = (2, 3, 4, 5, 6)
SECURITIES_COLUMNS: tuple[str, ...] = ("scaf", "scaa", "schf", "scha", "rbct1j", "asset")

#: Core frames each page table reads (``case_study_*`` load their own frame).
CORE_DEPENDENCIES: dict[str, frozenset[str]] = {
    "drivers": frozenset({"scores", "failures"}),
    "case_study_2023": frozenset(),
    "case_study_series": frozenset(),
    "rate_shock_scores": frozenset({"scores"}),
    "map_quarters": frozenset({"scores", "banks", "failures"}),
}


def core_dependencies(tables) -> set[str]:
    """Core tables that must be built in memory before the page ``tables``."""
    return set().union(*(CORE_DEPENDENCIES.get(t, frozenset()) for t in tables))


def feature_labels(version: str = "v2") -> dict[str, str]:
    """Plain-English label per registry feature: the first clause of its explanation.

    The registry explanation reads ``<what it is>; <why it matters>. <monotone note>.``,
    so the text before the first ``;``, ``:`` or sentence end is the title, capitalised.
    """
    from bankcanary.features.registry import specs

    out: dict[str, str] = {}
    for item in specs(version):
        text = item.explanation.strip()
        for sep in (";", ". ", ":"):
            text = text.split(sep, 1)[0]
        text = text.rstrip(". ").strip()
        out[item.name] = text[:1].upper() + text[1:]
    return out


def driver_keys(
    scores: pd.DataFrame,
    failures: pd.DataFrame,
    latest_quarters: int = DRIVER_LATEST_QUARTERS,
    top_frac: float = DRIVER_TOP_FRAC,
) -> pd.DataFrame:
    """``cert, repdte`` pairs the ``drivers`` table keeps (CONTRACT 16 subset rule).

    The latest ``latest_quarters`` scored quarters of every bank, every scored quarter
    of every bank on the failures list, and every bank-quarter ranked in the top
    ``top_frac`` of its quarter (``ceil``, at least one) by the ``gbdt_mono`` score.
    """
    s = scores.loc[scores["model"] == DRIVER_MODEL, ["cert", "repdte", "rank"]].copy()
    s["repdte"] = pd.to_datetime(s["repdte"])
    s = s.sort_values(["cert", "repdte"], kind="mergesort")
    latest = s.groupby("cert").tail(latest_quarters)
    failed = s[s["cert"].isin(set(failures["cert"].dropna().astype("int64")))]
    n = s.groupby("repdte")["cert"].transform("size")
    top = s[s["rank"] <= n.map(lambda k: _head_size(top_frac, int(k)))]
    keys = pd.concat([latest, failed, top])[["cert", "repdte"]].drop_duplicates()
    return keys.sort_values(["cert", "repdte"]).reset_index(drop=True)


DRIVER_COLUMNS: tuple[str, ...] = (
    "cert",
    "repdte",
    "model",
    "rank",
    "feature",
    "feature_label",
    "shap_value",
    "feature_value",
    "direction",
)


def build_drivers(
    drivers: pd.DataFrame,
    scores: pd.DataFrame,
    failures: pd.DataFrame,
    labels: dict[str, str] | None = None,
    rows: int = DRIVER_ROWS,
) -> pd.DataFrame:
    """The ``drivers`` table: warehouse SHAP rows of the monotone booster, subset rule applied.

    ``drivers`` is the warehouse table :mod:`bankcanary.explain.shap_drivers` builds
    (``gbdt_mono`` rows for the backtest years, ``gbdt_mono_production`` rows past them);
    both are published under ``model = 'gbdt_mono'``, the model whose ``scores`` rows they
    explain. Only bank-quarters in :func:`driver_keys` are kept, and of each one's ten
    warehouse rows the ``rows`` largest absolute contributions, re-ranked ``1..rows`` by
    ``|shap_value|`` (``direction`` still says which way each pushes; the size budget
    deviation is recorded in CONTRACT 16). Every row gets the registry's plain-English
    ``feature_label`` (the feature name when unregistered).
    """
    labels = feature_labels() if labels is None else labels
    d = drivers[drivers["model"].isin((DRIVER_MODEL, DRIVER_PRODUCTION_LABEL))].copy()
    d["model"] = DRIVER_MODEL
    d["repdte"] = pd.to_datetime(d["repdte"])
    keys = driver_keys(scores, failures)
    out = d.merge(keys, on=["cert", "repdte"], how="inner")
    out = out.drop_duplicates(["cert", "repdte", "model", "rank"], keep="last")
    out["_abs"] = out["shap_value"].abs()
    out = out.sort_values(
        ["cert", "repdte", "_abs", "rank"], ascending=[True, True, False, True], kind="mergesort"
    )
    out = out[out.groupby(["cert", "repdte"]).cumcount() < rows].copy()
    out["rank"] = out.groupby(["cert", "repdte"]).cumcount() + 1
    out["feature_label"] = out["feature"].map(labels).fillna(out["feature"])
    out["rank"] = out["rank"].astype(int)
    out["repdte"] = out["repdte"].dt.date
    out = out[list(DRIVER_COLUMNS)].sort_values(["cert", "repdte", "rank"], kind="mergesort")
    log.info("drivers: %d rows for %d bank-quarters", len(out), len(keys))
    return out.reset_index(drop=True)


def build_map_quarters(
    scores: pd.DataFrame, banks: pd.DataFrame, failures: pd.DataFrame, horizon: int = HORIZON
) -> pd.DataFrame:
    """``map_quarters``: every ``gbdt_mono`` bank-quarter with head-office coordinates.

    ``failed_this_quarter`` marks the report a failure follows: the bank's ``fail_date``
    lies after ``repdte`` and on or before the next quarter end, or the row is the bank's
    last scored report and the failure comes within ``horizon`` quarters of it (a bank
    whose final report predates its failure by more than one quarter is still marked
    once, on that final report).
    """
    s = scores.loc[scores["model"] == DRIVER_MODEL, ["cert", "repdte", "band", "probability"]]
    s = s.copy()
    s["repdte"] = pd.to_datetime(s["repdte"])
    coords = banks[["cert", "latitude", "longitude"]].dropna(subset=["latitude", "longitude"])
    out = s.merge(coords, on="cert", how="inner")
    last = out["repdte"].eq(out.groupby("cert")["repdte"].transform("max"))
    fails = failures.loc[failures["cert"].notna(), ["cert", "fail_date"]].copy()
    fails["cert"] = fails["cert"].astype("int64")
    fails["fail_date"] = pd.to_datetime(fails["fail_date"])
    hit = out[["cert", "repdte"]].assign(_last=last).merge(fails, on="cert", how="inner")
    after = hit["fail_date"].gt(hit["repdte"])
    next_q = hit["repdte"] + pd.offsets.QuarterEnd(1)
    within = hit["fail_date"].le(hit["repdte"] + pd.offsets.QuarterEnd(horizon))
    hit["failed_this_quarter"] = after & (hit["fail_date"].le(next_q) | (hit["_last"] & within))
    flags = hit.groupby(["cert", "repdte"], as_index=False)["failed_this_quarter"].any()
    out = out.merge(flags, on=["cert", "repdte"], how="left")
    out["failed_this_quarter"] = out["failed_this_quarter"].fillna(False).astype(bool)
    out["repdte"] = out["repdte"].dt.date
    columns = ["repdte", "cert", "latitude", "longitude", "band", "probability"]
    out = out[[*columns, "failed_this_quarter"]].sort_values(["repdte", "cert"])
    return out.reset_index(drop=True)


def shocked_ratios(rows: pd.DataFrame, shock_bp: int, duration_years: int) -> pd.DataFrame:
    """``extra_loss, unrealized_loss_to_tier1, adjusted_tier1_leverage`` after a rate shock.

    A parallel shock of ``shock_bp`` basis points on securities of ``duration_years``
    modified duration loses ``duration x shock / 10000`` of the amortised cost of both
    books (``scaa`` + ``scha``, missing = 0); the two sensitivity ratios are recomputed
    exactly as :mod:`bankcanary.features.sensitivity` does, with the extra loss added to
    the unrealised total. A simplified approximation: no convexity, no hedges, no tax.
    """
    from bankcanary.features.sensitivity import unrealized
    from bankcanary.features.spec import safe_ratio

    afs_cost = rows["scaa"].astype("float64").fillna(0.0)
    htm_cost = rows["scha"].astype("float64").fillna(0.0)
    securities = afs_cost + htm_cost
    extra = -float(duration_years) * float(shock_bp) / 10_000.0 * securities
    afs, htm = unrealized(rows)
    total = afs + htm + extra
    tier1 = rows["rbct1j"].astype("float64")
    out = pd.DataFrame(index=rows.index)
    out["extra_loss"] = extra
    out["unrealized_loss_to_tier1"] = safe_ratio(total, tier1)
    out["adjusted_tier1_leverage"] = safe_ratio(tier1 + total.clip(upper=0.0), rows["asset"])
    out["adjusted_tier1_leverage"] *= 100.0
    return out


def build_rate_shock(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    scorer: Callable[[pd.DataFrame], pd.DataFrame],
    keys: pd.DataFrame,
    shocks=SHOCKS_BP,
    durations=DURATIONS_YEARS,
) -> pd.DataFrame:
    """``rate_shock_scores``: the shock x duration grid re-scored for one quarter's banks.

    ``keys`` (``cert, repdte``) names the bank-quarters (the latest scored quarter);
    ``scorer(rows)`` returns ``cert, score, score_calibrated`` for feature rows in the
    ``features_v2`` shape (:func:`bankcanary.publish.core.score_with_artefacts` bound to
    ``models/production/gbdt_mono``). Rank and band are per scenario, as in ``scores``.
    """
    keys = keys[["cert", "repdte"]].drop_duplicates().copy()
    keys["repdte"] = pd.to_datetime(keys["repdte"])
    base = features.merge(keys, on=["cert", "repdte"], how="inner")
    facts = panel[["cert", "repdte", *SECURITIES_COLUMNS]].copy()
    facts["repdte"] = pd.to_datetime(facts["repdte"])
    base = base.merge(facts, on=["cert", "repdte"], how="left", validate="one_to_one")
    parts = []
    for shock in shocks:
        for duration in durations:
            rows = base.copy()
            shocked = shocked_ratios(rows, shock, duration)
            rows["unrealized_loss_to_tier1"] = shocked["unrealized_loss_to_tier1"]
            rows["adjusted_tier1_leverage"] = shocked["adjusted_tier1_leverage"]
            scored = scorer(rows)
            frame = pd.DataFrame(
                {
                    "cert": scored["cert"].to_numpy(),
                    "repdte": rows["repdte"].to_numpy(),
                    "model": DRIVER_MODEL,
                    "score": np.asarray(scored["score"], dtype=float),
                    "probability": np.asarray(scored["score_calibrated"], dtype=float),
                    "shock_bp": int(shock),
                    "duration_years": int(duration),
                }
            )
            ranked = rank_scores(frame).set_index("cert")
            out = shocked.set_index(rows["cert"].to_numpy()).join(ranked, how="inner")
            parts.append(out.rename_axis("cert").reset_index())
            log.info("rate shock %d bp x %d y: %d banks scored", shock, duration, len(out))
    columns = ["cert", "shock_bp", "duration_years", "extra_loss", "adjusted_tier1_leverage"]
    columns += ["unrealized_loss_to_tier1", "probability", "rank", "band"]
    table = pd.concat(parts, ignore_index=True)[columns]
    return table.sort_values(["shock_bp", "duration_years", "rank"]).reset_index(drop=True)


CASE_STUDY_COLUMNS: tuple[str, ...] = (
    "cert",
    "quarter",
    "view",
    "model",
    "probability",
    "rank",
    "percentile",
    "n_scored",
)
SERIES_COLUMNS: tuple[str, ...] = (
    "cert",
    "repdte",
    "unrealized_loss_to_tier1",
    "uninsured_share",
    "peer_p50_unrealized",
    "peer_p05_unrealized",
    "peer_p50_uninsured",
    "peer_p95_uninsured",
)


def build_case_study_series(
    frame: pd.DataFrame, banks=None, window=None, min_assets: float | None = None
) -> pd.DataFrame:
    """``case_study_series``: the tracked banks' two rate-risk ratios against the peer bands.

    Peers are the banks above ``min_assets`` (thousands of dollars; the case study's
    $10 billion) in the same quarter, inside ``window``; the bands are the median and
    the 5th percentile of the unrealised-loss ratio (losses are negative, so the 5th
    percentile is the worst tail) and the median and 95th percentile of the uninsured
    share, exactly as :func:`bankcanary.evaluation.case_study_2023.peer_bands` draws them.
    """
    from bankcanary.evaluation import case_study_2023 as cs

    banks = cs.BANKS if banks is None else banks
    window = cs.CHART_WINDOW if window is None else window
    min_assets = cs.PEER_MIN_ASSETS if min_assets is None else min_assets
    lo, hi = (pd.Timestamp(w) for w in window)
    data = frame.assign(repdte=pd.to_datetime(frame["repdte"]))
    rows = data.loc[
        data["cert"].isin(list(banks)) & data["repdte"].between(lo, hi),
        ["cert", "repdte", "unrealized_loss_to_tier1", "uninsured_share"],
    ]
    unreal = cs.peer_bands(data, "unrealized_loss_to_tier1", min_assets, window)
    uninsured = cs.peer_bands(data, "uninsured_share", min_assets, window)
    peers = pd.DataFrame(
        {
            "repdte": unreal.index,
            "peer_p50_unrealized": unreal.get(0.5, np.nan).to_numpy(),
            "peer_p05_unrealized": unreal.get(0.05, np.nan).to_numpy(),
        }
    ).merge(
        pd.DataFrame(
            {
                "repdte": uninsured.index,
                "peer_p50_uninsured": uninsured.get(0.5, np.nan).to_numpy(),
                "peer_p95_uninsured": uninsured.get(0.95, np.nan).to_numpy(),
            }
        ),
        on="repdte",
        how="outer",
    )
    out = rows.merge(peers, on="repdte", how="left")
    out["cert"] = out["cert"].astype("int64")
    out["repdte"] = out["repdte"].dt.date
    return out[list(SERIES_COLUMNS)].sort_values(["cert", "repdte"]).reset_index(drop=True)


def build_case_study(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(case_study_2023, case_study_series)`` by re-running the case study's fits.

    :func:`bankcanary.evaluation.case_study_2023.run_case_study` fits the two learners on
    the two views under the cut-off rule and ranks SVB, Signature and First Republic in
    2022Q3-2023Q1; no run record is logged here (the study's own command logs those).
    """
    from bankcanary.evaluation import case_study_2023 as cs

    study = cs.run_case_study(frame, settings, log_run=False)
    table = study.table[list(CASE_STUDY_COLUMNS)].copy()
    table["cert"] = table["cert"].astype("int64")
    table = table.sort_values(["cert", "quarter", "view", "model"]).reset_index(drop=True)
    return table, build_case_study_series(frame)


def latest_scored_keys(scores: pd.DataFrame, model: str = DRIVER_MODEL) -> pd.DataFrame:
    """``cert, repdte`` of every bank in the latest quarter ``model`` scored."""
    s = scores.loc[scores["model"] == model, ["cert", "repdte"]].copy()
    s["repdte"] = pd.to_datetime(s["repdte"])
    return s[s["repdte"] == s["repdte"].max()].reset_index(drop=True)


def production_scorer(
    settings, model: str = DRIVER_MODEL
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """The ``models/production/<model>`` pipeline + calibrator as a ``rows -> scores`` callable."""
    from pathlib import Path

    from bankcanary.publish.core import score_with_artefacts

    model_dir = Path(settings.models_dir) / "production" / model
    return lambda rows: score_with_artefacts(rows, model_dir, model)


def build_all(
    wh, frames: dict[str, pd.DataFrame], settings, tables=None
) -> dict[str, pd.DataFrame]:
    """Every requested page table (publish order) as a ``{name: frame}`` dict.

    ``wh`` is the :class:`bankcanary.publish.core.Warehouse` (``extra['drivers']`` holds
    the warehouse SHAP table) and ``frames`` the core frames named by
    :func:`core_dependencies`. The case study loads its own frame from the warehouse.
    """
    want = [t for t in PAGE_TABLES if tables is None or t in set(tables)]
    missing = core_dependencies(want) - set(frames)
    if missing:
        raise KeyError(f"page tables {want} need core frames {sorted(missing)}")
    out: dict[str, pd.DataFrame] = {}
    if "drivers" in want:
        out["drivers"] = build_drivers(wh.extra["drivers"], frames["scores"], frames["failures"])
    if {"case_study_2023", "case_study_series"} & set(want):
        from bankcanary.evaluation.case_study_2023 import load_frame

        table, series = build_case_study(load_frame(settings), settings)
        out["case_study_2023"] = table
        out["case_study_series"] = series
    if "rate_shock_scores" in want:
        keys = latest_scored_keys(frames["scores"])
        scorer = production_scorer(settings)
        out["rate_shock_scores"] = build_rate_shock(wh.features, wh.panel, scorer, keys)
    if "map_quarters" in want:
        out["map_quarters"] = build_map_quarters(
            frames["scores"], frames["banks"], frames["failures"]
        )
    return {t: out[t] for t in want if t in out}
