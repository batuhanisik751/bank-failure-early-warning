"""``bankcanary refresh``: the scheduled pipeline (CONTRACT section 17, spec E2).

The job probes the FDIC API for the newest ``REPDTE`` and compares it with the newest
quarter already published (``quarters.max(repdte)`` in Postgres, or the warehouse when the
database is unreachable). When a newer quarter exists it ingests that quarter, rebuilds
``panel``, ``labels``, ``macro_state`` and ``features_v2`` from the raw cache, scores the
new quarter with ``models/production/``, explains it with SHAP and publishes the rows the
new quarter adds. It is incremental by design: historical rows (walk-forward scores,
drivers, case study, metrics) stay untouched, so the job never needs the walk-forward
artefacts, which are not committed. ``bankcanary publish`` remains the full rebuild.

Every run, new quarter or not, writes one ``pipeline_runs`` row and a ``runs/refresh/``
tracking record. Educational project, not a credit rating, not investment advice, not a
supervisory assessment.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RUN_NAME = "refresh"
#: Structure tables re-pulled on every new quarter (small; the failures list and the
#: institution attributes move between quarters). ``history`` is served from cache.
STRUCTURE_TABLES: tuple[str, ...] = ("failures", "institutions", "history")
FORCED_STRUCTURE: frozenset[str] = frozenset({"failures", "institutions"})
#: Quarters of ``map_quarters`` rewritten with the new one, so ``failed_this_quarter`` of
#: the reports a new failure follows is refreshed too.
MAP_LOOKBACK_QUARTERS = 4


@dataclass(frozen=True)
class Decision:
    """What the run will do: the two latest quarters compared and the outcome."""

    latest_api: dt.date | None
    latest_published: dt.date | None
    source: str
    new_quarter: bool
    quarter: dt.date | None
    reason: str


def decide(
    latest_api: dt.date | None,
    latest_published: dt.date | None,
    source: str = "database",
    force_quarter: dt.date | None = None,
) -> Decision:
    """Pure decision rule: new quarter when the API is ahead of what is published.

    ``force_quarter`` publishes that quarter regardless (it must be one the API has);
    ``latest_published`` ``None`` means nothing is published yet, so the latest
    quarter is treated as new.
    """
    if force_quarter is not None:
        return Decision(
            latest_api, latest_published, source, True, force_quarter, "forced by --force-quarter"
        )
    if latest_api is None:
        return Decision(None, latest_published, source, False, None, "FDIC probe gave no quarter")
    if latest_published is None:
        return Decision(latest_api, None, source, True, latest_api, f"nothing published ({source})")
    if latest_api > latest_published:
        why = f"API {latest_api} is newer than {source} {latest_published}"
        return Decision(latest_api, latest_published, source, True, latest_api, why)
    why = f"API {latest_api} is not newer than {source} {latest_published}"
    return Decision(latest_api, latest_published, source, False, None, why)


def probe_latest(settings) -> dt.date:
    """The newest ``REPDTE`` the FDIC financials endpoint publishes (one uncached request)."""
    from bankcanary.ingest.financials import latest_repdte
    from bankcanary.sources.fdic import FdicClient

    with FdicClient(settings) as client:
        return latest_repdte(client)


def warehouse_latest(settings) -> dt.date | None:
    """``max(repdte)`` of the warehouse ``labels`` table, or ``None`` when it is absent."""
    from bankcanary.storage.parquet import read_table, table_path

    if not table_path(settings, "labels").exists():
        return None
    return pd.to_datetime(read_table("labels", settings)["repdte"]).max().date()


def published_latest(settings, url: str | None = None) -> tuple[dt.date | None, str]:
    """``(quarter, source)``: ``quarters.max(repdte)`` from Postgres, else the warehouse.

    ``source`` is ``'database'`` when Postgres answered (``quarter`` may still be ``None``
    when nothing is published yet), ``'warehouse'`` when it did not, and ``'none'`` when
    neither holds a quarter. The fallback is named in the run log (CONTRACT 17).
    """
    import psycopg

    from bankcanary.publish import db

    if db.reachable(url):
        with db.connect(url, autocommit=True) as conn:
            try:
                row = conn.execute("SELECT max(repdte) FROM quarters").fetchone()
            except psycopg.errors.UndefinedTable:
                row = None
        found = row[0] if row else None
        return (found if found is None else pd.Timestamp(found).date()), "database"
    log.warning("database unreachable: comparing with the warehouse instead")
    found = warehouse_latest(settings)
    return found, ("warehouse" if found is not None else "none")


@dataclass
class Timings:
    """Wall-clock seconds per step, in order, for the run log."""

    steps: list[tuple[str, float]] = field(default_factory=list)

    def add(self, name: str, seconds: float) -> None:
        self.steps.append((name, round(seconds, 1)))
        log.info("refresh step %s: %.1f s", name, seconds)

    def text(self) -> str:
        return ", ".join(f"{n} {s:.0f} s" for n, s in self.steps)

    def total(self) -> float:
        return float(sum(s for _, s in self.steps))


def _timed(timings: Timings, name: str, fn, *args, **kwargs):
    clock = time.perf_counter()
    out = fn(*args, **kwargs)
    timings.add(name, time.perf_counter() - clock)
    return out


def ingest_quarter(settings, quarter: dt.date, timings: Timings) -> None:
    """Pull the structure tables and ``quarter``'s financials, then rebuild ``financials_raw``.

    ``failures`` and ``institutions`` are re-downloaded (they change between quarters);
    ``history`` and every other financials quarter come from ``data/raw/fdic``.
    """
    from bankcanary.ingest import failures, financials, history, institutions
    from bankcanary.sources.fdic import FdicClient
    from bankcanary.storage.duckdb import replace_table
    from bankcanary.storage.parquet import write_table

    fetchers = {
        "failures": failures.fetch_failures,
        "institutions": institutions.fetch_institutions,
        "history": history.fetch_history,
    }
    clock = time.perf_counter()
    with FdicClient(settings) as client:
        for name in STRUCTURE_TABLES:
            frame = fetchers[name](settings, client=client, force=name in FORCED_STRUCTURE)
            path = write_table(frame, name, settings=settings)
            replace_table(name, path, settings=settings)
            log.info("refresh: %s %d rows", name, len(frame))
    timings.add("ingest structure", time.perf_counter() - clock)
    _timed(
        timings,
        f"ingest financials {quarter}",
        financials.ingest_financials,
        settings,
        start=quarter,
        end=quarter,
        force=False,
        build=True,
    )


def refresh_macro(settings, timings: Timings) -> bool:
    """Re-pull every FRED series and rebuild ``macro_state``; ``False`` (and a warning) on failure.

    The macro features are point-in-time, so a stale cache still yields valid values; a
    network or key problem therefore degrades to the cached table instead of failing.
    """
    from bankcanary.config import load_secrets
    from bankcanary.ingest import macro
    from bankcanary.sources.fred import FredClient

    clock = time.perf_counter()
    try:
        with FredClient(settings, secrets=load_secrets()) as client:
            for sid in macro.series_ids():
                client.series(sid, force=True)
        if macro.missing_series(settings):
            raise RuntimeError(f"{len(macro.missing_series(settings))} FRED series missing")
        macro.build_macro(settings)
    except Exception as exc:  # noqa: BLE001 - degrade to the cached table, never abort
        log.warning("macro refresh skipped, cached macro_state kept: %s", exc)
        timings.add("macro (skipped)", time.perf_counter() - clock)
        return False
    timings.add("macro", time.perf_counter() - clock)
    return True


def rebuild_warehouse(settings, timings: Timings) -> None:
    """``panel`` -> ``labels`` -> ``macro_state`` -> ``features_v2``, each step timed."""
    from bankcanary.features.build import build_features_table
    from bankcanary.labels.build import build_labels_table
    from bankcanary.panel.build import build_panel

    _timed(timings, "panel", build_panel, settings)
    _timed(timings, "labels", build_labels_table, settings)
    refresh_macro(settings, timings)
    _timed(timings, "features_v2", build_features_table, settings, version="v2")


SCORE_COLUMNS: tuple[str, ...] = (
    "cert", "repdte", "horizon", "model", "test_year", "score", "score_calibrated",
)  # fmt: skip
#: Tables rebuilt whole on a refresh (small, or defined as "the latest quarter").
REPLACED_TABLES: frozenset[str] = frozenset({"banks", "failures", "rate_shock_scores"})
#: Publish order of the tables a refresh writes (upsert unless in :data:`REPLACED_TABLES`).
REFRESH_TABLES: tuple[str, ...] = (
    "model_versions", "banks", "quarters", "scores", "drivers", "ratios", "peer_stats",
    "failures", "rate_shock_scores", "map_quarters",
)  # fmt: skip


def prior_quarter(quarter: dt.date) -> dt.date:
    return (pd.Timestamp(quarter) - pd.offsets.QuarterEnd(1)).date()


def production_versions(settings) -> tuple[pd.DataFrame, dict]:
    """``model_versions`` rows and lookup for ``models/production/`` only (no walk-forward)."""
    from bankcanary.publish.core import build_model_versions

    models_dir = Path(settings.models_dir)
    repo = Path(settings.data_dir).resolve().parent
    runs_dir = Path(settings.runs_dir)
    return build_model_versions(models_dir / "production", models_dir, runs_dir, repo)


def quarter_rows(wh, quarters: list[dt.date]) -> pd.DataFrame:
    """``features_v2`` rows of ``quarters`` that are not post-failure reports."""
    labels = wh.labels
    keep = ~labels["dropped_failed_before_avail"].fillna(True).astype(bool)
    keep &= pd.to_datetime(labels["repdte"]).dt.date.isin(set(quarters))
    keys = labels.loc[keep, ["cert", "repdte"]]
    return wh.features.merge(keys, on=["cert", "repdte"], how="inner")


def score_quarter(settings, wh, quarter: dt.date, lookup: dict) -> pd.DataFrame:
    """``scores`` rows of ``quarter`` from ``models/production/`` (ranked within the quarter).

    The prior quarter is scored too, only to fill ``delta_prob_prior_q``; its rows are
    not returned (they are already published).
    """
    from bankcanary.publish.core import MODELS, build_scores, score_with_artefacts

    rows = quarter_rows(wh, [prior_quarter(quarter), quarter])
    production_dir = Path(settings.models_dir) / "production"
    scored = pd.concat(
        [score_with_artefacts(rows, production_dir / m, m) for m in MODELS], ignore_index=True
    )
    table = build_scores(scored.iloc[0:0], scored, lookup)
    out = table[pd.to_datetime(table["repdte"]).dt.date == quarter]
    log.info("refresh: scored %d rows for %s", len(out), quarter)
    return out.reset_index(drop=True)


def explain_quarter(settings, wh, quarter: dt.date) -> pd.DataFrame:
    """Warehouse-shaped SHAP drivers of ``quarter`` from the production ``gbdt_mono``."""
    import json

    import joblib

    from bankcanary.explain.shap_drivers import drivers_table, shap_matrix
    from bankcanary.publish.pages import DRIVER_MODEL, DRIVER_PRODUCTION_LABEL

    model_dir = Path(settings.models_dir) / "production" / DRIVER_MODEL
    pipeline = joblib.load(model_dir / "pipeline.joblib")
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    meta = json.loads((model_dir / "model_version.json").read_text(encoding="utf-8"))
    model_year = pd.Timestamp(meta["train_end_repdte"]).year
    rows = quarter_rows(wh, [quarter])
    values, _, clipped = shap_matrix(pipeline, rows[list(features)])
    keys = rows[["cert", "repdte"]]
    return drivers_table(keys, values, clipped, features, DRIVER_PRODUCTION_LABEL, model_year)


def merge_quarter_versions(quarters: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """Keep the published ``model_year``/``model_version`` of quarters already in the database.

    ``quarters`` is a fresh :func:`bankcanary.publish.core.build_quarters` frame that only
    knows the production version; ``existing`` holds ``repdte, model_year, model_version``
    as published (walk-forward years included). Label counts and completeness come from
    the fresh frame, so they keep moving as the label windows close.
    """
    if existing is None or existing.empty:
        return quarters
    out = quarters.copy()
    old = existing.copy()
    old["repdte"] = pd.to_datetime(old["repdte"]).dt.date
    old = old.set_index("repdte")
    hit = out["repdte"].isin(old.index)
    out.loc[hit, "model_year"] = out.loc[hit, "repdte"].map(old["model_year"]).astype("Int64")
    out.loc[hit, "model_version"] = out.loc[hit, "repdte"].map(old["model_version"])
    return out


def build_quarter_frames(
    settings,
    wh,
    quarter: dt.date,
    existing_quarters: pd.DataFrame | None = None,
    recent_scores: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Every :data:`REFRESH_TABLES` frame the new ``quarter`` adds, in publish order.

    ``recent_scores`` (``cert, repdte, model, band, probability`` of the last
    :data:`MAP_LOOKBACK_QUARTERS` quarters as published) lets ``map_quarters`` re-flag
    the reports a new failure follows.
    """
    from bankcanary.publish import core, pages

    versions, lookup = production_versions(settings)
    scores = score_quarter(settings, wh, quarter, lookup)
    banks = core.build_banks(wh.institutions, wh.panel)
    failures = core.build_failures(wh.failures)
    by_year = {"production": lookup.get(("gbdt_mono", "production"))}
    quarters = merge_quarter_versions(
        core.build_quarters(wh.panel, wh.labels, by_year), existing_quarters
    )
    ratios = core.build_ratios(wh.features, wh.panel)
    in_quarter = pd.to_datetime(ratios["repdte"]).dt.date == quarter
    peer_stats = core.build_peer_stats(ratios.loc[in_quarter])
    ratios_q = ratios.loc[in_quarter].drop(columns=["size_bucket", "region"])
    drivers = pages.build_drivers(explain_quarter(settings, wh, quarter), scores, failures)
    keys = pages.latest_scored_keys(scores)
    shock = pages.build_rate_shock(wh.features, wh.panel, pages.production_scorer(settings), keys)
    map_scores = scores[["cert", "repdte", "model", "band", "probability"]]
    if recent_scores is not None and len(recent_scores):
        map_scores = pd.concat([recent_scores, map_scores], ignore_index=True)
        map_scores = map_scores.drop_duplicates(["cert", "repdte", "model"], keep="last")
    map_quarters = pages.build_map_quarters(map_scores, banks, failures)
    frames = {
        "model_versions": versions,
        "banks": banks,
        "quarters": quarters,
        "scores": scores,
        "drivers": drivers,
        "ratios": ratios_q.reset_index(drop=True),
        "peer_stats": peer_stats,
        "failures": failures,
        "rate_shock_scores": shock,
        "map_quarters": map_quarters,
    }
    return {t: frames[t] for t in REFRESH_TABLES}


def load_refresh_warehouse(settings):
    """The warehouse frames a refresh needs; never the walk-forward or drivers tables."""
    from bankcanary.publish.core import Warehouse
    from bankcanary.storage.parquet import read_table

    return Warehouse(
        institutions=read_table("institutions", settings),
        panel=read_table("panel", settings),
        labels=read_table("labels", settings),
        features=read_table("features_v2", settings),
        failures=read_table("failures", settings),
        walkforward=pd.DataFrame(columns=list(SCORE_COLUMNS)),
    )


def read_published(conn, quarter: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(existing quarters, recent gbdt_mono scores)`` as published; empty on a first publish."""
    import psycopg

    since = (pd.Timestamp(quarter) - pd.offsets.QuarterEnd(MAP_LOOKBACK_QUARTERS)).date()
    quarter_cols = ["repdte", "model_year", "model_version"]
    score_cols = ["cert", "repdte", "model", "band", "probability"]
    try:
        quarters = conn.execute("SELECT repdte, model_year, model_version FROM quarters").fetchall()
        scores = conn.execute(
            "SELECT cert, repdte, model, band, probability FROM scores "
            "WHERE model = 'gbdt_mono' AND repdte >= %s",
            (since,),
        ).fetchall()
    except psycopg.errors.UndefinedTable:
        return pd.DataFrame(columns=quarter_cols), pd.DataFrame(columns=score_cols)
    return pd.DataFrame(quarters, columns=quarter_cols), pd.DataFrame(scores, columns=score_cols)


def publish_frames(conn, frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Apply the schema, then load every frame (replace or upsert per table); rows per table."""
    from bankcanary.publish import writer

    writer.apply_schema(conn)
    written: dict[str, int] = {}
    for name, frame in frames.items():
        mode = "replace" if name in REPLACED_TABLES else "upsert"
        written[name] = writer.write_table(conn, name, frame, mode=mode)
        log.info("refresh: %s %s %d rows", name, mode, written[name])
    writer.analyze(conn, list(written))
    return written


def write_github_output(decision: Decision, status: str, path: str | None = None) -> bool:
    """Append ``new_quarter``, ``quarter`` and ``status`` to ``$GITHUB_OUTPUT`` when set.

    The refresh workflow reads ``new_quarter`` to decide whether to call the web app's
    revalidate route. Returns ``False`` (nothing written) outside GitHub Actions.
    """
    target = path or os.environ.get("GITHUB_OUTPUT")
    if not target:
        return False
    published = decision.new_quarter and status == "ok"
    lines = [
        f"new_quarter={'true' if published else 'false'}",
        f"quarter={decision.quarter.isoformat() if decision.quarter else ''}",
        f"status={status}",
    ]
    with open(target, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return True


@dataclass
class RefreshResult:
    decision: Decision
    status: str
    timings: Timings
    rows_written: dict[str, int]
    run_id: str | None
    log_text: str
    db_bytes: int | None = None


def plan(settings, url: str | None = None, force_quarter: dt.date | None = None) -> Decision:
    """Probe the API and the published state, then :func:`decide`."""
    latest_api = probe_latest(settings)
    published, source = published_latest(settings, url)
    decision = decide(latest_api, published, source, force_quarter)
    log.info("refresh: %s", decision.reason)
    return decision


def run(
    settings, url: str | None = None, force_quarter: dt.date | None = None, dry_run: bool = False
) -> RefreshResult:
    """One refresh: decide, then ingest, rebuild, score, explain and publish when needed.

    Always writes a ``pipeline_runs`` row (when the database answers) and a
    ``runs/refresh/`` record; ``dry_run`` stops after the decision and writes nothing.
    A failure is recorded with ``status = 'failed'`` and re-raised.
    """
    from bankcanary import tracking
    from bankcanary.publish import db
    from bankcanary.publish.core import HORIZON, pipeline_run_row
    from bankcanary.publish.writer import write_table

    started = dt.datetime.now(dt.UTC)
    timings = Timings()
    decision = _timed(timings, "probe", plan, settings, url, force_quarter)
    if dry_run:
        text = f"dry run: {decision.reason}; nothing written"
        return RefreshResult(decision, "dry_run", timings, {}, None, text)
    config = {"horizon": HORIZON, "started_at": started.isoformat(), "force": str(force_quarter)}
    record = tracking.start_run(RUN_NAME, config, settings)
    status, written, size, error = "ok", {}, None, ""
    try:
        if decision.new_quarter and decision.quarter is not None:
            ingest_quarter(settings, decision.quarter, timings)
            rebuild_warehouse(settings, timings)
            if decision.source != "database" and not db.reachable(url):
                status = "no_database"
            else:
                wh = _timed(timings, "load warehouse", load_refresh_warehouse, settings)
                with db.connect(url, autocommit=True) as conn:
                    existing, recent = read_published(conn, decision.quarter)
                frames = _timed(
                    timings, "build frames", build_quarter_frames, settings, wh,
                    decision.quarter, existing, recent,
                )  # fmt: skip
                with db.connect(url) as conn:
                    written = _timed(timings, "publish", publish_frames, conn, frames)
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finished = dt.datetime.now(dt.UTC)
        source_note = "" if decision.source == "database" else f" [compared with {decision.source}]"
        text = f"{decision.reason}{source_note}; {status}; steps: {timings.text()}"
        if error:
            text += f"; error: {error}"
        row = pipeline_run_row(
            record.run_id, started, finished, status, written, text,
            latest_repdte=decision.latest_api, new_quarter=decision.new_quarter,
        )  # fmt: skip
        if db.reachable(url):
            with db.connect(url) as conn:
                write_table(conn, "pipeline_runs", row, mode="upsert")
                size = db.database_size_bytes(conn)
        record.log_metrics(
            {**{f"rows_{k}": v for k, v in written.items()}, "status": status,
             "new_quarter": decision.new_quarter, "seconds": timings.total(),
             "db_bytes": size, "steps": dict(timings.steps)}
        )  # fmt: skip
        record.finish()
        write_github_output(decision, status)
    return RefreshResult(decision, status, timings, written, record.run_id, text, size)
