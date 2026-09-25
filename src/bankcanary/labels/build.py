"""Build the ``labels`` table: did the bank fail inside the prediction window?

Spec section 5 defines, for a report dated ``q`` that becomes usable on ``avail(q)``
(``repdte + availability_lag_days``) and a horizon of ``H`` quarters, the window
``(avail(q), avail(q) + 3*H months]`` and the rules:

1. ``y_Hq = 1`` when the bank's failure date falls inside the window, else 0.
2. Rows whose failure date is on or before ``avail(q)`` are *dropped*: the bank had
   already failed by the time this report could have been read. They stay in the table
   with ``dropped_failed_before_avail = True`` so the count can be audited, and every
   downstream consumer filters them out.
3. A bank that left the industry without failing (merger, voluntary closing, ...) inside
   the window keeps ``y = 0`` and is flagged ``censored_in_window_Hq`` so a sensitivity
   run can exclude it and survival models can treat it as censoring.
4. A window that ends after the last date the failures list is known for is
   ``label_complete_Hq = False``: such rows are scored but never trained or evaluated on.
5. ``window_end_Hq`` is stored on every row because the leakage rules of section 6 use it.

``as_of_date`` is the ``fetched_at`` date of the cached failures pull rather than
"today", so rebuilding from cache gives identical labels.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from bankcanary.config import Settings
from bankcanary.sources.fdic import cache_path, read_cache

log = logging.getLogger(__name__)

#: Columns the panel must carry for labels to be computed.
PANEL_INPUTS = ("cert", "repdte", "avail_date", "fail_date", "exit_date")

#: Columns of the labels table that do not depend on the horizon.
LABEL_COLUMNS = ("cert", "repdte", "window_start", "dropped_failed_before_avail")


def horizon_columns(horizon: int) -> list[str]:
    """Names of the four per-horizon label columns, e.g. ``y_4q`` ... ``label_complete_4q``."""
    h = int(horizon)
    return [f"y_{h}q", f"window_end_{h}q", f"censored_in_window_{h}q", f"label_complete_{h}q"]


def failures_as_of_date(settings: Settings) -> dt.date:
    """The date the cached failures list was pulled: the last day failures are known for.

    Read from ``data/raw/fdic/failures/all.json`` (``fetched_at``). Windows ending after
    this date cannot be labelled with confidence, because a failure inside them may simply
    not have happened yet.
    """
    payload = read_cache(cache_path(settings, "failures", "all"))
    fetched = str(payload["fetched_at"])[:10]
    return dt.date.fromisoformat(fetched)


def _dates(series: pd.Series) -> pd.Series:
    """Coerce a panel date column to ``datetime64[ns]`` (missing stays ``NaT``)."""
    return pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")


def build_labels(
    panel: pd.DataFrame,
    horizons: list[int],
    as_of_date: dt.date | pd.Timestamp,
    lag_days: int | None = None,
) -> pd.DataFrame:
    """Label every panel row for each horizon; keyed by ``(cert, repdte)``.

    Vectorised over the whole panel. Comparisons against a missing ``fail_date`` or
    ``exit_date`` (``NaT``) are always ``False``, so open banks get ``y = 0``, are never
    dropped and never censored. Returns one row per panel row with ``window_start``
    (= ``avail_date``), ``dropped_failed_before_avail`` and, per horizon H, ``y_Hq``
    (int 0/1), ``window_end_Hq``, ``censored_in_window_Hq`` and ``label_complete_Hq``.

    ``lag_days`` overrides the panel's ``avail_date`` with ``repdte + lag_days`` so the
    availability-lag sensitivity can relabel in memory (a report that is assumed to
    arrive later has a later window, and a bank that failed in the meantime is dropped);
    the stored ``labels`` table is never touched by this.
    """
    missing = [c for c in PANEL_INPUTS if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing column(s) {missing}")
    if not horizons:
        raise ValueError("horizons must not be empty")
    if lag_days is None:
        avail = _dates(panel["avail_date"])
    else:
        if int(lag_days) < 0:
            raise ValueError(f"lag_days must be non-negative, got {lag_days}")
        avail = _dates(panel["repdte"]) + pd.Timedelta(days=int(lag_days))
    fail = _dates(panel["fail_date"])
    exit_ = _dates(panel["exit_date"])
    as_of = pd.Timestamp(as_of_date)

    out = pd.DataFrame(
        {
            "cert": panel["cert"].to_numpy(),
            "repdte": _dates(panel["repdte"]).to_numpy(),
            "window_start": avail.to_numpy(),
            # Rule 2: the bank was already gone when this report became usable.
            "dropped_failed_before_avail": (fail <= avail).fillna(False).to_numpy(dtype=bool),
        }
    )
    for horizon in horizons:
        y_col, end_col, cens_col, complete_col = horizon_columns(horizon)
        window_end = avail + pd.DateOffset(months=3 * int(horizon))
        window_end = window_end.astype("datetime64[ns]")
        inside_fail = (fail > avail) & (fail <= window_end)
        # Rule 3: a non-failure exit inside the window censors the observation.
        inside_exit = fail.isna() & (exit_ > avail) & (exit_ <= window_end)
        out[y_col] = inside_fail.fillna(False).to_numpy(dtype=bool).astype("int64")
        out[end_col] = window_end.to_numpy()
        out[cens_col] = inside_exit.fillna(False).to_numpy(dtype=bool)
        # Rule 4: a window still open on the failures-list date is not a complete label.
        out[complete_col] = (window_end <= as_of).fillna(False).to_numpy(dtype=bool)
    log.info(
        "labels: %d rows, %d dropped (failed before avail), as_of %s, horizons %s",
        len(out),
        int(out["dropped_failed_before_avail"].sum()),
        as_of.date(),
        list(horizons),
    )
    return out


def label_summary(labels: pd.DataFrame, horizons: list[int]) -> dict:
    """Positives per report year and the overall positive rate, per horizon.

    Only rows that are usable for training count: ``label_complete_Hq`` and not
    ``dropped_failed_before_avail``. Returns ``{"per_year": DataFrame indexed by year with
    columns rows_Hq / pos_Hq, "rate_Hq": float, "rows_Hq": int, "pos_Hq": int, ...}``.
    """
    year = labels["repdte"].dt.year
    summary: dict = {}
    per_year: dict[str, pd.Series] = {}
    for horizon in horizons:
        y_col, _, _, complete_col = horizon_columns(horizon)
        usable = labels[complete_col] & ~labels["dropped_failed_before_avail"]
        rows = usable.groupby(year).sum().astype("int64")
        pos = labels.loc[usable, y_col].groupby(year[usable]).sum().astype("int64")
        per_year[f"rows_{horizon}q"] = rows
        per_year[f"pos_{horizon}q"] = pos.reindex(rows.index, fill_value=0)
        n_rows = int(usable.sum())
        n_pos = int(labels.loc[usable, y_col].sum())
        summary[f"rows_{horizon}q"] = n_rows
        summary[f"pos_{horizon}q"] = n_pos
        summary[f"rate_{horizon}q"] = (n_pos / n_rows) if n_rows else float("nan")
    table = pd.DataFrame(per_year)
    table.index.name = "year"
    summary["per_year"] = table
    summary["rows"] = int(len(labels))
    summary["dropped"] = int(labels["dropped_failed_before_avail"].sum())
    return summary


def build_labels_table(settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Read ``panel``, build ``labels`` for ``settings.horizons_quarters``, write it.

    Writes Parquet + DuckDB through the storage layer and returns the labels and
    :func:`label_summary` (plus ``as_of_date`` and ``duckdb_rows``). Idempotent: reads
    only local tables and the cached failures pull.
    """
    from bankcanary.storage.duckdb import replace_table
    from bankcanary.storage.parquet import read_table, write_table

    panel = read_table("panel", settings=settings)
    as_of = failures_as_of_date(settings)
    horizons = [int(h) for h in settings.horizons_quarters]
    labels = build_labels(panel, horizons, as_of)
    path = write_table(labels, "labels", settings=settings)
    rows = replace_table("labels", path, settings=settings)
    summary = label_summary(labels, horizons)
    summary["as_of_date"] = as_of
    summary["duckdb_rows"] = rows
    log.info("labels: %d rows x %d columns -> %s", len(labels), len(labels.columns), path)
    return labels, summary
