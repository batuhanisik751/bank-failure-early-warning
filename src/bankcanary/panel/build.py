"""Build the ``panel`` table: every ``financials_raw`` row plus availability and exit facts.

The panel is the point-in-time view a supervisor could have had: a report dated
``repdte`` becomes usable on ``avail_date = repdte + availability_lag_days``. Institution
attributes come from ``institutions``; ``fail_date``/``exit_date``/``exit_reason`` come
from :mod:`bankcanary.panel.exits`. Quarter-level values already in ``financials_raw``
(``bkclass``, ``stalp``, ``rssdhcr``, ...) are kept because they are point-in-time; only
their gaps are filled from the institution record, except ``rssdhcr`` (a null there means
"no holding company that quarter", and filling it would leak later affiliations).
"""

from __future__ import annotations

import logging

import pandas as pd

from bankcanary.config import Settings
from bankcanary.panel.exits import EXIT_COLUMNS, build_exits
from bankcanary.storage.duckdb import replace_table
from bankcanary.storage.parquet import read_table, write_table

log = logging.getLogger(__name__)

#: Institution attributes joined onto the panel (CONTRACT section 5).
INSTITUTION_COLUMNS = [
    "estymd",
    "bkclass",
    "stalp",
    "rssdhcr",
    "fed_rssd",
    "cb",
    "latitude",
    "longitude",
]
#: Of those, the ones whose financials value is kept and only filled where missing.
FILL_ONLY = ("estymd", "bkclass", "stalp", "cb")
#: Always taken from institutions (``/financials`` drops FED_RSSD; lat/long are not there).
INSTITUTION_ONLY = ("fed_rssd", "latitude", "longitude")


def assemble_panel(
    financials: pd.DataFrame,
    institutions: pd.DataFrame,
    exits: pd.DataFrame,
    availability_lag_days: int,
) -> pd.DataFrame:
    """Join the three inputs into the panel frame (pure; no I/O).

    Keeps every ``financials`` column and row (left joins on ``cert``), adds the
    institution attributes, the exit columns, ``avail_date`` and ``has_holding_company``
    (``rssdhcr`` present and non-zero).
    """
    inst = institutions.loc[institutions["cert"].notna(), ["cert", *INSTITUTION_COLUMNS]]
    inst = inst.astype({"cert": "Int64"}).drop_duplicates("cert")
    out = financials.astype({"cert": "Int64"}).merge(
        inst, on="cert", how="left", suffixes=("", "_inst"), validate="many_to_one"
    )
    for col in INSTITUTION_COLUMNS:
        alt = f"{col}_inst"
        if alt not in out.columns:
            continue  # financials did not carry this column; the institution value stands
        if col in INSTITUTION_ONLY:
            out[col] = out[alt]
        elif col in FILL_ONLY:
            out[col] = out[col].where(out[col].notna(), out[alt])
        out = out.drop(columns=alt)
    ex = exits.loc[:, EXIT_COLUMNS].astype({"cert": "Int64"}).drop_duplicates("cert")
    out = out.merge(ex, on="cert", how="left", validate="many_to_one")
    out = out.copy()  # defragment after the column-by-column fills
    out["assisted"] = out["assisted"].fillna(False).astype(bool)
    out["avail_date"] = out["repdte"] + pd.Timedelta(days=availability_lag_days)
    hc = pd.to_numeric(out["rssdhcr"], errors="coerce")
    out["has_holding_company"] = (hc.notna() & (hc != 0)).astype(bool)
    return out


def panel_summary(
    panel: pd.DataFrame,
    exits: pd.DataFrame,
    institutions: pd.DataFrame | None = None,
    since_year: int = 2001,
) -> dict:
    """Headline counts for the CLI: rows, banks, failed-bank coverage and exit reasons.

    ``failed_banks`` counts banks whose ``fail_date`` is in ``since_year`` or later;
    ``failed_matched`` is how many of them have at least one panel report dated before
    the failure (the spec's sanity check: at least 95% should).
    """
    failed = exits.loc[exits["fail_date"].notna() & (exits["fail_date"].dt.year >= since_year)]
    first = panel.groupby("cert")["repdte"].min()
    prior = failed["cert"].map(first)
    matched = int((prior < failed["fail_date"]).sum())
    reasons = exits["exit_reason"].fillna("none").value_counts()
    known = institutions["cert"] if institutions is not None else panel["cert"]
    orphan = panel.loc[~panel["cert"].isin(known), "cert"].nunique()
    return {
        "rows": int(len(panel)),
        "banks": int(panel["cert"].nunique()),
        "banks_missing_institution": int(orphan),
        "failed_banks": int(len(failed)),
        "failed_matched": matched,
        "failed_matched_share": (matched / len(failed)) if len(failed) else float("nan"),
        "exit_reasons": {str(k): int(v) for k, v in reasons.items()},
    }


def build_panel(settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Read the four ingested tables, build ``panel``, write Parquet + DuckDB.

    Returns the panel and :func:`panel_summary`. Idempotent: reads only local tables.
    """
    financials = read_table("financials_raw", settings=settings)
    institutions = read_table("institutions", settings=settings)
    failures = read_table("failures", settings=settings)
    history = read_table("history", settings=settings)
    exits = build_exits(failures, institutions, history)
    panel = assemble_panel(financials, institutions, exits, settings.availability_lag_days)
    path = write_table(panel, "panel", settings=settings)
    rows = replace_table("panel", path, settings=settings)
    summary = panel_summary(panel, exits, institutions)
    summary["duckdb_rows"] = rows
    log.info("panel: %d rows x %d columns -> %s", len(panel), len(panel.columns), path)
    return panel, summary
