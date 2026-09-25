"""Ingest ``/financials``: one Call Report snapshot per bank and quarter, 2001Q1 onward.

Every FDIC-insured bank files a quarterly Call Report; the ``/financials`` endpoint
exposes the balance sheet, income statement (year-to-date), capital and asset-quality
items keyed by certificate number and report date (``REPDTE``, a calendar quarter end).
The pull is one request per quarter for the verified codes in ``config/fields.yaml``,
cached as ``data/raw/fdic/financials/<YYYYMMDD>.json`` so a re-run is free and a
partially finished pull resumes where it stopped. Dollar amounts are thousands of dollars.
"""

from __future__ import annotations

import calendar
import datetime as dt
import logging

import pandas as pd

from bankcanary.config import Settings
from bankcanary.fields import ATTRIBUTE_GROUPS, FieldSpec, load_field_map
from bankcanary.ingest import parse_dates, to_bool, to_float, to_int, to_str
from bankcanary.sources.fdic import FdicClient, cache_path, read_cache

log = logging.getLogger(__name__)

ENDPOINT = "financials"
KEY = ("cert", "repdte")
#: A bank that has filed every quarter since 2001 (Wells Fargo Bank, N.A.); used to ask the
#: API which quarter end is the latest one published.
PROBE_CERT = 3511
PAGE_LIMIT = 10_000


def quarter_end(day: dt.date) -> dt.date:
    """The calendar quarter end (Mar 31, Jun 30, Sep 30, Dec 31) on or after ``day``."""
    month = ((day.month - 1) // 3 + 1) * 3
    return dt.date(day.year, month, calendar.monthrange(day.year, month)[1])


def quarter_ends(start: dt.date, end: dt.date) -> list[dt.date]:
    """Every calendar quarter end from ``start`` to ``end`` inclusive (both snapped up)."""
    out: list[dt.date] = []
    current = quarter_end(start)
    last = quarter_end(end)
    while current <= last:
        out.append(current)
        month = current.month + 3
        year = current.year + (month > 12)
        month = month - 12 if month > 12 else month
        current = dt.date(year, month, calendar.monthrange(year, month)[1])
    return out


def repdte_code(day: dt.date) -> str:
    """``REPDTE`` as the API spells it in filters and cache names: ``YYYYMMDD``."""
    return day.strftime("%Y%m%d")


def parse_repdte(code: str) -> dt.date:
    return dt.datetime.strptime(str(code), "%Y%m%d").date()


def pull_codes(specs: list[FieldSpec] | None = None) -> list[str]:
    """The FDIC codes requested for every quarter: every entry of ``config/fields.yaml``."""
    specs = load_field_map() if specs is None else specs
    return [s.code for s in specs]


def latest_repdte(client: FdicClient) -> dt.date:
    """Ask the API for the newest quarter end published (one tiny, never-cached request)."""
    rows = client.get(
        ENDPOINT,
        filters=f"CERT:{PROBE_CERT}",
        fields=["REPDTE"],
        sort_by="REPDTE",
        sort_order="DESC",
        limit=1,
        cache_name="latest_probe",
        force=True,
    )
    if not rows:
        raise RuntimeError("financials probe returned no rows; cannot find the latest REPDTE")
    return parse_repdte(rows[0]["REPDTE"])


def resolve_range(
    settings: Settings,
    client: FdicClient | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> list[dt.date]:
    """Quarter ends to pull: ``settings.start_quarter`` to the latest published (or overrides)."""
    start = start or settings.start_quarter
    if end is None:
        end = settings.end_quarter
    if end is None:
        if client is None:
            raise ValueError("no end quarter configured and no client to probe the API with")
        end = latest_repdte(client)
        log.info("financials: latest REPDTE published by the API is %s", end)
    return quarter_ends(start, end)


def fetch_quarter(
    client: FdicClient, repdte: dt.date, codes: list[str], force: bool = False
) -> list[dict]:
    """Pull (or read from cache) every bank's record for one quarter end."""
    return client.get(
        ENDPOINT,
        filters=f"REPDTE:{repdte_code(repdte)}",
        fields=codes,
        sort_by="CERT",
        limit=PAGE_LIMIT,
        cache_name=repdte_code(repdte),
        force=force,
    )


def fetch_financials(
    settings: Settings,
    client: FdicClient | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    force: bool = False,
) -> dict[dt.date, int]:
    """Pull every quarter in range into the cache; returns ``{repdte: rows}`` per quarter.

    Quarters already on disk are served from cache instantly, so the command is
    idempotent and resumable after an interrupted run.
    """
    codes = pull_codes()
    own = client is None
    client = client or FdicClient(settings)
    counts: dict[dt.date, int] = {}
    try:
        for repdte in resolve_range(settings, client, start, end):
            rows = fetch_quarter(client, repdte, codes, force=force)
            counts[repdte] = len(rows)
            log.info("financials %s: %d rows", repdte, len(rows))
    finally:
        if own:
            client.close()
    return counts


def _num(series: pd.Series) -> pd.Series:
    """float64 from API numbers; the string path only when the column is mixed/object."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    return to_float(series)


def _attribute(series: pd.Series, spec: FieldSpec) -> pd.Series:
    if spec.unit == "date":
        return parse_dates(series)
    if spec.unit == "count":
        return to_int(series)
    if spec.unit == "flag":
        return to_bool(series)
    return to_str(series)


def clean_financials(rows: list[dict], specs: list[FieldSpec] | None = None) -> pd.DataFrame:
    """Map raw ``/financials`` records to the typed ``financials_raw`` table.

    Columns are the lower-case codes of ``config/fields.yaml`` in file order. ``repdte``
    (``YYYYMMDD`` in the API) and ``estymd`` become dates, identifiers nullable ints, the
    community-bank flag a bool and every financial code float64 (thousands of dollars or
    percent, as the field map says). Rows without a positive ``asset`` are dropped: they are
    empty shells (no balance sheet) that would only distort ratios.
    """
    specs = load_field_map() if specs is None else specs
    raw = pd.DataFrame(list(rows)).reindex(columns=[s.code for s in specs])
    cols: dict[str, pd.Series] = {}
    for spec in specs:
        series = raw[spec.code]
        if spec.code == "REPDTE":
            text = series.astype("string").str.strip().replace("", pd.NA)
            cols["repdte"] = pd.to_datetime(text, format="%Y%m%d", errors="coerce").astype(
                "datetime64[ns]"
            )
        elif spec.group in ATTRIBUTE_GROUPS:
            cols[spec.column] = _attribute(series, spec)
        else:
            cols[spec.column] = _num(series)
    df = pd.DataFrame(cols)
    keep = df["asset"].notna() & (df["asset"] > 0)
    if (~keep).any():
        log.info("financials: dropping %d rows with null or non-positive asset", int((~keep).sum()))
    df = df.loc[keep]
    dupes = df.duplicated(list(KEY), keep="last")
    if dupes.any():
        log.warning("financials: dropping %d duplicate (cert, repdte) rows", int(dupes.sum()))
        df = df.loc[~dupes]
    return df.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def cached_quarters(settings: Settings) -> list[dt.date]:
    """Quarter ends that already have a cache file, oldest first."""
    folder = cache_path(settings, ENDPOINT, "x").parent
    if not folder.exists():
        return []
    found = [p.stem for p in folder.glob("[0-9]" * 8 + ".json")]
    return sorted(parse_repdte(name) for name in found)


def build_financials_raw(settings: Settings, quarters: list[dt.date] | None = None) -> pd.DataFrame:
    """Assemble ``financials_raw`` from the cached quarter files (no network).

    Each quarter is cleaned on its own and the frames are concatenated once, which keeps a
    ~100-quarter build well under the two-minute budget.
    """
    specs = load_field_map()
    quarters = cached_quarters(settings) if quarters is None else quarters
    frames: list[pd.DataFrame] = []
    for repdte in quarters:
        path = cache_path(settings, ENDPOINT, repdte_code(repdte))
        if not path.exists():
            log.warning("financials: quarter %s is not cached; skipping", repdte)
            continue
        rows = read_cache(path)["data"]
        if rows:
            frames.append(clean_financials(rows, specs))
    if not frames:
        raise FileNotFoundError("no cached financials quarters under data/raw/fdic/financials")
    df = pd.concat(frames, ignore_index=True)
    dupes = df.duplicated(list(KEY), keep="last")
    if dupes.any():  # a quarter file whose rows carry another REPDTE would collide here
        log.warning("financials_raw: dropping %d duplicate (cert, repdte) rows", int(dupes.sum()))
        df = df.loc[~dupes]
    return df.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def quarter_counts(df: pd.DataFrame) -> pd.Series:
    """Rows (banks) per report date: the spec's sanity check (~8,300 in 2009Q1, ~4,500 now)."""
    return df.groupby("repdte").size().rename("banks")


def ingest_financials(
    settings: Settings,
    start: dt.date | None = None,
    end: dt.date | None = None,
    force: bool = False,
    build: bool = True,
) -> pd.DataFrame | None:
    """CLI entry: pull the quarters in range, then rebuild the table from every cached quarter."""
    from bankcanary.storage.duckdb import replace_table
    from bankcanary.storage.parquet import write_table

    with FdicClient(settings) as client:
        counts = fetch_financials(settings, client=client, start=start, end=end, force=force)
    log.info(
        "financials: %d quarters requested, %d quarters in cache",
        len(counts),
        len(cached_quarters(settings)),
    )
    if not build:
        return None
    df = build_financials_raw(settings)
    path = write_table(df, "financials_raw", key=KEY, settings=settings)
    rows = replace_table("financials_raw", path, settings=settings)
    log.info("financials_raw: %d rows -> %s and duckdb (%d rows)", len(df), path, rows)
    return df
