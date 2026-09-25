"""Ingest ``/failures``: every FDIC-insured failure and open-bank assistance since 1934.

``RESTYPE == 'FAILURE'`` rows are the label events for the model; ``ASSISTANCE`` rows are
open-bank transactions (the bank kept operating with FDIC help) and are kept so the
panel can flag them separately. Amounts are thousands of dollars.
"""

from __future__ import annotations

import logging

import pandas as pd

from bankcanary.config import Settings
from bankcanary.ingest import parse_dates, to_float, to_frame, to_int, to_str
from bankcanary.sources.fdic import FdicClient

log = logging.getLogger(__name__)

FIELDS = [
    "CERT", "NAME", "FAILDATE", "FAILYR", "RESTYPE", "RESTYPE1", "COST", "QBFASSET",
    "QBFDEP", "CITYST", "PSTALP", "CHCLASS1", "RESDATE", "UNINSDEP",
]  # fmt: skip
KEY = ("cert", "fail_date")
#: Published FDIC failed-bank counts used as a sanity check (spec section 5).
BENCHMARK_YEARS = {2009: 140, 2010: 157, 2011: 92}


def clean_failures(rows: list[dict]) -> pd.DataFrame:
    """Map raw API records to the typed ``failures`` table (CONTRACT section 5)."""
    raw = to_frame(rows, [*FIELDS, "ID"])
    df = pd.DataFrame(
        {
            "cert": to_int(raw["CERT"]),
            "fail_date": parse_dates(raw["FAILDATE"]),
            "name": to_str(raw["NAME"]),
            "fail_year": to_int(raw["FAILYR"]),
            "restype": to_str(raw["RESTYPE"]).str.upper(),
            "restype1": to_str(raw["RESTYPE1"]),
            "cost": to_float(raw["COST"]),
            "qbfasset": to_float(raw["QBFASSET"]),
            "qbfdep": to_float(raw["QBFDEP"]),
            "cityst": to_str(raw["CITYST"]),
            "pstalp": to_str(raw["PSTALP"]),
            "chclass1": to_str(raw["CHCLASS1"]),
            "resdate": parse_dates(raw["RESDATE"]),
            "uninsdep": to_float(raw["UNINSDEP"]),
            "fdic_id": to_int(raw["ID"]),
        }
    )
    # FAILYR is a string in the API; derive it from the date when it is missing.
    df["fail_year"] = df["fail_year"].fillna(df["fail_date"].dt.year.astype("Int64"))
    # 488 pre-1966 failures have no CERT, so (cert, fail_date) is not unique for them;
    # every row is kept and fdic_id breaks ties so the file is reproducible.
    return df.sort_values([*KEY, "fdic_id"], kind="mergesort").reset_index(drop=True)


def failures_per_year(df: pd.DataFrame, start_year: int = 2000) -> pd.Series:
    """Count of ``FAILURE`` events per year (excludes open-bank assistance)."""
    sub = df[(df["restype"] == "FAILURE") & (df["fail_year"] >= start_year)]
    return sub.groupby("fail_year").size().rename("failures")


def check_benchmarks(df: pd.DataFrame) -> None:
    """Assert the 2009-2011 failure counts match the FDIC's published list."""
    counts = failures_per_year(df, start_year=min(BENCHMARK_YEARS))
    for year, expected in BENCHMARK_YEARS.items():
        got = int(counts.get(year, 0))
        if got != expected:
            raise AssertionError(f"{year}: {got} FAILURE rows, FDIC list says {expected}")


def fetch_failures(settings: Settings, client: FdicClient | None = None, force: bool = False):
    """Pull the whole ``/failures`` table (cached as ``failures/all.json``) and clean it."""
    own = client is None
    client = client or FdicClient(settings)
    try:
        rows = client.get("failures", fields=FIELDS, sort_by="CERT", cache_name="all", force=force)
    finally:
        if own:
            client.close()
    df = clean_failures(rows)
    log.info("failures: %d rows (%d FAILURE)", len(df), int((df["restype"] == "FAILURE").sum()))
    return df
