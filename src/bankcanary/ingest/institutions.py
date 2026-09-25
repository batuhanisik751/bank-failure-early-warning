"""Ingest ``/institutions``: one row per FDIC certificate, open or closed.

This is the bank master: charter class, dates of establishment and closure, the
RSSD identifiers that link a bank to its holding company (``RSSDHCR``) and the
current total assets. ``ENDEFYMD`` is null for banks that are still open.
"""

from __future__ import annotations

import logging

import pandas as pd

from bankcanary.config import Settings
from bankcanary.ingest import parse_dates, to_bool, to_float, to_frame, to_int, to_str
from bankcanary.sources.fdic import FdicClient

log = logging.getLogger(__name__)

FIELDS = [
    "CERT", "NAME", "STALP", "CITY", "BKCLASS", "CHARTER", "ESTYMD", "ENDEFYMD", "ACTIVE",
    "INACTIVE", "FED_RSSD", "RSSDHCR", "ASSET", "ULTCERT", "NEWCERT", "LATITUDE",
    "LONGITUDE", "PROCDATE", "CLCODE", "CB",
]  # fmt: skip
KEY = ("cert",)


def clean_institutions(rows: list[dict]) -> pd.DataFrame:
    """Map raw API records to the typed ``institutions`` table (CONTRACT section 5).

    ``ENDEFYMD`` of ``""`` or ``12/31/9999`` (still open) becomes ``NaT``. ``FED_RSSD`` and
    ``RSSDHCR`` arrive as strings (sometimes empty) and become nullable ints; an empty
    ``RSSDHCR`` means the bank has no holding company.
    """
    raw = to_frame(rows, FIELDS)
    df = pd.DataFrame(
        {
            "cert": to_int(raw["CERT"]),
            "name": to_str(raw["NAME"]),
            "stalp": to_str(raw["STALP"]),
            "city": to_str(raw["CITY"]),
            "bkclass": to_str(raw["BKCLASS"]),
            "charter": to_str(raw["CHARTER"]),
            "estymd": parse_dates(raw["ESTYMD"]),
            "endefymd": parse_dates(raw["ENDEFYMD"]),
            "active": to_bool(raw["ACTIVE"]),
            "inactive": to_bool(raw["INACTIVE"]),
            "fed_rssd": to_int(raw["FED_RSSD"]),
            "rssdhcr": to_int(raw["RSSDHCR"]),
            "asset": to_float(raw["ASSET"]),
            "ultcert": to_int(raw["ULTCERT"]),
            "newcert": to_int(raw["NEWCERT"]),
            "latitude": to_float(raw["LATITUDE"]),
            "longitude": to_float(raw["LONGITUDE"]),
            "procdate": parse_dates(raw["PROCDATE"]),
            "clcode": to_int(raw["CLCODE"]),
            "cb": to_bool(raw["CB"]),
        }
    )
    df = df.drop_duplicates(subset=list(KEY), keep="first")
    return df.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def fetch_institutions(
    settings: Settings, client: FdicClient | None = None, force: bool = False
) -> pd.DataFrame:
    """Pull the whole ``/institutions`` table (cached as ``institutions/all.json``)."""
    own = client is None
    client = client or FdicClient(settings)
    try:
        rows = client.get(
            "institutions", fields=FIELDS, sort_by="CERT", cache_name="all", force=force
        )
    finally:
        if own:
            client.close()
    df = clean_institutions(rows)
    log.info("institutions: %d rows (%d active)", len(df), int(df["active"].sum()))
    return df
