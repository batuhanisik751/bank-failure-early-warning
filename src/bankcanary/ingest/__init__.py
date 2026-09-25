"""Ingest: FDIC API records -> typed pandas frames -> Parquet + DuckDB tables.

Shared parsing helpers live here. The FDIC returns every field as JSON with loose
types (numbers as strings, empty strings for missing values, several date formats),
so each ingest module maps the raw upper-case columns to the snake_case, typed
columns fixed in CONTRACT section 5.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Sentinel the FDIC uses for "still open" in ``ENDEFYMD``.
OPEN_ENDED = "12/31/9999"


def to_frame(rows: Iterable[dict], fields: list[str]) -> pd.DataFrame:
    """Build a frame with exactly ``fields`` (in order), adding missing ones as NA."""
    df = pd.DataFrame(list(rows))
    for col in fields:
        if col not in df.columns:
            df[col] = pd.NA
    return df.loc[:, fields]


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse FDIC dates ("3/10/2023", "03/10/2023" or "2021-07-01T00:00:00") to datetime64[ns].

    Empty strings and the ``12/31/9999`` open-ended sentinel become ``NaT``.
    """
    text = series.astype("string").str.strip()
    text = text.mask(text.isin(["", OPEN_ENDED]))
    slash = text.str.contains("/", na=False)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    out.loc[slash] = pd.to_datetime(text[slash], format="%m/%d/%Y", errors="coerce")
    out.loc[~slash] = pd.to_datetime(text[~slash], format="ISO8601", errors="coerce")
    return out.astype("datetime64[ns]")


def to_int(series: pd.Series) -> pd.Series:
    """Coerce numbers-as-strings (``"573401"``, ``""``, ``None``) to nullable ``Int64``."""
    text = series.astype("string").str.strip().replace("", pd.NA)
    return pd.to_numeric(text, errors="coerce").round().astype("Int64")


def to_float(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().replace("", pd.NA)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def to_str(series: pd.Series) -> pd.Series:
    """Strings with empty values as missing (pandas 3 ``str`` dtype)."""
    text = series.astype("string").str.strip().replace("", pd.NA)
    return text.astype("str")


def to_bool(series: pd.Series) -> pd.Series:
    """FDIC flags arrive as 0/1 ints or "0"/"1" strings."""
    return to_int(series).fillna(0).astype("int64").astype(bool)


__all__ = ["to_frame", "parse_dates", "to_int", "to_float", "to_str", "to_bool", "np"]
