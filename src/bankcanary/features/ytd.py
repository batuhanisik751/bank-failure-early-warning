"""Year-to-date de-accumulation, quarter averages and lags for Call Report panels.

Call Report income-statement items (net income, provisions, charge-offs, interest
income and expense) are reported *cumulative within the calendar year*: the value at
Q3 is the sum of Q1, Q2 and Q3. Every helper here works on a frame keyed by
``(cert, repdte)`` and matches the previous observation by *exact quarter-end date*
for the same cert, never by row position, so gaps and mergers never leak one bank's
numbers into another quarter.
"""

from __future__ import annotations

import pandas as pd

KEY = ["cert", "repdte"]
YTD_PREV_MISSING = "ytd_prev_missing"
QUARTERLY_SUFFIX = "_q"


def quarter_number(repdte: pd.Series) -> pd.Series:
    """Calendar quarter (1..4) of each report date."""
    return pd.to_datetime(repdte).dt.quarter.astype("int64")


def previous_quarter_end(repdte: pd.Series) -> pd.Series:
    """Quarter-end date immediately before each report date (e.g. 2008-06-30 -> 2008-03-31)."""
    dates = pd.to_datetime(repdte)
    return (dates - pd.offsets.QuarterEnd(1)).dt.normalize()


def _shifted_lookup(df: pd.DataFrame, cols: list[str], target_dates: pd.Series) -> pd.DataFrame:
    """Return ``cols`` of ``df`` observed by the same cert at ``target_dates`` (NaN if absent).

    The result is aligned with ``df``'s index and carries an extra boolean column
    ``_present`` that is True when the same cert has *any* row at the target date. Only
    the row whose ``repdte`` equals the target date is used, so a bank that skipped a
    quarter gets NaN instead of the value from whatever row happens to sit before it.

    Both ``repdte`` and ``target_dates`` are coerced to datetime and normalised to
    midnight before matching, so date objects, second-resolution timestamps and dates
    carrying a time-of-day all match on the calendar day. Rows are aligned by position
    (the index is copied back onto the result), so the index name, non-unique labels
    or a column literally called ``index`` cannot break the lookup. A duplicated
    ``(cert, repdte)`` key has no well-defined previous quarter and raises ``ValueError``.
    """
    certs = df["cert"].to_numpy()
    observed = pd.to_datetime(df["repdte"]).dt.normalize().to_numpy()
    targets = pd.to_datetime(target_dates).dt.normalize().to_numpy()
    right = pd.DataFrame({"cert": certs, "_target": observed})
    if right.duplicated().any():
        raise ValueError("duplicated (cert, repdte) rows: previous quarter is ambiguous")
    for col in cols:
        right[col] = df[col].array
    right["_present"] = True
    left = pd.DataFrame({"cert": certs, "_target": targets})
    merged = left.merge(right, on=["cert", "_target"], how="left", sort=False)
    merged.index = df.index
    merged["_present"] = merged["_present"].fillna(False).astype(bool)
    return merged[cols + ["_present"]]


def deaccumulate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Turn year-to-date income items into single-quarter flows.

    For every column in ``cols`` a new ``<col>_q`` column holds
    ``YTD(q) - YTD(previous quarter-end of the same calendar year, same cert)``. Q1 rows
    are already a single quarter, so ``<col>_q = YTD``. When a Q2-Q4 row has no
    observation at the previous quarter-end (a bank that first reports mid-year, a
    merger, or a gap in filings) the boolean ``ytd_prev_missing`` is set and the quarter
    is approximated as ``YTD / quarter_number`` (the average quarter so far). The flag is
    False for Q1 rows. Rows where the previous quarter exists but the YTD value itself is
    NaN keep NaN, so structural missingness is preserved rather than invented.
    """
    if not cols:
        return df.copy()
    out = df.copy()
    qnum = quarter_number(out["repdte"])
    prev_date = previous_quarter_end(out["repdte"])
    prev = _shifted_lookup(out, cols, prev_date)
    prev_present = prev["_present"]
    is_q1 = qnum == 1
    flag = (~is_q1) & (~prev_present)
    for col in cols:
        ytd = out[col].astype("float64")
        diff = ytd - prev[col].astype("float64")
        fallback = ytd / qnum
        quarterly = ytd.where(is_q1, diff.where(~flag, fallback))
        out[col + QUARTERLY_SUFFIX] = quarterly
    out[YTD_PREV_MISSING] = flag.to_numpy()
    return out


def annualize(series_q: pd.Series) -> pd.Series:
    """Scale a single-quarter flow to a yearly rate (four quarters)."""
    return series_q * 4


def average_with_previous(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of the current and previous quarter-end value of ``col`` for the same cert.

    Denominators such as average assets are conventionally two-point averages. When the
    previous quarter-end is absent (or its value is NaN) the current value is used alone.
    """
    current = df[col].astype("float64")
    prev = _shifted_lookup(df, [col], previous_quarter_end(df["repdte"]))[col].astype("float64")
    return current.where(prev.isna(), (current + prev) / 2).rename(col)


def lag(df: pd.DataFrame, col: str, n_quarters: int) -> pd.Series:
    """Value of ``col`` reported by the same cert ``n_quarters`` quarter-ends earlier.

    Matched by exact date, so a bank with a filing gap gets NaN rather than the value
    from an older row. ``n_quarters`` must be positive.
    """
    if n_quarters < 1:
        raise ValueError("n_quarters must be >= 1")
    target = (pd.to_datetime(df["repdte"]) - pd.offsets.QuarterEnd(n_quarters)).dt.normalize()
    return _shifted_lookup(df, [col], target)[col].rename(f"{col}_lag{n_quarters}q")
