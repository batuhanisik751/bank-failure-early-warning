"""The point-in-time state macro table (``macro_state``, CONTRACT section 11).

A bank's own Call Report says how it is doing; the macro table says what its home
state's economy was doing *as far as anyone could know on the prediction date*. For
every distinct ``(stalp, avail_date)`` in the panel the table holds the latest FRED
observation whose reporting period had ended **and** whose usual publication lag had
elapsed by ``avail_date``:

* state unemployment ``<ST>UR`` (monthly, BLS; released about 45 days after month end),
* state house price index ``<ST>STHPI`` (quarterly, FHFA; about 75 days after quarter end),
* ``FEDFUNDS`` (monthly average, known the next day), ``T10Y3M`` and ``DGS10`` (daily).

The four-quarter changes compare the value used at ``avail_date`` with the value that
would have been used four quarter-ends earlier, on the same point-in-time basis: the
lagged date comes from shifting the report quarter on the calendar, never from row
offsets, so a state that is missing a quarter in the panel still gets the right change.

Limitation: FRED serves the current vintage of every series. Revisions (benchmarked
unemployment, re-estimated house price indices) are therefore visible earlier than they
were in real time. Modelling vintages needs ALFRED and is out of scope for Prototype 2.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bankcanary.config import Settings
from bankcanary.sources.fred import cached_series
from bankcanary.storage.duckdb import replace_table
from bankcanary.storage.parquet import table_path, write_table

log = logging.getLogger(__name__)

TABLE = "macro_state"
KEY = ("stalp", "avail_date")

#: The 50 states plus DC. Territories (PR, GU, VI, AS, MP, FM, MH, PW) have no FRED
#: state series and get NaN for the state columns.
STATES: tuple[str, ...] = (
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
)

#: column -> (FRED id pattern, frequency, publication lag in days). ``{st}`` = state code.
SERIES: dict[str, tuple[str, str, int]] = {
    "unemp_rate": ("{st}UR", "M", 45),
    "hpi": ("{st}STHPI", "Q", 75),
    "fedfunds": ("FEDFUNDS", "M", 1),
    "t10y3m": ("T10Y3M", "D", 1),
    "dgs10": ("DGS10", "D", 1),
}
STATE_COLUMNS = ("unemp_rate", "hpi")
COLUMNS = (
    "unemp_rate",
    "unemp_change_4q",
    "hpi_change_4q",
    "fedfunds",
    "fedfunds_change_4q",
    "t10y3m",
    "dgs10",
)


def series_ids() -> list[str]:
    """Every FRED series the table needs (state series first, then the national ones)."""
    ids: list[str] = []
    for column, (pattern, _freq, _lag) in SERIES.items():
        if column in STATE_COLUMNS:
            ids.extend(pattern.format(st=st) for st in STATES)
        else:
            ids.append(pattern)
    return ids


def period_end(dates: pd.Series, freq: str) -> pd.Series:
    """FRED dates are period starts; the observation is only known once the period ends."""
    dates = pd.to_datetime(dates)
    if freq == "M":
        return dates + pd.offsets.MonthEnd(0)
    if freq == "Q":
        return dates + pd.offsets.QuarterEnd(0)
    if freq == "D":
        return dates
    raise ValueError(f"unknown frequency {freq!r}")


def usable_from(frame: pd.DataFrame, freq: str, lag_days: int) -> pd.DataFrame:
    """``DataFrame(usable_from, value)``: period end + publication lag, missing values dropped."""
    out = frame.dropna(subset=["value"])
    usable = period_end(out["date"], freq) + pd.Timedelta(days=int(lag_days))
    return (
        pd.DataFrame({"usable_from": usable.astype("datetime64[ns]"), "value": out["value"]})
        .sort_values("usable_from", kind="mergesort")
        .reset_index(drop=True)
    )


def point_in_time(
    grid: pd.DataFrame, observations: pd.DataFrame, by: str | None = None
) -> pd.Series:
    """The latest observation usable at each grid ``avail_date`` (``usable_from <= avail_date``).

    ``grid`` has ``avail_date`` (and ``by`` when given); ``observations`` has
    ``usable_from``, ``value`` (and ``by``). Returns a float Series aligned to ``grid``'s
    index; rows with no usable observation (or an unknown ``by`` key) are NaN.
    """
    if grid.empty:
        return pd.Series(np.nan, index=grid.index, dtype="float64")
    left_cols = ["avail_date"] + ([by] if by else [])
    left = (
        grid.loc[:, left_cols]
        .astype({"avail_date": "datetime64[ns]"})
        .reset_index(names="_row")
        .sort_values("avail_date", kind="mergesort")
    )
    right_cols = ["usable_from", "value"] + ([by] if by else [])
    right = (
        observations.loc[:, right_cols]
        .astype({"usable_from": "datetime64[ns]"})
        .sort_values("usable_from", kind="mergesort")
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="avail_date",
        right_on="usable_from",
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.set_index("_row")["value"].reindex(grid.index).astype("float64")


def _observations(series: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """Stack the cached FRED frames behind ``column`` into one ``usable_from`` table."""
    pattern, freq, lag = SERIES[column]
    if column not in STATE_COLUMNS:
        frame = series.get(pattern)
        empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": []})
        return usable_from(frame if frame is not None else empty, freq, lag)
    parts = []
    for st in STATES:
        frame = series.get(pattern.format(st=st))
        if frame is None or frame.empty:
            continue
        parts.append(usable_from(frame, freq, lag).assign(stalp=st))
    if not parts:
        return pd.DataFrame(
            {"usable_from": pd.Series(dtype="datetime64[ns]"), "value": [], "stalp": []}
        ).astype({"stalp": "str"})
    return pd.concat(parts, ignore_index=True)


def build_macro_state(
    grid: pd.DataFrame, series: dict[str, pd.DataFrame], availability_lag_days: int = 60
) -> pd.DataFrame:
    """Evaluate every macro column point-in-time on the panel's ``(stalp, avail_date)`` grid.

    ``series`` maps FRED ids to ``DataFrame(date, value)`` frames (period-start dates, as
    FRED serves them). The four-quarter change compares each row with the value that
    would have been used at the same state's availability date four report quarters
    earlier: ``repdte = avail_date - availability_lag_days`` is shifted back four quarter
    ends on the calendar and the lag re-added, so the comparison never depends on which
    quarters the panel happens to contain.
    """
    grid = (
        grid.loc[:, ["stalp", "avail_date"]]
        .dropna()
        .drop_duplicates()
        .astype({"stalp": "str", "avail_date": "datetime64[ns]"})
        .sort_values(["stalp", "avail_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    repdte = grid["avail_date"] - pd.Timedelta(days=int(availability_lag_days))
    repdte_4q = (repdte + pd.offsets.QuarterEnd(0)) - pd.offsets.QuarterEnd(4)
    lagged = grid.assign(avail_date=repdte_4q + pd.Timedelta(days=int(availability_lag_days)))
    now: dict[str, pd.Series] = {}
    then: dict[str, pd.Series] = {}
    for column in SERIES:
        obs = _observations(series, column)
        by = "stalp" if column in STATE_COLUMNS else None
        now[column] = point_in_time(grid, obs, by=by)
        if column in ("unemp_rate", "hpi", "fedfunds"):
            then[column] = point_in_time(lagged, obs, by=by)
    with np.errstate(divide="ignore", invalid="ignore"):
        hpi_change = np.log(now["hpi"]) - np.log(then["hpi"])
    out = grid.assign(
        unemp_rate=now["unemp_rate"],
        unemp_change_4q=now["unemp_rate"] - then["unemp_rate"],
        hpi_change_4q=hpi_change.astype("float64"),
        fedfunds=now["fedfunds"],
        fedfunds_change_4q=now["fedfunds"] - then["fedfunds"],
        t10y3m=now["t10y3m"],
        dgs10=now["dgs10"],
    )
    return out.loc[:, list(KEY) + list(COLUMNS)]


def missing_series(settings: Settings) -> list[str]:
    """Series ids that are not in the FRED cache yet (the CLI pulls these in chunks)."""
    from bankcanary.sources.fred import cache_path

    return [sid for sid in series_ids() if not cache_path(settings, sid).exists()]


def load_series(settings: Settings) -> dict[str, pd.DataFrame]:
    """Every cached series as ``DataFrame(date, value)``; missing ones are logged and skipped."""
    series: dict[str, pd.DataFrame] = {}
    for sid in series_ids():
        frame = cached_series(settings, sid)
        if frame is None:
            log.warning("FRED %s: not cached; its column will be NaN", sid)
        else:
            series[sid] = frame
    return series


def panel_grid(settings: Settings) -> pd.DataFrame:
    """Distinct ``(stalp, avail_date)`` pairs of the panel, reading only those two columns."""
    table = pq.read_table(table_path(settings, "panel"), columns=["stalp", "avail_date"])
    grid = table.to_pandas().dropna().drop_duplicates().reset_index(drop=True)
    return grid.astype({"stalp": "str", "avail_date": "datetime64[ns]"})


def build_macro(settings: Settings) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build ``macro_state`` from the cache, write Parquet + DuckDB and return summary stats."""
    series = load_series(settings)
    grid = panel_grid(settings)
    df = build_macro_state(grid, series, settings.availability_lag_days)
    path = write_table(df, TABLE, key=KEY, settings=settings)
    rows = replace_table(TABLE, path, settings=settings)
    stats: dict[str, object] = {
        "rows": len(df),
        "states": int(df["stalp"].nunique()),
        "series_cached": len(series),
        "series_expected": len(series_ids()),
        "duckdb_rows": rows,
        "path": path,
        "coverage": {c: float(df[c].notna().mean()) for c in COLUMNS},
    }
    return df, stats
