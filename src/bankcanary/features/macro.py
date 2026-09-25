"""Macro features joined point-in-time from ``macro_state`` (P2, CONTRACT sections 11-12).

Bank failures come in regional waves that ride the local economy: unemployment and
house prices drive loan losses with a lag of several quarters, and the rate cycle sets
the value of the securities book and the cost of deposits. ``macro_state`` (built by
:mod:`bankcanary.ingest.macro`) already holds every series evaluated as of each panel
``avail_date`` (spec rule 6.4: the latest observation whose period end plus publication
lag is on or before the date the Call Report itself became available), so this module is
a pure join: state series on ``(stalp, avail_date)``, national series on ``avail_date``
alone. Nothing here looks at the panel's financials.

Territories (AS, FM, GU, PR, VI) have no FRED state series and rows without a ``stalp``
cannot be placed in a state, so their three state columns are NaN and the tree models
take that as its own signal; the national columns are filled for every row whose
``avail_date`` is on the macro grid.
"""

from __future__ import annotations

import pandas as pd

from bankcanary.features.spec import FeatureSpec, spec

#: Source table name and its key (CONTRACT section 11).
TABLE = "macro_state"
KEY = ["stalp", "avail_date"]

#: State-level series: feature name -> ``macro_state`` column.
STATE_COLUMNS: dict[str, str] = {
    "macro_unemp_rate": "unemp_rate",
    "macro_unemp_change_4q": "unemp_change_4q",
    "macro_hpi_change_4q": "hpi_change_4q",
}

#: National series: feature name -> ``macro_state`` column (identical across states).
NATIONAL_COLUMNS: dict[str, str] = {
    "macro_t10y3m": "t10y3m",
    "macro_dgs10": "dgs10",
    "macro_fedfunds_change_4q": "fedfunds_change_4q",
}

SPECS: list[FeatureSpec] = [
    spec(
        "macro_unemp_rate",
        "macro",
        "state unemployment rate (<ST>UR) as of avail_date",
        "percent",
        "Seasonally adjusted unemployment rate of the head-office state, latest month "
        "published 45 days or more before the Call Report became available. Job losses "
        "turn into missed loan payments, so a higher level means a weaker local book.",
        monotone=1,
        prototype="P2",
    ),
    spec(
        "macro_unemp_change_4q",
        "macro",
        "unemp_rate - unemp_rate 12 months earlier",
        "percentage_points",
        "Twelve-month change in the state unemployment rate. A rising rate marks a local "
        "downturn that has not yet reached the loan book, so the change adds lead time "
        "over the level.",
        monotone=1,
        prototype="P2",
    ),
    spec(
        "macro_hpi_change_4q",
        "macro",
        "log(<ST>STHPI / value four quarters earlier)",
        "log_ratio",
        "Four-quarter log change in the FHFA all-transactions house price index of the "
        "state. Falling prices erode the collateral behind mortgage and construction "
        "loans and turn delinquencies into losses, so risk falls as prices rise.",
        monotone=-1,
        prototype="P2",
    ),
    spec(
        "macro_t10y3m",
        "macro",
        "T10Y3M (10-year minus 3-month Treasury) as of avail_date",
        "percentage_points",
        "Slope of the Treasury curve. An inverted curve (negative spread) has preceded "
        "every recent recession and squeezes banks that fund long assets with short "
        "deposits, so a steeper curve means lower risk.",
        monotone=-1,
        prototype="P2",
    ),
    spec(
        "macro_dgs10",
        "macro",
        "DGS10 (10-year Treasury yield) as of avail_date",
        "percent",
        "Level of the 10-year Treasury yield. A high level raises funding costs but also "
        "the yield on new loans, and low levels came with both the calm 2010s and the "
        "2008 crisis, so no direction is imposed.",
        monotone=0,
        prototype="P2",
    ),
    spec(
        "macro_fedfunds_change_4q",
        "macro",
        "FEDFUNDS - FEDFUNDS 12 months earlier",
        "percentage_points",
        "Twelve-month change in the effective federal funds rate. A fast tightening "
        "cycle marks down fixed-rate securities and drains deposits to money funds (the "
        "2023 mechanism), so risk rises with the change.",
        monotone=1,
        prototype="P2",
    ),
]


def load_macro_state(settings=None) -> pd.DataFrame:
    """Read ``macro_state`` from Parquet (the default source when no frame is passed)."""
    from bankcanary.storage.parquet import read_table

    return read_table(TABLE, settings=settings)


def _dates(values: pd.Series) -> pd.Series:
    """``datetime64[ns]`` so both merge sides share one unit (pandas 3 infers ``[us]``)."""
    return pd.to_datetime(values).astype("datetime64[ns]")


def _states(values: pd.Series) -> pd.Series:
    """Upper-cased, stripped state codes with missing values kept missing."""
    return values.astype("string").str.strip().str.upper()


def national_rates(macro_state: pd.DataFrame) -> pd.DataFrame:
    """One row per ``avail_date`` with the national columns.

    The national series are repeated on every state row of ``macro_state``; the first
    non-missing value per date is taken, which is the same value on every row.
    """
    cols = list(NATIONAL_COLUMNS.values())
    frame = macro_state.assign(avail_date=_dates(macro_state["avail_date"]))
    return frame.groupby("avail_date", sort=True)[cols].first().reset_index()


def build(panel: pd.DataFrame, macro_state: pd.DataFrame | None = None, **deps: object):
    """Macro columns aligned to ``panel``'s index, joined on ``(stalp, avail_date)``.

    ``macro_state`` must be passed (``build_features`` loads it from Parquet when the
    caller does not). A duplicated macro key would fan the panel out, so it is an error.
    """
    if macro_state is None:
        raise ValueError("macro.build needs macro_state=<macro_state frame>")
    missing = [c for c in KEY if c not in panel.columns]
    if missing:
        raise ValueError(f"panel lacks the macro join columns {missing}")
    needed = KEY + list(STATE_COLUMNS.values()) + list(NATIONAL_COLUMNS.values())
    absent = [c for c in needed if c not in macro_state.columns]
    if absent:
        raise ValueError(f"macro_state lacks columns {absent}")
    state = macro_state[KEY + list(STATE_COLUMNS.values())].assign(
        stalp=_states(macro_state["stalp"]), avail_date=_dates(macro_state["avail_date"])
    )
    state = state.dropna(subset=["stalp"])
    if state.duplicated(KEY).any():
        raise ValueError("macro_state has duplicate (stalp, avail_date) keys")
    keys = pd.DataFrame(
        {"stalp": _states(panel["stalp"]), "avail_date": _dates(panel["avail_date"])}
    )
    merged = keys.merge(state, how="left", on=KEY, validate="many_to_one")
    merged = merged.merge(national_rates(macro_state), how="left", on="avail_date")
    if len(merged) != len(panel):
        raise RuntimeError("macro join changed the row count")
    out = pd.DataFrame(index=panel.index)
    for name, column in {**STATE_COLUMNS, **NATIONAL_COLUMNS}.items():
        out[name] = merged[column].astype("float64").to_numpy()
    return out[[s.name for s in SPECS]]
