"""macro_state builder tests on synthetic FRED-shaped frames (no data/, no network)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bankcanary.ingest.macro import (
    COLUMNS,
    STATES,
    build_macro_state,
    period_end,
    point_in_time,
    series_ids,
    usable_from,
)


def frame(pairs: list[tuple[str, float | None]]) -> pd.DataFrame:
    df = pd.DataFrame(pairs, columns=["date", "value"])
    return df.assign(date=pd.to_datetime(df["date"]), value=df["value"].astype("float64"))


def monthly(start: str, end: str, fn) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="MS")
    return frame([(d.strftime("%Y-%m-%d"), fn(d)) for d in dates])


def quarterly(start: str, end: str, fn) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="QS")
    return frame([(d.strftime("%Y-%m-%d"), fn(d)) for d in dates])


def grid(rows: list[tuple[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["stalp", "avail_date"])
    return df.assign(avail_date=pd.to_datetime(df["avail_date"]))


def test_series_ids_cover_states_dc_and_national_rates():
    ids = series_ids()
    assert len(STATES) == 51 and "DC" in STATES and "PR" not in STATES
    assert len(ids) == 2 * 51 + 3
    assert {"CAUR", "CASTHPI", "DCUR", "FEDFUNDS", "T10Y3M", "DGS10"} <= set(ids)


def test_period_end_by_frequency():
    d = pd.Series(pd.to_datetime(["2008-01-01", "2008-02-15"]))
    assert period_end(d, "M").tolist() == list(pd.to_datetime(["2008-01-31", "2008-02-29"]))
    assert period_end(d, "Q").tolist() == list(pd.to_datetime(["2008-03-31", "2008-03-31"]))
    assert period_end(d, "D").tolist() == d.tolist()
    with pytest.raises(ValueError):
        period_end(d, "W")


def test_lag_rule_boundary_monthly_state_series():
    # Dec 2007 usable from 2008-02-14 (Dec 31 + 45); Jan 2008 usable from 2008-03-16.
    obs = usable_from(frame([("2007-12-01", 5.0), ("2008-01-01", 6.0)]), "M", 45).assign(stalp="CA")
    assert obs["usable_from"].tolist() == list(pd.to_datetime(["2008-02-14", "2008-03-16"]))
    g = grid(
        [("CA", "2008-02-13"), ("CA", "2008-03-15"), ("CA", "2008-03-16"), ("CA", "2009-01-01")]
    )
    got = point_in_time(g, obs, by="stalp")
    assert math.isnan(got.iloc[0])  # nothing published yet
    assert got.iloc[1] == 5.0  # one day before Jan becomes usable
    assert got.iloc[2] == 6.0  # period end + lag == avail_date is used
    assert got.iloc[3] == 6.0


def test_lag_rule_boundary_quarterly_and_daily_and_missing_values():
    hpi = usable_from(frame([("2007-07-01", 200.0), ("2007-10-01", 210.0)]), "Q", 75)
    assert hpi["usable_from"].tolist() == list(pd.to_datetime(["2007-12-14", "2008-03-15"]))
    g = grid([("CA", "2008-03-14"), ("CA", "2008-03-15")])
    assert point_in_time(g, hpi).tolist() == [200.0, 210.0]
    daily = usable_from(
        frame([("2008-03-13", 3.5), ("2008-03-14", None), ("2008-03-15", 3.7)]), "D", 1
    )
    assert len(daily) == 2  # the "." observation is dropped, not carried as zero
    got = point_in_time(grid([("CA", "2008-03-15"), ("CA", "2008-03-16")]), daily)
    assert got.tolist() == [3.5, 3.7]


def synthetic_series() -> dict[str, pd.DataFrame]:
    """Values encode the period they belong to so the chosen observation is identifiable."""
    ym = lambda d: d.year + d.month / 100  # noqa: E731 - 2008.03 == March 2008
    quarters_since_2005 = lambda d: (d.year - 2005) * 4 + (d.month - 1) // 3  # noqa: E731
    return {
        "CAUR": monthly("2005-01-01", "2009-12-01", ym),
        "CASTHPI": quarterly(
            "2005-01-01", "2009-10-01", lambda d: 100 * 1.1 ** quarters_since_2005(d)
        ),
        "FEDFUNDS": monthly("2005-01-01", "2009-12-01", ym),
        "T10Y3M": frame([("2008-05-28", 1.5), ("2008-05-29", 1.6), ("2008-05-30", 1.7)]),
        "DGS10": frame([("2008-05-29", 4.0), ("2008-05-30", 4.1)]),
    }


def test_four_quarter_changes_shift_the_calendar_not_the_rows():
    # CA is missing 2007Q2 and 2007Q4 from the grid; changes must still be year-over-year.
    g = grid([("CA", "2007-05-30"), ("CA", "2007-11-29"), ("CA", "2008-05-30")])
    out = build_macro_state(g, synthetic_series(), availability_lag_days=60)
    assert list(out.columns) == ["stalp", "avail_date"] + list(COLUMNS)
    row = out.set_index("avail_date").loc[pd.Timestamp("2008-05-30")]
    # avail 2008-05-30: unemployment usable when month end + 45 <= avail -> March 2008
    assert row["unemp_rate"] == pytest.approx(2008.03)
    # four quarter-ends earlier: repdte 2007-03-31 -> avail 2007-05-30 -> March 2007
    assert row["unemp_change_4q"] == pytest.approx(1.0)
    # HPI: quarter end + 75 <= 2008-05-30 -> 2007Q4; a year earlier 2006Q4; 4 quarters of 10 %
    assert row["hpi_change_4q"] == pytest.approx(4 * math.log(1.1))
    # FEDFUNDS lag 1 day: April 2008 (month end 04-30 + 1 <= 05-30); a year earlier April 2007
    assert row["fedfunds"] == pytest.approx(2008.04)
    assert row["fedfunds_change_4q"] == pytest.approx(1.0)
    assert row["t10y3m"] == 1.6 and row["dgs10"] == 4.0  # 05-30 observations usable from 05-31
    mid = out.set_index("avail_date").loc[pd.Timestamp("2007-11-29")]
    assert mid["unemp_rate"] == pytest.approx(2007.09)
    assert mid["unemp_change_4q"] == pytest.approx(1.0)  # vs 2006-11-29 -> September 2006
    assert mid["hpi_change_4q"] == pytest.approx(4 * math.log(1.1))


def test_unknown_state_gets_nan_state_columns_but_national_values():
    g = grid([("PR", "2008-05-30"), ("CA", "2008-05-30")])
    out = build_macro_state(g, synthetic_series()).set_index("stalp")
    pr = out.loc["PR"]
    assert np.isnan(pr[["unemp_rate", "unemp_change_4q", "hpi_change_4q"]].astype(float)).all()
    assert pr["fedfunds"] == pytest.approx(2008.04) and pr["dgs10"] == 4.0
    assert out.loc["CA", "unemp_rate"] == pytest.approx(2008.03)


def test_missing_series_and_duplicate_grid_rows():
    g = grid([("CA", "2008-05-30"), ("CA", "2008-05-30"), ("TX", "2008-05-30")])
    out = build_macro_state(g, {})
    assert len(out) == 2 and out[list(COLUMNS)].isna().all().all()
    for col in COLUMNS:
        assert str(out[col].dtype) == "float64"
    assert str(out["avail_date"].dtype) == "datetime64[ns]"
