"""Hand-computed checks for year-to-date de-accumulation, averages and lags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankcanary.features.ytd import (
    annualize,
    average_with_previous,
    deaccumulate,
    lag,
    previous_quarter_end,
    quarter_number,
)


def make_frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (cert, repdte, netinc, asset)."""
    df = pd.DataFrame(rows, columns=["cert", "repdte", "netinc", "asset"])
    df["repdte"] = pd.to_datetime(df["repdte"])
    return df


@pytest.fixture
def full_year() -> pd.DataFrame:
    # Cert 1 files every quarter of 2008 and Q1 2009; Q3 is a loss quarter.
    return make_frame(
        [
            (1, "2008-03-31", 10.0, 100.0),
            (1, "2008-06-30", 25.0, 110.0),
            (1, "2008-09-30", 15.0, 120.0),
            (1, "2008-12-31", 40.0, 130.0),
            (1, "2009-03-31", 7.0, 140.0),
        ]
    )


def test_quarter_number_and_previous_quarter_end() -> None:
    dates = pd.Series(pd.to_datetime(["2008-03-31", "2008-06-30", "2008-09-30", "2008-12-31"]))
    assert quarter_number(dates).tolist() == [1, 2, 3, 4]
    expected = pd.to_datetime(["2007-12-31", "2008-03-31", "2008-06-30", "2008-09-30"])
    assert previous_quarter_end(dates).tolist() == list(expected)


def test_q1_equals_ytd_and_q2_q3_q4_chain(full_year: pd.DataFrame) -> None:
    out = deaccumulate(full_year, ["netinc"])
    # Q1 = 10; Q2 = 25 - 10; Q3 = 15 - 25 (loss); Q4 = 40 - 15.
    assert out["netinc_q"].tolist() == [10.0, 15.0, -10.0, 25.0, 7.0]
    assert not out["ytd_prev_missing"].any()
    assert out["ytd_prev_missing"].dtype == bool
    # Original column and row order untouched.
    assert out["netinc"].tolist() == full_year["netinc"].tolist()
    assert list(out.index) == list(full_year.index)


def test_year_boundary_q1_is_not_differenced_against_q4(full_year: pd.DataFrame) -> None:
    out = deaccumulate(full_year, ["netinc"])
    q1_2009 = out.loc[out["repdte"] == "2009-03-31"].iloc[0]
    assert q1_2009["netinc_q"] == 7.0  # not 7 - 40
    assert not q1_2009["ytd_prev_missing"]


def test_negative_ytd_swings() -> None:
    df = make_frame(
        [
            (5, "2010-03-31", -30.0, 1.0),
            (5, "2010-06-30", -50.0, 1.0),
            (5, "2010-09-30", -20.0, 1.0),
            (5, "2010-12-31", -60.0, 1.0),
        ]
    )
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [-30.0, -20.0, 30.0, -40.0]


def test_bank_first_reporting_in_q3_is_flagged_and_divided_by_three() -> None:
    df = make_frame([(2, "2008-09-30", 30.0, 200.0), (2, "2008-12-31", 50.0, 220.0)])
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [10.0, 20.0]
    assert out["ytd_prev_missing"].tolist() == [True, False]


def test_gap_in_q2_flags_q3_and_uses_ytd_over_quarter_number() -> None:
    df = make_frame(
        [
            (3, "2008-03-31", 10.0, 100.0),
            (3, "2008-09-30", 45.0, 120.0),
            (3, "2008-12-31", 60.0, 130.0),
        ]
    )
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [10.0, 15.0, 15.0]
    assert out["ytd_prev_missing"].tolist() == [False, True, False]


def test_deaccumulate_matches_by_cert_not_row_position() -> None:
    # Cert 9's Q2 row sits directly above cert 8's Q3 row; cert 8 must not use it.
    df = make_frame([(9, "2008-06-30", 100.0, 1.0), (8, "2008-09-30", 30.0, 1.0)])
    out = deaccumulate(df, ["netinc"])
    assert out.loc[1, "netinc_q"] == 10.0
    assert bool(out.loc[1, "ytd_prev_missing"])


def test_deaccumulate_handles_multiple_columns_and_nan_ytd() -> None:
    df = make_frame([(4, "2008-03-31", 10.0, 100.0), (4, "2008-06-30", np.nan, 130.0)])
    out = deaccumulate(df, ["netinc", "asset"])
    assert out["asset_q"].tolist() == [100.0, 30.0]
    assert np.isnan(out.loc[1, "netinc_q"])
    assert not out.loc[1, "ytd_prev_missing"]


def test_annualize() -> None:
    assert annualize(pd.Series([2.5, -1.0])).tolist() == [10.0, -4.0]


def test_average_with_previous_and_fallback(full_year: pd.DataFrame) -> None:
    avg = average_with_previous(full_year, "asset")
    # First quarter has no previous -> current value; then two-point means.
    assert avg.tolist() == [100.0, 105.0, 115.0, 125.0, 135.0]


def test_average_falls_back_across_a_gap() -> None:
    df = make_frame([(6, "2008-03-31", 0.0, 100.0), (6, "2008-09-30", 0.0, 300.0)])
    assert average_with_previous(df, "asset").tolist() == [100.0, 300.0]


def test_lag_exact_dates_and_gap_returns_nan(full_year: pd.DataFrame) -> None:
    lag1 = lag(full_year, "asset", 1)
    assert lag1.name == "asset_lag1q"
    assert lag1.tolist()[1:] == [100.0, 110.0, 120.0, 130.0]
    assert np.isnan(lag1.iloc[0])
    lag4 = lag(full_year, "asset", 4)
    assert np.isnan(lag4.iloc[3]) and lag4.iloc[4] == 100.0
    gappy = make_frame([(7, "2008-03-31", 0.0, 100.0), (7, "2008-09-30", 0.0, 300.0)])
    assert lag(gappy, "asset", 1).isna().all()
    assert lag(gappy, "asset", 2).tolist()[1] == 100.0
    with pytest.raises(ValueError):
        lag(gappy, "asset", 0)
