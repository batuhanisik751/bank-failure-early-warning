"""Adversarial checks for year-to-date de-accumulation, averages and lags.

Every expected value is derived by hand from PROJECT_SPEC.md section 7.3 and
docs/CONTRACT.md section 7: quarterly = YTD(q) - YTD(previous quarter-end of the same
calendar year); Q1 = YTD; missing prior quarter -> ``ytd_prev_missing`` and quarterly =
YTD / quarters elapsed; annualised = quarterly x 4; averages are two-point means with a
fallback to the current value. Matching is by exact quarter-end date for the same cert.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bankcanary.features.ytd import annualize, average_with_previous, deaccumulate, lag


def frame(rows: list[tuple], cols: tuple[str, ...] = ("cert", "repdte", "netinc")) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(cols))
    df["repdte"] = pd.to_datetime(df["repdte"])
    return df


def test_bank_whose_first_report_is_q4_is_flagged_and_divided_by_four() -> None:
    # Spec: no prior quarter -> flag and annualise YTD. Quarters elapsed at Q4 = 4.
    df = frame([(1, "2008-12-31", 80.0), (1, "2009-03-31", 6.0)])
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [20.0, 6.0]
    assert out["ytd_prev_missing"].tolist() == [True, False]
    # Annualising the flagged Q4 gives back the full-year YTD.
    assert annualize(out["netinc_q"]).iloc[0] == 80.0


def test_annualised_flagged_quarter_equals_ytd_scaled_to_a_year() -> None:
    # First report at Q3 with YTD -30: quarterly = -30 / 3 = -10, annualised = -40,
    # which is exactly "annualise the YTD": -30 * 4 / 3.
    df = frame([(2, "2008-09-30", -30.0)])
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [-10.0]
    assert bool(out["ytd_prev_missing"].iloc[0])
    assert annualize(out["netinc_q"]).tolist() == [-40.0]


def test_q1_row_is_never_flagged_even_without_a_prior_q4() -> None:
    df = frame([(3, "2009-03-31", 7.0)])
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [7.0]
    assert out["ytd_prev_missing"].tolist() == [False]


def test_ytd_resets_at_q1_after_a_multi_quarter_gap() -> None:
    # Q1 2008 filed, nothing until Q1 2009: Q1 is a single quarter regardless of gaps.
    df = frame([(4, "2008-03-31", 10.0), (4, "2009-03-31", 9.0), (4, "2009-06-30", 12.0)])
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [10.0, 9.0, 3.0]
    assert not out["ytd_prev_missing"].any()


def test_bank_skips_q2_and_q3_then_files_q4() -> None:
    # Contract: quarterly = YTD / quarters elapsed (4), flagged; not (YTD4 - YTD1) / 3.
    df = frame([(5, "2008-03-31", 12.0), (5, "2008-12-31", 100.0)])
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [12.0, 25.0]
    assert out["ytd_prev_missing"].tolist() == [False, True]


def test_two_banks_interleaved_and_unsorted_are_kept_apart() -> None:
    # Bank 10 (YTD 10, 25, 15, 40) and bank 20 (YTD 4, 4, 12) shuffled together and
    # shuffled in time. Each row must find *its own* bank's previous quarter-end.
    df = frame(
        [
            (20, "2008-09-30", 12.0),
            (10, "2008-12-31", 40.0),
            (10, "2008-03-31", 10.0),
            (20, "2008-03-31", 4.0),
            (10, "2008-09-30", 15.0),
            (20, "2008-06-30", 4.0),
            (10, "2008-06-30", 25.0),
        ],
    )
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [8.0, 25.0, 10.0, 4.0, -10.0, 0.0, 15.0]
    assert not out["ytd_prev_missing"].any()
    # Row order and index are those of the input.
    assert out.index.tolist() == df.index.tolist()
    assert out["cert"].tolist() == df["cert"].tolist()


def test_gap_for_one_bank_does_not_borrow_from_the_other_bank_on_the_same_date() -> None:
    # Bank 30 has Q2; bank 31 skipped Q2. Bank 31's Q3 must not use bank 30's Q2 row.
    df = frame([(30, "2008-06-30", 100.0), (31, "2008-03-31", 6.0), (31, "2008-09-30", 30.0)])
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [50.0, 6.0, 10.0]
    assert out["ytd_prev_missing"].tolist() == [True, False, True]


def test_non_default_index_is_preserved_and_values_stay_aligned() -> None:
    df = frame([(6, "2008-09-30", 15.0), (6, "2008-03-31", 10.0), (6, "2008-06-30", 25.0)])
    df.index = pd.Index([9, 3, 5])
    out = deaccumulate(df, ["netinc"])
    assert out.index.tolist() == [9, 3, 5]
    assert out.loc[9, "netinc_q"] == -10.0
    assert out.loc[3, "netinc_q"] == 10.0
    assert out.loc[5, "netinc_q"] == 15.0
    assert average_with_previous(df, "netinc").tolist() == [20.0, 10.0, 17.5]
    assert lag(df, "netinc", 1).tolist()[0] == 25.0
    assert np.isnan(lag(df, "netinc", 1).tolist()[1])


def test_non_unique_index_from_concat_is_handled() -> None:
    a = frame([(7, "2008-03-31", 10.0), (7, "2008-06-30", 25.0)])
    b = frame([(8, "2008-03-31", 5.0), (8, "2008-06-30", 9.0)])
    out = deaccumulate(pd.concat([a, b]), ["netinc"])
    assert out["netinc_q"].tolist() == [10.0, 15.0, 5.0, 4.0]
    assert not out["ytd_prev_missing"].any()


def test_duplicated_cert_repdte_rows_never_yield_silent_numbers() -> None:
    # (cert, repdte) is the table key; a duplicated key has no well-defined previous
    # quarter, so the helper must refuse rather than pick one silently.
    a = frame([(7, "2008-03-31", 10.0), (7, "2008-06-30", 25.0)])
    df = pd.concat([a, a], ignore_index=True)
    with pytest.raises(ValueError):
        deaccumulate(df, ["netinc"])


def test_multiple_income_columns_share_one_row_level_flag() -> None:
    rows = [
        (9, "2008-06-30", 20.0, 8.0),
        (9, "2008-09-30", 26.0, 8.0),
        (9, "2008-12-31", 30.0, 9.0),
    ]
    df = frame(rows, ("cert", "repdte", "netinc", "provisions"))
    out = deaccumulate(df, ["netinc", "provisions"])
    assert out["netinc_q"].tolist() == [10.0, 6.0, 4.0]
    assert out["provisions_q"].tolist() == [4.0, 0.0, 1.0]
    assert out["ytd_prev_missing"].tolist() == [True, False, False]


def test_integer_and_nullable_integer_ytd_columns() -> None:
    df = frame([(11, "2008-03-31", 10), (11, "2008-06-30", 25), (11, "2008-09-30", pd.NA)])
    df["netinc"] = pd.array([10, 25, pd.NA], dtype="Int64")
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist()[:2] == [10.0, 15.0]
    assert np.isnan(out["netinc_q"].iloc[2])
    assert not out["ytd_prev_missing"].iloc[2]


def test_repdte_as_datetime64_seconds_resolution() -> None:
    df = frame([(12, "2008-03-31", 10.0), (12, "2008-06-30", 25.0)])
    df["repdte"] = df["repdte"].astype("datetime64[s]")
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [10.0, 15.0]
    assert average_with_previous(df, "netinc").tolist() == [10.0, 17.5]
    assert lag(df, "netinc", 1).tolist()[1] == 10.0


def test_repdte_as_python_date_objects_gives_the_same_answer_as_datetime64() -> None:
    # quarter_number/previous_quarter_end call pd.to_datetime, so date objects are
    # accepted there; the lookup must match them on the same calendar day too.
    df = pd.DataFrame(
        {
            "cert": [13, 13],
            "repdte": [dt.date(2008, 3, 31), dt.date(2008, 6, 30)],
            "netinc": [10.0, 25.0],
        }
    )
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [10.0, 15.0]
    assert out["ytd_prev_missing"].tolist() == [False, False]
    assert average_with_previous(df, "netinc").tolist() == [10.0, 17.5]
    assert lag(df, "netinc", 1).tolist()[1] == 10.0


def test_named_index_does_not_break_the_lookup() -> None:
    df = frame([(14, "2008-03-31", 10.0), (14, "2008-06-30", 25.0)]).rename_axis("row")
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [10.0, 15.0]
    assert average_with_previous(df, "netinc").tolist() == [10.0, 17.5]
    assert lag(df, "netinc", 1).tolist()[1] == 10.0


def test_repdte_with_time_of_day_still_matches_the_same_quarter_end() -> None:
    # The target date is normalised to midnight but the looked-up rows are not, so a
    # timestamp at noon never matches itself and Q2 is wrongly flagged as 25 / 2.
    df = frame([(15, "2008-03-31 12:00", 10.0), (15, "2008-06-30 12:00", 25.0)])
    out = deaccumulate(df, ["netinc"])
    assert out["netinc_q"].tolist() == [10.0, 15.0]
    assert out["ytd_prev_missing"].tolist() == [False, False]


def test_average_first_row_prev_nan_and_current_nan() -> None:
    # Spec: mean of current and previous quarter-end, fallback current.
    # 2007Q4: no previous -> 100. 2008Q1: current NaN -> no defined mean, NaN.
    # 2008Q2: previous NaN -> falls back to current 120. 2008Q3: mean(130, 120) = 125.
    df = frame(
        [
            (16, "2007-12-31", 100.0),
            (16, "2008-03-31", np.nan),
            (16, "2008-06-30", 120.0),
            (16, "2008-09-30", 130.0),
        ],
    )
    avg = average_with_previous(df, "netinc")
    assert avg.iloc[0] == 100.0
    assert np.isnan(avg.iloc[1])
    assert avg.tolist()[2:] == [120.0, 125.0]


def test_average_with_previous_two_banks_interleaved_and_gap() -> None:
    df = frame(
        [
            (17, "2008-06-30", 200.0),
            (18, "2008-06-30", 20.0),
            (17, "2008-03-31", 100.0),
            (18, "2008-12-31", 40.0),
            (17, "2008-09-30", 300.0),
        ],
    )
    # 17: Q2 mean(200,100)=150; Q1 alone 100; Q3 mean(300,200)=250.
    # 18: Q2 alone 20; Q4 has no Q3 -> 40 (not 30 from Q2, not 200 from bank 17).
    assert average_with_previous(df, "netinc").tolist() == [150.0, 20.0, 100.0, 40.0, 250.0]


def test_lag_across_year_boundaries_and_certs() -> None:
    df = frame(
        [
            (19, "2009-03-31", 3.0),
            (19, "2008-09-30", 1.0),
            (21, "2008-12-31", 99.0),
            (19, "2008-12-31", 2.0),
            (19, "2009-12-31", 6.0),
            (19, "2009-09-30", 5.0),
        ],
    )
    lag1 = lag(df, "netinc", 1)
    assert lag1.iloc[0] == 2.0  # 2009Q1 <- 2008Q4 of the same cert, not cert 21
    assert np.isnan(lag1.iloc[1]) and np.isnan(lag1.iloc[2])
    assert lag1.iloc[3] == 1.0
    assert lag1.iloc[4] == 5.0
    assert np.isnan(lag1.iloc[5])  # 2009Q2 was never filed
    lag4 = lag(df, "netinc", 4)
    assert lag4.iloc[4] == 2.0  # 2009Q4 <- 2008Q4
    assert lag4.iloc[5] == 1.0  # 2009Q3 <- 2008Q3
    assert lag(df, "netinc", 5).iloc[4] == 1.0  # 2009Q4 <- 2008Q3


def test_helpers_do_not_mutate_the_input_frame() -> None:
    df = frame([(22, "2008-03-31", 10.0), (22, "2008-06-30", 25.0)])
    snapshot = df.copy()
    deaccumulate(df, ["netinc"])
    average_with_previous(df, "netinc")
    lag(df, "netinc", 1)
    pd.testing.assert_frame_equal(df, snapshot)
