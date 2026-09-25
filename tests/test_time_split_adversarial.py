"""Adversarial checks of the Rule 6.2 split utility (spec section 6.2, 5.4, 8.2).

Every expected value below is derived by hand from the spec text, never from the code:
with ``T = avail(first test q) = test_start + lag`` a training row needs ``window_end < T``
(strictly), a complete label and no post-failure drop flag; test rows are the usable
reports dated inside the test range. Synthetic frames only: no data files, no network.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from bankcanary.config import FixedSplit, Settings
from bankcanary.labels.build import horizon_columns
from bankcanary.splits import (
    assert_no_leakage,
    fixed_split_masks,
    prediction_date,
    test_mask,
    training_mask,
    walk_forward_folds,
)

LAG = 60
TEST_START = pd.Timestamp("2010-03-31")
T = pd.Timestamp("2010-05-30")  # avail(2010-03-31) with the 60-day lag
DAY = pd.Timedelta(days=1)


def rows(*specs: dict) -> pd.DataFrame:
    """Build a labels frame from explicit per-row values (defaults: usable, both horizons).

    Each spec may set ``repdte``, ``window_end`` (applied to both horizons unless
    ``window_end_4q``/``window_end_8q`` are given), ``complete`` (both horizons unless
    ``complete_4q``/``complete_8q``), ``dropped`` and ``cert``.
    """
    out = []
    for i, spec in enumerate(specs):
        row = {
            "cert": spec.get("cert", 100 + i),
            "repdte": pd.Timestamp(spec.get("repdte", "2005-03-31")),
            "dropped_failed_before_avail": spec.get("dropped", False),
        }
        for horizon in (4, 8):
            y_col, end_col, cens_col, complete_col = horizon_columns(horizon)
            end = spec.get(end_col, spec.get("window_end", pd.Timestamp("2006-12-31")))
            row[y_col] = 0
            row[end_col] = pd.Timestamp(end) if end is not None else pd.NaT
            row[cens_col] = False
            row[complete_col] = spec.get(f"complete_{horizon}q", spec.get("complete", True))
        out.append(row)
    frame = pd.DataFrame(out)
    for horizon in (4, 8):
        end_col = horizon_columns(horizon)[1]
        frame[end_col] = pd.to_datetime(frame[end_col]).astype("datetime64[ns]")
    return frame


def make_labels(certs=(1, 2), start="2001-03-31", end="2015-12-31", as_of="2026-09-25", lag=LAG):
    """Formula-built frame: avail = repdte + lag, window_end = avail + 3H months."""
    quarters = pd.date_range(start, end, freq="QE-DEC")
    frame = pd.concat(
        [pd.DataFrame({"cert": cert, "repdte": quarters}) for cert in certs], ignore_index=True
    )
    avail = frame["repdte"] + pd.Timedelta(days=lag)
    cols = {"dropped_failed_before_avail": False}
    for horizon in (4, 8):
        y_col, end_col, cens_col, complete_col = horizon_columns(horizon)
        window_end = (avail + pd.DateOffset(months=3 * horizon)).astype("datetime64[ns]")
        cols[y_col] = 0
        cols[end_col] = window_end
        cols[cens_col] = False
        cols[complete_col] = window_end <= pd.Timestamp(as_of)
    return frame.assign(**cols)


def quarters(labels: pd.DataFrame, mask: pd.Series) -> list[pd.Timestamp]:
    return sorted(labels.loc[mask, "repdte"].unique())


def qrange(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="QE-DEC"))


# --- Rule 6.2: strict inequality, prediction date uses the lag ---------------------------


def test_window_end_equal_to_first_test_prediction_date_is_excluded():
    # Spec 6.2: "window_end < T". The day before T is usable; T itself and T + 1 are not.
    labels = rows({"window_end": T - DAY}, {"window_end": T}, {"window_end": T + DAY})
    mask = training_mask(labels, 4, TEST_START, LAG)
    assert mask.tolist() == [True, False, False]
    mask8 = training_mask(labels, 8, TEST_START, LAG)
    assert mask8.tolist() == [True, False, False]


def test_cutoff_is_the_lagged_prediction_date_not_the_raw_report_date():
    # T = avail(2010-03-31) = 2010-05-30. A window closing on 2010-04-30 lies after the raw
    # test_start but before T, so its outcome is already known and the row must train.
    labels = rows({"window_end": "2010-04-30"}, {"window_end": "2010-03-31"})
    assert training_mask(labels, 4, TEST_START, LAG).tolist() == [True, True]
    # With a zero lag the prediction date is the report date itself: 2010-04-30 is now open.
    assert training_mask(labels, 4, TEST_START, 0).tolist() == [False, False]
    assert training_mask(rows({"window_end": "2010-03-30"}), 4, TEST_START, 0).tolist() == [True]


def test_prediction_date_accepts_dates_and_drops_time_of_day():
    assert prediction_date(dt.date(2010, 3, 31), LAG) == T
    assert prediction_date("2010-03-31", LAG) == T
    assert prediction_date(pd.Timestamp("2010-03-31 23:59"), LAG) == T
    # A test_start with a time component must not shift the cut by a day.
    labels = rows({"window_end": T - DAY}, {"window_end": T})
    assert training_mask(labels, 4, pd.Timestamp("2010-03-31 23:59"), LAG).tolist() == [
        True,
        False,
    ]


def test_training_mask_reads_the_columns_of_the_requested_horizon_only():
    closed, open_ = T - DAY, T + pd.DateOffset(months=12)
    labels = rows(
        # 4q window closed, 8q window still open: trains for H=4 only.
        {"window_end_4q": closed, "window_end_8q": open_},
        # both windows closed but only the 8q label is complete: trains for H=8 only.
        {"window_end": closed, "complete_4q": False, "complete_8q": True},
        # both closed, both complete: trains for both.
        {"window_end": closed},
    )
    assert training_mask(labels, 4, TEST_START, LAG).tolist() == [True, False, True]
    assert training_mask(labels, 8, TEST_START, LAG).tolist() == [False, True, True]


def test_eight_quarter_windows_trim_the_fixed_split_two_years_before_the_test_start():
    # Formula: window_end_8q = repdte + 60d + 24 months. 2008-03-31 -> 2008-05-30 -> 2010-05-30
    # == T (excluded); 2007-12-31 -> 2008-02-29 -> 2010-02-28 < T (kept).
    labels = make_labels(certs=(1,))
    settings = Settings(
        fixed_split=FixedSplit(
            train_start="2002-03-31",
            train_end="2008-12-31",
            test_start="2010-03-31",
            test_end="2013-12-31",
        ),
        availability_lag_days=LAG,
    )
    train8, _ = fixed_split_masks(labels, settings, 8)
    assert quarters(labels, train8) == qrange("2002-03-31", "2007-12-31")
    assert labels.loc[labels["repdte"] == "2008-03-31", "window_end_8q"].iloc[0] == T
    train4, _ = fixed_split_masks(labels, settings, 4)
    assert quarters(labels, train4) == qrange("2002-03-31", "2008-12-31")


# --- Usability: incomplete labels and dropped rows never train or test --------------------


def test_incomplete_label_with_a_closed_window_never_trains():
    # Spec 5.4: label_complete = False rows are "never used for training or evaluation",
    # even if the stored window_end happens to lie before T.
    labels = rows({"window_end": T - DAY, "complete": False}, {"window_end": T - DAY})
    assert training_mask(labels, 4, TEST_START, LAG).tolist() == [False, True]
    assert training_mask(labels, 8, TEST_START, LAG).tolist() == [False, True]


def test_missing_window_end_never_trains_even_when_flagged_complete():
    # Rule 6.2 cannot be verified without window_end, so the row is unusable.
    labels = rows({"window_end": None, "complete": True}, {"window_end": T - DAY})
    assert training_mask(labels, 4, TEST_START, LAG).tolist() == [False, True]
    with pytest.raises(ValueError, match="Rule 6.2"):
        assert_no_leakage(labels, 4, TEST_START, LAG)


@pytest.mark.xfail(
    strict=True,
    reason="_usable coerces a missing (NaN) label_complete flag to True via astype(bool)",
)
def test_unknown_completeness_is_not_complete():
    # Only a label known to be complete may train; a missing flag is not a True flag.
    labels = rows({"window_end": T - DAY}, {"window_end": T - DAY})
    labels = labels.assign(label_complete_4q=pd.Series([True, float("nan")], dtype=object))
    assert training_mask(labels, 4, TEST_START, LAG).tolist() == [True, False]
    assert test_mask(labels.assign(repdte=TEST_START), 4, TEST_START, TEST_START).tolist() == [
        True,
        False,
    ]


def test_nullable_boolean_flags_are_accepted():
    labels = rows({"window_end": T - DAY, "dropped": True}, {"window_end": T - DAY})
    labels = labels.astype(
        {"label_complete_4q": "boolean", "dropped_failed_before_avail": "boolean"}
    )
    assert training_mask(labels, 4, TEST_START, LAG).tolist() == [False, True]


def test_dropped_rows_never_enter_test_or_training():
    labels = rows(
        {"repdte": "2010-06-30", "window_end": "2011-08-29", "dropped": True},
        {"repdte": "2010-06-30", "window_end": "2011-08-29"},
        {"repdte": "2005-03-31", "window_end": "2006-05-30", "dropped": True},
        {"repdte": "2005-03-31", "window_end": "2006-05-30"},
    )
    assert test_mask(labels, 4, TEST_START, "2013-12-31").tolist() == [False, True, False, False]
    assert training_mask(labels, 4, TEST_START, LAG).tolist() == [False, False, False, True]
    # Integer 0/1 flags (as a DuckDB round-trip may yield) mean the same thing.
    dropped_int = labels["dropped_failed_before_avail"].astype(int)
    as_int = labels.assign(dropped_failed_before_avail=dropped_int)
    assert training_mask(as_int, 4, TEST_START, LAG).tolist() == [False, False, False, True]


def test_incomplete_labels_never_enter_the_test_set():
    labels = rows(
        {"repdte": "2010-06-30", "window_end": "2011-08-29", "complete_4q": False},
        {"repdte": "2010-06-30", "window_end": "2011-08-29", "complete_8q": False},
    )
    assert test_mask(labels, 4, TEST_START, "2013-12-31").tolist() == [False, True]
    assert test_mask(labels, 8, TEST_START, "2013-12-31").tolist() == [True, False]


def test_test_mask_bounds_are_inclusive_and_ignore_the_outcome_window():
    end = pd.Timestamp("2013-12-31")
    labels = rows(
        {"repdte": TEST_START - DAY, "window_end": "2011-05-30"},
        {"repdte": TEST_START, "window_end": "2011-05-30"},
        {"repdte": end, "window_end": "2015-03-01"},
        {"repdte": end + DAY, "window_end": "2015-03-02"},
        # A test row's window is far in the future of T; that is expected, not a leak.
        {"repdte": "2012-06-30", "window_end": "2013-08-29"},
    )
    assert test_mask(labels, 4, TEST_START, end).tolist() == [False, True, True, False, True]
    single_day = test_mask(labels, 4, TEST_START, TEST_START)
    assert single_day.tolist() == [False, True, False, False, False]


# --- Walk-forward folds ------------------------------------------------------------------


@pytest.mark.parametrize(("horizon", "years_back"), [(4, 2), (8, 3)])
def test_walk_forward_training_stops_before_any_open_window(horizon, years_back):
    # For test year Y, T = avail(Y-03-31) = Y-05-30. With window_end = repdte + 60d + 3H
    # months, the last closed report is Q4 of Y-2 (H=4) or Y-3 (H=8); Q1 of the next
    # year closes exactly on T and is excluded.
    labels = make_labels()
    folds = list(walk_forward_folds(labels, horizon, 2009, 2013, LAG))
    assert [year for year, _, _ in folds] == [2009, 2010, 2011, 2012, 2013]
    for year, train, test in folds:
        first_prediction = pd.Timestamp(year=year, month=5, day=30)
        end_col = horizon_columns(horizon)[1]
        assert (labels.loc[train, end_col] < first_prediction).all()
        assert labels.loc[train, "repdte"].max() == pd.Timestamp(year - years_back, 12, 31)
        assert set(labels.loc[train, "repdte"].dt.year) == set(range(2001, year - years_back + 1))
        assert quarters(labels, test) == qrange(f"{year}-03-31", f"{year}-12-31")
        assert not (train & test).any()


def test_walk_forward_consecutive_test_years_are_disjoint_and_exhaustive():
    labels = make_labels(certs=(1,), start="2005-03-31", end="2012-12-31")
    folds = list(walk_forward_folds(labels, 4, 2008, 2012, LAG))
    union = pd.Series(False, index=labels.index)
    for _, _, test in folds:
        assert not (union & test).any()
        union |= test
    assert quarters(labels, union) == qrange("2008-03-31", "2012-12-31")


def test_walk_forward_test_rows_exclude_dropped_and_incomplete_reports():
    labels = make_labels(certs=(1,), start="2005-03-31", end="2012-12-31")
    flagged = labels["repdte"] == "2010-06-30"
    incomplete = labels["repdte"] == "2010-09-30"
    labels = labels.assign(
        dropped_failed_before_avail=flagged,
        label_complete_4q=labels["label_complete_4q"] & ~incomplete,
    )
    ((year, train, test),) = walk_forward_folds(labels, 4, 2010, 2010, LAG)
    assert year == 2010
    assert quarters(labels, test) == [pd.Timestamp("2010-03-31"), pd.Timestamp("2010-12-31")]
    assert not train[flagged].any() and not train[incomplete].any()


def test_walk_forward_respects_the_lag_argument():
    # Stored windows come from a 60-day lag; with lag_days=120 the first prediction date
    # of 2010 is 2010-07-29, so 2009Q1 (window_end 2010-05-30) is already closed and trains.
    labels = make_labels(certs=(1,))
    ((year, train, _),) = walk_forward_folds(labels, 4, 2010, 2010, 120)
    assert year == 2010
    assert labels.loc[train, "repdte"].max() == pd.Timestamp("2009-03-31")
    ((_, train0, _),) = walk_forward_folds(labels, 4, 2010, 2010, 0)
    # lag 0: T = 2010-03-31; 2008Q4 closes 2010-03-01 (kept), 2009Q1 closes 2010-05-30 (out).
    assert labels.loc[train0, "repdte"].max() == pd.Timestamp("2008-12-31")


# --- Index alignment, overlapping bounds, configured lag, empty frames -------------------


def _reindexed_variants(labels: pd.DataFrame) -> list[pd.DataFrame]:
    shuffled = labels.sample(frac=1, random_state=7)
    strings = labels.set_axis([f"r{i}" for i in range(len(labels))])
    multi = labels.set_index(["cert", "repdte"], drop=False)
    duplicated = labels.set_axis([i // 2 for i in range(len(labels))])
    return [shuffled, strings, multi, duplicated]


def test_masks_align_to_whatever_index_the_labels_frame_carries():
    base = make_labels()
    expected_train = set(
        zip(
            base.loc[training_mask(base, 4, TEST_START, LAG), "cert"],
            base.loc[training_mask(base, 4, TEST_START, LAG), "repdte"],
        )
    )
    assert expected_train  # sanity: the RangeIndex baseline selects rows
    for variant in _reindexed_variants(base):
        train = training_mask(variant, 4, TEST_START, LAG)
        test = test_mask(variant, 4, TEST_START, "2013-12-31")
        assert train.index.equals(variant.index) and test.index.equals(variant.index)
        assert train.dtype == bool and test.dtype == bool
        kept = variant.loc[train]
        assert set(zip(kept["cert"], kept["repdte"])) == expected_train
        assert (kept["window_end_4q"] < T).all()
        assert quarters(variant, test) == qrange("2010-03-31", "2013-12-31")
        assert not (train & test).any()


def test_fixed_split_test_rows_dated_before_the_nominal_train_end_are_test_not_train():
    # Nominal ranges overlap: train 2002Q1..2010Q4, test 2010Q1..2011Q4. Rule 6.2 still cuts
    # training at 2008Q4 (2009Q1 closes exactly on T) and every 2010 report is a test row.
    labels = make_labels(certs=(1,))
    settings = Settings(
        fixed_split=FixedSplit(
            train_start="2002-03-31",
            train_end="2010-12-31",
            test_start="2010-03-31",
            test_end="2011-12-31",
        ),
        availability_lag_days=LAG,
    )
    train, test = fixed_split_masks(labels, settings, 4)
    assert quarters(labels, train) == qrange("2002-03-31", "2008-12-31")
    assert quarters(labels, test) == qrange("2010-03-31", "2011-12-31")
    assert not (train & test).any()


def test_fixed_split_uses_the_configured_lag_not_the_default():
    # Windows stored with a 60-day lag; settings say 120 days -> T = 2010-07-29, so 2009Q1
    # (window_end 2010-05-30) is closed and trains, 2009Q2 (2010-08-29) is not.
    labels = make_labels(certs=(1,))
    settings = Settings(
        fixed_split=FixedSplit(
            train_start="2002-03-31",
            train_end="2009-12-31",
            test_start="2010-03-31",
            test_end="2013-12-31",
        ),
        availability_lag_days=120,
    )
    train, _ = fixed_split_masks(labels, settings, 4)
    assert quarters(labels, train) == qrange("2002-03-31", "2009-03-31")


def test_empty_and_fully_unusable_frames_give_empty_masks_without_errors():
    empty = make_labels().iloc[0:0]
    assert training_mask(empty, 4, TEST_START, LAG).tolist() == []
    assert test_mask(empty, 4, TEST_START, "2013-12-31").tolist() == []
    settings = Settings(
        fixed_split=FixedSplit(
            train_start="2002-03-31",
            train_end="2008-12-31",
            test_start="2010-03-31",
            test_end="2013-12-31",
        ),
        availability_lag_days=LAG,
    )
    train, test = fixed_split_masks(empty, settings, 4)
    assert len(train) == 0 and len(test) == 0 and train.dtype == bool
    assert list(walk_forward_folds(empty, 4, 2008, 2012, LAG)) == []
    unusable = make_labels().assign(dropped_failed_before_avail=True)
    assert not training_mask(unusable, 4, TEST_START, LAG).any()
    assert not test_mask(unusable, 4, TEST_START, "2013-12-31").any()
    assert list(walk_forward_folds(unusable, 4, 2008, 2012, LAG)) == []
    assert_no_leakage(unusable.iloc[0:0], 4, TEST_START, LAG)


def test_assert_no_leakage_treats_equality_as_a_violation():
    assert_no_leakage(rows({"window_end": T - DAY}), 4, TEST_START, LAG)
    with pytest.raises(ValueError, match="1 of 1"):
        assert_no_leakage(rows({"window_end": T}), 4, TEST_START, LAG)
    with pytest.raises(ValueError, match="Rule 6.2"):
        assert_no_leakage(rows({"window_end": "2010-04-30"}), 4, TEST_START, 0)
