"""Rule 6.2 split utility on a synthetic labels frame (no data files, no network)."""

from __future__ import annotations

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
AS_OF = pd.Timestamp("2026-09-25")
TEST_START = pd.Timestamp("2010-03-31")
# prediction_date(2010-03-31) = 2010-05-30. Report 2009-03-31 -> avail 2009-05-30 -> 4q
# window ends exactly 2010-05-30 (not usable); 2008-12-31 -> ends 2010-03-01 (usable).
FIRST_TEST_PREDICTION = pd.Timestamp("2010-05-30")


def make_labels(certs=(1, 2), start="2001-03-31", end="2015-12-31", as_of=AS_OF):
    quarters = pd.date_range(start, end, freq="QE-DEC")
    frame = pd.concat(
        [pd.DataFrame({"cert": cert, "repdte": quarters}) for cert in certs], ignore_index=True
    )
    avail = frame["repdte"] + pd.Timedelta(days=LAG)
    cols = {"window_start": avail, "dropped_failed_before_avail": False}
    for horizon in (4, 8):
        y_col, end_col, cens_col, complete_col = horizon_columns(horizon)
        window_end = (avail + pd.DateOffset(months=3 * horizon)).astype("datetime64[ns]")
        cols[y_col] = 0
        cols[end_col] = window_end
        cols[cens_col] = False
        cols[complete_col] = window_end <= as_of
    return frame.assign(**cols)


def _quarters(labels: pd.DataFrame, mask: pd.Series) -> list[pd.Timestamp]:
    return sorted(labels.loc[mask, "repdte"].unique())


def _range(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="QE-DEC"))


def test_prediction_date_adds_the_availability_lag():
    assert prediction_date("2010-03-31", LAG) == FIRST_TEST_PREDICTION
    assert prediction_date(pd.Timestamp("2010-03-31 13:00"), 0) == TEST_START


def test_training_mask_applies_rule_6_2_and_excludes_incomplete_and_dropped_rows():
    labels = make_labels(certs=(1,))
    incomplete = labels["repdte"] == "2005-03-31"
    dropped = labels["repdte"] == "2006-06-30"
    labels = labels.assign(
        label_complete_4q=labels["label_complete_4q"] & ~incomplete,
        dropped_failed_before_avail=dropped,
    )
    mask = training_mask(labels, 4, TEST_START, LAG)
    assert mask.dtype == bool and mask.index.equals(labels.index)
    kept = labels.loc[mask]
    assert kept["repdte"].max() == pd.Timestamp("2008-12-31")
    assert (kept["window_end_4q"] < FIRST_TEST_PREDICTION).all()
    assert pd.Timestamp("2009-03-31") not in set(kept["repdte"])  # window_end == T
    assert pd.Timestamp("2005-03-31") not in set(kept["repdte"])  # incomplete label
    assert pd.Timestamp("2006-06-30") not in set(kept["repdte"])  # dropped row
    assert len(kept) == 32 - 2  # 2001Q1..2008Q4 minus the two flagged rows


def test_assert_no_leakage_rejects_a_leaky_selection():
    labels = make_labels(certs=(1,))
    leaky = labels.loc[labels["repdte"] <= "2009-06-30"]  # 2009Q1 (== T) and 2009Q2 (> T)
    assert (leaky["window_end_4q"] == FIRST_TEST_PREDICTION).sum() == 1
    with pytest.raises(ValueError, match="Rule 6.2 violated .* 2 of 34"):
        assert_no_leakage(leaky, 4, TEST_START, LAG)
    with pytest.raises(ValueError, match="Rule 6.2"):
        assert_no_leakage(leaky.assign(window_end_4q=pd.NaT), 4, TEST_START, LAG)


def test_assert_no_leakage_accepts_a_compliant_selection():
    labels = make_labels()
    for horizon in (4, 8):
        assert_no_leakage(
            labels.loc[training_mask(labels, horizon, TEST_START, LAG)], horizon, TEST_START, LAG
        )
    assert_no_leakage(labels.iloc[0:0], 4, TEST_START, LAG)


def _settings() -> Settings:
    split = FixedSplit(
        train_start="2002-03-31",
        train_end="2008-12-31",
        test_start="2010-03-31",
        test_end="2013-12-31",
    )
    return Settings(fixed_split=split, availability_lag_days=LAG)


def test_fixed_split_masks_keep_the_nominal_train_end_for_4q_and_cut_earlier_for_8q():
    labels = make_labels()
    settings = _settings()
    expected_test = _range("2010-03-31", "2013-12-31")

    train4, test4 = fixed_split_masks(labels, settings, 4)
    assert _quarters(labels, train4) == _range("2002-03-31", "2008-12-31")
    assert _quarters(labels, test4) == expected_test
    assert_no_leakage(labels.loc[train4], 4, settings.fixed_split.test_start, LAG)

    train8, test8 = fixed_split_masks(labels, settings, 8)
    # 2008Q1 -> avail 2008-05-30 -> 8q window ends 2010-05-30 == T, so 2007Q4 is the last cut.
    assert _quarters(labels, train8) == _range("2002-03-31", "2007-12-31")
    assert _quarters(labels, test8) == expected_test
    assert_no_leakage(labels.loc[train8], 8, settings.fixed_split.test_start, LAG)
    assert not (train4 & test4).any() and not (train8 & test8).any()


def test_test_mask_respects_bounds_and_usability_flags():
    labels = make_labels(certs=(1,), as_of=pd.Timestamp("2011-12-31"))
    mask = test_mask(labels, 4, "2010-03-31", "2013-12-31")
    # Windows for reports after 2010Q3 close after the 2011-12-31 as-of date -> incomplete.
    assert _quarters(labels, mask) == _range("2010-03-31", "2010-09-30")
    with pytest.raises(ValueError, match="before"):
        test_mask(labels, 4, "2011-01-01", "2010-01-01")


def test_walk_forward_folds_never_overlap_and_pass_the_leakage_check():
    labels = make_labels()
    folds = list(walk_forward_folds(labels, 4, 2008, 2012, LAG))
    assert [year for year, _, _ in folds] == [2008, 2009, 2010, 2011, 2012]
    seen_test = pd.Series(False, index=labels.index)
    previous_train = None
    for year, train, test in folds:
        assert not (train & test).any()
        assert not (seen_test & test).any()
        seen_test |= test
        assert set(labels.loc[test, "repdte"].dt.year) == {year}
        assert labels.loc[test, "repdte"].nunique() == 4
        test_start = pd.Timestamp(year=year, month=3, day=31)
        assert_no_leakage(labels.loc[train], 4, test_start, LAG)
        # 4q rule: the last usable training report is Q4 two years before the test year.
        assert labels.loc[train, "repdte"].max() == pd.Timestamp(year=year - 2, month=12, day=31)
        if previous_train is not None:
            assert (previous_train & ~train).sum() == 0  # training sets are nested
        previous_train = train


def test_walk_forward_folds_skip_years_without_complete_test_labels():
    labels = make_labels(certs=(1,), as_of=pd.Timestamp("2010-12-31"))
    years = [year for year, _, _ in walk_forward_folds(labels, 4, 2008, 2011, LAG)]
    assert years == [2008, 2009]
    with pytest.raises(ValueError):
        list(walk_forward_folds(labels, 4, 2010, 2009, LAG))


def test_missing_flags_are_treated_conservatively():
    # A NaN completeness flag is not a known-complete label; a NaN drop flag is not a
    # known-clean report. Both keep the row out of training and test (spec 5.4).
    labels = make_labels(certs=(1,), end="2008-12-31")
    nan_complete = labels["repdte"] == "2005-03-31"
    nan_dropped = labels["repdte"] == "2006-06-30"
    complete = labels["label_complete_4q"].astype(object).mask(nan_complete, float("nan"))
    dropped = labels["dropped_failed_before_avail"].astype(object).mask(nan_dropped, float("nan"))
    labels = labels.assign(label_complete_4q=complete, dropped_failed_before_avail=dropped)
    train = training_mask(labels, 4, TEST_START, LAG)
    assert not train[nan_complete | nan_dropped].any()
    assert train[~(nan_complete | nan_dropped)].all()
    test = test_mask(labels, 4, "2005-03-31", "2006-06-30")
    assert not test[nan_complete | nan_dropped].any()
    assert test.sum() == 4  # 2005-06-30 .. 2006-03-31
