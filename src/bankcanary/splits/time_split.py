"""Time-based train/test splits that enforce the outcome-window rule (spec section 6.2).

A bank-quarter report dated ``repdte`` becomes usable ``lag_days`` later (its prediction
date). Its label is only known once its outcome window has closed, at ``window_end_Hq``.
When the test set starts at report date ``T``, the model is "standing" on
``prediction_date(T)``: it may only learn from rows whose outcome had already been
observed by then, i.e. ``window_end_Hq < prediction_date(T)``. Rows whose window is still
open on the failures-list date (``label_complete_Hq == False``) and reports that were
published after the bank had already failed (``dropped_failed_before_avail``) never enter
training either.

Rule 6.2 lives in exactly one place, :func:`training_mask`; every training path (fixed
split, walk-forward backtest, case studies) builds its selection through it and can be
checked afterwards with :func:`assert_no_leakage`.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterator

import pandas as pd

from bankcanary.config import Settings
from bankcanary.labels.build import horizon_columns

log = logging.getLogger(__name__)

DEFAULT_LAG_DAYS: int = Settings.model_fields["availability_lag_days"].default
DROPPED_COL = "dropped_failed_before_avail"

DateLike = str | dt.date | pd.Timestamp


def prediction_date(repdte: DateLike, lag_days: int = DEFAULT_LAG_DAYS) -> pd.Timestamp:
    """First calendar day on which the report dated ``repdte`` can be used for a prediction.

    This is the same ``avail_date = repdte + availability_lag_days`` the panel stores; it
    is recomputed here so the split utility works from a labels frame alone.
    """
    return pd.Timestamp(repdte).normalize() + pd.Timedelta(days=int(lag_days))


def _require_columns(labels: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in labels.columns]
    if missing:
        raise KeyError(f"labels frame is missing required column(s): {missing}")


def _flag(series: pd.Series, *, missing: bool) -> pd.Series:
    """Coerce a boolean-like flag column to plain ``bool``, filling missing values.

    The canonical labels table stores flags as ``bool``, but a DuckDB round-trip yields
    ``int`` 0/1 and a join can introduce ``NaN``/``pd.NA`` (``object`` or nullable
    ``boolean`` dtype). ``astype(bool)`` alone would turn a float ``NaN`` into ``True``,
    so the caller states which side is safe when the flag is unknown.
    """
    return series.fillna(missing).astype(bool)


def _usable(labels: pd.DataFrame, horizon: int) -> pd.Series:
    """Rows with a complete label that were not published after the bank had failed.

    Only a label *known* to be complete may train or be scored (spec 5.4), so an unknown
    completeness flag counts as incomplete; an unknown drop flag counts as dropped.
    """
    complete_col = horizon_columns(horizon)[3]
    _require_columns(labels, [complete_col, DROPPED_COL])
    complete = _flag(labels[complete_col], missing=False)
    dropped = _flag(labels[DROPPED_COL], missing=True)
    return complete & ~dropped


def training_mask(
    labels: pd.DataFrame,
    horizon: int,
    test_start_repdte: DateLike,
    lag_days: int = DEFAULT_LAG_DAYS,
) -> pd.Series:
    """Boolean mask of rows a model may train on when the test period starts at a report date.

    Implements Rule 6.2: with ``T = prediction_date(test_start_repdte)``, a row is usable
    only if ``window_end_Hq < T`` (its outcome was already known on the first test
    prediction date), its label is complete and it is not a dropped post-failure report.
    Because ``window_end`` sits ``lag + 3H months`` after ``repdte``, this cuts training
    off well before the test period: the gap grows with the horizon.
    """
    end_col = horizon_columns(horizon)[1]
    _require_columns(labels, [end_col])
    first_test_prediction = prediction_date(test_start_repdte, lag_days)
    closed = labels[end_col] < first_test_prediction
    mask = closed & _usable(labels, horizon)
    return mask.fillna(False).astype(bool).rename(f"train_{horizon}q")


def test_mask(
    labels: pd.DataFrame,
    horizon: int,
    test_start_repdte: DateLike,
    test_end_repdte: DateLike,
) -> pd.Series:
    """Boolean mask of the bank-quarters a model is scored on.

    Test rows are the reports dated between ``test_start_repdte`` and ``test_end_repdte``
    inclusive whose label is complete and which are not dropped post-failure reports. No
    outcome-window condition applies here: a test row's outcome is *supposed* to lie in
    the future of its prediction date.
    """
    _require_columns(labels, ["repdte"])
    start = pd.Timestamp(test_start_repdte).normalize()
    end = pd.Timestamp(test_end_repdte).normalize()
    if end < start:
        raise ValueError(f"test_end_repdte {end.date()} is before test_start_repdte {start.date()}")
    in_period = (labels["repdte"] >= start) & (labels["repdte"] <= end)
    mask = in_period & _usable(labels, horizon)
    return mask.fillna(False).astype(bool).rename(f"test_{horizon}q")


def assert_no_leakage(
    train_labels: pd.DataFrame,
    horizon: int,
    test_start_repdte: DateLike,
    lag_days: int = DEFAULT_LAG_DAYS,
) -> None:
    """Raise ``ValueError`` if any training row's outcome window closes on or after the
    first test prediction date.

    Call it on the *selected* training rows (``labels[train_mask]``) right before fitting.
    A missing ``window_end`` counts as a violation because the rule cannot be verified.
    """
    end_col = horizon_columns(horizon)[1]
    _require_columns(train_labels, [end_col])
    first_test_prediction = prediction_date(test_start_repdte, lag_days)
    ends = train_labels[end_col]
    leaky = (ends >= first_test_prediction) | ends.isna()
    n_leaky = int(leaky.sum())
    if n_leaky:
        latest = ends.max()
        raise ValueError(
            f"Rule 6.2 violated for horizon {horizon}q: {n_leaky} of {len(train_labels)} "
            f"training rows have {end_col} >= first test prediction date "
            f"{first_test_prediction.date()} (latest window_end {latest})"
        )


def fixed_split_masks(
    labels: pd.DataFrame, settings: Settings, horizon: int
) -> tuple[pd.Series, pd.Series]:
    """Train/test masks for the fixed out-of-time split in ``settings.fixed_split``.

    ``train_start``/``train_end`` in settings are *nominal* report-date bounds. Rule 6.2
    trims the training set further: a row is kept only if its ``window_end_Hq`` falls
    before the prediction date of ``test_start``. With the default 60-day lag and
    ``test_start = 2010-03-31``, the last usable training quarter is 2008Q4 for H=4 but
    only 2007Q4 for H=8, which is earlier than the nominal ``train_end`` of 2008-12-31.
    The effective cut is computed from the data and logged.
    """
    split = settings.fixed_split
    lag = settings.availability_lag_days
    train_start = pd.Timestamp(split.train_start)
    train_end = pd.Timestamp(split.train_end)
    in_train_period = (labels["repdte"] >= train_start) & (labels["repdte"] <= train_end)
    train = training_mask(labels, horizon, split.test_start, lag) & in_train_period
    test = test_mask(labels, horizon, split.test_start, split.test_end)
    last_train = labels.loc[train, "repdte"].max()
    log.info(
        "fixed split %dq: train %s..%s (nominal end %s, %d rows), test %s..%s (%d rows)",
        horizon,
        _fmt(labels.loc[train, "repdte"].min()),
        _fmt(last_train),
        train_end.date(),
        int(train.sum()),
        split.test_start,
        split.test_end,
        int(test.sum()),
    )
    if pd.notna(last_train) and last_train < train_end:
        log.info("fixed split %dq: Rule 6.2 trims training to %s", horizon, _fmt(last_train))
    return train, test


def _fmt(ts: pd.Timestamp | float) -> str:
    return "n/a" if pd.isna(ts) else str(pd.Timestamp(ts).date())


def walk_forward_folds(
    labels: pd.DataFrame,
    horizon: int,
    first_test_year: int,
    last_test_year: int,
    lag_days: int = DEFAULT_LAG_DAYS,
) -> Iterator[tuple[int, pd.Series, pd.Series]]:
    """Yield ``(year, train_mask, test_mask)`` for the walk-forward backtest (spec 8.2).

    For each test year ``Y`` the test rows are the usable reports dated in ``Y`` (report
    quarters 03-31 .. 12-31) and the training rows are every earlier report whose outcome
    window had closed before the prediction date of ``Y``'s first quarter, via
    :func:`training_mask`. Folds are therefore nested (each year's training set contains
    the previous year's) while test sets never overlap. Years with no usable test row
    (e.g. beyond the last complete label year) are logged and skipped; a year with
    fewer than four usable quarters is yielded with a warning.
    """
    if last_test_year < first_test_year:
        raise ValueError(f"last_test_year {last_test_year} < first_test_year {first_test_year}")
    for year in range(int(first_test_year), int(last_test_year) + 1):
        test_start = pd.Timestamp(year=year, month=3, day=31)
        test_end = pd.Timestamp(year=year, month=12, day=31)
        test = test_mask(labels, horizon, test_start, test_end).rename(f"test_{horizon}q_{year}")
        if not test.any():
            log.warning("walk-forward %dq: no usable test rows in %d, fold skipped", horizon, year)
            continue
        n_quarters = labels.loc[test, "repdte"].nunique()
        if n_quarters < 4:
            log.warning(
                "walk-forward %dq: %d has only %d usable test quarter(s); pass an earlier "
                "last_test_year for a complete-year backtest",
                horizon,
                year,
                n_quarters,
            )
        train = training_mask(labels, horizon, test_start, lag_days).rename(
            f"train_{horizon}q_{year}"
        )
        log.info(
            "walk-forward %dq: test year %d, %d train rows (last report %s), %d test rows",
            horizon,
            year,
            int(train.sum()),
            _fmt(labels.loc[train, "repdte"].max()),
            int(test.sum()),
        )
        yield year, train, test


# The name starts with ``test_`` so pytest would otherwise collect it from test modules.
test_mask.__test__ = False  # type: ignore[attr-defined]
