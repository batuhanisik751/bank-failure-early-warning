"""Adversarial checks of the spec section 5 label rules on synthetic data (no network).

Every expected value below is derived by hand from the spec text (window
``(avail(q), avail(q) + 3H months]``, drop when ``fail_date <= avail(q)``, censor on a
non-failure exit inside the window, incomplete when the window ends after the last known
date), never from the implementation's current behaviour.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from bankcanary.config import load_settings
from bankcanary.labels.build import build_labels, failures_as_of_date, horizon_columns
from bankcanary.panel.build import assemble_panel
from bankcanary.panel.exits import build_exits

AS_OF = dt.date(2026, 9, 25)
DAY = pd.Timedelta(days=1)
LAG = pd.Timedelta(days=60)


def _dates(*values):
    return pd.to_datetime(list(values)).astype("datetime64[ns]")


def _panel(rows: list[dict]) -> pd.DataFrame:
    """Rows of ``{cert, repdte, fail?, exit?}``; avail_date = repdte + 60 days."""
    frame = pd.DataFrame(
        {
            "cert": [r["cert"] for r in rows],
            "repdte": _dates(*[r["repdte"] for r in rows]),
            "fail_date": _dates(*[r.get("fail") for r in rows]),
            "exit_date": _dates(*[r.get("exit") for r in rows]),
        }
    )
    frame["avail_date"] = frame["repdte"] + LAG
    return frame


def _labels(rows: list[dict], as_of=AS_OF) -> pd.DataFrame:
    return build_labels(_panel(rows), [4, 8], as_of).set_index("cert")


# Hand-derived from the spec: quarter end + 60 days, then + 12 and + 24 calendar months.
# 12-31 + 60 days is 03-01 in a common year and 02-29 in a leap year; a Feb-29 start
# plus whole months lands on Feb-28 (the day is clamped, not rolled into March).
WINDOWS = [
    ("2009-03-31", "2009-05-30", "2010-05-30", "2011-05-30"),
    ("2009-06-30", "2009-08-29", "2010-08-29", "2011-08-29"),
    ("2009-09-30", "2009-11-29", "2010-11-29", "2011-11-29"),
    ("2009-12-31", "2010-03-01", "2011-03-01", "2012-03-01"),
    ("2011-12-31", "2012-02-29", "2013-02-28", "2014-02-28"),
    ("2012-12-31", "2013-03-01", "2014-03-01", "2015-03-01"),
    ("2019-12-31", "2020-02-29", "2021-02-28", "2022-02-28"),
    ("2023-12-31", "2024-02-29", "2025-02-28", "2026-02-28"),
]


@pytest.mark.parametrize(("repdte", "avail", "end_4q", "end_8q"), WINDOWS)
def test_window_arithmetic_for_every_quarter_end(repdte, avail, end_4q, end_8q):
    lab = _labels([{"cert": 1, "repdte": repdte}])
    assert lab.loc[1, "window_start"] == pd.Timestamp(avail)
    assert lab.loc[1, "window_end_4q"] == pd.Timestamp(end_4q)
    assert lab.loc[1, "window_end_8q"] == pd.Timestamp(end_8q)


@pytest.mark.parametrize(("repdte", "avail", "end_4q", "end_8q"), WINDOWS)
@pytest.mark.parametrize("horizon", [4, 8])
def test_failure_boundaries_for_each_horizon(repdte, avail, end_4q, end_8q, horizon):
    """avail and avail-1 -> dropped; avail+1, end-1, end -> 1; end+1 -> 0 for this H."""
    a = pd.Timestamp(avail)
    end = pd.Timestamp(end_4q if horizon == 4 else end_8q)
    rows = [
        {"cert": 1, "repdte": repdte, "fail": a - DAY},
        {"cert": 2, "repdte": repdte, "fail": a},
        {"cert": 3, "repdte": repdte, "fail": a + DAY},
        {"cert": 4, "repdte": repdte, "fail": end - DAY},
        {"cert": 5, "repdte": repdte, "fail": end},
        {"cert": 6, "repdte": repdte, "fail": end + DAY},
    ]
    lab = _labels(rows)
    y_col, _, cens_col, _ = horizon_columns(horizon)
    assert lab["dropped_failed_before_avail"].tolist() == [True, True, False, False, False, False]
    assert lab[y_col].tolist() == [0, 0, 1, 1, 1, 0]
    assert not lab[cens_col].any()  # a failure is never a censoring exit
    if horizon == 4:
        assert lab.loc[6, "y_8q"] == 1  # one day past the 4q end is still inside 8q


@pytest.mark.parametrize(("repdte", "avail", "end_4q", "end_8q"), WINDOWS)
def test_exit_boundaries(repdte, avail, end_4q, end_8q):
    """The window is open at avail and closed at window_end, for exits as for failures."""
    a, e4, e8 = (pd.Timestamp(x) for x in (avail, end_4q, end_8q))
    rows = [
        {"cert": 1, "repdte": repdte, "exit": a - DAY},  # gone before the report was usable
        {"cert": 2, "repdte": repdte, "exit": a},  # on avail: not inside (avail, end]
        {"cert": 3, "repdte": repdte, "exit": a + DAY},
        {"cert": 4, "repdte": repdte, "exit": e4},
        {"cert": 5, "repdte": repdte, "exit": e4 + DAY},
        {"cert": 6, "repdte": repdte, "exit": e8},
        {"cert": 7, "repdte": repdte, "exit": e8 + DAY},
    ]
    lab = _labels(rows)
    assert lab["censored_in_window_4q"].tolist() == [False, False, True, True, False, False, False]
    assert lab["censored_in_window_8q"].tolist() == [False, False, True, True, True, True, False]
    # A non-failure exit is never a positive and never a rule-2 drop.
    assert not lab["y_4q"].any() and not lab["y_8q"].any()
    assert not lab["dropped_failed_before_avail"].any()


# Report 2024-12-31 -> avail 2025-03-01 -> 4q end 2026-03-01, 8q end 2027-03-01.
RECENT = {"cert": 1, "repdte": "2024-12-31"}


@pytest.mark.parametrize(
    ("as_of", "complete_4q"),
    [
        (dt.date(2026, 3, 1), True),  # window ends exactly on the last known date
        (dt.date(2026, 2, 28), False),  # one day short
        (pd.Timestamp("2026-03-01 09:30"), True),  # a pull time on that day is the same day
        ("2026-03-01", True),
        (dt.date(2027, 3, 1), True),
    ],
)
def test_label_complete_on_and_around_as_of_date(as_of, complete_4q):
    lab = _labels([RECENT], as_of=as_of)
    assert bool(lab.loc[1, "label_complete_4q"]) is complete_4q
    complete_8q = pd.Timestamp(as_of) >= pd.Timestamp("2027-03-01")
    assert bool(lab.loc[1, "label_complete_8q"]) is bool(complete_8q)


def test_incomplete_window_still_records_a_known_failure():
    """Rule 1 sets y from the failure date; rule 4 only marks the row unusable for training."""
    lab = _labels([{**RECENT, "fail": "2026-09-18"}], as_of=dt.date(2026, 9, 25))
    assert lab.loc[1, "label_complete_4q"] and not lab.loc[1, "label_complete_8q"]
    assert lab.loc[1, "y_4q"] == 0 and lab.loc[1, "y_8q"] == 1
    assert lab.loc[1, "window_end_8q"] == pd.Timestamp("2027-03-01")


def _reference(avail: pd.Timestamp, fail, exit_, horizon: int, as_of: dt.date) -> dict:
    """Scalar re-derivation of the rules with dateutil, independent of pandas offsets."""
    end = pd.Timestamp(avail.to_pydatetime() + relativedelta(months=3 * horizon))
    dropped = fail is not pd.NaT and fail <= avail
    y = int(fail is not pd.NaT and avail < fail <= end)
    cens = fail is pd.NaT and exit_ is not pd.NaT and avail < exit_ <= end
    return {"end": end, "dropped": dropped, "y": y, "cens": cens, "complete": end.date() <= as_of}


def _random_panel(seed: int = 7, n: int = 400) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    quarters = pd.date_range("2001-03-31", "2026-06-30", freq="QE")
    repdte = quarters[rng.randint(0, len(quarters), n)]
    fail = repdte + pd.to_timedelta(rng.randint(-100, 900, n), unit="D")
    exit_ = repdte + pd.to_timedelta(rng.randint(-100, 900, n), unit="D")
    kind = rng.randint(0, 3, n)  # 0 open, 1 failed, 2 exited without failing
    frame = pd.DataFrame(
        {
            "cert": np.arange(1, n + 1),
            "repdte": repdte,
            "fail_date": fail.where(kind == 1, pd.NaT),
            "exit_date": exit_.where(kind == 2, pd.NaT),
        }
    )
    frame["avail_date"] = frame["repdte"] + LAG
    return frame.astype({c: "datetime64[ns]" for c in ("repdte", "fail_date", "exit_date")})


def test_random_panel_matches_scalar_reference():
    panel = _random_panel()
    lab = build_labels(panel, [4, 8], AS_OF)
    for i, row in panel.iterrows():
        for h in (4, 8):
            ref = _reference(row["avail_date"], row["fail_date"], row["exit_date"], h, AS_OF)
            y_col, end_col, cens_col, complete_col = horizon_columns(h)
            got = lab.iloc[i]
            assert got[end_col] == ref["end"], (i, h)
            assert (got[y_col], got[cens_col], got[complete_col]) == (
                ref["y"],
                ref["cens"],
                ref["complete"],
            ), (i, h)
            assert got["dropped_failed_before_avail"] == ref["dropped"], (i, h)


def test_random_panel_horizon_invariants():
    lab = build_labels(_random_panel(seed=11), [4, 8], AS_OF)
    # The synthetic panel must exercise every branch, or the invariants prove nothing.
    assert lab["dropped_failed_before_avail"].any() and lab["y_4q"].any()
    assert lab["censored_in_window_4q"].any() and (~lab["label_complete_8q"]).any()
    assert (lab["y_4q"] <= lab["y_8q"]).all()  # the 8q window contains the 4q window
    assert (~lab["censored_in_window_4q"] | lab["censored_in_window_8q"]).all()
    assert (~lab["label_complete_8q"] | lab["label_complete_4q"]).all()
    dropped = lab["dropped_failed_before_avail"]
    assert (lab.loc[dropped, ["y_4q", "y_8q"]] == 0).all().all()
    assert not lab.loc[dropped, ["censored_in_window_4q", "censored_in_window_8q"]].any().any()
    positive = lab["y_8q"] == 1
    assert not lab.loc[positive, ["censored_in_window_4q", "censored_in_window_8q"]].any().any()
    assert (lab["window_end_4q"] > lab["window_start"]).all()
    assert (lab["window_end_8q"] > lab["window_end_4q"]).all()
    assert lab["window_end_4q"].notna().all() and lab["window_end_8q"].notna().all()


def test_dropped_rows_are_kept_with_their_windows():
    """Rule 2 drops are flagged, not removed; rule 5 stores window_end on every row."""
    rows = [
        {"cert": 1, "repdte": "2009-03-31", "fail": "2009-01-16"},  # report after the failure
        {"cert": 1, "repdte": "2008-12-31", "fail": "2009-01-16"},  # last usable report
        {"cert": 1, "repdte": "2008-12-31", "fail": "2009-01-16"},  # duplicate key: kept as is
    ]
    lab = build_labels(_panel(rows), [4, 8], AS_OF)
    assert len(lab) == 3
    assert lab["dropped_failed_before_avail"].tolist() == [True, True, True]
    assert (
        lab["window_end_4q"].tolist()
        == [pd.Timestamp("2010-05-30")] + [pd.Timestamp("2010-03-01")] * 2
    )
    assert (
        lab["window_start"].tolist()
        == [pd.Timestamp("2009-05-30")] + [pd.Timestamp("2009-03-01")] * 2
    )
    assert (lab[["y_4q", "y_8q"]] == 0).all().all()


def test_rows_align_positionally_with_a_shuffled_panel():
    panel = _random_panel(seed=3, n=50)
    shuffled = panel.sample(frac=1, random_state=1)
    shuffled.index = np.arange(1000, 1050)[::-1]  # non-default, decreasing index
    lab = build_labels(shuffled, [4], AS_OF)
    assert lab["cert"].tolist() == shuffled["cert"].tolist()
    expected = build_labels(panel, [4], AS_OF).set_index("cert").loc[shuffled["cert"]]
    assert lab["y_4q"].tolist() == expected["y_4q"].tolist()
    assert lab["window_end_4q"].tolist() == expected["window_end_4q"].tolist()
    col = "dropped_failed_before_avail"
    assert lab[col].tolist() == expected[col].tolist()


def test_input_panel_is_not_mutated():
    panel = _random_panel(seed=5, n=20)
    before = panel.copy(deep=True)
    build_labels(panel, [4, 8], AS_OF)
    pd.testing.assert_frame_equal(panel, before)


# Report 2008-12-31 (avail 2009-03-01, 4q end 2010-03-01) and 2009-03-31 (avail 2009-05-30):
# cert 1 fails 2009-06-01 (y_4q = 1), cert 2 merges 2009-09-30 (censored).
ODD_DTYPES = {
    "object_none": ([None, None], [None, None], [0, 0], [False, False]),
    "float_nan": ([np.nan, np.nan], [np.nan, np.nan], [0, 0], [False, False]),
    "str_dtype": (
        pd.Series(["2009-06-01", pd.NA], dtype="str"),
        pd.Series([pd.NA, "2009-09-30"], dtype="str"),
        [1, 0],
        [False, True],
    ),
    "date_objects": (
        [dt.date(2009, 6, 1), None],
        [None, dt.date(2009, 9, 30)],
        [1, 0],
        [False, True],
    ),
    "datetime64_s": (
        _dates("2009-06-01", None).astype("datetime64[s]"),
        _dates(None, "2009-09-30").astype("datetime64[s]"),
        [1, 0],
        [False, True],
    ),
}


@pytest.mark.parametrize("case", list(ODD_DTYPES))
def test_date_columns_in_other_dtypes_are_coerced(case):
    fail, exit_, y, cens = ODD_DTYPES[case]
    panel = pd.DataFrame(
        {
            "cert": pd.array([1, 2], dtype="Int64"),
            "repdte": ["2008-12-31", "2009-03-31"],
            "avail_date": ["2009-03-01", "2009-05-30"],
            "fail_date": fail,
            "exit_date": exit_,
        }
    )
    lab = build_labels(panel, [4, 8], AS_OF)
    assert lab["y_4q"].tolist() == y and lab["censored_in_window_4q"].tolist() == cens
    assert not lab["dropped_failed_before_avail"].any()
    assert lab["label_complete_8q"].all()
    assert str(lab["window_end_4q"].dtype) == "datetime64[ns]"
    assert str(lab["repdte"].dtype) == "datetime64[ns]"
    assert lab["window_end_4q"].tolist() == [pd.Timestamp("2010-03-01"), pd.Timestamp("2010-05-30")]


def test_empty_panel_gives_empty_labels_with_full_schema():
    empty = _panel([]).iloc[0:0]
    lab = build_labels(empty, [4, 8], AS_OF)
    assert len(lab) == 0
    expected = ["cert", "repdte", "window_start", "dropped_failed_before_avail"]
    assert list(lab.columns) == expected + horizon_columns(4) + horizon_columns(8)
    assert str(lab["y_4q"].dtype) == "int64"
    assert str(lab["censored_in_window_8q"].dtype) == "bool"
    assert str(lab["window_end_8q"].dtype) == "datetime64[ns]"


# ---- exits -> panel -> labels on the spec's edge cases ---------------------------------
# All reports are dated 2008-12-31: avail 2009-03-01, 4q end 2010-03-01, 8q end 2011-03-01.
# cert 10: failed 2009-10-30, no institutions row at all (and a stale report after failing)
# cert 20: two identical FAILURE records for 2010-04-16 (beyond 4q, inside 8q); its
#          institutions row says it ended 2010-04-09, a week before the FDIC failure date
# cert 30: only ASSISTANCE records, one of them (2009-06-15) inside the window; still open
# cert 40: ASSISTANCE 2009-06-15, then FAILURE 2009-12-04
# cert 50: cert reused: a FAILURE in 1985 before estymd 1990, then a FAILURE 2009-08-14
# cert 60: merged on avail day (2009-03-01), history event on that day
# cert 70: reporting bank with no institutions row and no failure: plain negative
# cert 80: FAILURE record whose date is after the 8q window: negative for both horizons


def _failures():
    return pd.DataFrame(
        {
            "cert": pd.array([10, 20, 20, 30, 30, 40, 40, 50, 50, 80], dtype="Int64"),
            "fail_date": _dates(
                "2009-10-30",
                "2010-04-16",
                "2010-04-16",
                "1992-02-01",
                "2009-06-15",
                "2009-06-15",
                "2009-12-04",
                "1985-03-01",
                "2009-08-14",
                "2011-03-02",
            ),  # fmt: skip
            "restype": [
                "FAILURE",
                "FAILURE",
                "FAILURE",
                "ASSISTANCE",
                "ASSISTANCE",
                "ASSISTANCE",
                "FAILURE",
                "FAILURE",
                "FAILURE",
                "FAILURE",
            ],  # fmt: skip
        }
    )


def _institutions():
    certs = [20, 30, 40, 50, 60, 80]
    return pd.DataFrame(
        {
            "cert": pd.array(certs, dtype="Int64"),
            "estymd": _dates(*["1990-01-01"] * 6),
            "active": [False, True, False, False, False, False],
            "endefymd": _dates(
                "2010-04-09", None, "2009-12-04", "2009-08-14", "2009-03-01", "2011-03-02"
            ),
            "bkclass": ["N"] * 6,
            "stalp": ["IL"] * 6,
            "rssdhcr": pd.array([0] * 6, dtype="Int64"),
            "fed_rssd": pd.array(range(6), dtype="Int64"),
            "cb": [True] * 6,
            "latitude": [0.0] * 6,
            "longitude": [0.0] * 6,
        }
    )


def _history():
    return pd.DataFrame(
        {
            "cert": pd.array([60], dtype="Int64"),
            "transnum": pd.array([1], dtype="Int64"),
            "subject_cert": pd.array([60], dtype="Int64"),
            "changecode": pd.array([223], dtype="Int64"),
            "effdate": _dates("2009-03-01"),
        }
    )


def _financials():
    certs = [10, 10, 20, 30, 40, 50, 60, 70, 80]
    repdte = ["2008-12-31", "2009-12-31"] + ["2008-12-31"] * 7
    return pd.DataFrame(
        {
            "cert": pd.array(certs, dtype="Int64"),
            "repdte": _dates(*repdte),
            "asset": [1000.0] * 9,
            "rssdhcr": pd.array([None] * 9, dtype="Int64"),
        }
    )


@pytest.fixture(scope="module")
def pipeline():
    exits = build_exits(_failures(), _institutions(), _history())
    panel = assemble_panel(_financials(), _institutions(), exits, 60)
    labels = build_labels(panel, [4, 8], AS_OF)
    return exits.set_index("cert"), panel, labels


def test_exits_have_one_row_per_cert(pipeline):
    exits, panel, labels = pipeline
    assert exits.index.tolist() == [10, 20, 30, 40, 50, 60, 80]
    assert len(panel) == 9 and len(labels) == 9  # duplicate failure rows never fan out
    assert exits.loc[20, "fail_date"] == pd.Timestamp("2010-04-16")
    assert exits.loc[50, "fail_date"] == pd.Timestamp("2009-08-14")  # not the 1985 record
    assert exits.loc[40, "fail_date"] == pd.Timestamp("2009-12-04")
    assert not exits.loc[40, "assisted"]
    assert bool(exits.loc[30, "assisted"]) and pd.isna(exits.loc[30, "fail_date"])
    assert exits["fail_date"].notna().tolist() == [True, True, False, True, True, False, True]
    # A failed bank exits by failing: no separate non-failure exit, even with an endefymd.
    assert exits.loc[[10, 20, 40, 50, 80], "exit_date"].isna().all()
    assert exits.loc[60, "exit_date"] == pd.Timestamp("2009-03-01")
    assert exits.loc[60, "exit_reason"] == "merger"


def test_failed_cert_without_institution_row_is_labelled(pipeline):
    _, _, labels = pipeline
    rows = labels[labels["cert"] == 10].set_index("repdte")
    first, stale = rows.loc[pd.Timestamp("2008-12-31")], rows.loc[pd.Timestamp("2009-12-31")]
    assert first["y_4q"] == 1 and first["y_8q"] == 1 and not first["dropped_failed_before_avail"]
    assert stale["dropped_failed_before_avail"] and stale["y_4q"] == 0 and stale["y_8q"] == 0


def test_failure_after_recorded_end_date_is_positive_not_censored(pipeline):
    """cert 20 ended 2010-04-09 per institutions but failed 2010-04-16 per failures."""
    _, _, labels = pipeline
    row = labels.set_index("cert").loc[20]
    assert row["y_4q"] == 0 and row["y_8q"] == 1
    assert not row["censored_in_window_4q"] and not row["censored_in_window_8q"]
    assert not row["dropped_failed_before_avail"]


def test_assistance_inside_window_is_a_plain_negative(pipeline):
    _, panel, labels = pipeline
    lab = labels.set_index("cert")
    assert lab.loc[30, "y_4q"] == 0 and lab.loc[30, "y_8q"] == 0
    assert not lab.loc[30, "censored_in_window_8q"]
    assert not lab.loc[30, "dropped_failed_before_avail"]
    assert bool(panel.set_index("cert").loc[30, "assisted"])
    assert lab.loc[40, "y_4q"] == 1  # assistance followed by a real failure counts


def test_reused_cert_with_stale_failure_is_not_dropped(pipeline):
    _, _, labels = pipeline
    row = labels.set_index("cert").loc[50]
    assert not row["dropped_failed_before_avail"] and row["y_4q"] == 1 and row["y_8q"] == 1


def test_merger_on_avail_day_and_failure_after_window(pipeline):
    _, _, labels = pipeline
    lab = labels.set_index("cert")
    for cert in (60, 70, 80):
        assert lab.loc[cert, "y_4q"] == 0 and lab.loc[cert, "y_8q"] == 0
        assert not lab.loc[cert, "censored_in_window_4q"]
        assert not lab.loc[cert, "censored_in_window_8q"]
        assert not lab.loc[cert, "dropped_failed_before_avail"]
    assert lab.loc[80, "window_end_8q"] == pd.Timestamp("2011-03-01")


def test_failures_as_of_date_keeps_the_calendar_day(tmp_path: Path):
    settings = load_settings().model_copy(update={"data_dir": tmp_path / "data"})
    path = settings.data_dir / "raw" / "fdic" / "failures" / "all.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"fetched_at": "2026-09-25T23:30:00-04:00", "data": []}))
    assert failures_as_of_date(settings) == AS_OF
