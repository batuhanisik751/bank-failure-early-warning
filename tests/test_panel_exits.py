"""Exit logic on tiny synthetic frames: every branch of fail_date / exit_date / exit_reason."""

from __future__ import annotations

import logging

import pandas as pd

from bankcanary.panel.exits import build_exits, classify_event, derive_fail_dates


def _dates(*values):
    return pd.to_datetime(list(values)).astype("datetime64[ns]")


def failures_frame():
    # cert 1: failed twice (old cert reuse, first failure before estymd); cert 2: assisted only;
    # cert 3: assistance then a real failure; cert 4: failure before estymd only; null cert row.
    return pd.DataFrame(
        {
            "cert": pd.array([1, 1, 2, 3, 3, 4, None], dtype="Int64"),
            "fail_date": _dates(
                "1985-03-01",
                "2009-05-01",
                "1991-01-01",
                "1988-01-01",
                "2010-08-20",
                "1970-01-01",
                "1940-01-01",
            ),  # fmt: skip
            "restype": [
                "FAILURE",
                "FAILURE",
                "ASSISTANCE",
                "ASSISTANCE",
                "FAILURE",
                "FAILURE",
                "FAILURE",
            ],  # fmt: skip
        }
    )


def institutions_frame():
    # 5 merged, 6 open, 7 inactive with no event, 8 inactive with no endefymd, 9 unmatched
    # failure code, 10 voluntary closing with a same-day regulator change, 11 no estymd.
    certs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    return pd.DataFrame(
        {
            "cert": pd.array(certs, dtype="Int64"),
            "estymd": _dates(
                "2000-01-01",
                "1980-01-01",
                "1980-01-01",
                "1990-01-01",
                "1990-01-01",
                "1990-01-01",
                "1990-01-01",
                "1990-01-01",
                "1990-01-01",
                "1990-01-01",
                None,
            ),  # fmt: skip
            "endefymd": _dates(
                "2009-05-01",
                None,
                "2010-08-20",
                "1970-01-01",
                "2012-06-30",
                None,
                "2013-01-01",
                None,
                "1995-04-01",
                "2014-09-30",
                "2015-01-01",
            ),  # fmt: skip
            "active": [False, True, False, False, False, True, False, False, False, False, False],
        }
    )


def history_frame():
    rows = [
        # cert, transnum, subject_cert, effdate, changecode
        (5, 1, 5, "2012-07-03", 223),  # merger three days after endefymd -> in window
        (9, 1, 9, "1995-04-01", 230),  # failure payoff, but no failures record
        (10, 1, 10, "2014-09-30", 470),  # same-day regulator change loses to ...
        (10, 2, 10, "2014-09-30", 240),  # ... the voluntary closing
        (11, 1, 11, "2015-01-01", 420),  # charter change
        (7, 1, 7, "2013-02-01", 223),  # 31 days out: outside the window
        (None, 1, 3, "2010-08-20", 211),  # failure event with null cert (out_cert form)
    ]
    df = pd.DataFrame(rows, columns=["cert", "transnum", "subject_cert", "effdate", "changecode"])
    return df.astype(
        {"cert": "Int64", "transnum": "Int64", "subject_cert": "Int64", "changecode": "Int64"}
    ).assign(effdate=lambda d: _dates(*d["effdate"]))


def test_fail_date_is_earliest_failure_on_or_after_estymd():
    fd = derive_fail_dates(failures_frame(), institutions_frame()).set_index("cert")
    assert fd.loc[1, "fail_date"] == pd.Timestamp("2009-05-01")
    assert fd.loc[3, "fail_date"] == pd.Timestamp("2010-08-20") and not fd.loc[3, "assisted"]
    assert bool(fd.loc[2, "assisted"]) and pd.isna(fd.loc[2, "fail_date"])
    assert fd.loc[4, "fail_date"] == pd.Timestamp("1970-01-01")  # only failure pre-dates estymd
    assert fd.index.tolist() == [1, 2, 3, 4]  # the null-cert row is ignored


def test_classify_event_branches():
    assert classify_event(223, False) == "merger"
    assert classify_event(224, False) == "affiliated_merger"
    assert classify_event(240, False) == "voluntary_closing"
    assert classify_event(430, False) == "charter_change"
    assert classify_event(211, False) == "failure_unmatched"
    assert classify_event(211, True) == "other"
    assert classify_event(470, False) == "other"


def test_build_exits_covers_every_branch(caplog):
    caplog.set_level(logging.INFO, logger="bankcanary.panel.exits")
    ex = build_exits(failures_frame(), institutions_frame(), history_frame()).set_index("cert")
    assert ex.index.tolist() == list(range(1, 12))
    # failed banks: fail_date set, no exit_date / exit_reason
    assert ex.loc[1, "fail_date"] == pd.Timestamp("2009-05-01") and pd.isna(ex.loc[1, "exit_date"])
    assert pd.isna(ex.loc[3, "exit_reason"])
    # assisted-only bank is open: nothing set except the flag
    assert bool(ex.loc[2, "assisted"]) and pd.isna(ex.loc[2, "exit_date"])
    assert not ex.loc[[1, 3, 5, 6], "assisted"].any()
    # open bank
    assert pd.isna(ex.loc[6, "exit_date"]) and pd.isna(ex.loc[6, "exit_reason"])
    reasons = ex["exit_reason"]
    assert reasons[5] == "merger"
    assert reasons[7] == "unknown"  # event outside the 7-day window
    assert pd.isna(ex.loc[8, "exit_date"]) and pd.isna(reasons[8])  # inactive, no endefymd
    assert reasons[9] == "failure_unmatched"
    assert reasons[10] == "voluntary_closing"  # beats the same-day 470
    assert reasons[11] == "charter_change"
    assert ex.loc[5, "exit_date"] == pd.Timestamp("2012-06-30")
    assert str(ex["fail_date"].dtype) == "datetime64[ns]" and ex["assisted"].dtype == bool
    assert "1 banks exit on a failure-type" in caplog.text
