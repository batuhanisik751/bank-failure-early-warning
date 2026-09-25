"""Per-bank exit facts: failure date, assistance flag, non-failure exit date and reason.

A bank leaves the panel in one of three ways. It *fails* (a ``FAILURE`` row in the FDIC
``failures`` table, the model's positive event), it is *assisted* (an open-bank
``ASSISTANCE`` transaction: the bank survived with FDIC help, so it is not a failure), or
it *exits without failing* (merged, absorbed, closed voluntarily, changed charter). The
non-failure exits matter for labels because a merged bank cannot fail afterwards, so its
rows are censored rather than negative.

The exit reason comes from the institution-level ``history`` event closest to the
bank's ``endefymd`` (within :data:`EXIT_WINDOW_DAYS`), matched on ``subject_cert``: the
FDIC records failure events with a null ``CERT`` and the failed bank in ``OUT_CERT``.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

#: A history event counts as explaining an exit when it falls this close to ``endefymd``.
EXIT_WINDOW_DAYS = 7

#: History change codes for failure resolutions (CONTRACT section 5 keeps codes < 500).
FAILURE_CODES = frozenset({211, 213, 215, 216, 217, 230, 235, 260})

#: Change code -> exit reason for non-failure exits.
REASON_BY_CODE: dict[int, str] = {
    223: "merger",
    221: "absorption",
    222: "consolidation",
    224: "affiliated_merger",
    240: "voluntary_closing",
    420: "charter_change",
    430: "charter_change",
    440: "charter_change",
}

#: When several events fall in the window, prefer the closest; on a tie, the most
#: "exit-like" one (a merger beats a same-day change of regulator).
_PRIORITY = {
    "failure_unmatched": 0,
    "merger": 1,
    "absorption": 1,
    "consolidation": 1,
    "affiliated_merger": 1,
    "voluntary_closing": 2,
    "charter_change": 3,
    "other": 4,
}

EXIT_COLUMNS = ["cert", "fail_date", "assisted", "exit_date", "exit_reason"]


def derive_fail_dates(failures: pd.DataFrame, institutions: pd.DataFrame) -> pd.DataFrame:
    """One row per cert in ``failures``: ``fail_date`` and ``assisted``.

    ``fail_date`` is the earliest ``FAILURE`` on or after the bank's ``estymd`` (a cert
    can be reused, so an older failure under the same number belongs to a predecessor);
    when ``estymd`` is unknown the earliest failure of any date is taken. ``assisted`` is
    True only for certs whose every record is open-bank ``ASSISTANCE`` (they never failed).
    Rows with a null cert (pre-1966 failures) cannot be joined and are ignored.
    """
    f = failures.loc[failures["cert"].notna(), ["cert", "fail_date", "restype"]]
    f = f.astype({"cert": "Int64"})
    est = institutions.loc[institutions["cert"].notna(), ["cert", "estymd"]]
    est = est.astype({"cert": "Int64"}).drop_duplicates("cert")
    f = f.merge(est, on="cert", how="left")
    fails = f[f["restype"] == "FAILURE"]
    eligible = fails[fails["estymd"].isna() | (fails["fail_date"] >= fails["estymd"])]
    first = eligible.groupby("cert")["fail_date"].min()
    # A cert with FAILURE rows that all pre-date its estymd falls back to the earliest one.
    fallback = fails.groupby("cert")["fail_date"].min()
    fail_date = first.reindex(fallback.index).fillna(fallback)
    certs = pd.Index(f["cert"].unique(), name="cert")
    out = pd.DataFrame(
        {
            "fail_date": fail_date.reindex(certs).astype("datetime64[ns]"),
            "assisted": ~certs.isin(fail_date.index),
        },
        index=certs,
    ).reset_index()
    return out.astype({"cert": "Int64", "assisted": bool})


def classify_event(code: int, has_failure_record: bool) -> str:
    """Map one history change code to an exit reason (see the module docstring)."""
    if code in FAILURE_CODES:
        return "other" if has_failure_record else "failure_unmatched"
    return REASON_BY_CODE.get(int(code), "other")


def derive_exit_reasons(
    exits: pd.DataFrame, history: pd.DataFrame, window_days: int = EXIT_WINDOW_DAYS
) -> pd.Series:
    """Exit reason per row of ``exits`` (``cert``, ``exit_date``, ``has_failure_record``).

    Takes the institution-level history event on the bank (``subject_cert``) within
    ``window_days`` of ``exit_date``; the closest wins, ties go to the most exit-like
    code. Banks with no event in the window get ``unknown``; banks without an
    ``exit_date`` get NA. Returned as a ``str`` Series aligned to ``exits.index``.
    """
    dated = exits.loc[exits["exit_date"].notna(), ["cert", "exit_date", "has_failure_record"]]
    h = history.loc[
        history["subject_cert"].notna(), ["subject_cert", "effdate", "changecode", "transnum"]
    ]
    h = h.rename(columns={"subject_cert": "cert"}).astype({"cert": "Int64"})
    cand = dated.reset_index(names="_row").merge(h, on="cert", how="inner")
    cand["gap"] = (cand["effdate"] - cand["exit_date"]).dt.days.abs()
    cand = cand[cand["gap"] <= window_days]
    cand["reason"] = [
        classify_event(int(c), bool(r))
        for c, r in zip(cand["changecode"], cand["has_failure_record"], strict=True)
    ]
    cand["rank"] = cand["reason"].map(_PRIORITY).fillna(max(_PRIORITY.values()))
    cand = cand.sort_values(["_row", "gap", "rank", "transnum"], kind="mergesort")
    best = cand.drop_duplicates("_row").set_index("_row")["reason"]
    out = pd.Series(pd.NA, index=exits.index, dtype="str")
    out.loc[dated.index] = "unknown"
    out.loc[best.index] = best.astype("str")
    unmatched = int((out == "failure_unmatched").sum())
    if unmatched:
        log.info(
            "%d banks exit on a failure-type history event but have no failures record", unmatched
        )
    return out


def build_exits(
    failures: pd.DataFrame, institutions: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """One row per cert with ``fail_date, assisted, exit_date, exit_reason``.

    Covers every cert in ``institutions`` plus any failed cert missing from it. A bank
    that failed has ``fail_date`` set and no ``exit_date`` (failure is its exit). An
    inactive bank that did not fail exits on ``institutions.endefymd`` with a reason from
    :func:`derive_exit_reasons`; an open bank (or a failed bank) has both exit fields NA.
    A bank flagged inactive with no ``endefymd`` is logged and left with no exit date.
    """
    inst = institutions.loc[institutions["cert"].notna(), ["cert", "active", "endefymd"]]
    inst = inst.astype({"cert": "Int64"}).drop_duplicates("cert")
    fd = derive_fail_dates(failures, institutions)
    out = inst.merge(fd, on="cert", how="outer")
    out["assisted"] = out["assisted"].fillna(False).astype(bool)
    out["has_failure_record"] = out["cert"].isin(fd["cert"])
    failed = out["fail_date"].notna()
    inactive = out["active"].fillna(True).astype(bool).eq(False)
    exit_mask = inactive & ~failed
    no_date = exit_mask & out["endefymd"].isna()
    if no_date.any():
        log.warning("%d inactive banks have no endefymd; exit_date left null", int(no_date.sum()))
    out["exit_date"] = out["endefymd"].where(exit_mask).astype("datetime64[ns]")
    out["exit_reason"] = derive_exit_reasons(out, history)
    out = out.loc[:, EXIT_COLUMNS].sort_values("cert", kind="mergesort").reset_index(drop=True)
    log.info(
        "exits: %d banks, %d failed, %d assisted-only, %d non-failure exits; reasons: %s",
        len(out),
        int(out["fail_date"].notna().sum()),
        int(out["assisted"].sum()),
        int(out["exit_date"].notna().sum()),
        out["exit_reason"].value_counts().to_dict(),
    )
    return out
