"""Ingest ``/history``: institution-level structure events (CHANGECODE < 500).

These events say *why* a bank stopped filing: failure (211/213/215/216/217/230/235),
merger or absorption without assistance (221-224), voluntary closing (240), closure
by the chartering agency (260), conservatorship (350/360) and charter or class changes
(420-470). The panel uses them to classify non-failure exits (CONTRACT section 6).
Branch-level events (codes 5xx-8xx) are not ingested (see docs/DECISIONS.md).
"""

from __future__ import annotations

import logging

import pandas as pd

from bankcanary.config import Settings
from bankcanary.ingest import parse_dates, to_frame, to_int, to_str
from bankcanary.sources.fdic import FdicClient

log = logging.getLogger(__name__)

FIELDS = [
    "CERT", "TRANSNUM", "CHANGECODE", "CHANGECODE_DESC", "EFFDATE", "ACQ_CERT", "OUT_CERT",
    "ACQ_UNINUM", "OUT_UNINUM", "ACQ_INSTNAME", "OUT_INSTNAME",
]  # fmt: skip
FILTERS = "CHANGECODE:[1 TO 499]"
KEY = ("cert", "transnum")

#: Coarse exit reasons per change code, used downstream for ``panel.exit_reason``.
EXIT_REASON = {
    211: "failure", 213: "failure", 215: "failure", 216: "failure", 217: "failure",
    230: "failure", 235: "failure",
    221: "absorption", 222: "consolidation", 223: "merger", 224: "merger", 225: "merger",
    240: "voluntary_closing", 260: "other", 350: "other", 360: "other",
    150: "charter_change", 420: "charter_change", 430: "charter_change",
    440: "charter_change", 470: "charter_change", 110: "other",
}  # fmt: skip


def clean_history(rows: list[dict]) -> pd.DataFrame:
    """Map raw API records to the typed ``history`` table (CONTRACT section 5).

    ``EFFDATE`` is ISO (``2021-07-01T00:00:00``). ``(cert, transnum)`` is the contract key
    but is not unique in the FDIC data (about 10k repeated pairs, mostly ``transnum`` 0),
    so only whole-row duplicates are dropped and the sort continues through every column
    to stay reproducible. Downstream code should match events on ``(cert, effdate,
    changecode)`` or, better, on ``subject_cert``.
    """
    raw = to_frame(rows, FIELDS)
    df = pd.DataFrame(
        {
            "cert": to_int(raw["CERT"]),
            "transnum": to_int(raw["TRANSNUM"]),
            "changecode": to_int(raw["CHANGECODE"]),
            "changecode_desc": to_str(raw["CHANGECODE_DESC"]),
            "effdate": parse_dates(raw["EFFDATE"]),
            "acq_cert": to_int(raw["ACQ_CERT"]),
            "out_cert": to_int(raw["OUT_CERT"]),
            "acq_uninum": to_int(raw["ACQ_UNINUM"]),
            "out_uninum": to_int(raw["OUT_UNINUM"]),
            "acq_instname": to_str(raw["ACQ_INSTNAME"]),
            "out_instname": to_str(raw["OUT_INSTNAME"]),
        }
    )
    # Failure and merger records carry the exiting bank in OUT_CERT and often a null
    # CERT (e.g. SVB's 211 event: CERT null, OUT_CERT 24735). ``subject_cert`` is the
    # cert the event happened to and is what the panel should join on.
    df["subject_cert"] = df["out_cert"].fillna(df["cert"]).astype("Int64")
    df = df[df["changecode"] < 500].drop_duplicates()
    order = ["cert", "transnum", "effdate", "changecode", "acq_cert", "out_cert"]
    return df.sort_values(order, kind="mergesort").reset_index(drop=True)


def fetch_history(
    settings: Settings, client: FdicClient | None = None, force: bool = False
) -> pd.DataFrame:
    """Pull institution-level ``/history`` events (cached as ``history/all.json``)."""
    own = client is None
    client = client or FdicClient(settings)
    try:
        rows = client.get(
            "history", filters=FILTERS, fields=FIELDS, sort_by="TRANSNUM", cache_name="all",
            force=force,
        )  # fmt: skip
    finally:
        if own:
            client.close()
    df = clean_history(rows)
    log.info("history: %d rows, %d certs", len(df), df["cert"].nunique())
    return df
