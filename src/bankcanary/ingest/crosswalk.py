"""The ``crosswalk_rssd`` table and the securities / uninsured-deposit cross-check (step D1).

Two identifier systems name the same bank: the FDIC certificate number (``CERT``, the key of
every table in this project) and the Federal Reserve's ``RSSD`` id (``IDRSSD``), which keys the
raw FFIEC Call Report files and the Fed's structure data. The FDIC ``/institutions`` table
carries both (``FED_RSSD``), so the crosswalk is a projection of it; certs without an RSSD id
are kept with a null so that the count of unmatched banks is visible.

The cross-check answers the question that decides whether raw FFIEC files are needed at all:
do the FDIC API fields ``SCHA``/``SCHF`` (held-to-maturity securities at amortised cost and
fair value), ``SCAA``/``SCAF`` (available-for-sale, same two bases) and ``DEPUNINS``
(estimated uninsured deposits) reproduce the figures the banks published? Silicon Valley
Bank's 2022 10-K is the reference; the same items are read from the FFIEC RC-B and RC-O
schedules when a bulk ZIP is available.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TABLE = "crosswalk_rssd"
KEY = ("cert",)
SOURCE = "institutions"

#: FDIC field -> (label, FFIEC MDRM item, schedule). The MDRM code is looked up as ``RCFD``
#: (consolidated) first and ``RCON`` (domestic offices) otherwise, per spec section 3.2.
ITEMS: dict[str, tuple[str, str, str]] = {
    "scha": ("HTM securities, amortised cost", "1754", "RC-B"),
    "schf": ("HTM securities, fair value", "1771", "RC-B"),
    "scaa": ("AFS securities, amortised cost", "1772", "RC-B"),
    "scaf": ("AFS securities, fair value", "1773", "RC-B"),
    "depunins": ("Estimated uninsured deposits", "5597", "RC-O"),
}
#: Uninsured deposits are a domestic-office item (Schedule RC-O has no RCFD version).
DOMESTIC_ONLY = {"5597"}

CROSSCHECK_BANKS: dict[int, str] = {
    24735: "Silicon Valley Bank",
    57053: "Signature Bank",
    59017: "First Republic Bank",
}
CROSSCHECK_DATES = ("2022-12-31", "2023-03-31")

#: Silicon Valley Bank, 2022-12-31, thousands of dollars, as in its 2022 Form 10-K.
PUBLISHED_SVB_2022Q4: dict[str, float] = {
    "scha": 91_327_000,
    "schf": 76_168_000,
    "scaa": 28_502_000,
    "scaf": 25_976_000,
    "depunins": 151_592_000,
}


def build_crosswalk(institutions: pd.DataFrame) -> pd.DataFrame:
    """Project ``institutions`` onto ``(cert, fed_rssd, name, source)``.

    ``fed_rssd`` of 0 means "no RSSD id assigned" in the FDIC data and becomes null, so
    that :func:`missing_rssd_count` and any join on the id treat it as absent.
    """
    out = pd.DataFrame(
        {
            "cert": institutions["cert"].astype("Int64"),
            "fed_rssd": institutions["fed_rssd"].astype("Int64"),
            "name": institutions["name"].astype("str"),
            "source": SOURCE,
        }
    )
    out.loc[out["fed_rssd"] == 0, "fed_rssd"] = pd.NA
    out = out.drop_duplicates(subset=list(KEY), keep="first")
    return out.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def missing_rssd_count(crosswalk: pd.DataFrame) -> int:
    """Number of certificates that have no RSSD id (cannot be matched to FFIEC files)."""
    return int(crosswalk["fed_rssd"].isna().sum())


def securities_crosscheck(
    financials: pd.DataFrame,
    certs: Iterable[int] | None = None,
    repdtes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per (bank, quarter, item) with the FDIC value of each cross-check field.

    ``financials`` is ``financials_raw`` (or any frame with ``cert, repdte, name`` and the
    five FDIC columns). Columns: ``cert, name, repdte, item, fdic_field, mdrm, fdic_value``.
    Quarters a bank never filed (Silicon Valley Bank and Signature Bank after March 2023)
    are simply absent.
    """
    certs = list(certs if certs is not None else CROSSCHECK_BANKS)
    dates = pd.to_datetime(list(repdtes if repdtes is not None else CROSSCHECK_DATES))
    rows = financials.loc[
        financials["cert"].isin(certs) & pd.to_datetime(financials["repdte"]).isin(dates)
    ]
    records = []
    for _, row in rows.sort_values(["cert", "repdte"]).iterrows():
        for field, (label, mdrm, _schedule) in ITEMS.items():
            records.append(
                {
                    "cert": int(row["cert"]),
                    "name": row["name"],
                    "repdte": pd.Timestamp(row["repdte"]),
                    "item": label,
                    "fdic_field": field,
                    "mdrm": mdrm,
                    "fdic_value": float(row[field]) if pd.notna(row[field]) else float("nan"),
                }
            )
    return pd.DataFrame(records)


def check_published(table: pd.DataFrame, cert: int = 24735, repdte: str = "2022-12-31") -> None:
    """Raise ``AssertionError`` unless the FDIC values equal the bank's published figures."""
    expected = PUBLISHED_SVB_2022Q4
    sel = table.loc[(table["cert"] == cert) & (table["repdte"] == pd.Timestamp(repdte))]
    if sel.empty:
        raise AssertionError(f"no cross-check rows for cert {cert} at {repdte}")
    got = dict(zip(sel["fdic_field"], sel["fdic_value"], strict=True))
    bad = {f: (got.get(f), v) for f, v in expected.items() if got.get(f) != v}
    if bad:
        raise AssertionError(f"FDIC fields differ from published figures for {cert}: {bad}")


def add_ffiec_values(
    table: pd.DataFrame, zip_path: Path | str, crosswalk: pd.DataFrame
) -> pd.DataFrame:
    """Add ``ffiec_value`` (and ``ffiec_code``) read from the bulk ZIP's RC-B / RC-O files.

    Only rows whose ``repdte`` matches the ZIP's period get a value; for each item the
    ``RCFD`` (consolidated) column is used when the bank reports one, else ``RCON``.
    """
    from bankcanary.sources import ffiec

    period = pd.Timestamp(Path(zip_path).stem.rsplit("_", 1)[-1])
    schedules = {s: ffiec.parse_schedule(zip_path, s) for s in {v[2] for v in ITEMS.values()}}
    rssd_of = dict(zip(crosswalk["cert"], crosswalk["fed_rssd"], strict=True))
    values, codes = [], []
    for _, row in table.iterrows():
        rssd = rssd_of.get(row["cert"])
        _label, mdrm, schedule = ITEMS[row["fdic_field"]]
        frame = schedules[schedule]
        value, code = float("nan"), None
        if row["repdte"] == period and pd.notna(rssd) and int(rssd) in frame.index:
            prefixes = ("RCON",) if mdrm in DOMESTIC_ONLY else ("RCFD", "RCON")
            for prefix in prefixes:
                col = f"{prefix}{mdrm}"
                if col in frame.columns and pd.notna(frame.at[int(rssd), col]):
                    value, code = float(frame.at[int(rssd), col]), col
                    break
        values.append(value)
        codes.append(code)
    return table.assign(ffiec_code=codes, ffiec_value=values)


REPORT_INTRO = """\
# FFIEC cross-check: securities and uninsured deposits

Amounts in thousands of dollars. Silicon Valley Bank's 2022-12-31 row is checked in code
(`bankcanary.ingest.crosswalk.check_published`) against its 2022 Form 10-K: HTM securities
$91,327,000 amortised cost vs $76,168,000 fair value, AFS $28,502,000 vs $25,976,000,
uninsured deposits $151,592,000. Signature Bank and First Republic Bank are listed for every
quarter they filed; SVB and Signature closed before 2023-03-31 and have no Q1 2023 report.
"""

REPORT_RATIONALE = """\
## Why the FDIC fields are used directly

Spec section 3.2 says to prefer the FDIC API whenever it already exposes an item. The table
shows that `SCHA`/`SCHF`, `SCAA`/`SCAF` and `DEPUNINS` are the Call Report items RC-B
1754/1771/1772/1773 and RC-O 5597 to the dollar, because the FDIC builds its `/financials`
endpoint from the same filings. Using them keeps one client, one cache and one identifier
(`CERT`) for every feature, avoids the 50-150 MB per quarter of the FFIEC bulk files, and
reaches back to 2001 with the same field names. The raw FFIEC route stays available
(`bankcanary.sources.ffiec`) for items the API does not carry.
"""


def render_report(table: pd.DataFrame, n_certs: int, n_missing: int, zip_name: str | None) -> str:
    """Markdown for ``reports/ffiec_crosscheck.md``: the cross-check table and the rationale."""

    def money(value: float) -> str:
        return "" if pd.isna(value) else f"{value:,.0f}"

    has_ffiec = "ffiec_value" in table.columns
    cols = ["Bank (cert)", "Quarter", "Item", "FDIC field", "FDIC value"]
    align = ["---", "---", "---", "---", "---:"]
    if has_ffiec:
        cols += ["FFIEC item", "FFIEC value", "Match"]
        align += ["---", "---:", "---"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"]
    for _, r in table.iterrows():
        cells = [
            f"{r['name'].title()} ({r['cert']})",
            r["repdte"].strftime("%Y-%m-%d"),
            r["item"],
            f"`{r['fdic_field']}`",
            money(r["fdic_value"]),
        ]
        if has_ffiec:
            same = pd.notna(r["ffiec_value"]) and r["ffiec_value"] == r["fdic_value"]
            cells += [
                f"`{r['ffiec_code']}`" if pd.notna(r["ffiec_code"]) else "-",
                money(r["ffiec_value"]),
                "yes" if same else ("" if pd.isna(r["ffiec_value"]) else "NO"),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    if zip_name:
        note = (
            f"FFIEC values come from `{zip_name}` (Schedules RC-B and RC-O, keyed by `IDRSSD`"
            " through `crosswalk_rssd`); `RCFD` (consolidated) is taken when the bank reports"
            " it, else `RCON`."
        )
    else:
        note = "No FFIEC bulk ZIP was available, so only the FDIC values are shown."
    tally = (
        f"Crosswalk: {n_certs:,} certificates in `institutions`, {n_missing:,} without an RSSD id."
    )
    return "\n".join([REPORT_INTRO, *lines, "", note, "", REPORT_RATIONALE, tally, ""])
