"""FFIEC Central Data Repository bulk download (spec section 3.2).

The CDR publishes every bank's raw Call Report as one ZIP per quarter ("Call Reports --
Single Period"): a tab-delimited file per schedule (RC-B securities, RC-O deposit
insurance, ...) plus a Panel of Reporters (POR) file, all keyed by ``IDRSSD``, the Federal
Reserve's RSSD identifier. The FDIC API already exposes the securities and uninsured-deposit
items this project needs (see ``docs/DECISIONS.md``), so this module exists for the
cross-check in ``reports/ffiec_crosscheck.md`` and for anyone who wants the raw schedules.

The download page is an ASP.NET WebForms form: selecting a product posts back (with the
``__VIEWSTATE`` round trip) and fills the period list, whose option values are opaque ids;
the download itself is a second post that names the period id, the TSV format and the
Download button. Downloads are cached under ``data/raw/ffiec/`` and never repeated.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from bankcanary.config import Settings

log = logging.getLogger(__name__)

BULK_URL = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
PRODUCT_SINGLE_PERIOD = "ReportingSeriesSinglePeriod"
FIELD_PRODUCT = "ctl00$MainContentHolder$ListBox1"
FIELD_PERIOD = "ctl00$MainContentHolder$DatesDropDownList"
FIELD_FORMAT = "ctl00$MainContentHolder$FormatType"
FIELD_DOWNLOAD = "ctl00$MainContentHolder$TabStrip1$Download_0"
CONFIDENTIAL = "CONF"  # the CDR's marker for items withheld from the public file
USER_AGENT = "bankcanary/0.1 (+https://github.com/batuhanisik751)"

_HIDDEN_RE = re.compile(r'<input type="hidden" name="([^"]+)" id="[^"]+" value="([^"]*)"')
_PERIOD_SELECT_RE = re.compile(r"<select[^>]*DatesDropDownList[^>]*>(.*?)</select>", re.S)
_OPTION_RE = re.compile(r'<option[^>]*value="([^"]*)"[^>]*>\s*([^<]*?)\s*</option>')


def hidden_fields(html: str) -> dict[str, str]:
    """Return the WebForms hidden inputs (``__VIEWSTATE`` & co.) of a page."""
    return {name: value for name, value in _HIDDEN_RE.findall(html)}


def period_options(html: str) -> dict[str, str]:
    """Map period labels (``MM/DD/YYYY``) to the option ids of the period drop-down."""
    match = _PERIOD_SELECT_RE.search(html)
    if match is None:
        return {}
    return {label: value for value, label in _OPTION_RE.findall(match.group(1))}


def period_label(period: str | dt.date) -> str:
    """Format a quarter end as the CDR shows it, ``MM/DD/YYYY``."""
    day = pd.Timestamp(period).date()
    return f"{day.month:02d}/{day.day:02d}/{day.year}"


def cache_dir(settings: Settings) -> Path:
    return settings.data_dir / "raw" / "ffiec"


def zip_path(period: str | dt.date, dest: Path) -> Path:
    day = pd.Timestamp(period).date()
    return Path(dest) / f"call_single_period_{day.isoformat()}.zip"


def download_single_period(
    period: str | dt.date,
    dest: Path | None = None,
    client: httpx.Client | None = None,
    *,
    force: bool = False,
    settings: Settings | None = None,
) -> Path:
    """Download the "Call Reports -- Single Period" TSV bundle for one quarter end.

    ``period`` is the quarter end (``2022-12-31`` or a date). The ZIP (50-150 MB for a
    recent quarter) lands in ``dest`` (default ``data/raw/ffiec/``) as
    ``call_single_period_<YYYY-MM-DD>.zip`` and is reused on later calls unless ``force``.
    Three requests are made: GET the form, POST the product selection (a WebForms postback
    that fills the period list) and POST the download. Raises ``ValueError`` when the CDR
    does not list the period and ``RuntimeError`` when the response is not a ZIP.
    """
    if dest is None:
        if settings is None:
            from bankcanary.config import load_settings

            settings = load_settings()
        dest = cache_dir(settings)
    target = zip_path(period, Path(dest))
    if target.exists() and not force:
        log.info("ffiec: using cached %s", target)
        return target
    own = client is None
    client = client or httpx.Client(
        timeout=httpx.Timeout(600.0, connect=30.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        page = client.get(BULK_URL)
        page.raise_for_status()
        form = hidden_fields(page.text)
        form.update({"__EVENTTARGET": FIELD_PRODUCT, "__EVENTARGUMENT": ""})
        form[FIELD_PRODUCT] = PRODUCT_SINGLE_PERIOD
        listed = client.post(BULK_URL, data=form)
        listed.raise_for_status()
        options = period_options(listed.text)
        label = period_label(period)
        if label not in options:
            raise ValueError(f"FFIEC CDR does not list period {label}; known: {sorted(options)}")
        form = hidden_fields(listed.text)
        form.update({"__EVENTTARGET": "", "__EVENTARGUMENT": ""})
        form[FIELD_PRODUCT] = PRODUCT_SINGLE_PERIOD
        form[FIELD_PERIOD] = options[label]
        form[FIELD_FORMAT] = "TSVRadioButton"
        form[FIELD_DOWNLOAD] = "Download"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        with client.stream("POST", BULK_URL, data=form) as resp:
            resp.raise_for_status()
            disposition = resp.headers.get("content-disposition", "")
            if ".zip" not in disposition.lower():
                raise RuntimeError(f"FFIEC CDR returned no ZIP for {label}: {disposition!r}")
            with open(partial, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
        partial.replace(target)
    finally:
        if own:
            client.close()
    log.info("ffiec: downloaded %s (%d bytes)", target, target.stat().st_size)
    return target


def schedule_members(zip_file: zipfile.ZipFile, schedule: str) -> list[str]:
    """Names of the ZIP members holding ``schedule`` (``RC-B`` matches ``Schedule RCB``).

    Large schedules are split into several files (``... RCB 12312022(1 of 2).txt``); the
    members are returned in part order so :func:`parse_schedule` can join them.
    """
    code = schedule.upper().replace("-", "").replace(" ", "")
    pattern = re.compile(rf"Schedule {re.escape(code)} \d{{8}}(?:\((\d+) of \d+\))?\.txt$")
    found = []
    for name in zip_file.namelist():
        match = pattern.search(name)
        if match:
            found.append((int(match.group(1) or 1), name))
    return [name for _, name in sorted(found)]


def parse_schedule(zip_path: Path | str, schedule: str) -> pd.DataFrame:
    """Read one Call Report schedule out of a bulk ZIP, keyed by ``IDRSSD``.

    The TSV has the MDRM codes (``RCON1754``, ...) in its first row and their descriptions
    in the second; the descriptions are dropped and kept in ``df.attrs["descriptions"]``.
    ``IDRSSD`` becomes the integer index; every other column is numeric where it parses
    (blank or ``CONF``, a confidential item, = missing) and left as text otherwise.
    Multi-part schedules are joined on ``IDRSSD``. Dollar amounts are in thousands, as on
    the Call Report.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = schedule_members(zf, schedule)
        if not members:
            raise KeyError(f"schedule {schedule!r} not found in {zip_path}")
        parts = []
        descriptions: dict[str, str] = {}
        for name in members:
            with zf.open(name) as fh:
                raw = pd.read_csv(
                    fh,
                    sep="\t",
                    dtype=str,
                    keep_default_na=False,
                    index_col=False,
                    encoding="latin-1",
                )
            raw = raw.loc[:, [c for c in raw.columns if not c.startswith("Unnamed")]]
            descriptions.update(dict(zip(raw.columns, raw.iloc[0].tolist(), strict=True)))
            body = raw.iloc[1:]
            body = body.set_index(pd.to_numeric(body["IDRSSD"]).astype("int64")).drop(
                columns=["IDRSSD"]
            )
            body.index.name = "IDRSSD"
            parts.append(body)
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.join(part.loc[:, [c for c in part.columns if c not in joined.columns]])
    out = joined.apply(_numeric_or_text)
    out.attrs["descriptions"] = descriptions
    return out.sort_index()


def _numeric_or_text(col: pd.Series) -> pd.Series:
    cleaned = col.str.strip().replace({"": None, CONFIDENTIAL: None})
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().sum() == cleaned.notna().sum():
        return numeric.astype("float64")
    return cleaned
