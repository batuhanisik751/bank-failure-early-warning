"""Client for FRED, the St. Louis Fed's economic data service.

FRED carries the macro context the model needs beside each bank's own balance sheet:
state unemployment (``<ST>UR``), the FHFA state house price index (``<ST>STHPI``), the
federal funds rate (``FEDFUNDS``) and Treasury yields (``DGS10``, ``T10Y3M``). Two
transports serve the same observations:

* the JSON API (``fred.api_url``) when ``Secrets.fred_api_key`` is set, and
* the keyless CSV export (``fred.csv_url``, header ``observation_date,<ID>``) otherwise.

Both mark a missing observation with ``"."``; both are cached as one JSON file per
series under ``data/raw/fred/<ID>.json`` so a rebuild never re-downloads. Only the
current vintage is available through these endpoints: revisions (ALFRED) are not
modelled, which is a documented limitation of the point-in-time macro table.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import tenacity

from bankcanary.config import Secrets, Settings
from bankcanary.sources.fdic import TokenBucket, write_cache

log = logging.getLogger(__name__)

DEFAULT_START = dt.date(1999, 1, 1)


def cache_root(settings: Settings) -> Path:
    return settings.data_dir / "raw" / "fred"


def cache_path(settings: Settings, series_id: str) -> Path:
    """``data/raw/fred/<ID>.json``: ``{"series_id", "fetched_at", "source", "start", "data"}``."""
    return cache_root(settings) / f"{series_id}.json"


class FredApiError(RuntimeError):
    """An HTTP status worth retrying (429 or 5xx) or the one that ended the retries."""

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"FRED returned HTTP {status_code} for {url}")
        self.status_code = status_code
        self.url = url


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, FredApiError) and (exc.status_code == 429 or exc.status_code >= 500)


def _parse_value(raw: str | None) -> float | None:
    """FRED writes ``"."`` for a missing observation; everything else is a decimal."""
    text = (raw or "").strip()
    if text in ("", "."):
        return None
    return float(text)


def parse_api_json(body: dict[str, Any]) -> list[tuple[str, float | None]]:
    """``observations: [{"date": ..., "value": ...}]`` from the JSON API."""
    return [(o["date"], _parse_value(o.get("value"))) for o in body.get("observations") or []]


def parse_csv(text: str, series_id: str) -> list[tuple[str, float | None]]:
    """``observation_date,<ID>`` rows from ``fredgraph.csv``; the ID column is checked."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or header[0].strip() != "observation_date":
        raise ValueError(f"FRED CSV for {series_id}: unexpected header {header!r}")
    if len(header) < 2 or header[1].strip() != series_id:
        log.warning("FRED CSV for %s: value column is named %r", series_id, header[1:])
    return [(row[0], _parse_value(row[1] if len(row) > 1 else None)) for row in reader if row]


def to_frame(data: list[tuple[str, float | None]] | list[list[Any]]) -> pd.DataFrame:
    """``DataFrame(date, value)`` with ``datetime64[ns]`` dates and float values (NaN = missing)."""
    frame = pd.DataFrame(list(data), columns=["date", "value"])
    frame = frame.assign(
        date=pd.to_datetime(frame["date"]).astype("datetime64[ns]"),
        value=pd.to_numeric(frame["value"], errors="coerce").astype("float64"),
    )
    return frame.sort_values("date", kind="mergesort").reset_index(drop=True)


class FredClient:
    """Cached, rate-limited FRED client with the same shape as :class:`FdicClient`.

    ``transport`` lets tests inject :class:`httpx.MockTransport`; ``wait`` overrides the
    retry back-off (tests pass ``tenacity.wait_none()``).
    """

    def __init__(
        self,
        settings: Settings,
        secrets: Secrets | None = None,
        transport: httpx.BaseTransport | None = None,
        wait: tenacity.wait.wait_base | None = None,
    ) -> None:
        self.settings = settings
        self.fred = settings.fred
        self.api_key = secrets.fred_api_key if secrets is not None else None
        self.bucket = TokenBucket(self.fred.requests_per_second)
        self.wait = (
            wait if wait is not None else tenacity.wait_exponential_jitter(initial=1, max=60)
        )
        self._client = httpx.Client(timeout=self.fred.timeout_seconds, transport=transport)
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FredClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def source(self) -> str:
        return "api" if self.api_key else "csv"

    def _request(self, url: str, params: dict[str, Any]) -> httpx.Response:
        """One GET with rate limiting and retries on 429, 5xx and transport errors."""
        retryer = tenacity.Retrying(
            retry=tenacity.retry_if_exception(_is_retryable),
            stop=tenacity.stop_after_attempt(max(int(self.fred.max_retries), 0) + 1),
            wait=self.wait,
            reraise=True,
            before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
        )
        for attempt in retryer:
            with attempt:
                self.bucket.acquire()
                self.request_count += 1
                response = self._client.get(url, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    raise FredApiError(response.status_code, str(response.url))
                response.raise_for_status()
        return response

    def fetch(self, series_id: str, start: dt.date) -> list[tuple[str, float | None]]:
        """Download one series (no cache) through the API or the CSV export."""
        if self.api_key:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start.isoformat(),
            }
            return parse_api_json(self._request(self.fred.api_url, params).json())
        params = {"id": series_id, "cosd": start.isoformat()}
        return parse_csv(self._request(self.fred.csv_url, params).text, series_id)

    def series(
        self, series_id: str, start: dt.date | None = None, force: bool = False
    ) -> pd.DataFrame:
        """Return ``DataFrame(date, value)`` for a series, from the cache when it covers ``start``.

        A cached file is reused unless ``force=True`` or it was pulled with a later start
        than the one requested. Dates are FRED period starts (the first day of the month
        or quarter for monthly/quarterly series); ``ingest.macro`` turns them into period
        ends before applying publication lags.
        """
        start = start or DEFAULT_START
        path = cache_path(self.settings, series_id)
        cached = read_cached(path)
        if cached is not None and not force and cached.get("start", "9999") <= start.isoformat():
            log.info("FRED %s: cache hit (%d observations)", series_id, len(cached["data"]))
            return to_frame(cached["data"])
        data = self.fetch(series_id, start)
        payload = {
            "series_id": series_id,
            "fetched_at": dt.date.today().isoformat(),
            "source": self.source,
            "start": start.isoformat(),
            "data": [[d, v] for d, v in data],
        }
        write_cache(path, payload)
        log.info("FRED %s: fetched %d observations via %s", series_id, len(data), self.source)
        return to_frame(data)


def read_cached(path: Path) -> dict[str, Any] | None:
    """The cached payload for a series, or ``None`` when it has not been pulled."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cached_series(settings: Settings, series_id: str) -> pd.DataFrame | None:
    """Read a series from the cache only (never the network); ``None`` when absent."""
    cached = read_cached(cache_path(settings, series_id))
    return None if cached is None else to_frame(cached["data"])
