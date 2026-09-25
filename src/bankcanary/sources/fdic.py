"""Client for the FDIC BankFind Suite API (``https://api.fdic.gov/banks/``).

The API exposes the public record of every FDIC-insured institution: quarterly Call
Report financials (``financials``), the failure list (``failures``), charter attributes
(``institutions``), structure events such as mergers (``history``) and branch locations
(``locations``). This module wraps it with three things the ingest pipeline relies on:

* an on-disk cache (one JSON file per query) so a rebuild never re-downloads a quarter,
* polite rate limiting plus retries with backoff, and
* transparent pagination, because a quarter of financials (~4,000-9,000 banks) or the
  full institutions table (~28,000 rows) exceeds the 10,000-record page limit.

No API key is required; ``Secrets.fdic_api_key`` is forwarded only when it is set.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import tenacity
import yaml

from bankcanary.config import Secrets, Settings

log = logging.getLogger(__name__)

ENDPOINTS = ("failures", "institutions", "history", "financials", "locations")
DEFINITION_NAMES = (
    "risview_properties",
    "failure_properties",
    "institution_properties",
    "history_properties",
    "location_properties",
)
MAX_PAGE_SIZE = 10_000


def canonical_query(
    endpoint: str,
    filters: str | None,
    fields: list[str] | tuple[str, ...] | None,
    sort_by: str | None,
    sort_order: str,
) -> dict[str, Any]:
    """Return the query as a stable dict: cache keys must not depend on field order."""
    return {
        "endpoint": endpoint,
        "filters": filters or "",
        "fields": sorted(set(fields)) if fields else [],
        "sort_by": sort_by or "",
        "sort_order": (sort_order or "ASC").upper(),
    }


def query_hash(query: dict[str, Any]) -> str:
    """sha1 of the canonical query, used as the cache file name when none is given."""
    payload = json.dumps(query, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def cache_root(settings: Settings) -> Path:
    return settings.data_dir / "raw" / "fdic"


def cache_path(settings: Settings, endpoint: str, cache_name_or_hash: str) -> Path:
    """``data/raw/fdic/<endpoint>/<name>.json`` (see CONTRACT section 3)."""
    return cache_root(settings) / endpoint / f"{cache_name_or_hash}.json"


class TokenBucket:
    """Simple token bucket: at most ``rate`` requests per second on average, burst ``capacity``."""

    def __init__(self, rate: float, capacity: int = 1) -> None:
        self.rate = max(float(rate), 1e-9)
        self.capacity = max(int(capacity), 1)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rate
                time.sleep(wait)
                self._last = time.monotonic()
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class FdicApiError(RuntimeError):
    """An HTTP status that should be retried (429 or 5xx) or that ended the retries."""

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"FDIC API returned HTTP {status_code} for {url}")
        self.status_code = status_code
        self.url = url


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, FdicApiError) and (exc.status_code == 429 or exc.status_code >= 500)


def _flatten(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The API wraps each row as ``{"data": {...}, "score": 0}``; keep only the row."""
    return [r.get("data", r) if isinstance(r, dict) else r for r in records]


def _warn_dropped_fields(endpoint: str, fields: list[str] | None, rows: list[dict]) -> list[str]:
    """Unknown field names are silently dropped by the API; surface them as warnings."""
    if not fields or not rows:
        return []
    present: set[str] = set()
    for row in rows:
        present.update(row.keys())
    missing = [f for f in fields if f not in present]
    if missing:
        log.warning(
            "FDIC %s: requested field(s) absent from every returned record: %s",
            endpoint,
            ", ".join(missing),
        )
    return missing


class FdicClient:
    """Thin, cached, rate-limited client for the BankFind Suite API.

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
        self.secrets = secrets
        self.fdic = settings.fdic
        self.api_key = secrets.fdic_api_key if secrets is not None else None
        self.page_size = min(int(self.fdic.page_size), MAX_PAGE_SIZE)
        self.bucket = TokenBucket(self.fdic.requests_per_second)
        self.wait = (
            wait if wait is not None else tenacity.wait_exponential_jitter(initial=1, max=60)
        )
        self._client = httpx.Client(
            base_url=self.fdic.base_url,
            timeout=self.fdic.timeout_seconds,
            transport=transport,
            headers={"Accept": "application/json"},
        )
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FdicClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low-level HTTP ---------------------------------------------------------------

    def _request(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """One HTTP GET with rate limiting and retries on 429, 5xx and transport errors."""
        retryer = tenacity.Retrying(
            retry=tenacity.retry_if_exception(_is_retryable),
            stop=tenacity.stop_after_attempt(max(int(self.fdic.max_retries), 0) + 1),
            wait=self.wait,
            reraise=True,
            before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
        )
        query = dict(params or {})
        if self.api_key:
            query["api_key"] = self.api_key
        for attempt in retryer:
            with attempt:
                self.bucket.acquire()
                self.request_count += 1
                response = self._client.get(path, params=query)
                if response.status_code == 429 or response.status_code >= 500:
                    raise FdicApiError(response.status_code, str(response.url))
                response.raise_for_status()
        return response

    # -- pages ------------------------------------------------------------------------

    def get_page(
        self,
        endpoint: str,
        *,
        filters: str | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        sort_by: str | None = None,
        sort_order: str = "ASC",
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Fetch one page and return ``(flattened records, total matching rows)``."""
        params: dict[str, Any] = {
            "format": "json",
            "limit": min(int(limit or self.page_size), MAX_PAGE_SIZE),
            "offset": int(offset),
        }
        if filters:
            params["filters"] = filters
        if fields:
            params["fields"] = ",".join(fields)
        if sort_by:
            params["sort_by"] = sort_by
            params["sort_order"] = (sort_order or "ASC").upper()
        response = self._request(endpoint, params)
        body = response.json()
        rows = _flatten(body.get("data") or [])
        meta = body.get("meta") or {}
        total = meta.get("total")
        if total is None:
            total = (body.get("totals") or {}).get("count", len(rows))
        return rows, int(total)

    def _fetch_all(
        self,
        endpoint: str,
        *,
        filters: str | None,
        fields: list[str] | tuple[str, ...] | None,
        sort_by: str | None,
        sort_order: str,
        limit: int | None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Walk every page via ``offset`` until ``limit`` rows or the reported total."""
        rows: list[dict[str, Any]] = []
        offset = 0
        total = 0
        while True:
            page_size = self.page_size if limit is None else min(self.page_size, limit - len(rows))
            if page_size <= 0:
                break
            page, total = self.get_page(
                endpoint,
                filters=filters,
                fields=fields,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=page_size,
                offset=offset,
            )
            rows.extend(page)
            offset += len(page)
            if not page or offset >= total or (limit is not None and len(rows) >= limit):
                break
        if limit is not None:
            rows = rows[:limit]
        log.info("FDIC %s: fetched %d of %d rows (filters=%r)", endpoint, len(rows), total, filters)
        return rows, total

    # -- cached queries ---------------------------------------------------------------

    def get(
        self,
        endpoint: str,
        *,
        filters: str | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        sort_by: str | None = None,
        sort_order: str = "ASC",
        limit: int | None = None,
        cache_name: str | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Return all records for a query, from the on-disk cache when available.

        ``cache_name`` names the file (ingest uses ``<REPDTE>`` for financials, ``all``
        for whole-table pulls); otherwise the file is named by the query hash. The cache
        is only bypassed with ``force=True``. A cached file whose query differs from the
        one requested (same name, different fields) is re-downloaded and overwritten.
        """
        query = canonical_query(endpoint, filters, fields, sort_by, sort_order)
        path = cache_path(self.settings, endpoint, cache_name or query_hash(query))
        if not force and path.exists():
            cached = read_cache(path)
            if cached.get("query") == query and (limit is None or limit == cached.get("limit")):
                log.info(
                    "FDIC %s: cache hit %s (%d rows)", endpoint, path.name, len(cached["data"])
                )
                return list(cached["data"])
            log.info(
                "FDIC %s: cache file %s has a different query; refetching", endpoint, path.name
            )
        rows, total = self._fetch_all(
            endpoint,
            filters=filters,
            fields=fields,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
        )
        _warn_dropped_fields(endpoint, list(fields) if fields else None, rows)
        payload: dict[str, Any] = {
            "query": query,
            "fetched_at": dt.date.today().isoformat(),
            "total": total,
            "data": rows,
        }
        if limit is not None:
            payload["limit"] = limit
        write_cache(path, payload)
        return rows

    def fetch_definitions(self, name: str, force: bool = False) -> dict[str, Any]:
        """Download ``<docs_url><name>.yaml`` and return its ``properties.data.properties``.

        These are the FDIC's own field dictionaries (``risview_properties`` covers the
        financial codes: title, description, type). They are the source of truth for the
        field list in ``config/fields.yaml``.
        """
        if name not in DEFINITION_NAMES:
            log.warning("FDIC definitions: %r is not one of %s", name, DEFINITION_NAMES)
        path = cache_root(self.settings) / "docs" / f"{name}.yaml"
        if force or not path.exists():
            url = f"{self.fdic.docs_url}{name}.yaml"
            response = self._request(url)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text, encoding="utf-8")
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return doc.get("properties", {}).get("data", {}).get("properties", {}) or {}


def read_cache(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Write atomically (temp file + rename) so an interrupted pull never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    tmp.replace(path)
