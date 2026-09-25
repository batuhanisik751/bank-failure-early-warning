"""FdicClient tests: every request goes through httpx.MockTransport, never the network."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
import tenacity

from bankcanary.config import Secrets, load_settings
from bankcanary.sources.fdic import (
    FdicApiError,
    FdicClient,
    cache_path,
    canonical_query,
    query_hash,
)

FIXTURES = Path(__file__).parent / "fixtures" / "fdic"


def load_fixture(name: str) -> dict:
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def settings(tmp_path: Path):
    base = load_settings()
    fdic = base.fdic.model_copy(
        update={"page_size": 2, "requests_per_second": 10_000.0, "max_retries": 3}
    )
    return base.model_copy(update={"data_dir": tmp_path / "data", "fdic": fdic})


def make_client(settings, handler, secrets: Secrets | None = None) -> FdicClient:
    return FdicClient(
        settings,
        secrets=secrets,
        transport=httpx.MockTransport(handler),
        wait=tenacity.wait_none(),
    )


def page_response(rows: list[dict], total: int) -> httpx.Response:
    body = {
        "meta": {"total": total, "parameters": {}},
        "data": [{"data": r, "score": 0} for r in rows],
        "totals": {"count": total},
    }
    return httpx.Response(200, json=body)


class Recorder:
    """Handler that serves the same fixture rows page by page and records each request."""

    def __init__(self, rows: list[dict], statuses: list[int] | None = None) -> None:
        self.rows = rows
        self.requests: list[httpx.Request] = []
        self.statuses = list(statuses or [])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                return httpx.Response(status, json={"error": status})
        params = request.url.params
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 10_000))
        return page_response(self.rows[offset : offset + limit], len(self.rows))


INSTITUTIONS = [r["data"] for r in load_fixture("institutions_page.json")["data"]]
FAILURES = [r["data"] for r in load_fixture("failures_page.json")["data"]]


def test_get_paginates_across_three_pages(settings):
    handler = Recorder(INSTITUTIONS)  # 3 rows, page_size 2 -> pages of 2, 1, done
    with make_client(settings, handler) as client:
        rows = client.get("institutions", fields=["CERT", "NAME", "STALP"], cache_name="all")
    assert [r["CERT"] for r in rows] == [3511, 24735, 628]
    offsets = [int(r.url.params["offset"]) for r in handler.requests]
    assert offsets == [0, 2]
    assert handler.requests[0].url.params["limit"] == "2"
    assert handler.requests[0].url.params["fields"] == "CERT,NAME,STALP"
    assert handler.requests[0].url.params["format"] == "json"
    assert str(handler.requests[0].url).startswith("https://api.fdic.gov/banks/institutions?")


def test_get_page_returns_records_and_total(settings):
    handler = Recorder(FAILURES)
    with make_client(settings, handler) as client:
        rows, total = client.get_page("failures", filters="FAILYR:2023", sort_by="FAILDATE")
    assert total == 2 and rows[0]["NAME"] == "SILICON VALLEY BANK"
    params = handler.requests[0].url.params
    assert params["filters"] == "FAILYR:2023"
    assert params["sort_by"] == "FAILDATE" and params["sort_order"] == "ASC"


def test_cache_hit_makes_no_request(settings):
    handler = Recorder(FAILURES)
    with make_client(settings, handler) as client:
        first = client.get("failures", filters="FAILYR:2023", cache_name="all")
        n_requests = len(handler.requests)
        second = client.get("failures", filters="FAILYR:2023", cache_name="all")
    assert first == second == FAILURES
    assert len(handler.requests) == n_requests == 1
    path = cache_path(settings, "failures", "all")
    assert path == settings.data_dir / "raw" / "fdic" / "failures" / "all.json"
    cached = json.loads(path.read_text(encoding="utf-8"))
    assert set(cached) == {"query", "fetched_at", "total", "data"}
    assert cached["total"] == 2 and len(cached["data"]) == 2
    assert cached["query"] == canonical_query("failures", "FAILYR:2023", None, None, "ASC")
    assert len(cached["fetched_at"]) == 10  # YYYY-MM-DD


def test_force_redownloads_and_overwrites_cache(settings):
    handler = Recorder(FAILURES)
    with make_client(settings, handler) as client:
        client.get("failures", cache_name="all")
        path = cache_path(settings, "failures", "all")
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["data"] = []
        path.write_text(json.dumps(cached), encoding="utf-8")
        assert client.get("failures", cache_name="all") == []
        assert client.get("failures", cache_name="all", force=True) == FAILURES
    assert len(handler.requests) == 2
    assert len(json.loads(path.read_text(encoding="utf-8"))["data"]) == 2


def test_hash_named_cache_is_independent_of_field_order(settings):
    handler = Recorder(INSTITUTIONS)
    with make_client(settings, handler) as client:
        client.get("institutions", fields=["NAME", "CERT", "STALP"])
        client.get("institutions", fields=["STALP", "CERT", "NAME"])
    assert len(handler.requests) == 2  # one query, two pages, second call served from cache
    q = canonical_query("institutions", None, ["NAME", "CERT", "STALP"], None, "ASC")
    assert cache_path(settings, "institutions", query_hash(q)).exists()


def test_retries_on_500_then_succeeds(settings):
    handler = Recorder(FAILURES, statuses=[500, 503, 200])
    with make_client(settings, handler) as client:
        rows = client.get("failures", cache_name="all")
    assert rows == FAILURES
    assert len(handler.requests) == 3


def test_retries_on_429(settings):
    handler = Recorder(FAILURES, statuses=[429, 200])
    with make_client(settings, handler) as client:
        rows, total = client.get_page("failures")
    assert total == 2 and len(handler.requests) == 2


def test_gives_up_after_max_retries(settings):
    handler = Recorder(FAILURES, statuses=[500] * 10)
    with make_client(settings, handler) as client, pytest.raises(FdicApiError) as excinfo:
        client.get_page("failures")
    assert excinfo.value.status_code == 500
    assert len(handler.requests) == settings.fdic.max_retries + 1
    assert not cache_path(settings, "failures", "all").exists()


def test_retries_on_transport_error(settings):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("boom", request=request)
        return page_response(FAILURES, 2)

    with make_client(settings, handler) as client:
        rows, _ = client.get_page("failures")
    assert len(rows) == 2 and len(calls) == 2


def test_does_not_retry_client_errors(settings):
    handler = Recorder(FAILURES, statuses=[404])
    with make_client(settings, handler) as client, pytest.raises(httpx.HTTPStatusError):
        client.get_page("nonsense")
    assert len(handler.requests) == 1


def test_warns_when_requested_field_is_dropped(settings, caplog):
    handler = Recorder(INSTITUTIONS)
    with make_client(settings, handler) as client, caplog.at_level(logging.WARNING):
        client.get("institutions", fields=["CERT", "NAME", "STALP", "NOSUCHFIELD"])
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("NOSUCHFIELD" in m and "institutions" in m for m in messages)
    assert not any("STALP" in m for m in messages)


def test_api_key_sent_only_when_set(settings):
    handler = Recorder(FAILURES)
    with make_client(settings, handler) as client:
        client.get_page("failures")
    assert "api_key" not in handler.requests[0].url.params

    handler = Recorder(FAILURES)
    secrets = Secrets(fdic_api_key="test-key", _env_file=None)
    with make_client(settings, handler, secrets=secrets) as client:
        client.get_page("failures")
    assert handler.requests[0].url.params["api_key"] == "test-key"

    handler = Recorder(FAILURES)
    with make_client(settings, handler, secrets=Secrets(_env_file=None)) as client:
        client.get_page("failures")
    assert "api_key" not in handler.requests[0].url.params


def test_fetch_definitions_parses_yaml_and_caches(settings):
    yaml_text = (FIXTURES / "risview_properties_small.yaml").read_text(encoding="utf-8")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=yaml_text, headers={"content-type": "text/yaml"})

    with make_client(settings, handler) as client:
        props = client.fetch_definitions("risview_properties")
        again = client.fetch_definitions("risview_properties")
        forced = client.fetch_definitions("risview_properties", force=True)
    assert str(requests[0].url) == "https://api.fdic.gov/banks/docs/risview_properties.yaml"
    assert len(requests) == 2  # second call came from disk, force re-downloaded
    assert props == again == forced
    assert set(props) == {"CERT", "REPDTE", "ASSET", "NCLNLS"}
    assert props["ASSET"]["title"] == "Total assets" and props["ASSET"]["type"] == "number"
    assert (settings.data_dir / "raw" / "fdic" / "docs" / "risview_properties.yaml").exists()


def test_canonical_query_is_independent_of_field_order():
    a = canonical_query("financials", "REPDTE:20230331", ["ASSET", "CERT", "REPDTE"], None, "asc")
    b = canonical_query("financials", "REPDTE:20230331", ("REPDTE", "ASSET", "CERT"), None, "ASC")
    assert a == b and query_hash(a) == query_hash(b)
    assert a["fields"] == ["ASSET", "CERT", "REPDTE"] and a["sort_order"] == "ASC"
    c = canonical_query("financials", "REPDTE:20230630", ["ASSET", "CERT", "REPDTE"], None, "ASC")
    assert query_hash(c) != query_hash(a)
    assert canonical_query("failures", None, None, None, "ASC")["fields"] == []


def test_limit_caps_rows_and_pages(settings):
    handler = Recorder(INSTITUTIONS)
    with make_client(settings, handler) as client:
        rows = client.get("institutions", limit=1)
    assert len(rows) == 1 and len(handler.requests) == 1
    assert handler.requests[0].url.params["limit"] == "1"
