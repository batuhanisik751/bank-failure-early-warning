"""FredClient tests: every request goes through httpx.MockTransport, never the network."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import pytest
import tenacity

from bankcanary.config import Secrets, load_settings
from bankcanary.sources.fred import FredApiError, FredClient, cache_path, cached_series

FIXTURES = Path(__file__).parent / "fixtures" / "fred"


@pytest.fixture
def settings(tmp_path: Path):
    base = load_settings()
    fred = base.fred.model_copy(update={"requests_per_second": 10_000.0, "max_retries": 2})
    return base.model_copy(update={"data_dir": tmp_path / "data", "fred": fred})


class Recorder:
    """Serves the CSV fixture or its JSON-API equivalent and records every request."""

    def __init__(self, statuses: list[int] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.statuses = list(statuses or [])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                return httpx.Response(status, text="error")
        params = request.url.params
        if request.url.host == "api.stlouisfed.org":
            series_id = params["series_id"]
            rows = _fixture_rows(series_id)
            body = {"observations": [{"date": d, "value": v} for d, v in rows]}
            return httpx.Response(200, json=body)
        return httpx.Response(200, text=(FIXTURES / f"{params['id']}.csv").read_text())


def _fixture_rows(series_id: str) -> list[tuple[str, str]]:
    lines = (FIXTURES / f"{series_id}.csv").read_text().splitlines()[1:]
    return [tuple(line.split(",")) for line in lines]


def make_client(settings, handler, secrets: Secrets | None = None) -> FredClient:
    return FredClient(
        settings, secrets=secrets, transport=httpx.MockTransport(handler), wait=tenacity.wait_none()
    )


def test_csv_path_without_key_parses_dot_as_missing(settings):
    handler = Recorder()
    with make_client(settings, handler) as client:
        frame = client.series("CAUR", start=dt.date(2007, 1, 1))
    req = handler.requests[0]
    assert req.url.host == "fred.stlouisfed.org" and req.url.path == "/graph/fredgraph.csv"
    assert req.url.params["id"] == "CAUR" and req.url.params["cosd"] == "2007-01-01"
    assert "api_key" not in req.url.params
    assert list(frame.columns) == ["date", "value"]
    assert str(frame["date"].dtype) == "datetime64[ns]"
    assert frame["value"].tolist()[:2] == [4.9, 4.9]
    assert frame["value"].isna().tolist() == [False, False, True, False]


def test_api_path_with_key_uses_json_endpoint(settings):
    handler = Recorder()
    secrets = Secrets(fred_api_key="secret", _env_file=None)
    with make_client(settings, handler, secrets=secrets) as client:
        frame = client.series("DGS10", start=dt.date(2007, 1, 1))
    req = handler.requests[0]
    assert req.url.host == "api.stlouisfed.org"
    assert req.url.params["series_id"] == "DGS10"
    assert req.url.params["api_key"] == "secret"
    assert req.url.params["file_type"] == "json"
    assert req.url.params["observation_start"] == "2007-01-01"
    assert frame["value"].isna().sum() == 1 and len(frame) == 4
    payload = json.loads(cache_path(settings, "DGS10").read_text())
    assert payload["source"] == "api" and payload["series_id"] == "DGS10"


def test_cache_is_reused_and_bypassed_with_force(settings):
    handler = Recorder()
    with make_client(settings, handler) as client:
        client.series("CAUR", start=dt.date(2007, 1, 1))
        again = client.series("CAUR", start=dt.date(2007, 1, 1))
        assert len(handler.requests) == 1
        payload = json.loads(cache_path(settings, "CAUR").read_text())
        assert set(payload) >= {"series_id", "fetched_at", "source", "data"}
        assert payload["source"] == "csv" and payload["data"][2] == ["2007-03-01", None]
        assert cached_series(settings, "CAUR").equals(again)
        assert cached_series(settings, "NOPE") is None
        # a later start is served from the cache; an earlier one triggers a refetch
        client.series("CAUR", start=dt.date(2007, 3, 1))
        assert len(handler.requests) == 1
        client.series("CAUR", start=dt.date(2006, 1, 1))
        assert len(handler.requests) == 2
        client.series("CAUR", start=dt.date(2006, 1, 1), force=True)
        assert len(handler.requests) == 3


def test_retries_on_429_and_5xx_then_succeeds(settings):
    handler = Recorder(statuses=[429, 503, 200])
    with make_client(settings, handler) as client:
        frame = client.series("CAUR")
    assert len(handler.requests) == 3 and len(frame) == 4


def test_gives_up_after_max_retries(settings):
    handler = Recorder(statuses=[500, 500, 500, 500])
    with make_client(settings, handler) as client, pytest.raises(FredApiError):
        client.series("CAUR")
    assert len(handler.requests) == 3  # max_retries=2 -> three attempts
    assert not cache_path(settings, "CAUR").exists()


def test_client_error_is_not_retried(settings):
    handler = Recorder(statuses=[404])
    with make_client(settings, handler) as client, pytest.raises(httpx.HTTPStatusError):
        client.series("CAUR")
    assert len(handler.requests) == 1
