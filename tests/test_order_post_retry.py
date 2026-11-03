"""An order-creating POST must never be blindly resent after an ambiguous failure."""
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.clients.kalshi_client import (
    KalshiAPIError, KalshiClient, KalshiOrderOutcomeUnknown)

POST_PATH = "/trade-api/v2/portfolio/events/orders"
ORDERS_PATH = "/trade-api/v2/portfolio/orders"
BODY = {"ticker": "T", "client_order_id": "coid-1", "side": "bid"}


def _client(handler):
    with patch.object(KalshiClient, "_load_private_key", lambda self: None):
        c = KalshiClient()
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


@pytest.fixture(autouse=True)
def _patches():
    with patch.object(KalshiClient, "_sign_request", lambda self, *a: "sig"), \
         patch("src.clients.kalshi_client.asyncio.sleep", AsyncMock()):
        yield


def _router(post_behaviors, orders):
    log = []

    def handle(req):
        log.append((req.method, req.url.path))
        if req.method == "POST":
            b = post_behaviors.pop(0)
            if isinstance(b, Exception):
                raise b
            return b
        assert req.url.path == ORDERS_PATH
        return httpx.Response(200, json={"orders": orders, "cursor": ""})
    return handle, log


async def test_timeout_then_found_returns_order_without_resend():
    h, log = _router([httpx.ReadTimeout("t")],
                     [{"order_id": "o9", "client_order_id": "coid-1"}])
    r = await _client(h)._make_authenticated_request("POST", POST_PATH, json_data=BODY, ambiguous_order_create=True)
    assert r["order_id"] == "o9" and r["recovered_by_client_order_id"]
    assert [m for m, _ in log].count("POST") == 1


async def test_5xx_not_found_raises_unknown_and_never_resends():
    h, log = _router([httpx.Response(503)], [{"order_id": "x", "client_order_id": "other"}])
    with pytest.raises(KalshiOrderOutcomeUnknown):
        await _client(h)._make_authenticated_request("POST", POST_PATH, json_data=BODY, ambiguous_order_create=True)
    assert [m for m, _ in log].count("POST") == 1


async def test_timeout_not_found_raises_unknown_with_exactly_one_post():
    h, log = _router([httpx.ReadTimeout("t")], [])
    with pytest.raises(KalshiOrderOutcomeUnknown):
        await _client(h)._make_authenticated_request(
            "POST", POST_PATH, json_data=BODY, ambiguous_order_create=True)
    assert [m for m, _ in log].count("POST") == 1


async def test_409_duplicate_is_looked_up():
    h, log = _router([httpx.Response(409, json={"error": "dup"})],
                     [{"order_id": "o1", "client_order_id": "coid-1"}])
    r = await _client(h)._make_authenticated_request("POST", POST_PATH, json_data=BODY, ambiguous_order_create=True)
    assert r["order_id"] == "o1"


async def test_unknown_is_a_kalshi_api_error():
    assert issubclass(KalshiOrderOutcomeUnknown, KalshiAPIError)


async def test_connect_error_still_retried():
    ok = httpx.Response(200, json={"order_id": "o2"})
    h, log = _router([httpx.ConnectError("c"), ok], [])
    r = await _client(h)._make_authenticated_request("POST", POST_PATH, json_data=BODY, ambiguous_order_create=True)
    assert r == {"order_id": "o2"}
    assert [m for m, _ in log].count("POST") == 2


async def test_get_still_retries_on_5xx():
    n = {"i": 0}

    def handle(req):
        n["i"] += 1
        return httpx.Response(503) if n["i"] < 3 else httpx.Response(200, json={"ok": 1})
    r = await _client(handle)._make_authenticated_request("GET", "/trade-api/v2/portfolio/balance")
    assert r == {"ok": 1} and n["i"] == 3
