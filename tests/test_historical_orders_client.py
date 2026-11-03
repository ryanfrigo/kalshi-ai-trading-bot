"""The real KalshiClient over httpx.MockTransport: 404 classification and the archive lookup."""
import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.clients.kalshi_client import (
    KalshiAPIError, KalshiClient, KalshiOrderOutcomeUnknown)
from src.jobs.execute import reconcile_resting_exit
from tests.test_resting_exit_reconcile import _db, _order, _row

HIST_PATH = "/trade-api/v2/historical/orders"


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


# ---- (1) status_code is set only for a definitive error response --------------------------

@pytest.mark.parametrize("status", [404, 400])
async def test_definitive_error_status_is_carried(status):
    c = _client(lambda req: httpx.Response(status, json={"error": "x"}))
    with pytest.raises(KalshiAPIError) as ei:
        await c.get_order("o1")
    assert ei.value.status_code == status


def _timeout(req):
    raise httpx.ReadTimeout("t")


@pytest.mark.parametrize("handler", [lambda req: httpx.Response(503), _timeout],
                         ids=["503-after-retries", "read-timeout"])
async def test_retried_failures_have_no_status_code(handler):
    c = _client(handler)
    with pytest.raises(KalshiAPIError) as ei:
        await c.get_order("o1")
    assert ei.value.status_code is None


def test_outcome_unknown_has_no_status_code():
    assert KalshiOrderOutcomeUnknown("x").status_code is None


# ---- (2) get_historical_orders request shape ---------------------------------------------

async def test_historical_orders_request_shape():
    seen = []

    def handle(req):
        seen.append((req.method, req.url.path, dict(req.url.params)))
        return httpx.Response(200, json={"orders": [], "cursor": ""})
    c = _client(handle)
    await c.get_historical_orders("RX-1")
    await c.get_historical_orders("RX-1", cursor="c2")
    assert seen[0] == ("GET", HIST_PATH, {"ticker": "RX-1", "limit": "1000"})
    assert seen[1] == ("GET", HIST_PATH, {"ticker": "RX-1", "limit": "1000", "cursor": "c2"})


# ---- (3) end to end: get_order 404 -> archived page 2 -> booked once ---------------------

def test_404_then_archived_page_two_books_the_fill_once(tmp_path):
    calls = []

    def handle(req):
        calls.append((req.url.path, req.url.params.get("cursor")))
        if req.url.path == "/trade-api/v2/portfolio/orders/o1":
            return httpx.Response(404, json={"error": "not_found"})
        assert req.method == "GET" and req.url.path == HIST_PATH
        assert req.url.params["ticker"] == "RX-1" and req.url.params["limit"] == "1000"
        if not req.url.params.get("cursor"):
            return httpx.Response(200, json={
                "orders": [_order(3, "executed", oid="other")["order"]], "cursor": "c2"})
        assert req.url.params["cursor"] == "c2"
        return httpx.Response(200, json={
            "orders": [_order(10, "executed")["order"]], "cursor": ""})

    async def go():
        db = await _db(tmp_path)
        (pos,) = await db.get_open_live_positions()
        result = await reconcile_resting_exit(pos, _client(handle), db)
        return result, await _row(db), await db.get_all_trade_logs()
    result, row, logs = asyncio.run(go())
    assert result == "closed" and row == ("closed", 10, None, None)
    assert len(logs) == 1 and logs[0].quantity == 10 and logs[0].pnl == pytest.approx(2.0)
    assert calls == [("/trade-api/v2/portfolio/orders/o1", None), (HIST_PATH, None),
                     (HIST_PATH, "c2")]
