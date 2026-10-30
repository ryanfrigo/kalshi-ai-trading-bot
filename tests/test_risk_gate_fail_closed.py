"""Regression tests for issue #98: the order-placement risk gate fails CLOSED.

If the risk governor raises (infrastructure failure, not a limit breach), a buy
must be refused before anything is posted to Kalshi. Sells reduce risk, so they
still go out with a warning.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.clients.kalshi_client import KalshiAPIError, KalshiClient


def _client_with_mocked_request():
    client = KalshiClient.__new__(KalshiClient)
    client._make_authenticated_request = AsyncMock(return_value={"order_id": "abc"})
    return client


def _place(client, action):
    return asyncio.run(client.place_order(
        ticker="KXTEST-26OCT30",
        client_order_id="cid-1",
        side="yes",
        action=action,
        count=1,
        type_="limit",
        yes_price=50,
        no_price=None,
        expiration_ts=None,
    ))


def test_buy_fails_closed_when_governor_raises():
    client = _client_with_mocked_request()
    with patch("src.risk.risk_governor.RiskGovernor.check",
               new=AsyncMock(side_effect=RuntimeError("balance endpoint down"))):
        with pytest.raises(KalshiAPIError, match="RISK_GATE_UNAVAILABLE: balance endpoint down"):
            _place(client, "buy")
    client._make_authenticated_request.assert_not_called()


def test_sell_proceeds_when_governor_raises():
    client = _client_with_mocked_request()
    with patch("src.risk.risk_governor.RiskGovernor.check",
               new=AsyncMock(side_effect=RuntimeError("balance endpoint down"))):
        _place(client, "sell")
    client._make_authenticated_request.assert_awaited_once()
    assert client._make_authenticated_request.await_args.args[0] == "POST"
