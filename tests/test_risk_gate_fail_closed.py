"""Regression tests for issue #98: the order-placement risk gate fails CLOSED.

If the risk governor raises (infrastructure failure, not a limit breach), the
order is refused before anything is posted to Kalshi, whether it buys or sells:
a sell can open a short, and there is no reduce_only flag. The position cap is
checked against the price of the leg being bought.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.clients.kalshi_client import KalshiAPIError, KalshiClient
from src.config.settings import settings
from src.risk.risk_governor import GovernorDecision


def _client_with_mocked_request():
    client = KalshiClient.__new__(KalshiClient)
    client._make_authenticated_request = AsyncMock(return_value={"order_id": "abc"})
    return client


def _place(client, action, side="yes", count=1, yes_price=50, no_price=None):
    return asyncio.run(client.place_order(
        ticker="KXTEST-26OCT30",
        client_order_id="cid-1",
        side=side,
        action=action,
        count=count,
        type_="limit",
        yes_price=yes_price,
        no_price=no_price,
        expiration_ts=None,
    ))


def _healthy_decision(equity_cents=10_000):
    return GovernorDecision(halted=False, reasons=[], current_equity_cents=equity_cents,
                            sod_equity_cents=equity_cents, peak_equity_cents=equity_cents)


def test_buy_fails_closed_when_governor_raises():
    client = _client_with_mocked_request()
    with patch("src.risk.risk_governor.RiskGovernor.check",
               new=AsyncMock(side_effect=RuntimeError("balance endpoint down"))):
        with pytest.raises(KalshiAPIError, match="RISK_GATE_UNAVAILABLE: balance endpoint down"):
            _place(client, "buy")
    client._make_authenticated_request.assert_not_called()


def test_sell_fails_closed_when_governor_raises():
    client = _client_with_mocked_request()
    with patch("src.risk.risk_governor.RiskGovernor.check",
               new=AsyncMock(side_effect=RuntimeError("balance endpoint down"))):
        with pytest.raises(KalshiAPIError, match="RISK_GATE_UNAVAILABLE"):
            _place(client, "sell")
    client._make_authenticated_request.assert_not_called()


def test_invalid_action_is_rejected_before_posting():
    client = _client_with_mocked_request()
    with pytest.raises(ValueError, match="action must be 'buy' or 'sell', got 'typo'"):
        _place(client, "typo")
    client._make_authenticated_request.assert_not_called()


def test_buy_no_cap_uses_no_price_and_blocks_over_cap():
    # Equity $100, cap 10% = $10. Buy 20 NO at 80c costs $16 -> blocked.
    # (The old code priced a NO buy off the YES leg: 100-80=20c -> $4, passed.)
    client = _client_with_mocked_request()
    with patch.object(settings.trading, "max_position_size_pct", 10.0), \
         patch("src.risk.risk_governor.RiskGovernor.check",
               new=AsyncMock(return_value=_healthy_decision())):
        with pytest.raises(KalshiAPIError, match="POSITION_CAP"):
            _place(client, "buy", side="no", count=20, yes_price=None, no_price=80)
    client._make_authenticated_request.assert_not_called()


def test_buy_no_within_cap_passes():
    # Buy 12 NO at 80c = $9.60, under the $10 cap -> posted.
    client = _client_with_mocked_request()
    with patch.object(settings.trading, "max_position_size_pct", 10.0), \
         patch("src.risk.risk_governor.RiskGovernor.check",
               new=AsyncMock(return_value=_healthy_decision())):
        _place(client, "buy", side="no", count=12, yes_price=None, no_price=80)
    client._make_authenticated_request.assert_awaited_once()


def test_buy_yes_over_cap_blocked():
    # Buy 20 YES at 80c = $16 -> blocked.
    client = _client_with_mocked_request()
    with patch.object(settings.trading, "max_position_size_pct", 10.0), \
         patch("src.risk.risk_governor.RiskGovernor.check",
               new=AsyncMock(return_value=_healthy_decision())):
        with pytest.raises(KalshiAPIError, match="POSITION_CAP"):
            _place(client, "buy", side="yes", count=20, yes_price=80)
    client._make_authenticated_request.assert_not_called()
