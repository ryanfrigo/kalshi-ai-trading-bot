"""Tests pinning the legacy->v2 order payload translation.

Kalshi deprecated POST /portfolio/orders (side=yes/no, action=buy/sell,
price in cents) with HTTP 410. The live endpoint is /portfolio/events/orders,
which quotes everything from the YES leg: side is bid/ask, price is the YES-leg
price in dollars, count is a string. Getting the side/price inversion wrong
means trading the OPPOSITE side with real money, so these are pinned hard.
"""
from src.clients.kalshi_client import build_order_v2_payload


def test_buy_yes_limit_is_bid_at_yes_price():
    p = build_order_v2_payload("MKT", "coid", side="yes", action="buy",
                               count=16, type_="limit", yes_price=94)
    assert p["side"] == "bid"
    assert p["price"] == "0.9400"
    assert p["count"] == "16.00"
    assert p["time_in_force"] == "good_till_canceled"
    assert p["ticker"] == "MKT"
    assert p["client_order_id"] == "coid"


def test_sell_yes_limit_is_ask_at_yes_price():
    p = build_order_v2_payload("MKT", "c", side="yes", action="sell",
                               count=5, type_="limit", yes_price=80)
    assert p["side"] == "ask"
    assert p["price"] == "0.8000"


def test_buy_no_is_ask_at_one_minus_no_price():
    # Buying NO at $0.96 == selling YES at $0.04.
    p = build_order_v2_payload("MKT", "c", side="no", action="buy",
                               count=16, type_="limit", no_price=96)
    assert p["side"] == "ask"
    assert p["price"] == "0.0400"


def test_sell_no_is_bid_at_one_minus_no_price():
    # Closing a NO at $0.90 == buying YES at $0.10.
    p = build_order_v2_payload("MKT", "c", side="no", action="sell",
                               count=10, type_="limit", no_price=90)
    assert p["side"] == "bid"
    assert p["price"] == "0.1000"


def test_market_order_is_immediate_or_cancel_taker():
    p = build_order_v2_payload("MKT", "c", side="no", action="buy",
                               count=3, type_="market", no_price=96)
    assert p["time_in_force"] == "immediate_or_cancel"
    assert p["self_trade_prevention_type"] == "taker_at_cross"


def test_count_formatted_with_two_decimals():
    p = build_order_v2_payload("MKT", "c", side="yes", action="buy",
                               count=1, type_="market", yes_price=50)
    assert p["count"] == "1.00"


def test_no_client_order_id_omitted_when_absent():
    p = build_order_v2_payload("MKT", "", side="yes", action="buy",
                               count=1, type_="market", yes_price=50)
    assert "client_order_id" not in p
