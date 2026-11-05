"""A risk halt caused only by a trading limit lets a reduce-only SELL through.

Everything else stays blocked: buys, MANUAL_HALT, STATE_CORRUPT, gate failure,
oversize sells, wrong-side sells, and any position read that cannot confirm the
held size. The HTTP layer is mocked; POSTs are counted.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.clients.kalshi_client import KalshiAPIError, KalshiClient
from src.risk.risk_governor import GovernorDecision

TICKER = "KXTEST-26OCT30"


def _halt(*reasons):
    return GovernorDecision(halted=True, reasons=list(reasons), current_equity_cents=8_000,
                            sod_equity_cents=10_000, peak_equity_cents=10_000)


LIMIT_HALT = _halt("DAILY_LOSS: down 20.0% from start-of-day (limit 10%)",
                   "DRAWDOWN: down 20.0% from peak (limit 15%)")


def _client(positions=None, positions_exc=None):
    client = KalshiClient.__new__(KalshiClient)
    posts = []

    async def fake(method, path, params=None, json_data=None, **kw):
        if method == "POST":
            posts.append(json_data)
            return {"order_id": "abc"}
        if positions_exc:
            raise positions_exc
        return {"market_positions": positions if positions is not None else []}

    client._make_authenticated_request = fake
    client.posts = posts
    return client


def _place(client, action="sell", side="yes", count=5, type_="limit"):
    return asyncio.run(client.place_order(
        ticker=TICKER, client_order_id="cid-1", side=side, action=action, count=count,
        type_=type_, yes_price=50 if side == "yes" else None,
        no_price=50 if side == "no" else None, expiration_ts=None))


def _gov(decision=None, exc=None):
    return patch("src.risk.risk_governor.RiskGovernor.check",
                 new=AsyncMock(return_value=decision, side_effect=exc))


def _pos(fp):
    return [{"ticker": TICKER, "position_fp": fp}]


def test_halted_reduce_only_sell_posts_once():
    client = _client(_pos("10.00"))
    with _gov(LIMIT_HALT):
        _place(client, "sell", "yes", 5)
    assert len(client.posts) == 1


def test_halted_reduce_only_sell_no_side_posts_once():
    client = _client(_pos("-10.00"))
    with _gov(LIMIT_HALT):
        _place(client, "sell", "no", 10)
    assert len(client.posts) == 1


def test_halted_oversize_sell_blocked():
    client = _client(_pos("3.00"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError, match="RISK_HALT"):
        _place(client, "sell", "yes", 5)
    assert client.posts == []


def test_halted_buy_blocked():
    client = _client(_pos("10.00"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError, match="RISK_HALT"):
        _place(client, "buy", "yes", 1)
    assert client.posts == []


def test_state_corrupt_sell_blocked():
    client = _client(_pos("10.00"))
    with _gov(_halt("STATE_CORRUPT: data/runtime/x.json")), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


def test_manual_halt_sell_blocked():
    client = _client(_pos("10.00"))
    with _gov(_halt("MANUAL_HALT: flag")), pytest.raises(KalshiAPIError, match="MANUAL_HALT"):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


def test_manual_halt_plus_limit_sell_blocked():
    client = _client(_pos("10.00"))
    with _gov(_halt("MANUAL_HALT: flag", "DAILY_LOSS: down 20.0%")), \
            pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


def test_gate_unavailable_sell_blocked():
    client = _client(_pos("10.00"))
    with _gov(exc=RuntimeError("balance down")), \
            pytest.raises(KalshiAPIError, match="RISK_GATE_UNAVAILABLE"):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


def test_positions_read_fails_sell_blocked():
    client = _client(positions_exc=RuntimeError("positions down"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError, match="RISK_HALT"):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


def test_ticker_absent_sell_blocked():
    client = _client([{"ticker": "OTHER", "position_fp": "10.00"}])
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


def test_unparseable_position_sell_blocked():
    client = _client(_pos("garbage"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


def test_wrong_side_sell_blocked():
    client = _client(_pos("10.00"))  # hold YES, try to sell NO
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "no", 1)
    assert client.posts == []


def test_halt_exempt_market_sell_carries_reduce_only_ioc():
    client = _client(_pos("10.00"))
    with _gov(LIMIT_HALT):
        _place(client, "sell", "yes", 5, type_="market")
    assert len(client.posts) == 1
    assert client.posts[0]["reduce_only"] is True
    assert client.posts[0]["time_in_force"] == "immediate_or_cancel"


def test_halt_exempt_limit_sell_is_forced_ioc_without_expiration():
    # Real exit paths send type_="limit" (GTC); Kalshi rejects reduce_only
    # unless IOC, so the exempt sell is forced to IOC and drops expiration_time.
    client = _client(_pos("10.00"))
    with _gov(LIMIT_HALT):
        asyncio.run(client.place_order(
            ticker=TICKER, client_order_id="cid-1", side="yes", action="sell", count=5,
            type_="limit", yes_price=50, no_price=None, expiration_ts=1_900_000_000))
    assert len(client.posts) == 1
    body = client.posts[0]
    assert body["reduce_only"] is True
    assert body["time_in_force"] == "immediate_or_cancel"
    assert "expiration_time" not in body


def test_non_halted_limit_sell_stays_gtc_without_reduce_only():
    client = _client(_pos("10.00"))
    with _gov(GovernorDecision(halted=False, reasons=[], current_equity_cents=10_000,
                               sod_equity_cents=10_000, peak_equity_cents=10_000)):
        asyncio.run(client.place_order(
            ticker=TICKER, client_order_id="cid-1", side="yes", action="sell", count=5,
            type_="limit", yes_price=50, no_price=None, expiration_ts=1_900_000_000))
    body = client.posts[0]
    assert body["time_in_force"] == "good_till_canceled"
    assert body["expiration_time"] == 1_900_000_000
    assert "reduce_only" not in body


@pytest.mark.parametrize("action", ["sell", "buy"])
def test_non_halted_orders_have_no_reduce_only_key(action):
    client = _client(_pos("10.00"))
    with _gov(GovernorDecision(halted=False, reasons=[], current_equity_cents=10_000,
                               sod_equity_cents=10_000, peak_equity_cents=10_000)):
        _place(client, action, "yes", 1)
    assert len(client.posts) == 1
    assert "reduce_only" not in client.posts[0]


@pytest.mark.parametrize("fp", ["Infinity", "-Infinity", "NaN", "1e999", "-1e999"])
def test_non_finite_position_sell_blocked(fp):
    client = _client(_pos(fp))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes" if not fp.startswith("-") else "no", 1)
    assert client.posts == []


def test_fractional_held_allows_int_count_within_it():
    client = _client(_pos("1.006"))
    with _gov(LIMIT_HALT):
        _place(client, "sell", "yes", 1)
    assert len(client.posts) == 1


def test_fractional_held_below_count_blocked():
    client = _client(_pos("1.996"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 2)
    assert client.posts == []


@pytest.mark.parametrize("count", [True, float("nan"), float("inf"), 0, -1, "5"])
def test_invalid_count_blocked(count):
    client = _client(_pos("5.00"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", count)
    assert client.posts == []


def test_duplicate_position_rows_blocked():
    client = _client(_pos("10.00") + _pos("10.00"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


@pytest.mark.parametrize("reasons", [[], ["WEIRD: x"], ["DAILY_LOSS: x", "WEIRD: x"]])
def test_empty_or_unknown_halt_reasons_blocked(reasons):
    client = _client(_pos("10.00"))
    with _gov(_halt(*reasons)), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 1)
    assert client.posts == []


@pytest.mark.parametrize("type_", ["limit", "market"])
def test_non_halted_sell_body_equals_default_payload(type_):
    from src.clients.kalshi_client import build_order_v2_payload
    client = _client(_pos("10.00"))
    with _gov(GovernorDecision(halted=False, reasons=[], current_equity_cents=10_000,
                               sod_equity_cents=10_000, peak_equity_cents=10_000)):
        _place(client, "sell", "yes", 3, type_=type_)
    assert client.posts == [build_order_v2_payload(
        TICKER, "cid-1", "yes", "sell", 3, type_, 50, None, None)]


def test_float_count_matches_held_no_side():
    # cli close-all passes round(abs(contracts), 2): always a float.
    client = _client(_pos("-40.00"))
    with _gov(LIMIT_HALT):
        _place(client, "sell", "no", 40.0, type_="limit")
    assert len(client.posts) == 1
    assert client.posts[0]["reduce_only"] is True
    assert client.posts[0]["time_in_force"] == "immediate_or_cancel"


def test_fractional_count_equal_to_held_passes():
    client = _client(_pos("12.34"))
    with _gov(LIMIT_HALT):
        _place(client, "sell", "yes", 12.34)
    assert len(client.posts) == 1


def test_count_serializing_to_held_passes():
    # 40.004 serializes to "40.00", equal to held, so the exemption applies; the
    # decision follows the wire value, not the raw float.
    assert f"{40.004:.2f}" == "40.00"
    client = _client(_pos("40.00"))
    with _gov(LIMIT_HALT):
        _place(client, "sell", "yes", 40.004)
    assert client.posts[0]["count"] == "40.00"


def test_count_serializing_above_held_blocked():
    # 40.005 is the float 40.00500000000000255, which serializes to "40.01" > held.
    assert f"{40.005:.2f}" == "40.01"
    client = _client(_pos("40.00"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 40.005)
    assert client.posts == []


def test_count_above_held_blocked():
    client = _client(_pos("40.00"))
    with _gov(LIMIT_HALT), pytest.raises(KalshiAPIError):
        _place(client, "sell", "yes", 40.01)
    assert client.posts == []
