"""Tests for the agent toolbelt order-sizing guard.

cap_count caps an intended contract count so the order never exceeds the
per-position equity cap or available cash — a hard backstop independent of
whatever size the agent (or strategy) asks for.
"""
from src.agent.toolbelt import cap_count


def test_cap_count_allows_within_limits():
    # equity 80000c -> 10% cap 8000c; price 95c -> up to 84 contracts; cash ample.
    n, reason = cap_count(16, price_cents=95, equity_cents=80000, cash_cents=60000)
    assert n == 16
    assert reason == ""


def test_cap_count_caps_to_position_pct():
    n, reason = cap_count(1000, price_cents=95, equity_cents=80000, cash_cents=60000)
    assert n == 84  # floor(8000 / 95)
    assert "position cap" in reason.lower()


def test_cap_count_caps_to_cash():
    n, reason = cap_count(1000, price_cents=95, equity_cents=80000, cash_cents=1000)
    assert n == 10  # floor(1000 / 95)
    assert "cash" in reason.lower()


def test_cap_count_zero_when_insufficient_cash():
    n, reason = cap_count(16, price_cents=95, equity_cents=80000, cash_cents=50)
    assert n == 0


def test_cap_count_custom_position_pct():
    # 5% of 80000 = 4000c -> floor(4000/95)=42
    n, reason = cap_count(1000, price_cents=95, equity_cents=80000,
                          cash_cents=60000, max_position_pct=0.05)
    assert n == 42


# ---------------------------------------------------------------------------
# Edge Policy gate — place_guarded_order consults the active policy
# ---------------------------------------------------------------------------
import asyncio
import json

import pytest

from src.agent.toolbelt import place_guarded_order


class _Decision:
    halted = False
    current_equity_cents = 80000
    def to_dict(self):
        return {"halted": False}


class _Governor:
    async def check(self):
        return _Decision()


class _ReachedMarket(Exception):
    """Sentinel: proves execution got PAST the policy gate to the market call."""


class _NoMarketClient:
    async def get_market(self, ticker):
        raise _ReachedMarket(ticker)


def _write_policy(tmp_path, blocks):
    p = tmp_path / "edge_policy.json"
    p.write_text(json.dumps({"meta": {}, "blocks": blocks,
                             "warnings": [], "haircuts": []}))
    return str(p)


def test_policy_blocks_a_trade_in_a_blocked_category(tmp_path):
    # Policy blocks the KXCPI series; a KXCPI trade must be refused BEFORE any
    # market call (the fake client would raise if reached).
    policy_path = _write_policy(tmp_path, [
        {"dimension": "category", "label": "KXCPI", "reason": "loses", "n": 6, "pnl": -66.0}])
    res = asyncio.run(place_guarded_order(
        _NoMarketClient(), ticker="KXCPI-26JUN-3", side="no", count=10,
        price=0.95, est_prob=0.97, governor=_Governor(), dry=True,
        policy_path=policy_path))
    assert res["ok"] is False
    assert res["reason"] == "blocked_by_policy"
    assert res["policy_verdict"]["verdict"] == "BLOCK"


def test_override_policy_bypasses_the_block(tmp_path):
    # With override, the gate is skipped and execution proceeds to the market
    # (proven by the sentinel the fake client raises).
    policy_path = _write_policy(tmp_path, [
        {"dimension": "category", "label": "KXCPI", "reason": "loses", "n": 6, "pnl": -66.0}])
    with pytest.raises(_ReachedMarket):
        asyncio.run(place_guarded_order(
            _NoMarketClient(), ticker="KXCPI-26JUN-3", side="no", count=10,
            price=0.95, est_prob=0.97, governor=_Governor(), dry=True,
            policy_path=policy_path, override_policy=True))


def test_no_policy_file_allows_trade_through_to_market(tmp_path):
    # Backward-compatible: no saved policy -> no gate -> proceeds to the market.
    missing = str(tmp_path / "does_not_exist.json")
    with pytest.raises(_ReachedMarket):
        asyncio.run(place_guarded_order(
            _NoMarketClient(), ticker="KXCPI-26JUN-3", side="no", count=10,
            price=0.95, est_prob=0.97, governor=_Governor(), dry=True,
            policy_path=missing))


# ---------------------------------------------------------------------------
# H1: a HAIRCUT that erases edge must abort (not silently place a -EV order)
# ---------------------------------------------------------------------------

class _MarketClient:
    """Minimal happy-path fake: tradeable market + ample cash, dry-run friendly."""
    def __init__(self, no_ask=0.95, yes_ask=0.10, cash=100000):
        self._no_ask, self._yes_ask, self._cash = no_ask, yes_ask, cash
        self.placed = []
    async def get_market(self, ticker):
        return {"market": {"status": "active",
                           "yes_bid_dollars": 0.03, "yes_ask_dollars": self._yes_ask,
                           "no_bid_dollars": round(1 - self._yes_ask - 0.02, 2),
                           "no_ask_dollars": self._no_ask}}
    async def get_balance(self):
        return {"balance": self._cash, "portfolio_value": 0}
    async def place_order(self, **kw):
        self.placed.append(kw)
        return {"order": {"order_id": "oid", "fill_count": kw.get("count")}}


def _write_policy_full(tmp_path, blocks=None, warnings=None, haircuts=None):
    p = tmp_path / "edge_policy.json"
    p.write_text(json.dumps({"meta": {}, "blocks": blocks or [],
                             "warnings": warnings or [], "haircuts": haircuts or []}))
    return str(p)


_HAIRCUT_9599 = {"band": "[0.95,0.99]", "lo": 0.95, "hi": 0.99,
                 "shrink_to": 0.72, "n": 8, "gap": 0.25}


def test_haircut_that_erases_edge_aborts(tmp_path):
    # est_prob 0.97 shrinks to 0.72; NO ask is 0.95 -> edge 0.72-0.95 < 0 -> abort.
    policy_path = _write_policy_full(tmp_path, haircuts=[_HAIRCUT_9599])
    client = _MarketClient(no_ask=0.95)
    res = asyncio.run(place_guarded_order(
        client, ticker="KXX-1", side="no", count=10, price=None, est_prob=0.97,
        governor=_Governor(), dry=True, policy_path=policy_path))
    assert res["ok"] is False
    assert res["reason"] == "haircut_erased_edge"
    assert client.placed == []  # nothing placed


def test_haircut_that_preserves_edge_proceeds(tmp_path):
    # est_prob 0.97 shrinks to 0.72; buy at 0.60 -> edge 0.72-0.60 > 0 -> proceeds.
    policy_path = _write_policy_full(tmp_path, haircuts=[_HAIRCUT_9599])
    client = _MarketClient(no_ask=0.95)
    res = asyncio.run(place_guarded_order(
        client, ticker="KXX-1", side="no", count=10, price=0.60, est_prob=0.97,
        governor=_Governor(), dry=True, policy_path=policy_path))
    assert res["ok"] is True
    assert res["policy_note"]["est_prob_shrunk_to"] == 0.72
    assert res["edge"] == round(0.72 - 0.60, 4)


def test_override_lets_an_edge_erasing_haircut_through(tmp_path):
    policy_path = _write_policy_full(tmp_path, haircuts=[_HAIRCUT_9599])
    client = _MarketClient(no_ask=0.95)
    res = asyncio.run(place_guarded_order(
        client, ticker="KXX-1", side="no", count=10, price=None, est_prob=0.97,
        governor=_Governor(), dry=True, policy_path=policy_path, override_policy=True))
    assert res["ok"] is True
    assert res["edge"] < 0  # -EV, but the agent explicitly overrode
