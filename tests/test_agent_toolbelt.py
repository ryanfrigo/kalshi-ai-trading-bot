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
