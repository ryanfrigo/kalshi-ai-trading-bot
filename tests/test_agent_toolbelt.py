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
