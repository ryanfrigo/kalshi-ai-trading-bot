"""Tests for fractional contract counts in the learnings P&L reconstruction.

Kalshi counts can be fractional ("2.50"); the per-contract reconstruction must
not truncate them to an int. Pure-function tests (no network, no IO).
"""
import pytest

from src.agent.learnings import _pnl_from_my_side


def test_fractional_count_win():
    record = {"side": "no", "count": 2.5, "price": 0.40}
    assert _pnl_from_my_side(record, {"held_side": "yes"}, True) == pytest.approx(1.5)


def test_fractional_count_loss():
    record = {"side": "no", "count": 2.5, "price": 0.40}
    assert _pnl_from_my_side(record, {"held_side": "yes"}, False) == pytest.approx(-1.0)


def test_integer_count_regression():
    record = {"side": "no", "count": 3, "price": 0.40}
    assert _pnl_from_my_side(record, {"held_side": "yes"}, True) == pytest.approx(1.8)


def test_matching_held_side_returns_settlement_pnl():
    record = {"side": "no"}
    settlement = {"held_side": "no", "pnl": "0.73"}
    assert _pnl_from_my_side(record, settlement, True) == pytest.approx(0.73)
