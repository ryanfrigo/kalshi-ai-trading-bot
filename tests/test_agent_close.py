"""Tests for resolve_sell — the guard that closes never sell more than held,
and always sell the side actually held (YES if position_fp>0 else NO)."""
from src.agent.toolbelt import resolve_sell


def test_sell_all_yes():
    assert resolve_sell(105.0, None) == ("yes", 105)


def test_sell_all_no():
    assert resolve_sell(-49.0, None) == ("no", 49)


def test_sell_partial():
    assert resolve_sell(105.0, 50) == ("yes", 50)


def test_sell_caps_to_held():
    assert resolve_sell(105.0, 500) == ("yes", 105)


def test_sell_flat_position_is_noop():
    assert resolve_sell(0.0, None) == (None, 0)


def test_sell_rounds_fractional_holding():
    assert resolve_sell(-48.7, None) == ("no", 49)
