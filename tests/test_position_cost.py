"""The position cap must bound the capital actually committed.

A Kalshi position is fully collateralised: a NO buy at p cents costs p per contract. The gate
previously priced NO orders off the YES leg (100 - p), understating a 95c fade by ~19x — enough
for a single NO position to pass a "10% of equity" cap at nearly 2x equity.
"""
from src.clients.kalshi_client import position_cost_dollars


def test_no_side_uses_the_price_actually_paid():
    # 1000 NO contracts at 95c commits $950, not the $50 the YES leg implies
    assert position_cost_dollars("no", 1000, no_price=95) == 950.0
    # a NO order with only a yes_price given has no NO price to size against
    assert position_cost_dollars("no", 1000, yes_price=5) is None


def test_yes_side_unchanged():
    assert position_cost_dollars("yes", 1000, yes_price=90) == 900.0


def test_the_old_yes_leg_mistake_is_what_it_was():
    # documents the trap: pricing NO off the YES leg looks cheap
    old_behaviour = 100 - 95
    assert position_cost_dollars("no", 1000, no_price=95) == 950.0
    assert old_behaviour / 100 * 1000 == 50.0


def test_missing_price_is_none_not_zero():
    assert position_cost_dollars("no", 10) is None
    assert position_cost_dollars("yes", 10) is None
