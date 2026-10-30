"""Market-making sizing must never use more capital than the 1000 cap (issue #96)."""

from unittest.mock import MagicMock

import pytest

from src.strategies import market_making
from src.strategies.market_making import (
    MARKET_MAKING_CAPITAL_CAP,
    AdvancedMarketMaker,
)


@pytest.fixture
def maker():
    return AdvancedMarketMaker(MagicMock(), MagicMock(), MagicMock())


def test_capital_cap_is_1000():
    assert MARKET_MAKING_CAPITAL_CAP == 1000


def test_no_getattr_on_nonexistent_setting():
    # settings.trading has no max_position_size, so the old getattr always
    # fell back to 1000. The sizing code must not look that name up.
    source = open(market_making.__file__).read()
    assert "'max_position_size'" not in source


def test_max_edge_sizes_match_pre_fix_values(maker):
    # Pre-fix behaviour: capital 1000 * kelly cap 0.25 = 250 for the YES side.
    yes_size, no_size = maker._calculate_optimal_sizes(
        yes_edge=1.0, no_edge=1.0, volatility=0.05, confidence=1.0
    )
    assert yes_size == 250
    assert no_size == 250


def test_unfavourable_edge_sizes_match_pre_fix_values(maker):
    # Pre-fix behaviour: 1000 * 0.05 = 50 when there is no edge.
    yes_size, no_size = maker._calculate_optimal_sizes(
        yes_edge=-0.1, no_edge=-0.1, volatility=0.05, confidence=0.5
    )
    assert yes_size == 50
    assert no_size == 50


@pytest.mark.parametrize("yes_edge,no_edge,confidence", [
    (1.0, 1.0, 1.0),
    (5.0, 5.0, 10.0),
    (-1.0, -1.0, 0.0),
    (0.0, 0.0, 0.5),
])
def test_sizes_never_exceed_cap(maker, yes_edge, no_edge, confidence):
    yes_size, no_size = maker._calculate_optimal_sizes(
        yes_edge=yes_edge, no_edge=no_edge, volatility=0.05, confidence=confidence
    )
    assert 0 < yes_size <= MARKET_MAKING_CAPITAL_CAP
    assert 0 < no_size <= MARKET_MAKING_CAPITAL_CAP
