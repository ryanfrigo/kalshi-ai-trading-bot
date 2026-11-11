"""The market-making strategy honours the configured position limit.

``execute_market_making_strategy`` read ``settings.trading.max_concurrent_markets``,
which does not exist, so every run used the hard-coded fallback of 10 and ignored
``max_positions`` (#96).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.config.settings import settings
from src.strategies.market_making import AdvancedMarketMaker


def _opportunity(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        market_id=f"m{index}",
        market_title=f"Market {index}",
        optimal_yes_size=10,
        optimal_no_size=10,
        total_expected_profit=1.0,
    )


def test_execute_market_making_strategy_uses_max_positions(monkeypatch):
    mm = AdvancedMarketMaker(None, None, None)  # type: ignore[arg-type]
    monkeypatch.setattr(settings.trading, "max_positions", 2)

    placed = []

    async def place(opportunity):
        placed.append(opportunity.market_id)

    monkeypatch.setattr(mm, "_place_market_making_orders", place)

    result = asyncio.run(
        mm.execute_market_making_strategy([_opportunity(i) for i in range(5)])
    )

    assert placed == ["m0", "m1"]
    assert result["markets_count"] == 2
