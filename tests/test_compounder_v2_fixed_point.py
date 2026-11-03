from unittest.mock import AsyncMock, MagicMock

import pytest

from src.strategies.safe_compounder import SafeCompounder, market_confidence_score


def test_confidence_keeps_fixed_point_sizes():
    m = {"volume_fp": "5000.00", "_days_to_expiry": 2, "last_price_dollars": "0.0400"}
    fp = {"yes_dollars": [["0.0300", "100.00"]], "no_dollars": [["0.9500", "100.00"]]}
    whole = {"yes_dollars": [["0.0300", "100"]], "no_dollars": [["0.9500", "100"]]}
    s_fp, r_fp = market_confidence_score("T", fp, m)
    s_whole, r_whole = market_confidence_score("T", whole, m)
    assert "no book" not in r_fp
    assert s_fp == s_whole


@pytest.mark.asyncio
async def test_resting_order_reports_v2_fill_count_fp(monkeypatch):
    c = MagicMock()
    c.get_positions = AsyncMock(return_value={"market_positions": [], "cursor": ""})
    c.get_orders = AsyncMock(return_value={"orders": [], "cursor": ""})
    c.place_order = AsyncMock(return_value={"order": {
        "order_id": "x", "fill_count_fp": "5.00", "status": "executed"}})
    sc = SafeCompounder(client=c, dry_run=False, use_kelly=False)
    opp = {"ticker": "T-1", "our_price": 0.90, "true_no_prob": 0.97, "profit": 0.10,
           "edge": 0.05, "roi_pct": 11.1, "annualized_roi": 100, "title": "t",
           "lowest_no_ask": 0.91, "days_to_expiry": 2}
    monkeypatch.setattr("src.strategies.safe_compounder.asyncio.sleep", AsyncMock())
    stats = await sc._place_resting_orders([opp], portfolio=100_000, cash=100_000)
    assert stats["placed"] == 1 and stats["filled"] == 5
