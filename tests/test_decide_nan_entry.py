"""A NaN/inf entry price must make make_decision_for_market skip the market
(return None, no Position), never crash the run or open a trade without a stop."""
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.jobs.decide import make_decision_for_market
from src.utils.database import Market

NAN = float("nan")
INF = float("inf")


def _market(yes, no, hours=2):
    return Market(market_id="NAN-1", title="t", yes_price=yes, no_price=no, volume=100_000,
                  expiration_ts=int(time.time() + hours * 3600), category="politics",
                  status="active", last_updated=datetime.now())


def _db():
    db = MagicMock()
    db.get_daily_ai_cost = AsyncMock(return_value=0.0)
    db.was_recently_analyzed = AsyncMock(return_value=False)
    db.get_market_analysis_count_today = AsyncMock(return_value=0)
    db.record_market_analysis = AsyncMock()
    db.upsert_daily_cost = AsyncMock()
    return db


def _clients(side, confidence=0.99):
    xai = MagicMock()
    xai.get_trading_decision = AsyncMock(return_value=SimpleNamespace(
        action="BUY", side=side, confidence=confidence, limit_price=50, reasoning="r"))
    xai.search = AsyncMock(return_value="news")
    kalshi = MagicMock()
    kalshi.get_balance = AsyncMock(return_value={"balance": 1_000_000})
    kalshi.get_market = AsyncMock(return_value={"market": {"rules": "r"}})
    return xai, kalshi


@pytest.fixture(autouse=True)
def gates_open():
    with patch("src.utils.position_limits.check_can_add_position", AsyncMock(return_value=(True, "ok"))), \
         patch("src.utils.cash_reserves.check_can_trade_with_cash_reserves", AsyncMock(return_value=(True, "ok"))):
        yield


@pytest.mark.parametrize("bad", [NAN, INF])
async def test_near_expiry_yes_path_skips_on_non_finite_entry(bad):
    """decide.py near-expiry YES branch (stop calc at the first call site)."""
    xai, kalshi = _clients("YES")
    assert await make_decision_for_market(_market(bad, 0.1), _db(), xai, kalshi) is None


@pytest.mark.parametrize("side,yes,no", [("YES", NAN, 0.5), ("NO", 0.5, NAN), ("YES", 0.5, INF)])
async def test_standard_path_skips_on_non_finite_entry(side, yes, no):
    xai, kalshi = _clients(side)
    assert await make_decision_for_market(_market(yes, no, hours=500), _db(), xai, kalshi) is None


async def test_forced_edge_approval_of_nan_entry_still_skips():
    """Force the edge filter to approve a NaN entry: sizing/stop math raises and the
    function's outer except turns it into a skipped market (None), not a crash."""
    xai, kalshi = _clients("YES")
    db = _db()
    with patch("src.utils.edge_filter.EdgeFilter.should_trade_market",
               return_value=(True, "forced", None)):
        out = await make_decision_for_market(_market(NAN, 0.5, hours=500), db, xai, kalshi)
    assert out is None


async def test_run_continues_after_a_skipped_nan_market():
    xai, kalshi = _clients("YES")
    assert await make_decision_for_market(_market(NAN, NAN, hours=500), _db(), xai, kalshi) is None
    assert await make_decision_for_market(_market(NAN, NAN, hours=500), _db(), xai, kalshi) is None
