"""AI spend cap fails closed. Fully offline: no network, no LLM calls."""
import asyncio
import pickle
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.clients.llm_types import DailyUsageTracker
from src.clients.openrouter_client import OpenRouterClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.jobs.decide import make_decision_for_market

TODAY = datetime.now().strftime("%Y-%m-%d")
YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def _make(cls, tmp_path, **kw):
    c = cls(api_key="test-key", **kw) if cls is OpenRouterClient else cls(api_key="test-key", **kw)
    c.usage_file = str(tmp_path / "usage.pkl")
    return c


@pytest.fixture(params=[OpenRouterClient, XAIClient])
def client(request, tmp_path):
    return _make(request.param, tmp_path)


def test_missing_file_is_fresh_day(client):
    assert not client._load_daily_tracker().is_exhausted


def test_corrupt_file_fails_closed(client):
    with open(client.usage_file, "wb") as fh:
        fh.write(b"\x00not a pickle")
    t = client._load_daily_tracker()
    assert t.is_exhausted and t.date == TODAY
    assert t.daily_limit == getattr(settings.trading, "daily_ai_cost_limit", t.daily_limit)


def test_wrong_type_fails_closed(client):
    with open(client.usage_file, "wb") as fh:
        pickle.dump({"date": TODAY}, fh)
    assert client._load_daily_tracker().is_exhausted


def test_old_date_file_is_fresh(client):
    with open(client.usage_file, "wb") as fh:
        pickle.dump(DailyUsageTracker(date=YESTERDAY, total_cost=99.0, is_exhausted=True), fh)
    t = client._load_daily_tracker()
    assert not t.is_exhausted and t.total_cost == 0.0 and t.date == TODAY


def test_same_date_file_unchanged(client):
    with open(client.usage_file, "wb") as fh:
        pickle.dump(DailyUsageTracker(date=TODAY, total_cost=1.25, request_count=3), fh)
    t = client._load_daily_tracker()
    assert t.total_cost == 1.25 and t.request_count == 3 and not t.is_exhausted


@pytest.mark.asyncio
async def test_exhausted_blocks_calls(client):
    with open(client.usage_file, "wb") as fh:
        fh.write(b"garbage")
    assert await client._check_daily_limits() is False


@pytest.mark.asyncio
async def test_update_daily_cost_records_to_db(tmp_path):
    db = MagicMock()
    db.upsert_daily_cost = AsyncMock()
    c = _make(OpenRouterClient, tmp_path, db_manager=db)
    c._update_daily_cost(0.5)
    await asyncio.gather(*list(c._db_cost_tasks))
    db.upsert_daily_cost.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_db_failure_does_not_skip_pickle_update(tmp_path):
    db = MagicMock()
    db.upsert_daily_cost = AsyncMock(side_effect=RuntimeError("db down"))
    c = _make(OpenRouterClient, tmp_path, db_manager=db)
    c._update_daily_cost(0.5)
    await asyncio.gather(*list(c._db_cost_tasks))
    assert c.daily_tracker.total_cost == 0.5
    with open(c.usage_file, "rb") as fh:
        assert pickle.load(fh).total_cost == 0.5


@pytest.mark.asyncio
async def test_decide_blocks_when_db_budget_exceeded():
    db = MagicMock()
    db.get_daily_ai_cost = AsyncMock(return_value=settings.trading.daily_ai_budget)
    xai = MagicMock()
    xai.get_completion = AsyncMock()
    xai.get_trading_decision = AsyncMock()
    kalshi = MagicMock()
    kalshi.get_balance = AsyncMock()
    market = MagicMock(market_id="M", title="t")

    assert await make_decision_for_market(market, db, xai, kalshi) is None
    xai.get_completion.assert_not_called()
    xai.get_trading_decision.assert_not_called()
    kalshi.get_balance.assert_not_called()


@pytest.mark.parametrize("cls", [OpenRouterClient, XAIClient])
@pytest.mark.asyncio
async def test_corrupt_refuses_across_instances_until_file_removed(cls, tmp_path):
    first = _make(cls, tmp_path)
    with open(first.usage_file, "wb") as fh:
        fh.write(b"corrupt")
    assert await first._check_daily_limits() is False  # call 1 refuses

    second = _make(cls, tmp_path)  # fresh instance, same file
    assert await second._check_daily_limits() is False  # still refuses
    with open(second.usage_file, "rb") as fh:
        persisted = pickle.load(fh)
    assert persisted.is_exhausted and persisted.date == TODAY

    import os
    os.remove(second.usage_file)
    third = _make(cls, tmp_path)
    assert await third._check_daily_limits() is True  # missing = fresh day


def test_update_daily_cost_without_event_loop_logs_and_keeps_pickle(tmp_path):
    db = MagicMock()
    db.upsert_daily_cost = AsyncMock()
    c = _make(OpenRouterClient, tmp_path, db_manager=db)
    c.logger = MagicMock()

    c._update_daily_cost(0.75)  # sync caller, no running loop

    assert c.daily_tracker.total_cost == 0.75
    with open(c.usage_file, "rb") as fh:
        assert pickle.load(fh).total_cost == 0.75
    c.logger.error.assert_called_once()
    assert "DB cost update" in c.logger.error.call_args[0][0]
    db.upsert_daily_cost.assert_not_called()
