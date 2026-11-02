"""DB spend total = actual request costs, written before the call returns (PR #107 P2s).

P2a: a short job (asyncio.run) must not lose its last spend row to task cancellation.
P2b: record_market_analysis must not add decide.py's estimate on top of the client's
actual cost. Offline: the model request is mocked; the DB is a temp SQLite file.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from src.clients.openrouter_client import OpenRouterClient
from src.utils.database import DatabaseManager


def _client(tmp_path, db):
    c = OpenRouterClient(api_key="test-key", db_manager=db)
    c.usage_file = str(tmp_path / "usage.pkl")
    c._request_single_model = AsyncMock(return_value=("ok", 0.25, 10, 10))
    c._log_query = AsyncMock()
    return c


async def _db(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    return db


def test_completion_spend_is_in_db_when_short_job_exits(tmp_path):
    async def job():
        db = await _db(tmp_path)
        c = _client(tmp_path, db)
        assert await c.get_completion("hi") == "ok"
        return db  # no close(), no gather: asyncio.run cancels leftovers

    db = asyncio.run(job())
    assert asyncio.run(db.get_daily_ai_cost()) == 0.25


def test_close_drains_pending_spend_writes(tmp_path):
    async def job():
        db = await _db(tmp_path)
        c = _client(tmp_path, db)
        c._update_daily_cost(0.4)  # schedules the DB write
        await c.close()
        return db

    db = asyncio.run(job())
    assert asyncio.run(db.get_daily_ai_cost()) == 0.4


def test_record_market_analysis_does_not_double_count(tmp_path):
    async def job():
        db = await _db(tmp_path)
        c = _client(tmp_path, db)
        await c.get_completion("hi")  # actual cost 0.25 recorded by the client
        await db.record_market_analysis("M-1", "BUY", 0.7, 0.035)  # decide.py estimate
        await db.record_market_analysis("M-2", "SKIP", 0.0, 0.02)
        return db, await db.get_daily_ai_cost()

    db, total = asyncio.run(job())
    assert total == 0.25


def test_record_market_analysis_still_counts_analyses(tmp_path):
    import aiosqlite

    async def job():
        db = await _db(tmp_path)
        await db.record_market_analysis("M-1", "BUY", 0.7, 0.035)
        await db.record_market_analysis("M-2", "SKIP", 0.0, 0.02)
        async with aiosqlite.connect(db.db_path) as conn:
            cur = await conn.execute(
                "SELECT analysis_count, decision_count FROM daily_cost_tracking")
            return await cur.fetchone(), await db.was_recently_analyzed("M-1")

    (analyses, decisions), recent = asyncio.run(job())
    assert (analyses, decisions) == (2, 1) and recent


def _market(mid):
    from unittest.mock import MagicMock
    import time
    return MagicMock(market_id=mid, title="t", volume=10**6, category="politics",
                     expiration_ts=time.time() + 30 * 86400, yes_price=0.5, no_price=0.5)


@pytest.mark.parametrize("fails", [False, True])
def test_sentiment_spend_counted_once_and_trips_the_budget(tmp_path, monkeypatch, fails):
    """Sentiment (own OpenRouter client) X + decision (client upsert) Y = X + Y, once."""
    import sys
    import types
    from unittest.mock import MagicMock
    from src.config.settings import settings
    from src.jobs.decide import make_decision_for_market

    class FakeAnalyzer:
        def __init__(self):
            self.total_cost = 0.0

        async def get_market_sentiment_summary(self, title):
            self.total_cost = 0.3
            if fails:
                raise RuntimeError("article call failed after spending")
            return "news"

    fake = types.ModuleType("src.data.sentiment_analyzer")
    fake.SentimentAnalyzer = FakeAnalyzer  # the real one needs feedparser, not in dev deps
    monkeypatch.setitem(sys.modules, "src.data.sentiment_analyzer", fake)
    monkeypatch.setattr(settings.sentiment, "enabled", True)
    monkeypatch.setattr(settings.trading, "skip_news_for_low_volume", False)
    monkeypatch.setattr(settings.trading, "enable_high_confidence_strategy", False)
    monkeypatch.setattr(settings.trading, "max_ai_cost_per_decision", 10.0)
    monkeypatch.setattr(settings.trading, "daily_ai_budget", 0.5)

    async def job():
        db = await _db(tmp_path)
        xai = MagicMock()
        xai.search = AsyncMock(return_value="fallback news")

        async def decision(**kw):
            await db.upsert_daily_cost(0.2)  # what OpenRouterClient records
            return None
        xai.get_trading_decision = AsyncMock(side_effect=decision)
        kalshi = MagicMock()
        kalshi.get_balance = AsyncMock(return_value={"balance": 100000})
        kalshi.get_market = AsyncMock(return_value={"market": {}})

        assert await make_decision_for_market(_market("M-1"), db, xai, kalshi) is None
        total = await db.get_daily_ai_cost()
        calls = kalshi.get_balance.await_count
        assert await make_decision_for_market(_market("M-2"), db, xai, kalshi) is None
        return total, calls, kalshi.get_balance.await_count

    total, before, after = asyncio.run(job())
    assert total == pytest.approx(0.5)
    assert after == before  # M-2 stopped at the budget check, before any spend


def test_cancelled_request_does_not_cancel_its_spend_write(tmp_path):
    from unittest.mock import MagicMock

    async def job():
        release, written = asyncio.Event(), []
        db = MagicMock()

        async def upsert(cost):
            await release.wait()
            written.append(cost)
        db.upsert_daily_cost = upsert
        c = _client(tmp_path, db)
        req = asyncio.create_task(c.get_completion("hi"))
        await asyncio.sleep(0.05)  # request is now waiting on the flush
        req.cancel()
        with pytest.raises(asyncio.CancelledError):
            await req
        release.set()
        await c.close()
        return written

    assert asyncio.run(job()) == [0.25]
