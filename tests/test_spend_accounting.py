"""DB spend total = actual request costs, written before the call returns (PR #107 P2s).

P2a: a short job (asyncio.run) must not lose its last spend row to task cancellation.
P2b: record_market_analysis must not add decide.py's estimate on top of the client's
actual cost. Offline: the model request is mocked; the DB is a temp SQLite file.
"""
import asyncio
from unittest.mock import AsyncMock

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
