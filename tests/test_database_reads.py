"""Read-path tests for DatabaseManager: positions by column, strategy and LLM stats."""

import asyncio
from datetime import datetime, timedelta

import aiosqlite
import pytest

from src.utils.database import DatabaseManager, LLMQuery, Position, TradeLog


def _pos(**kw) -> Position:
    values = dict(
        market_id="M",
        side="YES",
        entry_price=0.4,
        quantity=3,
        timestamp=datetime.now(),
        rationale="r",
        confidence=0.7,
        live=False,
        status="open",
        strategy="s1",
        stop_loss_price=0.2,
        take_profit_price=0.8,
        max_hold_hours=24,
        target_confidence_change=0.1,
    )
    values.update(kw)
    return Position(**values)


def _trade(pnl, strategy) -> TradeLog:
    now = datetime.now()
    return TradeLog(
        market_id="M",
        side="YES",
        entry_price=0.4,
        exit_price=0.6,
        quantity=3,
        pnl=pnl,
        entry_timestamp=now - timedelta(hours=2),
        exit_timestamp=now,
        rationale="r",
        strategy=strategy,
    )


def _query(strategy, prompt="p", response="r", tokens=10, cost=0.01,
           confidence=0.6, timestamp=None) -> LLMQuery:
    return LLMQuery(
        timestamp=timestamp or datetime.now(),
        strategy=strategy,
        query_type="market_analysis",
        market_id="M",
        prompt=prompt,
        response=response,
        tokens_used=tokens,
        cost_usd=cost,
        confidence_extracted=confidence,
    )


async def _db(tmp_path) -> DatabaseManager:
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    return db


def test_get_open_positions_round_trip(tmp_path):
    async def run():
        db = await _db(tmp_path)
        await db.add_position(_pos())

        positions = await db.get_open_positions()
        assert len(positions) == 1
        p = positions[0]
        assert p.id is not None
        assert p.strategy == "s1"
        assert p.stop_loss_price == pytest.approx(0.2)
        assert p.take_profit_price == pytest.approx(0.8)
        assert p.max_hold_hours == 24
        assert p.target_confidence_change == pytest.approx(0.1)
        assert p.quantity == 3
        assert p.entry_price == pytest.approx(0.4)

    asyncio.run(run())


def test_get_open_positions_excludes_closed_and_includes_live(tmp_path):
    async def run():
        db = await _db(tmp_path)
        paper_id = await db.add_position(_pos(market_id="M-PAPER", live=False))
        live_id = await db.add_position(_pos(market_id="M-LIVE", live=True))
        closed_id = await db.add_position(_pos(market_id="M-CLOSED"))
        await db.update_position_status(closed_id, "closed")

        positions = await db.get_open_positions()
        assert {p.id for p in positions} == {paper_id, live_id}
        assert {p.live for p in positions} == {False, True}

    asyncio.run(run())


def test_get_performance_by_strategy_empty_db(tmp_path):
    async def run():
        db = await _db(tmp_path)
        assert await db.get_performance_by_strategy() == {}

    asyncio.run(run())


def test_get_performance_by_strategy_completed_trades(tmp_path):
    async def run():
        db = await _db(tmp_path)
        for pnl in (2.0, -1.0, 0.0):
            await db.add_trade_log(_trade(pnl, "a"))

        stats = (await db.get_performance_by_strategy())["a"]
        assert stats["completed_trades"] == 3
        assert stats["total_pnl"] == pytest.approx(1.0)
        assert stats["winning_trades"] == 1
        assert stats["losing_trades"] == 2
        assert stats["win_rate_pct"] == pytest.approx(33.33, rel=1e-3)
        assert stats["best_trade"] == pytest.approx(2.0)
        assert stats["worst_trade"] == pytest.approx(-1.0)
        assert stats["open_positions"] == 0

    asyncio.run(run())


def test_get_performance_by_strategy_open_position_only(tmp_path):
    async def run():
        db = await _db(tmp_path)
        await db.add_position(_pos(market_id="M-B", strategy="b", quantity=3, entry_price=0.4))

        stats = (await db.get_performance_by_strategy())["b"]
        assert stats["completed_trades"] == 0
        assert stats["open_positions"] == 1
        assert stats["capital_deployed"] == pytest.approx(1.2)

    asyncio.run(run())


def test_get_performance_by_strategy_merges_trades_and_positions(tmp_path):
    async def run():
        db = await _db(tmp_path)
        await db.add_trade_log(_trade(1.5, "c"))
        await db.add_position(_pos(market_id="M-C", strategy="c"))

        performance = await db.get_performance_by_strategy()
        assert set(performance) == {"c"}
        assert performance["c"]["completed_trades"] == 1
        assert performance["c"]["open_positions"] == 1

    asyncio.run(run())


def test_get_performance_by_strategy_ignores_null_strategy(tmp_path):
    async def run():
        db = await _db(tmp_path)
        await db.add_trade_log(_trade(1.0, None))
        assert await db.get_performance_by_strategy() == {}

    asyncio.run(run())


def test_log_and_get_llm_query(tmp_path):
    async def run():
        db = await _db(tmp_path)
        await db.log_llm_query(_query("x", prompt="prompt-1", response="response-1",
                                      tokens=10, cost=0.01))

        queries = await db.get_llm_queries()
        assert len(queries) == 1
        q = queries[0]
        assert q.strategy == "x"
        assert q.prompt == "prompt-1"
        assert q.response == "response-1"
        assert q.tokens_used == 10
        assert q.cost_usd == pytest.approx(0.01)
        assert isinstance(q.timestamp, datetime)

    asyncio.run(run())


def test_get_llm_queries_filters_and_limits(tmp_path):
    async def run():
        db = await _db(tmp_path)
        now = datetime.now()
        await db.log_llm_query(_query("x", prompt="older", timestamp=now - timedelta(minutes=1)))
        await db.log_llm_query(_query("x", prompt="newer", timestamp=now))
        await db.log_llm_query(_query("other", prompt="other-strategy", timestamp=now))

        only_x = await db.get_llm_queries(strategy="x")
        assert {q.strategy for q in only_x} == {"x"}
        assert len(only_x) == 2

        limited = await db.get_llm_queries(strategy="x", limit=1)
        assert len(limited) == 1
        assert limited[0].prompt == "newer"

    asyncio.run(run())


def test_get_llm_queries_hours_back(tmp_path):
    async def run():
        db = await _db(tmp_path)
        await db.log_llm_query(_query("x", prompt="fresh", timestamp=datetime.now()))
        await db.log_llm_query(_query("x", prompt="stale",
                                      timestamp=datetime.now() - timedelta(hours=48)))

        queries = await db.get_llm_queries(hours_back=24)
        assert [q.prompt for q in queries] == ["fresh"]

    asyncio.run(run())


def test_get_llm_stats_by_strategy(tmp_path):
    async def run():
        db = await _db(tmp_path)
        await db.log_llm_query(_query("x", tokens=10, cost=0.01, confidence=0.6))
        await db.log_llm_query(_query("x", tokens=20, cost=0.02, confidence=0.8))
        await db.log_llm_query(_query("y", timestamp=datetime.now() - timedelta(days=10)))

        stats = await db.get_llm_stats_by_strategy()
        assert "y" not in stats
        assert stats["x"]["query_count"] == 2
        assert stats["x"]["total_tokens"] == 30
        assert stats["x"]["total_cost"] == pytest.approx(0.03)
        assert stats["x"]["avg_confidence"] == pytest.approx(0.7)

    asyncio.run(run())


def test_llm_reads_survive_missing_table(tmp_path):
    async def run():
        db = await _db(tmp_path)
        async with aiosqlite.connect(str(tmp_path / "t.db")) as conn:
            await conn.execute("DROP TABLE llm_queries")
            await conn.commit()

        assert await db.get_llm_queries() == []
        assert await db.get_llm_stats_by_strategy() == {}

    asyncio.run(run())
