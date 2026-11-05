"""A halt-exempt sell goes out IOC, so only its filled part is a close (PR #107 P1).

place_sell_limit_order must return True only when the whole position filled. A
partial fill logs the sold part and leaves the remainder open in the DB; no fill
leaves the position untouched. Kalshi is mocked; the DB is a temp SQLite file.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from src.jobs.execute import place_sell_limit_order
from src.utils.database import DatabaseManager, Position


def _run(tmp_path, response, quantity=10):
    async def go():
        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.initialize()
        pos = Position(market_id="IOC-1", side="YES", entry_price=0.40, quantity=quantity,
                       timestamp=datetime.now(), rationale="r", live=True, status="open")
        pos.id = await db.add_position(pos)
        client = AsyncMock()
        client.place_order = AsyncMock(return_value=response)
        # the sellable guard reads the exchange: we hold the full position, nothing resting
        client._held_on_side = AsyncMock(return_value=float(quantity))
        client.get_orders = AsyncMock(return_value={"orders": []})
        ok = await place_sell_limit_order(position=pos, limit_price=0.30,
                                          db_manager=db, kalshi_client=client)
        after = await db.get_position_by_market_id("IOC-1")
        logs = await db.get_all_trade_logs()
        return ok, after, logs
    return asyncio.run(go())


def _ioc(**order):
    return {"order": {"order_id": "o1", **order}, "reduce_only_ioc": True}


def test_ioc_full_fill_is_a_close(tmp_path):
    ok, after, logs = _run(tmp_path, _ioc(fill_count_fp="10.00"))
    assert ok is True
    assert logs == []  # the caller (track.py) logs the full close


def test_ioc_partial_fill_keeps_remainder_open(tmp_path):
    ok, after, logs = _run(tmp_path, _ioc(fill_count_fp="4.00", remaining_count_fp="0.00"))
    assert ok is False
    assert after is not None and after.status == "open"
    assert after.quantity == pytest.approx(6)
    assert len(logs) == 1 and logs[0].quantity == pytest.approx(4)
    assert logs[0].pnl == pytest.approx((0.30 - 0.40) * 4)


@pytest.mark.parametrize("order", [{"fill_count_fp": "0.00"}, {}, {"fill_count": "junk"},
                                   {"fill_count_fp": "nan"}, {"fill_count": -3}])
def test_ioc_no_or_unreadable_fill_leaves_position_untouched(tmp_path, order):
    ok, after, logs = _run(tmp_path, _ioc(**order))
    assert ok is False
    assert after.status == "open" and after.quantity == 10
    assert logs == []


def test_ioc_overreported_fill_is_capped_at_position(tmp_path):
    ok, after, logs = _run(tmp_path, _ioc(fill_count="12"))
    assert ok is True


def test_non_ioc_resting_sell_keeps_existing_behaviour(tmp_path):
    ok, after, logs = _run(tmp_path, {"order": {"order_id": "o1", "fill_count_fp": "0.00"}})
    assert ok is True  # GTC sell rests; unchanged pre-existing semantics


def test_place_order_flags_halt_exempt_response():
    from tests.test_reduce_only_exit import LIMIT_HALT, _client, _gov, _place, _pos
    client = _client(positions=_pos("5.00"))
    with _gov(LIMIT_HALT):
        resp = _place(client, count=5)
    assert resp["reduce_only_ioc"] is True


def test_place_order_does_not_flag_normal_response():
    from tests.test_reduce_only_exit import _client, _gov, _place
    from src.risk.risk_governor import GovernorDecision
    ok = GovernorDecision(halted=False, reasons=[], current_equity_cents=10_000,
                          sod_equity_cents=10_000, peak_equity_cents=10_000)
    client = _client()
    with _gov(ok):
        resp = _place(client, count=5)
    assert "reduce_only_ioc" not in resp


def test_partial_exit_is_atomic_when_quantity_write_fails(tmp_path):
    """If shrinking the position fails, the sale log must not survive alone."""
    import aiosqlite

    async def go():
        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.initialize()
        pos = Position(market_id="IOC-1", side="YES", entry_price=0.40, quantity=10,
                       timestamp=datetime.now(), rationale="r", live=True, status="open")
        pos.id = await db.add_position(pos)
        async with aiosqlite.connect(db.db_path) as conn:
            await conn.execute("""
                CREATE TRIGGER fail_qty BEFORE UPDATE OF quantity ON positions
                BEGIN SELECT RAISE(ABORT, 'injected'); END
            """)
            await conn.commit()
        client = AsyncMock()
        client.place_order = AsyncMock(return_value=_ioc(fill_count_fp="4.00"))
        ok = await place_sell_limit_order(position=pos, limit_price=0.30,
                                          db_manager=db, kalshi_client=client)
        return (ok, await db.get_position_by_market_id("IOC-1"),
                await db.get_all_trade_logs())

    ok, after, logs = asyncio.run(go())
    assert ok is False
    assert after.quantity == 10
    assert logs == []
