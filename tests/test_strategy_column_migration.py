"""A DB created before positions/trade_logs had a strategy column gets one on initialize."""
import asyncio
import sqlite3
from datetime import datetime

from src.utils.database import DatabaseManager, Position, TradeLog


def _columns(path, table):
    with sqlite3.connect(path) as conn:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_old_db_without_strategy_columns_migrates(tmp_path):
    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as conn:  # both tables as created before strategy existed
        conn.execute("""CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, market_id TEXT NOT NULL,
            side TEXT NOT NULL, entry_price REAL NOT NULL, quantity INTEGER NOT NULL,
            timestamp TEXT NOT NULL, rationale TEXT, confidence REAL,
            live BOOLEAN NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'open',
            UNIQUE(market_id, side))""")
        conn.execute("""CREATE TABLE trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, market_id TEXT NOT NULL,
            side TEXT NOT NULL, entry_price REAL NOT NULL, exit_price REAL NOT NULL,
            quantity INTEGER NOT NULL, pnl REAL NOT NULL, entry_timestamp TEXT NOT NULL,
            exit_timestamp TEXT NOT NULL, rationale TEXT)""")
        conn.execute("INSERT INTO positions (market_id, side, entry_price, quantity, timestamp, "
                     "rationale, live, status) VALUES ('SX-1', 'YES', 0.4, 10, ?, 'QUICK FLIP: x', "
                     "1, 'open')", (datetime.now().isoformat(),))

    async def go():
        db = DatabaseManager(db_path=path)
        await db.initialize()
        await db.initialize()  # idempotent
        (pos,) = await db.get_open_live_positions()
        now = datetime.now()
        # both inserts bind :strategy; they failed on this schema before the migration
        await db.record_partial_exit(pos.id, TradeLog(
            market_id="SX-1", side="YES", entry_price=0.4, exit_price=0.6, quantity=4,
            pnl=0.8, entry_timestamp=now, exit_timestamp=now, rationale="r"), 6)
        await db.add_trade_log(TradeLog(
            market_id="SX-1", side="YES", entry_price=0.4, exit_price=0.6, quantity=6,
            pnl=1.2, entry_timestamp=now, exit_timestamp=now, rationale="r"))
        return pos, await db.get_all_trade_logs()

    pos, logs = asyncio.run(go())
    assert _columns(path, "positions").count("strategy") == 1
    assert _columns(path, "trade_logs").count("strategy") == 1
    assert pos.strategy is None  # no backfill from the rationale
    assert [log.quantity for log in logs] == [4, 6]
