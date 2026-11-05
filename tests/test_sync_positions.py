"""scripts/sync_positions.py on fakes and a temp database. It replaces local position
rows, so nothing may be deleted until every read has worked. The fake has no order methods."""
import sqlite3

import pytest

from scripts import sync_positions as sp
from src.utils.database import DatabaseManager


def _row(ticker, fp="5.00"):
    return {"ticker": ticker, "position_fp": fp}


MARKET = {"market": {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.44",
                     "no_bid_dollars": "0.56", "no_ask_dollars": "0.60"}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A temp database holding one old row, and a fake client serving `world.pages`."""
    db_path = str(tmp_path / "t.db")

    class World:
        pages = {None: {"market_positions": [], "cursor": ""}}
        markets = {}
        closed = 0

    class Db(DatabaseManager):
        def __init__(self):
            super().__init__(db_path=db_path)

    class Fake:
        async def get_positions(self, cursor=None, **_):
            reply = World.pages[cursor]
            if isinstance(reply, Exception):
                raise reply
            return reply

        async def get_market(self, ticker):
            reply = World.markets.get(ticker, MARKET)
            if isinstance(reply, Exception):
                raise reply
            return reply

        async def close(self):
            World.closed += 1

    monkeypatch.setattr(sp, "KalshiClient", Fake)
    monkeypatch.setattr(sp, "DatabaseManager", Db)
    World.db_path = db_path
    return World


async def _seed(path):
    db = DatabaseManager(db_path=path)
    await db.initialize()
    con = sqlite3.connect(path)
    con.execute("INSERT INTO positions (market_id, side, entry_price, quantity, timestamp, status, "
                "live, stop_loss_price) VALUES ('OLD', 'YES', 0.33, 7, '2026-01-01', 'open', 1, 0.2)")
    con.execute("INSERT INTO markets (market_id, title, yes_price, no_price, volume, expiration_ts, "
                "category, status, last_updated, has_position) VALUES "
                "('NEW1', 't', 0.5, 0.5, 1, 1, 'c', 'active', '2026-01-01', 0)")
    con.commit()
    con.close()


def _rows(path):
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT market_id, side, quantity FROM positions ORDER BY market_id").fetchall()
    finally:
        con.close()


OLD_ONLY = [("OLD", "YES", 7)]


async def test_sync_replaces_rows_with_every_page(world):
    await _seed(world.db_path)
    world.pages = {None: {"market_positions": [_row("NEW1")], "cursor": "next"},
                   "next": {"market_positions": [_row("NEW2", "-3.00"), _row("FLAT", "0")], "cursor": ""}}
    assert await sp.sync_positions_to_database() is True
    assert _rows(world.db_path) == [("NEW1", "YES", 5), ("NEW2", "NO", 3)]
    con = sqlite3.connect(world.db_path)
    assert con.execute("SELECT has_position FROM markets WHERE market_id='NEW1'").fetchone() == (1,)
    con.close()
    assert world.closed == 1


@pytest.mark.parametrize("second_page", [ConnectionError("down"), {}, {"market_positions": None}])
async def test_a_bad_later_page_deletes_nothing(world, second_page):
    await _seed(world.db_path)
    world.pages = {None: {"market_positions": [_row("NEW1")], "cursor": "next"}, "next": second_page}
    assert await sp.sync_positions_to_database() is False
    assert _rows(world.db_path) == OLD_ONLY


@pytest.mark.parametrize("reply", [{}, None, [], {"market_positions": None}, {"market_positions": "x"}])
async def test_an_unreadable_first_reply_deletes_nothing(world, reply):
    await _seed(world.db_path)
    world.pages = {None: reply}
    assert await sp.sync_positions_to_database() is False
    assert _rows(world.db_path) == OLD_ONLY


@pytest.mark.parametrize("market", [ConnectionError("down"), {}, None])
async def test_a_failed_market_read_deletes_nothing(world, market):
    await _seed(world.db_path)
    world.pages = {None: {"market_positions": [_row("NEW1"), _row("NEW2")], "cursor": ""}}
    world.markets = {"NEW2": market}
    assert await sp.sync_positions_to_database() is False
    assert _rows(world.db_path) == OLD_ONLY


@pytest.mark.parametrize("row", [{"position_fp": "5.00"}, {"ticker": "X", "position_fp": "abc"}])
async def test_an_unreadable_row_deletes_nothing(world, row):
    await _seed(world.db_path)
    world.pages = {None: {"market_positions": [_row("NEW1"), row], "cursor": ""}}
    assert await sp.sync_positions_to_database() is False
    assert _rows(world.db_path) == OLD_ONLY


async def test_a_failed_insert_rolls_the_delete_back(world):
    await _seed(world.db_path)
    con = sqlite3.connect(world.db_path)
    con.execute("CREATE TRIGGER boom BEFORE INSERT ON positions WHEN NEW.market_id = 'NEW2' "
                "BEGIN SELECT RAISE(ABORT, 'boom'); END")
    con.commit()
    con.close()
    world.pages = {None: {"market_positions": [_row("NEW1"), _row("NEW2")], "cursor": ""}}
    assert await sp.sync_positions_to_database() is False
    assert _rows(world.db_path) == OLD_ONLY


async def test_an_empty_list_really_does_clear_the_table(world):
    await _seed(world.db_path)
    assert await sp.sync_positions_to_database() is True
    assert _rows(world.db_path) == []


async def test_verify_counts_every_page(world, capsys):
    await _seed(world.db_path)
    world.pages = {None: {"market_positions": [_row("NEW1")], "cursor": "next"},
                   "next": {"market_positions": [_row("NEW2")], "cursor": ""}}
    await sp.sync_positions_to_database()
    capsys.readouterr()
    await sp.verify_sync()
    out = capsys.readouterr().out
    assert "Kalshi active positions: 2" in out
    assert "counts match" in out


async def test_main_reports_failure(world):
    await _seed(world.db_path)
    world.pages = {None: {}}
    assert await sp.main() is False
    world.pages = {None: {"market_positions": [], "cursor": ""}}
    assert await sp.main() is True
