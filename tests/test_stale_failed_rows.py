"""Stale failed-buy rows: open live=0 rows older than 10 minutes are reconciled.

The pass is GET-only. A flat exchange voids the row; a holding or resting order
leaves it open with one ERROR line; a read failure leaves it open. The client
is a real KalshiClient with its HTTP layer replaced by a recorder, so the
"no non-GET request" check runs against the real get_positions/get_orders.
"""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.kalshi_client import KalshiClient
from src.jobs.execute import reconcile_stale_failed_rows
from src.utils.database import DatabaseManager, Position

from tests.test_strategies_unified_immediate import make_system, FakeKalshi, legacy_market


def run(coro):
    return asyncio.run(coro)


class Exchange:
    """Recording stand-in for KalshiClient._make_authenticated_request."""

    def __init__(self, positions=None, orders=None, fail_on=None):
        self.positions = positions or []
        self.orders = orders or []
        self.fail_on = fail_on
        self.requests = []

    async def __call__(self, method, path, **kw):
        self.requests.append((method, path))
        if self.fail_on and self.fail_on in path:
            raise RuntimeError("503")
        if path.endswith("/portfolio/positions"):
            return {"market_positions": self.positions}
        if path.endswith("/portfolio/orders"):
            return {"orders": self.orders}
        raise AssertionError(f"unexpected request {method} {path}")


def client(ex):
    c = object.__new__(KalshiClient)  # no credentials, no network
    c._make_authenticated_request = ex
    return c


@pytest.fixture
def db(tmp_path):
    d = DatabaseManager(db_path=str(tmp_path / "t.db"))
    run(d.initialize())
    return d


def add_row(db, age_min, market_id="MKT-1", side="YES"):
    pos = Position(market_id=market_id, side=side, entry_price=0.4, quantity=5,
                   timestamp=datetime.now() - timedelta(minutes=age_min))
    return run(db.add_position(pos))


def status(db, pid):
    import sqlite3
    c = sqlite3.connect(db.db_path)
    try:
        return c.execute("SELECT status, live FROM positions WHERE id=?", (pid,)).fetchone()
    finally:
        c.close()


def test_flat_exchange_voids_old_row(db):
    pid = add_row(db, 11)
    ex = Exchange()
    assert run(reconcile_stale_failed_rows(db, client(ex))) == 1
    assert status(db, pid) == ("voided", 0)


def test_held_is_left_open_and_logged(db):
    pid = add_row(db, 11, side="YES")
    ex = Exchange(positions=[{"ticker": "MKT-1", "position_fp": "7.00"}])
    log = MagicMock()
    with patch("src.jobs.execute.get_trading_logger", return_value=log):
        assert run(reconcile_stale_failed_rows(db, client(ex))) == 0
    assert status(db, pid) == ("open", 0)  # not voided, not adopted as live
    errs = [c.args[0] for c in log.error.call_args_list if "NOT RECONCILED" in c.args[0]]
    assert len(errs) == 1
    assert "MKT-1" in errs[0] and "side=YES" in errs[0] and "held=7" in errs[0]


def test_resting_order_is_left_open_and_logged(db):
    pid = add_row(db, 30, side="NO")
    ex = Exchange(orders=[{"ticker": "MKT-1", "outcome_side": "no", "remaining_count_fp": "4.00"}])
    log = MagicMock()
    with patch("src.jobs.execute.get_trading_logger", return_value=log):
        run(reconcile_stale_failed_rows(db, client(ex)))
    assert status(db, pid) == ("open", 0)
    errs = [c.args[0] for c in log.error.call_args_list if "NOT RECONCILED" in c.args[0]]
    assert len(errs) == 1 and "side=NO" in errs[0] and "resting_qty=4" in errs[0]


@pytest.mark.parametrize("fail_on", ["/portfolio/positions", "/portfolio/orders"])
def test_read_error_leaves_row_open(db, fail_on):
    pid = add_row(db, 11)
    assert run(reconcile_stale_failed_rows(db, client(Exchange(fail_on=fail_on)))) == 0
    assert status(db, pid) == ("open", 0)


def test_younger_than_ten_minutes_untouched(db):
    pid = add_row(db, 5)
    ex = Exchange()
    assert run(reconcile_stale_failed_rows(db, client(ex))) == 0
    assert status(db, pid) == ("open", 0)
    assert ex.requests == []  # not even read


def test_live_rows_untouched(db):
    pid = add_row(db, 60)
    run(db.update_position_to_live(pid, 0.4))
    ex = Exchange()
    run(reconcile_stale_failed_rows(db, client(ex)))
    assert status(db, pid) == ("open", 1)


def test_zero_non_get_requests_on_every_path(db):
    add_row(db, 11, market_id="A")
    add_row(db, 11, market_id="B")
    add_row(db, 11, market_id="C")
    ex = Exchange(positions=[{"ticker": "B", "position_fp": "-2.00"}])
    run(reconcile_stale_failed_rows(db, client(ex)))
    ex2 = Exchange(fail_on="/portfolio/orders")
    run(reconcile_stale_failed_rows(db, client(ex2)))
    assert ex.requests and ex2.requests
    assert all(m == "GET" for m, _ in ex.requests + ex2.requests)


def test_unified_cycle_runs_the_pass_first(db):
    """The pass runs at the top of the cycle; the next step is stubbed to stop it there."""
    s = make_system(db, FakeKalshi(legacy_market(40, 60)))
    with patch("src.jobs.execute.reconcile_stale_failed_rows", AsyncMock(return_value=0)) as m, \
         patch("src.utils.position_limits.PositionLimitsManager",
               side_effect=RuntimeError("stop here")):
        run(s.execute_unified_trading_strategy())
    m.assert_awaited_once()
