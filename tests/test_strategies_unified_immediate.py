"""Coverage for the strategy modules behind the default ``cli.py run`` path.

cli.py run -> BeastModeBot -> jobs.trade.run_trading_job ->
unified_trading_system (market making + directional) ->
portfolio.immediate / _execute_portfolio_allocations -> jobs.execute.execute_position
-> KalshiClient.place_order (risk-governor gate lives in the client).

Everything is faked: no network, no credentials, no real orders. Kalshi is a
stub that records what it is asked to do; the DB is a temp SQLite file.

Tests marked ``xfail(strict=True)`` pin REAL defects with a dollar example in
the reason; when the bug is fixed the xfail flips to XPASS(strict) and fails,
which is the signal to delete the marker.
"""
import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.kalshi_client import KalshiAPIError
from src.config.settings import settings
from src.jobs.execute import execute_position
from src.strategies.market_making import AdvancedMarketMaker, LimitOrder
from src.strategies.portfolio.immediate import (
    _calculate_simple_kelly,
    _evaluate_immediate_trade,
    create_market_opportunities_from_markets,
)
from src.strategies.portfolio.models import MarketOpportunity, PortfolioAllocation
from src.strategies.unified_trading_system import UnifiedAdvancedTradingSystem
from src.utils.database import DatabaseManager, Position


# ---------------------------------------------------------------- helpers

def run(coro):
    return asyncio.run(coro)


def v2_market(yes_bid, yes_ask, no_bid, no_ask, status="active"):
    """Kalshi API v2 market object: dollar-denominated fields only."""
    return {"market": {"status": status,
                       "yes_bid_dollars": yes_bid, "yes_ask_dollars": yes_ask,
                       "no_bid_dollars": no_bid, "no_ask_dollars": no_ask}}


def legacy_market(yes_cents, no_cents, status="active"):
    """Legacy market object: integer cents."""
    return {"market": {"status": status, "yes_price": yes_cents, "no_price": no_cents,
                       "yes_bid": yes_cents - 1, "yes_ask": yes_cents + 1,
                       "no_bid": no_cents - 1, "no_ask": no_cents + 1}}


class FakeKalshi:
    def __init__(self, market, balance_cents=100_000, portfolio_cents=0):
        self.market = market
        self.balance_cents = balance_cents
        self.portfolio_cents = portfolio_cents
        self.orders = []

    async def get_market(self, ticker):
        return self.market

    async def get_balance(self):
        return {"balance": self.balance_cents, "portfolio_value": self.portfolio_cents}

    async def get_positions(self, ticker=None, **kw):
        """Exchange positions for ``ticker``, derived from ``held``/``position_rows``."""
        if getattr(self, "positions_error", None):
            raise self.positions_error
        if hasattr(self, "position_rows"):
            return {"market_positions": list(self.position_rows)}
        rows = []
        held = getattr(self, "held", {})
        if held.get("yes"):
            rows.append({"ticker": ticker, "position_fp": f"{held['yes']:.2f}"})
        elif held.get("no"):
            rows.append({"ticker": ticker, "position_fp": f"{-held['no']:.2f}"})
        return {"market_positions": rows}

    async def get_orders(self, ticker=None, status=None, **kw):
        return {"orders": list(getattr(self, "resting", []))}

    async def _held_on_side(self, ticker, side):
        return getattr(self, "held", {}).get(side.lower())

    async def place_order(self, **kw):
        self.orders.append(kw)
        return {"order": {"order_id": "o1"}}


def make_opp(market_prob, predicted, conf=0.8, market_id="MKT-1"):
    edge = predicted - market_prob
    o = MarketOpportunity(
        market_id=market_id, market_title="t", predicted_probability=predicted,
        market_probability=market_prob, confidence=conf, edge=edge, volatility=0.2,
        expected_return=abs(edge) * conf, max_loss=0.5, time_to_expiry=30.0,
        correlation_score=0.0, kelly_fraction=0.0, fractional_kelly=0.0,
        risk_adjusted_fraction=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        max_drawdown_contribution=0.0)
    o.edge_percentage = abs(edge)
    o.recommended_side = "YES" if edge > 0 else "NO"
    return o


@pytest.fixture
def db(tmp_path):
    d = DatabaseManager(db_path=str(tmp_path / "t.db"))
    run(d.initialize())
    return d


@pytest.fixture(autouse=True)
def sane_settings(monkeypatch):
    monkeypatch.setattr(settings.trading, "kelly_fraction", 0.25)
    monkeypatch.setattr(settings.trading, "max_single_position", 0.03)
    monkeypatch.setattr(settings.trading, "live_trading_enabled", False)


class Captured:
    """Patch the three collaborators of the immediate/allocation paths."""

    def __init__(self, execute_result=True):
        self.execute = AsyncMock(return_value=execute_result)

    def __enter__(self):
        self._ps = [
            patch("src.utils.position_limits.check_can_add_position",
                  AsyncMock(return_value=(True, "ok"))),
            patch("src.utils.cash_reserves.check_can_trade_with_cash_reserves",
                  AsyncMock(return_value=(True, "ok"))),
            patch("src.jobs.execute.execute_position", self.execute),
        ]
        for p in self._ps:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._ps:
            p.stop()

    @property
    def position(self) -> Position:
        call = self.execute.await_args
        return call.args[0] if call.args else call.kwargs["position"]


# ------------------------------------------- immediate trade: side & price

def test_immediate_yes_edge_buys_yes_at_yes_price(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
    p = cap.position
    assert p.side == "YES"
    assert p.entry_price == pytest.approx(0.41)  # live YES ask, not the 40c snapshot
    # $1000 book, 3% cap = $30 -> int(30 / 0.41) = 73 contracts
    assert p.quantity == 73


def test_immediate_no_edge_buys_no_at_one_minus_p(db):
    """YES overpriced at 70c, AI says 50c: a NO position is priced at 30c (1-p)."""
    kalshi = FakeKalshi(legacy_market(70, 30))
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.70, 0.50), db, kalshi, 1000))
    p = cap.position
    assert p.side == "NO"
    assert p.entry_price == pytest.approx(0.31)  # live NO ask
    assert p.quantity * p.entry_price <= 30.0 + 1e-6  # 3% of $1000


def test_immediate_weak_edge_places_nothing(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.40, 0.46), db, kalshi, 1000))
    cap.execute.assert_not_awaited()


@pytest.mark.parametrize("status", ["closed", "settled", "finalized"])
def test_immediate_skips_non_tradeable_market_status(db, status):
    kalshi = FakeKalshi(legacy_market(40, 60, status=status))
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
    cap.execute.assert_not_awaited()


def test_immediate_skips_collection_ticker_both_sides_at_one_dollar(db):
    kalshi = FakeKalshi(v2_market(0.99, 1.0, 0.99, 1.0))
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
    cap.execute.assert_not_awaited()


def test_immediate_entry_price_is_live_ask(db):
    """NO snapshot 30c, live NO ask 45c, edge still 15pp: record 45c, size off 45c."""
    kalshi = FakeKalshi(v2_market(0.53, 0.58, 0.40, 0.45))
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.70, 0.40), db, kalshi, 1000))
    p = cap.position
    assert (p.side, p.entry_price) == ("NO", pytest.approx(0.45))
    assert p.quantity == 66  # int($30 / 0.45), was 100 at the 30c snapshot
    assert p.quantity * p.entry_price <= 30.0


def test_immediate_refuses_when_live_ask_eats_the_edge(db):
    """Snapshot NO 30c, AI says NO is worth 50c, live NO ask 45c: 5pp left < 10pp."""
    kalshi = FakeKalshi(v2_market(0.53, 0.58, 0.40, 0.45))
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.70, 0.50), db, kalshi, 1000))
    cap.execute.assert_not_awaited()


# ------------------------------------------------ immediate trade: sizing

def test_immediate_never_exceeds_cash_or_cap_with_ample_contracts(db):
    kalshi = FakeKalshi(legacy_market(40, 60), balance_cents=100_000)
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
    p = cap.position
    assert p.quantity * p.entry_price <= 1000 * settings.trading.max_single_position + 1e-6


def test_immediate_min_one_contract_respects_position_cap(db):
    kalshi = FakeKalshi(legacy_market(45, 55), balance_cents=1000)  # $10
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.45, 0.60), db, kalshi, 10))
    if cap.execute.await_args is None:
        return
    p = cap.position
    assert p.quantity * p.entry_price <= 10 * settings.trading.max_single_position + 1e-6


def test_kelly_no_side_uses_no_probability_and_is_capped():
    opp = make_opp(0.40, 0.05)  # AI: YES nearly impossible -> huge NO edge
    f = _calculate_simple_kelly(opp)
    assert 0 < f <= 0.2


def test_kelly_degenerate_no_price_falls_back_to_five_percent():
    opp = make_opp(0.0, -0.3)  # NO side, YES at 0 -> NO costs a full dollar
    opp.edge = -0.3
    assert _calculate_simple_kelly(opp) == 0.05


def test_kelly_zero_when_no_edge_is_negative_ev():
    opp = make_opp(0.60, 0.60)
    opp.edge = 0.0  # NO branch with p = 0.40 vs 60c YES: -EV
    assert _calculate_simple_kelly(opp) == 0.0


# -------------------------------------------- immediate trade: dedupe/halt

def test_immediate_skips_when_any_side_already_held(db):
    run(db.add_position(Position(market_id="MKT-1", side="YES", entry_price=0.40,
                                 quantity=5, timestamp=datetime.now())))
    kalshi = FakeKalshi(legacy_market(70, 30))
    with Captured() as cap:
        run(_evaluate_immediate_trade(make_opp(0.70, 0.50), db, kalshi, 1000))  # wants NO
    cap.execute.assert_not_awaited()


def _row_status(db, market_id="MKT-1"):
    import sqlite3
    c = sqlite3.connect(db.db_path)
    try:
        return c.execute("SELECT status, live FROM positions WHERE market_id=?", (market_id,)).fetchall()
    finally:
        c.close()


def _age_rows(db, minutes=11):
    import sqlite3
    from datetime import timedelta
    c = sqlite3.connect(db.db_path)
    try:
        c.execute("UPDATE positions SET timestamp=?",
                  ((datetime.now() - timedelta(minutes=minutes)).isoformat(),))
        c.commit()
    finally:
        c.close()


def test_failed_buy_leaves_row_open_and_next_cycle_does_not_rebuy(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    with Captured(execute_result=False) as cap:
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
    assert cap.execute.await_count == 1
    assert _row_status(db) == [("open", 0)]


def test_raising_buy_leaves_row_open_and_next_cycle_does_not_rebuy(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    with Captured() as cap:
        cap.execute.side_effect = RuntimeError("boom")
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
    assert cap.execute.await_count == 1
    assert _row_status(db) == [("open", 0)]


def test_allocation_failed_buy_leaves_row_open_and_next_cycle_does_not_rebuy(db):
    s = make_system(db, FakeKalshi(legacy_market(40, 60)))
    with Captured(execute_result=False) as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.40, 0.60)]))
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.40, 0.60)]))
    assert cap.execute.await_count == 1
    assert _row_status(db) == [("open", 0)]


def test_stale_pass_voids_failed_row_after_ten_minutes_when_exchange_flat(db):
    from src.jobs.execute import reconcile_stale_failed_rows
    kalshi = FakeKalshi(legacy_market(40, 60))
    with Captured(execute_result=False):
        run(_evaluate_immediate_trade(make_opp(0.40, 0.60), db, kalshi, 1000))
    assert run(reconcile_stale_failed_rows(db, kalshi)) == 0  # too young
    assert _row_status(db) == [("open", 0)]
    _age_rows(db, 11)
    assert run(reconcile_stale_failed_rows(db, kalshi)) == 1
    assert _row_status(db) == [("voided", 0)]


def test_halted_client_refuses_order_and_marks_nothing_live(db):
    """execute_position through the real code, with the client's governor gate tripping."""
    pos = Position(market_id="MKT-1", side="NO", entry_price=0.30, quantity=10,
                   timestamp=datetime.now())
    pos.id = run(db.add_position(pos))
    kalshi = FakeKalshi(v2_market(0.68, 0.70, 0.28, 0.30))
    kalshi.place_order = AsyncMock(side_effect=KalshiAPIError("RISK_HALT: MANUAL_HALT"))
    ok = run(execute_position(pos, True, db, kalshi))
    assert ok is False
    row = run(db.get_position_by_market_and_side("MKT-1", "NO"))
    assert row.live is False or row.live == 0


def test_live_buy_no_sends_no_price_not_yes_price(db):
    pos = Position(market_id="MKT-1", side="NO", entry_price=0.30, quantity=10,
                   timestamp=datetime.now())
    pos.id = run(db.add_position(pos))
    kalshi = FakeKalshi(v2_market(0.68, 0.70, 0.28, 0.30))
    assert run(execute_position(pos, True, db, kalshi)) is True
    sent = kalshi.orders[0]
    assert sent["side"] == "no" and sent["no_price"] == 30 and "yes_price" not in sent


# -------------------- discovery: v2 dollar-only market objects (root cause)

class _Mkt:
    market_id = "MKT-1"
    title = "t"
    volume = 5000
    expiration_ts = None


def test_discovery_reads_v2_dollar_prices():
    kalshi = FakeKalshi(v2_market(0.88, 0.90, 0.08, 0.10))
    with patch("src.strategies.portfolio.immediate._get_fast_ai_prediction",
               AsyncMock(return_value=(0.95, 0.9))):
        opps = run(create_market_opportunities_from_markets([_Mkt()], MagicMock(), kalshi))
    assert opps and opps[0].market_probability == pytest.approx(0.89, abs=0.02)


def test_discovery_skips_market_without_quotes():
    kalshi = FakeKalshi({"market": {"status": "active", "yes_price": 40}})  # no bid/ask
    with patch("src.strategies.portfolio.immediate._get_fast_ai_prediction",
               AsyncMock(return_value=(0.95, 0.9))):
        assert run(create_market_opportunities_from_markets([_Mkt()], MagicMock(), kalshi)) == []


def test_allocation_skips_market_without_ask(db):
    s = make_system(db, FakeKalshi({"market": {"status": "active", "no_price": 30}}))
    with Captured() as cap:
        res = run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    cap.execute.assert_not_awaited()
    assert res["failed_executions"] == 1


def test_discovery_reads_legacy_cent_prices():
    kalshi = FakeKalshi(legacy_market(40, 60))
    with patch("src.strategies.portfolio.immediate._get_fast_ai_prediction",
               AsyncMock(return_value=(0.60, 0.9))):
        opps = run(create_market_opportunities_from_markets([_Mkt()], MagicMock(), kalshi))
    assert opps[0].market_probability == pytest.approx(0.40)
    assert opps[0].edge == pytest.approx(0.20)


def test_discovery_skips_extreme_prices_and_failed_ai():
    kalshi = FakeKalshi(legacy_market(97, 3))
    with patch("src.strategies.portfolio.immediate._get_fast_ai_prediction",
               AsyncMock(return_value=(0.5, 0.9))):
        assert run(create_market_opportunities_from_markets([_Mkt()], MagicMock(), kalshi)) == []
    kalshi = FakeKalshi(legacy_market(40, 60))
    with patch("src.strategies.portfolio.immediate._get_fast_ai_prediction",
               AsyncMock(return_value=(None, None))):
        assert run(create_market_opportunities_from_markets([_Mkt()], MagicMock(), kalshi)) == []


# ------------------------------- unified system: allocations -> positions

def make_system(db, kalshi, directional=100.0):
    s = UnifiedAdvancedTradingSystem(db, kalshi, MagicMock())
    s.directional_capital = directional
    s.total_capital = directional / 0.7
    return s


def alloc(market_id="MKT-1", frac=0.10):
    return PortfolioAllocation(
        allocations={market_id: frac}, total_capital_used=frac, expected_portfolio_return=0,
        portfolio_volatility=0, portfolio_sharpe=0, max_portfolio_drawdown=0,
        diversification_ratio=1, portfolio_var_95=0, portfolio_cvar_95=0,
        aggregate_kelly_fraction=0, portfolio_growth_rate=0)


def test_discovery_reads_v2_dollar_prices():
    kalshi = FakeKalshi(v2_market(0.88, 0.90, 0.08, 0.10))
    with patch("src.strategies.portfolio.immediate._get_fast_ai_prediction",
               AsyncMock(return_value=(0.95, 0.9))):
        opps = run(create_market_opportunities_from_markets([_Mkt()], MagicMock(), kalshi))
    assert opps and opps[0].market_probability == pytest.approx(0.89)


def test_allocation_no_side_uses_v2_no_ask(db):
    s = make_system(db, FakeKalshi(v2_market(0.88, 0.90, 0.08, 0.10)))
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.90, 0.70)]))
    assert (cap.position.side, cap.position.entry_price) == ("NO", pytest.approx(0.10))
    assert cap.position.quantity == 100


def test_allocation_yes_side_v2_does_not_overspend(db):
    s = make_system(db, FakeKalshi(v2_market(0.88, 0.90, 0.08, 0.10)))
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.60, 0.95)]))
    assert cap.position.quantity == 11 and cap.position.quantity * 0.90 <= 10.0


def test_allocation_no_side_priced_at_no_price(db):
    kalshi = FakeKalshi(legacy_market(70, 30))
    s = make_system(db, kalshi)
    with Captured() as cap:
        res = run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    p = cap.position
    assert (p.side, p.entry_price) == ("NO", pytest.approx(0.31))  # live NO ask
    assert p.quantity == 32  # int($10 / 0.31)
    assert res["successful_executions"] == 1


def test_allocation_yes_side_priced_at_yes_price(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.40, 0.60)]))
    assert (cap.position.side, cap.position.entry_price) == ("YES", pytest.approx(0.41))  # live YES ask
    assert cap.position.quantity == 24


def test_allocation_no_side_uses_v2_no_ask(db):
    kalshi = FakeKalshi(v2_market(0.88, 0.90, 0.08, 0.10))
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.90, 0.70)]))
    assert cap.position.side == "NO"
    assert cap.position.entry_price == pytest.approx(0.10, abs=0.02)


def test_allocation_yes_side_v2_does_not_overspend(db):
    kalshi = FakeKalshi(v2_market(0.88, 0.90, 0.08, 0.10))
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.60, 0.95)]))
    p = cap.position
    assert p.quantity * 0.90 <= 10.0 * 1.05


def test_allocation_same_side_dedupe(db):
    run(db.add_position(Position(market_id="MKT-1", side="NO", entry_price=0.30,
                                 quantity=5, timestamp=datetime.now())))
    s = make_system(db, FakeKalshi(legacy_market(70, 30)))
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    cap.execute.assert_not_awaited()


def test_allocation_blocks_opposite_side_in_db(db):
    run(db.add_position(Position(market_id="MKT-1", side="YES", entry_price=0.70,
                                 quantity=5, timestamp=datetime.now())))
    s = make_system(db, FakeKalshi(legacy_market(70, 33)))
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    cap.execute.assert_not_awaited()


def test_allocation_blocks_opposite_side_held_on_exchange(db):
    kalshi = FakeKalshi(legacy_market(70, 33))
    kalshi.held = {"yes": 4.0}
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))  # wants NO
    cap.execute.assert_not_awaited()


@pytest.mark.parametrize("resting", [
    {"ticker": "MKT-1", "outcome_side": "yes", "remaining_count_fp": "3.00"},  # buy-YES rests
    {"ticker": "MKT-1", "side": "yes", "action": "buy", "remaining_count": 3},
    {"ticker": "MKT-1", "mystery": 1},  # unreadable -> fail closed
])
def test_allocation_blocks_opposite_side_resting(db, resting):
    kalshi = FakeKalshi(legacy_market(70, 33))
    kalshi.resting = [resting]
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    cap.execute.assert_not_awaited()


def test_allocation_allows_resting_order_on_same_side(db):
    kalshi = FakeKalshi(legacy_market(70, 33))
    kalshi.resting = [{"ticker": "MKT-1", "outcome_side": "no", "remaining_count_fp": "3.00"}]
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    cap.execute.assert_awaited_once()


def test_allocation_fails_closed_when_resting_read_fails(db):
    kalshi = FakeKalshi(legacy_market(70, 33))
    kalshi.get_orders = AsyncMock(side_effect=RuntimeError("503"))
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    cap.execute.assert_not_awaited()


@pytest.mark.parametrize("kw", [
    {"positions_error": RuntimeError("503")},                                  # read error
    {"position_rows": [{"ticker": "MKT-1", "position_fp": "1"},
                       {"ticker": "MKT-1", "position_fp": "2"}]},               # >1 row
    {"position_rows": [{"ticker": "MKT-1", "position_fp": "nan"}]},            # non-finite
    {"position_rows": [{"ticker": "MKT-1", "position_fp": "inf"}]},
])
def test_allocation_blocked_when_positions_read_is_unreliable(db, kw):
    kalshi = FakeKalshi(legacy_market(70, 33))
    for k, v in kw.items():
        setattr(kalshi, k, v)
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))  # wants NO
    cap.execute.assert_not_awaited()


def test_allocation_flat_positions_read_is_not_held(db):
    kalshi = FakeKalshi(legacy_market(70, 33))
    kalshi.position_rows = []
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    cap.execute.assert_awaited_once()


def test_allocation_blocked_when_resting_page_is_full(db):
    kalshi = FakeKalshi(legacy_market(70, 33))
    kalshi.resting = [{"ticker": "OTHER", "outcome_side": "no", "remaining_count_fp": "1.00"}] * 100
    orig = kalshi.get_orders

    async def paged(**kw):
        r = await orig(**kw)
        r["cursor"] = "next"
        return r
    kalshi.get_orders = paged
    s = make_system(db, kalshi)
    with Captured() as cap:
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.70, 0.50)]))
    cap.execute.assert_not_awaited()


# --- accepted order whose call then fails: one order only

def _accepted_then_4xx(kalshi):
    """place_order: the POST reaches the exchange (order rests), then the client raises."""
    async def place(**kw):
        kalshi.orders.append(kw)
        kalshi.resting = [{"ticker": "MKT-1", "outcome_side": "yes", "remaining_count_fp": "5.00"}]
        raise KalshiAPIError("409 duplicate client_order_id")
    kalshi.place_order = place


def test_immediate_keeps_row_open_when_order_reached_exchange_then_raised(db, monkeypatch):
    # in-memory flag only; the client is a fake, so no order can leave the process
    monkeypatch.setattr(settings.trading, "live_trading_enabled", True)
    kalshi = FakeKalshi(legacy_market(40, 60))
    _accepted_then_4xx(kalshi)
    opp = make_opp(0.40, 0.60)
    with patch("src.utils.position_limits.check_can_add_position",
               AsyncMock(return_value=(True, "ok"))), \
         patch("src.utils.cash_reserves.check_can_trade_with_cash_reserves",
               AsyncMock(return_value=(True, "ok"))):
        run(_evaluate_immediate_trade(opp, db, kalshi, 1000))
        run(_evaluate_immediate_trade(opp, db, kalshi, 1000))  # second run
    assert len(kalshi.orders) == 1  # no re-buy
    assert run(db.get_position_by_market_and_side("MKT-1", "YES")) is not None


def test_allocation_keeps_row_open_when_order_reached_exchange_then_raised(db, monkeypatch):
    # in-memory flag only; the client is a fake, so no order can leave the process
    monkeypatch.setattr(settings.trading, "live_trading_enabled", True)
    kalshi = FakeKalshi(legacy_market(40, 60))
    _accepted_then_4xx(kalshi)
    s = make_system(db, kalshi)
    with patch("src.utils.position_limits.check_can_add_position",
               AsyncMock(return_value=(True, "ok"))), \
         patch("src.utils.cash_reserves.check_can_trade_with_cash_reserves",
               AsyncMock(return_value=(True, "ok"))):
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.40, 0.60)]))
        run(s._execute_portfolio_allocations(alloc(), [make_opp(0.40, 0.60)]))
    assert len(kalshi.orders) == 1
    assert run(db.get_position_by_market_and_side("MKT-1", "YES")) is not None


def test_allocation_limit_failure_counts_failed_not_created(db):
    s = make_system(db, FakeKalshi(legacy_market(40, 60)))
    with Captured() as cap, \
            patch("src.utils.position_limits.check_can_add_position",
                  AsyncMock(return_value=(False, "over"))), \
            patch("src.utils.position_limits.PositionLimitsManager") as plm:
        mgr = plm.return_value
        mgr._get_position_count = AsyncMock(return_value=99)
        mgr.max_positions = 10
        res = run(s._execute_portfolio_allocations(alloc(), [make_opp(0.40, 0.60)]))
    cap.execute.assert_not_awaited()
    assert res["failed_executions"] == 1 and res["positions_created"] == 0


# ------------------------------------------------ unified system: halt/init

def _emergency_patches(action):
    limits = MagicMock()
    limits.get_position_limits_status = AsyncMock(
        return_value={"status": "OK", "position_utilization": "0/10"})
    cash = MagicMock()
    cash.get_cash_status = AsyncMock(return_value={
        "status": "EMERGENCY", "reserve_percentage": 1.0,
        "emergency_status": True, "recommendations": "x"})
    cash.handle_cash_emergency = AsyncMock(
        return_value=SimpleNamespace(action_type=action, reason="r", positions_to_close=0))
    return (patch("src.utils.position_limits.PositionLimitsManager", return_value=limits),
            patch("src.utils.cash_reserves.CashReservesManager", return_value=cash))


def test_cash_emergency_halt_stops_before_market_scan(db):
    s = make_system(db, FakeKalshi(legacy_market(40, 60)))
    s.db_manager = MagicMock()
    s.db_manager.get_eligible_markets = AsyncMock(return_value=[])
    p1, p2 = _emergency_patches("halt_trading")
    with p1, p2:
        res = run(s.execute_unified_trading_strategy())
    assert res.total_positions == 0
    s.db_manager.get_eligible_markets.assert_not_awaited()


@pytest.mark.xfail(strict=True, reason=(
    "async_initialize falls back to total_capital = $100 when the balance call "
    "fails ('Conservative fallback'). For a $20 account that is 5x reality: "
    "directional capital $70 and market-making capital $30 are sized off money "
    "that does not exist. Unreadable balance should mean no new buys (capital 0)."))
def test_unreadable_balance_does_not_invent_capital(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.get_balance = AsyncMock(side_effect=RuntimeError("boom"))
    s = UnifiedAdvancedTradingSystem(db, kalshi, MagicMock())
    run(s.async_initialize())
    assert s.total_capital == 0


def test_initialize_splits_capital_by_config(db):
    kalshi = FakeKalshi(legacy_market(40, 60), balance_cents=8000, portfolio_cents=2000)
    s = UnifiedAdvancedTradingSystem(db, kalshi, MagicMock())
    run(s.async_initialize())
    assert s.total_capital == pytest.approx(100.0)
    assert s.directional_capital == pytest.approx(70.0)
    assert s.market_making_capital == pytest.approx(30.0)


# ---------------------------------------------------------- market making

def _mm():
    return AdvancedMarketMaker(MagicMock(), FakeKalshi(None), MagicMock())


def _mm_market():
    return SimpleNamespace(market_id="MM-1", title="t", expiration_ts=None)


@pytest.mark.xfail(strict=True, reason=(
    "OWNER DECISION: activates market making. "
    "analyze_market_making_opportunities reads yes_bid_dollars/yes_price from the TOP "
    "LEVEL of get_market(); the API nests them under 'market'. Prices are always 0, "
    "the <0.02 guard skips every market, and market making can never fire: "
    "0 opportunities from a 40c/58c book with a 0.7 AI view."))
def test_mm_analysis_reads_nested_market_object():
    mm = _mm()
    mm.kalshi_client.market = v2_market(0.40, 0.42, 0.58, 0.60)
    mm._get_ai_analysis = AsyncMock(return_value={"probability": 0.7, "confidence": 0.9})
    opps = run(mm.analyze_market_making_opportunities([_mm_market()]))
    assert len(opps) == 1


def test_mm_analysis_works_only_on_flat_shape_real_api_never_returns():
    mm = _mm()
    mm.kalshi_client.market = v2_market(0.40, 0.42, 0.58, 0.60)["market"]  # flattened
    mm._get_ai_analysis = AsyncMock(return_value={"probability": 0.7, "confidence": 0.9})
    assert len(run(mm.analyze_market_making_opportunities([_mm_market()]))) == 1


def _opportunity(mm, yes=0.50, no=0.48, ai=0.7, conf=1.0):
    return run(mm._calculate_market_making_opportunity(_mm_market(), yes, no, ai, conf))


def test_mm_quotes_stay_in_bounds_and_bids_sum_below_a_dollar():
    mm = _mm()
    for yes, no in [(0.03, 0.95), (0.50, 0.48), (0.97, 0.01), (0.60, 0.38)]:
        for ai in (0.05, 0.5, 0.95):
            o = _opportunity(mm, yes, no, ai, 0.9)
            for px in (o.optimal_yes_bid, o.optimal_yes_ask, o.optimal_no_bid, o.optimal_no_ask):
                assert 0.01 <= px <= 0.99
            # Both bids filling must not cost more than the $1 payout.
            assert o.optimal_yes_bid + o.optimal_no_bid < 1.0


def test_mm_live_orders_map_side_to_price_field():
    mm = _mm()
    with patch.object(settings.trading, "live_trading_enabled", True):
        run(mm._place_limit_order(LimitOrder("MM-1", "YES", 40.0, 7)))
        run(mm._place_limit_order(LimitOrder("MM-1", "NO", 55.0, 7)))
    y, n = mm.kalshi_client.orders
    assert (y["side"], y["yes_price"], "no_price" in y) == ("yes", 40, False)
    assert (n["side"], n["no_price"], "yes_price" in n) == ("no", 55, False)
    assert y["action"] == n["action"] == "buy"


@pytest.mark.xfail(strict=True, reason=(
    "_place_limit_order does int(order.price) on a float made by dollars*100, which "
    "truncates: 0.58*100 = 57.99999999999999 -> a 57c bid instead of 58c, a "
    "systematic 1c/contract mispricing on ~25% of tick values (250 contracts: $2.50)."))
def test_mm_live_price_rounds_not_truncates():
    mm = _mm()
    with patch.object(settings.trading, "live_trading_enabled", True):
        run(mm._place_limit_order(LimitOrder("MM-1", "YES", 0.58 * 100, 10)))
    assert mm.kalshi_client.orders[0]["yes_price"] == 58


def test_mm_refused_order_is_marked_failed():
    mm = _mm()
    mm.kalshi_client.place_order = AsyncMock(side_effect=KalshiAPIError("RISK_HALT: x"))
    o = LimitOrder("MM-1", "YES", 40.0, 7)
    with patch.object(settings.trading, "live_trading_enabled", True):
        run(mm._place_limit_order(o))
    assert o.status == "failed"


def test_mm_results_do_not_count_refused_orders():
    mm = _mm()
    mm.kalshi_client.place_order = AsyncMock(side_effect=KalshiAPIError("RISK_HALT: x"))
    opp = _opportunity(mm)
    with patch.object(settings.trading, "live_trading_enabled", True):
        res = run(mm.execute_market_making_strategy([opp]))
    assert res["orders_placed"] == 0


def test_mm_unified_orders_fit_market_making_allotment(db):
    s = make_system(db, FakeKalshi(None))
    s.market_making_capital = 120.0
    s.market_maker = _mm()
    s.market_maker.analyze_market_making_opportunities = AsyncMock(
        return_value=[_opportunity(s.market_maker)] * 1)
    run(s._execute_market_making_strategy([_mm_market()]))
    notional = sum(o.quantity * o.price / 100.0
                   for orders in s.market_maker.active_orders.values() for o in orders)
    assert notional <= 120.0


def test_mm_unified_small_account_places_nothing(db):
    s = make_system(db, FakeKalshi(None))
    s.market_making_capital = 30.0  # int(30/100) == 0 opportunities allowed
    s.market_maker = _mm()
    s.market_maker.analyze_market_making_opportunities = AsyncMock(
        return_value=[_opportunity(s.market_maker)])
    res = run(s._execute_market_making_strategy([_mm_market()]))
    assert res["orders_placed"] == 0


# ------------------------------------------------- jobs.trade fallback path

@pytest.mark.xfail(strict=True, reason=(
    "OWNER DECISION: activates the legacy fallback. "
    "_fallback_legacy_trading calls execute_position(position, kalshi_client, "
    "db_manager) but the signature is (position, live_mode, db_manager, "
    "kalshi_client). The call raises TypeError on every market, is swallowed by "
    "the per-market except, and the fallback silently never trades. (A signature "
    "with a default for kalshi_client would instead treat the client object as "
    "live_mode=True, i.e. a truthy 'live' flag.)"))
def test_fallback_legacy_trading_calls_execute_position_correctly():
    from src.jobs import trade
    from src.jobs.execute import execute_position as real

    pos = Position(market_id="M", side="YES", entry_price=0.4, quantity=3,
                   timestamp=datetime.now())
    dbm = MagicMock()
    dbm.get_eligible_markets = AsyncMock(return_value=[SimpleNamespace(market_id="M")])
    with patch.object(trade, "DatabaseManager", return_value=dbm), \
            patch.object(trade, "KalshiClient", return_value=MagicMock()), \
            patch.object(trade, "XAIClient", return_value=MagicMock()), \
            patch.object(trade, "make_decision_for_market", AsyncMock(return_value=pos)), \
            patch.object(trade, "execute_position",
                         create_autospec_async(real)):
        res = run(trade._fallback_legacy_trading())
    assert res.total_positions == 1


def create_autospec_async(fn):
    from unittest.mock import create_autospec
    m = create_autospec(fn, return_value=True)
    return m


def test_mm_accepted_orders_are_counted_with_real_notional():
    mm = _mm()
    opp = _opportunity(mm)
    res = run(mm.execute_market_making_strategy([opp]))  # paper mode: simulated "placed"
    assert res["orders_placed"] == 2 and res["markets_count"] == 1
    assert res["total_exposure"] == pytest.approx(
        opp.optimal_yes_size * opp.optimal_yes_bid + opp.optimal_no_size * opp.optimal_no_bid)


def test_mm_half_refused_counts_only_the_placed_leg():
    mm = _mm()
    calls = []

    async def place(**kw):
        calls.append(kw)
        if kw["side"] == "no":
            raise KalshiAPIError("POSITION_CAP")
        return {"order": {"order_id": "o"}}
    mm.kalshi_client.place_order = place
    with patch.object(settings.trading, "live_trading_enabled", True):
        res = run(mm.execute_market_making_strategy([_opportunity(mm)]))
    assert res["orders_placed"] == 1


def test_mm_capital_scales_orders_to_the_allotment_exactly_once():
    mm = _mm()
    opps = [_opportunity(mm), _opportunity(mm)]
    res = run(mm.execute_market_making_strategy(opps, capital=60.0))
    assert res["total_exposure"] <= 60.0
    assert res["markets_count"] == 2


def test_mm_capital_too_small_for_one_contract_places_nothing():
    mm = _mm()
    res = run(mm.execute_market_making_strategy([_opportunity(mm)], capital=0.10))
    assert res["orders_placed"] == 0 and res["markets_count"] == 0
