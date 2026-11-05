"""The immediate and allocation order paths: guards that refuse a trade, and the rest of each path.

Everything is faked: no network, no credentials, no real orders.
"""
import math
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.strategies.portfolio.immediate import (
    _calculate_simple_kelly,
    _coerce_volume,
    _evaluate_immediate_trade,
    _get_fast_ai_prediction,
    create_market_opportunities_from_markets,
    holds_or_rests_opposite_side,
)
from src.strategies.unified_trading_system import (
    TradingSystemConfig,
    TradingSystemResults,
    UnifiedAdvancedTradingSystem,
    run_unified_trading_system,
)
from tests.test_strategies_unified_immediate import (  # noqa: F401  (fixtures)
    Captured,
    FakeKalshi,
    alloc,
    db,
    legacy_market,
    make_opp,
    make_system,
    run,
    sane_settings,
    v2_market,
)

UTS = "src.strategies.unified_trading_system"
IMM = "src.strategies.portfolio.immediate"


def immediate(db, kalshi, opp=None, capital=1000):
    run(_evaluate_immediate_trade(opp or make_opp(0.40, 0.60), db, kalshi, capital))


def allocate(db, kalshi, opp, directional=100.0, frac=0.10):
    return run(make_system(db, kalshi, directional)._execute_portfolio_allocations(alloc(frac=frac), [opp]))


# ---------------------------------------------- immediate path: opposite side

def test_immediate_buys_when_flat_and_no_orders_rest(db):
    with Captured() as cap:
        immediate(db, FakeKalshi(legacy_market(40, 60)))
    assert (cap.position.side, cap.position.quantity) == ("YES", 73)


def test_immediate_refuses_yes_while_exchange_holds_no(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.held = {"no": 5}
    with Captured() as cap:
        immediate(db, kalshi)
    assert cap.execute.await_count == 0


def test_immediate_refuses_no_while_exchange_holds_yes(db):
    kalshi = FakeKalshi(legacy_market(60, 40))
    kalshi.held = {"yes": 5}
    with Captured() as cap:
        immediate(db, kalshi, make_opp(0.60, 0.40))
    assert cap.execute.await_count == 0


def test_immediate_refuses_yes_while_a_no_buy_rests(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.resting = [{"ticker": "MKT-1", "side": "no", "action": "buy"}]
    with Captured() as cap:
        immediate(db, kalshi)
    assert cap.execute.await_count == 0


def test_immediate_refuses_when_the_positions_read_fails(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.positions_error = RuntimeError("503")
    with Captured() as cap:
        immediate(db, kalshi)
    assert cap.execute.await_count == 0


def test_immediate_refuses_when_the_orders_read_fails(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.get_orders = AsyncMock(side_effect=RuntimeError("503"))
    with Captured() as cap:
        immediate(db, kalshi)
    assert cap.execute.await_count == 0


def test_same_side_on_the_exchange_is_not_the_opposite_side(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.held = {"yes": 5}
    assert run(holds_or_rests_opposite_side(db, kalshi, "MKT-1", "YES")) is False


@pytest.mark.parametrize("response", [None, {"market_positions": "junk"}, {}])
def test_opposite_check_fails_closed_on_a_junk_positions_response(db, response):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.get_positions = AsyncMock(return_value=response)
    assert run(holds_or_rests_opposite_side(db, kalshi, "MKT-1", "YES")) is True


@pytest.mark.parametrize("response", [None, {"orders": "junk"}, {}])
def test_opposite_check_fails_closed_on_a_junk_orders_response(db, response):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.get_orders = AsyncMock(return_value=response)
    assert run(holds_or_rests_opposite_side(db, kalshi, "MKT-1", "YES")) is True


# --------------------------------------- allocation path: live ask and status

@pytest.mark.parametrize("predicted, book", [
    (0.70, (0.30, 0.80, 0.20, 0.70)),  # mid .55 looks like edge; the ask is above the estimate
    (0.60, (0.01, 0.99, 0.01, 0.99)),
    (0.60, (0.40, 0.60, 0.40, 0.60)),  # paying exactly the estimate has no edge
])
def test_allocation_refuses_yes_at_or_above_own_estimate(db, predicted, book):
    with Captured() as cap:
        result = allocate(db, FakeKalshi(v2_market(*book)), make_opp(0.40, predicted))
    assert cap.execute.await_count == 0
    assert result["failed_executions"] == 1


def test_allocation_refuses_no_at_or_above_own_estimate(db):
    # Estimate .30 YES means .70 NO; the NO ask is .75.
    with Captured() as cap:
        result = allocate(db, FakeKalshi(v2_market(0.20, 0.30, 0.70, 0.75)), make_opp(0.60, 0.30))
    assert cap.execute.await_count == 0
    assert result["failed_executions"] == 1


def test_allocation_buys_no_below_own_estimate(db):
    with Captured() as cap:
        allocate(db, FakeKalshi(v2_market(0.20, 0.30, 0.60, 0.65)), make_opp(0.60, 0.30))
    assert (cap.position.side, cap.position.entry_price) == ("NO", 0.65)


def test_allocation_buys_yes_one_cent_below_own_estimate(db):
    with Captured() as cap:
        allocate(db, FakeKalshi(v2_market(0.40, 0.59, 0.41, 0.60)), make_opp(0.40, 0.60))
    assert (cap.position.side, cap.position.entry_price) == ("YES", 0.59)


def test_allocation_refuses_a_nan_estimate(db):
    opp = make_opp(0.40, 0.60)
    opp.predicted_probability = float("nan")
    with Captured() as cap:
        result = allocate(db, FakeKalshi(v2_market(0.40, 0.41, 0.59, 0.60)), opp)
    assert cap.execute.await_count == 0
    assert result["failed_executions"] == 1


@pytest.mark.parametrize("status", ["closed", "settled", "paused", None])
def test_allocation_refuses_a_market_that_is_not_open(db, status):
    with Captured() as cap:
        result = allocate(db, FakeKalshi(v2_market(0.40, 0.41, 0.59, 0.60, status=status)), make_opp(0.40, 0.60))
    assert cap.execute.await_count == 0
    assert result["failed_executions"] == 1


@pytest.mark.parametrize("status", ["active", "open"])
def test_allocation_trades_an_open_market(db, status):
    with Captured() as cap:
        result = allocate(db, FakeKalshi(v2_market(0.40, 0.41, 0.59, 0.60, status=status)), make_opp(0.40, 0.60))
    assert result["successful_executions"] == 1
    assert result["total_capital_used"] == pytest.approx(10.0)
    assert cap.position.quantity == 24


# ------------------------------------------------- allocation path: the rest

def test_allocation_without_a_matching_opportunity_trades_nothing(db):
    kalshi = FakeKalshi(v2_market(0.40, 0.41, 0.59, 0.60))
    with Captured() as cap:
        run(make_system(db, kalshi)._execute_portfolio_allocations(alloc("OTHER"), [make_opp(0.40, 0.60)]))
    assert cap.execute.await_count == 0


def test_allocation_shrinks_to_the_first_size_the_limits_allow(db):
    kalshi = FakeKalshi(v2_market(0.40, 0.50, 0.50, 0.60))
    limits = AsyncMock(side_effect=[(False, "big"), (False, "big"), (True, "ok")])
    with Captured() as cap, patch("src.utils.position_limits.check_can_add_position", limits):
        allocate(db, kalshi, make_opp(0.40, 0.70))
    assert cap.position.quantity == 12  # $10 * 0.6 / $0.50
    assert limits.await_args_list[2].args[0] == pytest.approx(6.0)


@pytest.mark.parametrize("count", [50, 1])
def test_allocation_refuses_when_no_size_fits(db, count):
    kalshi = FakeKalshi(v2_market(0.40, 0.50, 0.50, 0.60))
    manager = MagicMock(max_positions=10, _get_position_count=AsyncMock(return_value=count))
    with Captured() as cap, \
            patch("src.utils.position_limits.check_can_add_position", AsyncMock(return_value=(False, "no"))), \
            patch("src.utils.position_limits.PositionLimitsManager", return_value=manager):
        result = allocate(db, kalshi, make_opp(0.40, 0.70))
    assert cap.execute.await_count == 0
    assert result["failed_executions"] == 1


def test_allocation_refuses_when_cash_reserves_say_no(db):
    kalshi = FakeKalshi(v2_market(0.40, 0.50, 0.50, 0.60))
    with Captured() as cap, \
            patch("src.utils.cash_reserves.check_can_trade_with_cash_reserves", AsyncMock(return_value=(False, "low"))):
        result = allocate(db, kalshi, make_opp(0.40, 0.70))
    assert cap.execute.await_count == 0
    assert result["failed_executions"] == 1


def test_allocation_without_market_data_trades_nothing(db):
    with Captured() as cap:
        allocate(db, FakeKalshi(None), make_opp(0.40, 0.70))
    assert cap.execute.await_count == 0


def test_allocation_allotment_below_one_contract_trades_nothing(db):
    with Captured() as cap:
        allocate(db, FakeKalshi(v2_market(0.40, 0.50, 0.50, 0.60)), make_opp(0.40, 0.70), directional=4.0)
    assert cap.execute.await_count == 0


def test_allocation_row_that_cannot_be_added_is_not_executed(db):
    db.add_position = AsyncMock(return_value=None)
    with Captured() as cap:
        allocate(db, FakeKalshi(v2_market(0.40, 0.50, 0.50, 0.60)), make_opp(0.40, 0.70))
    assert cap.execute.await_count == 0


def test_allocation_counts_a_failed_execution(db):
    with Captured(execute_result=False):
        result = allocate(db, FakeKalshi(v2_market(0.40, 0.50, 0.50, 0.60)), make_opp(0.40, 0.70))
    assert result == {"positions_created": 0, "total_capital_used": 0.0,
                      "successful_executions": 0, "failed_executions": 1}


def test_allocation_error_on_one_market_is_counted_and_trades_nothing(db):
    kalshi = FakeKalshi(v2_market(0.40, 0.50, 0.50, 0.60))
    kalshi.get_market = AsyncMock(side_effect=RuntimeError("boom"))
    with Captured() as cap:
        result = allocate(db, kalshi, make_opp(0.40, 0.70))
    assert cap.execute.await_count == 0
    assert result["failed_executions"] == 1


# ------------------------------------------------- immediate path: the rest

class EventKalshi(FakeKalshi):
    """Portfolio-wide positions read returns these event rows; a ticker read is the usual fake."""

    def __init__(self, market, events, **kw):
        super().__init__(market, **kw)
        self.events = events

    async def get_positions(self, ticker=None, **kw):
        if ticker is None:
            return {"event_positions": self.events}
        return await super().get_positions(ticker=ticker, **kw)


def test_immediate_sizes_from_cash_plus_positions(db):
    kalshi = EventKalshi(legacy_market(40, 60),
                         [{"event_ticker": "E", "event_exposure_dollars": "5"}],
                         balance_cents=100_000, portfolio_cents=100_000)
    with Captured() as cap:
        immediate(db, kalshi)
    assert cap.position.quantity == 146  # 3% of $2,000 at $0.41


def test_junk_exposure_row_falls_back_to_the_capital_it_was_given(db):
    kalshi = EventKalshi(legacy_market(40, 60), [{"event_exposure_dollars": "junk"}])
    with Captured() as cap:
        immediate(db, kalshi, capital=500)
    assert cap.position.quantity == 36  # 3% of $500 at $0.41


def test_unreadable_balance_falls_back_to_the_capital_it_was_given(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.get_balance = AsyncMock(side_effect=RuntimeError("503"))
    with Captured() as cap:
        immediate(db, kalshi, capital=500)
    assert cap.position.quantity == 36


def test_immediate_shrinks_to_the_first_size_the_limits_allow(db):
    limits = AsyncMock(side_effect=[(False, "big"), (True, "ok")])
    with Captured() as cap, patch("src.utils.position_limits.check_can_add_position", limits):
        immediate(db, FakeKalshi(legacy_market(40, 60)))
    assert cap.position.quantity == 58  # $30 * 0.8 / $0.41


@pytest.mark.parametrize("count", [50, 1])
def test_immediate_refuses_when_no_size_fits(db, count):
    manager = MagicMock(max_positions=10, _get_position_count=AsyncMock(return_value=count))
    with Captured() as cap, \
            patch("src.utils.position_limits.check_can_add_position", AsyncMock(return_value=(False, "no"))), \
            patch("src.utils.position_limits.PositionLimitsManager", return_value=manager):
        immediate(db, FakeKalshi(legacy_market(40, 60)))
    assert cap.execute.await_count == 0


def test_immediate_refuses_when_cash_reserves_say_no(db):
    with Captured() as cap, \
            patch("src.utils.cash_reserves.check_can_trade_with_cash_reserves", AsyncMock(return_value=(False, "low"))):
        immediate(db, FakeKalshi(legacy_market(40, 60)))
    assert cap.execute.await_count == 0


@pytest.mark.parametrize("market", [
    v2_market(0.39, 0.41, 0.59, 0.61, status="closed"),
    v2_market(0.39, 0.41, 0.0, 0.0),                # no NO quote, though YES is what we'd buy
])
def test_immediate_refuses_a_closed_or_half_quoted_market(db, market):
    with Captured() as cap:
        immediate(db, FakeKalshi(market))
    assert cap.execute.await_count == 0


def test_immediate_refuses_when_the_market_read_fails(db):
    kalshi = FakeKalshi(legacy_market(40, 60))
    kalshi.get_market = AsyncMock(side_effect=RuntimeError("503"))
    with Captured() as cap:
        immediate(db, kalshi)
    assert cap.execute.await_count == 0


def test_immediate_cap_below_one_contract_trades_nothing(db):
    with Captured() as cap:
        immediate(db, FakeKalshi(legacy_market(40, 60), balance_cents=1000))
    assert cap.execute.await_count == 0


def test_immediate_row_that_cannot_be_added_is_not_executed(db):
    db.add_position = AsyncMock(return_value=None)
    with Captured() as cap:
        immediate(db, FakeKalshi(legacy_market(40, 60)))
    assert cap.execute.await_count == 0


def test_failed_immediate_execution_leaves_the_row_open(db):
    with Captured(execute_result=False) as cap:
        immediate(db, FakeKalshi(legacy_market(40, 60)))
    assert cap.execute.await_count == 1
    assert run(db.get_position_by_market_and_side("MKT-1", "YES")) is not None


def test_immediate_error_in_a_gate_trades_nothing(db):
    with Captured() as cap, \
            patch("src.utils.position_limits.check_can_add_position", AsyncMock(side_effect=RuntimeError("boom"))):
        immediate(db, FakeKalshi(legacy_market(40, 60)))
    assert cap.execute.await_count == 0


# ----------------------------------------------------------- simple Kelly

def test_simple_kelly_yes_side():
    assert _calculate_simple_kelly(make_opp(0.60, 0.70)) == pytest.approx(0.2)  # capped from .25


def test_simple_kelly_no_side():
    # NO costs .40 and we give it .50: b = 1.5, f = (.50*1.5 - .50)/1.5.
    assert _calculate_simple_kelly(make_opp(0.60, 0.50)) == pytest.approx(0.25 / 1.5)


@pytest.mark.parametrize("market, predicted", [(float("nan"), 0.5), (0.5, float("inf")), (1.2, 0.5), (0.5, -0.1)])
def test_simple_kelly_sizes_corrupt_inputs_at_zero(market, predicted):
    opp = make_opp(0.5, 0.6)
    opp.market_probability, opp.predicted_probability = market, predicted
    assert _calculate_simple_kelly(opp) == 0.0


def test_coerce_volume_turns_junk_into_nan():
    assert _coerce_volume("12") == 12.0
    assert math.isnan(_coerce_volume(None))
    assert math.isnan(_coerce_volume("x"))
    assert math.isnan(_coerce_volume(float("inf")))


# ------------------------------------------------------- fast AI prediction

def ai(text):
    return SimpleNamespace(get_completion=AsyncMock(return_value=text))


def predict(text):
    return run(_get_fast_ai_prediction(SimpleNamespace(market_id="M", title="t"), ai(text), 0.5))


def test_prediction_reads_json_inside_prose():
    assert predict('Sure: {"probability": 0.7, "confidence": 0.6} done') == (0.7, 0.6)


@pytest.mark.parametrize("text", [
    None,
    "no json here",
    '{"probability": 0.7',
    '{"probability": true, "confidence": 0.6}',
    '{"probability": 0.7, "confidence": true}',
    '{"probability": 1.5, "confidence": 0.6}',
    '{"probability": 0.7, "confidence": -0.1}',
    '{"probability": "0.7", "confidence": 0.6}',
    '{"confidence": 0.6}',
])
def test_prediction_refuses_a_missing_or_invalid_answer(text):
    assert predict(text) == (None, None)


def test_prediction_refuses_when_the_call_raises():
    xai = SimpleNamespace(get_completion=AsyncMock(side_effect=RuntimeError("503")))
    assert run(_get_fast_ai_prediction(SimpleNamespace(market_id="M", title="t"), xai, 0.5)) == (None, None)


# ---------------------------------------------------------------- discovery

def mkt(market_id="MKT-1", volume=5000, expiration_ts=None):
    return SimpleNamespace(market_id=market_id, title="t", volume=volume, expiration_ts=expiration_ts)


def discover(markets, kalshi, prediction=(0.70, 0.9), db_manager=None, capital=10000):
    with patch(f"{IMM}._get_fast_ai_prediction", AsyncMock(return_value=prediction)) as fast:
        opps = run(create_market_opportunities_from_markets(markets, MagicMock(), kalshi, db_manager, capital))
    return opps, fast


def test_discovery_keeps_the_ten_largest_markets():
    kalshi = MagicMock(get_market=AsyncMock(return_value=v2_market(0.39, 0.41, 0.59, 0.61)))
    markets = [mkt(f"M{i}", volume=i) for i in range(12)]
    opps, _ = discover(markets, kalshi)
    assert sorted(o.market_id for o in opps) == sorted(f"M{i}" for i in range(2, 12))


def test_discovery_skips_an_empty_market_read():
    assert discover([mkt()], FakeKalshi(None))[0] == []


@pytest.mark.parametrize("book", [(0.96, 0.98, 0.02, 0.04), (0.01, 0.03, 0.97, 0.99)])
def test_discovery_skips_extreme_prices_before_paying_for_an_ai_call(book):
    opps, fast = discover([mkt()], FakeKalshi(v2_market(*book)))
    assert opps == [] and fast.await_count == 0


def test_discovery_skips_when_the_ai_has_no_answer():
    assert discover([mkt()], FakeKalshi(v2_market(0.39, 0.41, 0.59, 0.61)), prediction=(None, None))[0] == []


def test_discovery_filters_a_small_edge():
    assert discover([mkt()], FakeKalshi(v2_market(0.39, 0.41, 0.59, 0.61)), prediction=(0.41, 0.9))[0] == []


def test_discovery_yes_opportunity_fields():
    expiry = int(time.time()) + 5 * 86400
    opps, _ = discover([mkt(expiration_ts=expiry)], FakeKalshi(v2_market(0.39, 0.41, 0.59, 0.61)))
    o = opps[0]
    assert o.recommended_side == "YES"
    assert o.edge == pytest.approx(0.30)
    assert o.expected_return == pytest.approx(0.27)
    assert o.max_loss == pytest.approx(0.40)
    assert o.time_to_expiry == pytest.approx(5.0, abs=0.01)
    assert o.volume == 5000


def test_discovery_no_opportunity_keeps_negative_edge():
    opps, _ = discover([mkt()], FakeKalshi(v2_market(0.59, 0.61, 0.39, 0.41)), prediction=(0.30, 0.9))
    o = opps[0]
    assert o.recommended_side == "NO" and o.edge == pytest.approx(-0.30)
    assert o.max_loss == pytest.approx(0.40)
    assert o.time_to_expiry == 30.0  # no expiry on the market


def test_discovery_floors_a_past_expiry():
    opps, _ = discover([mkt(expiration_ts=1)], FakeKalshi(v2_market(0.39, 0.41, 0.59, 0.61)))
    assert opps[0].time_to_expiry == 0.1


def test_discovery_error_on_one_market_does_not_stop_the_next():
    good = v2_market(0.39, 0.41, 0.59, 0.61)
    kalshi = MagicMock(get_market=AsyncMock(side_effect=[RuntimeError("503"), good]))
    opps, _ = discover([mkt("BAD"), mkt("GOOD")], kalshi)
    assert [o.market_id for o in opps] == ["GOOD"]


def test_discovery_runs_the_immediate_check_only_with_a_db():
    kalshi = FakeKalshi(v2_market(0.39, 0.41, 0.59, 0.61))
    with patch(f"{IMM}._evaluate_immediate_trade", AsyncMock()) as check:
        discover([mkt()], kalshi)
        assert check.await_count == 0
        opps, _ = discover([mkt()], kalshi, db_manager="the-db", capital=123)
    check.assert_awaited_once_with(opps[0], "the-db", kalshi, 123)


# ------------------------------------------------------ unified: initialize

def test_initialize_splits_cash_plus_positions_by_config(db):
    kalshi = EventKalshi(legacy_market(40, 60),
                         [{"event_ticker": "E", "event_exposure_dollars": "5", "realized_pnl_dollars": "1"},
                          {"event_ticker": "F", "event_exposure_dollars": "0"}],
                         balance_cents=100_000, portfolio_cents=50_000)
    config = TradingSystemConfig(market_making_allocation=0.4, directional_trading_allocation=0.5,
                                 arbitrage_allocation=0.1)
    s = UnifiedAdvancedTradingSystem(db, kalshi, MagicMock(), config)
    run(s.async_initialize())
    assert s.total_capital == 1500
    assert (s.market_making_capital, s.directional_capital, s.arbitrage_capital) == pytest.approx((600, 750, 150))
    assert s.market_maker is not None and s.portfolio_optimizer is not None


def test_initialize_low_balance_stays_low(db):
    s = UnifiedAdvancedTradingSystem(db, FakeKalshi(None, balance_cents=500), MagicMock())
    run(s.async_initialize())
    assert s.total_capital == 5


@pytest.mark.parametrize("failure", ["balance", "junk_row"])
def test_initialize_unreadable_portfolio_means_no_capital(db, failure):
    kalshi = EventKalshi(None, [{"event_exposure_dollars": "junk"}] if failure == "junk_row" else [])
    if failure == "balance":
        kalshi.get_balance = AsyncMock(side_effect=RuntimeError("503"))
    s = UnifiedAdvancedTradingSystem(db, kalshi, MagicMock())
    run(s.async_initialize())
    assert (s.total_capital, s.directional_capital, s.market_making_capital) == (0, 0, 0)


# -------------------------------------------------- unified: the whole cycle

def empty_allocation():
    return alloc().__class__(
        allocations={}, total_capital_used=0, expected_portfolio_return=0, portfolio_volatility=0,
        portfolio_sharpe=0, max_portfolio_drawdown=0, diversification_ratio=1, portfolio_var_95=0,
        portfolio_cvar_95=0, aggregate_kelly_fraction=0, portfolio_growth_rate=0)


class Cycle:
    """The managers around one cycle, with both strategies stubbed out."""

    def __init__(self, db, limits_status="OK", cash=None, emergency="close_positions"):
        self.db = db
        self.limits = MagicMock(
            get_position_limits_status=AsyncMock(return_value={
                "status": limits_status, "position_utilization": "1/10", "recommendations": []}),
            enforce_position_limits=AsyncMock(return_value={"action": "positions_closed", "positions_closed": 2}),
        )
        status = {"status": "OK", "reserve_percentage": 50.0, "emergency_status": False} if cash is None else cash
        self.cash = MagicMock(
            get_cash_status=AsyncMock(return_value=status),
            handle_cash_emergency=AsyncMock(return_value=SimpleNamespace(
                action_type=emergency, reason="low", positions_to_close=1)),
        )
        self.system = make_system(db, FakeKalshi(None))
        self.mm = AsyncMock(return_value={"orders_placed": 2, "total_exposure": 40.0, "expected_profit": 1.0})
        self.directional = AsyncMock(return_value=empty_allocation())
        self.system._execute_market_making_strategy = self.mm
        self.system._execute_directional_trading_strategy = self.directional

    def run(self, markets=("a-market",)):
        self.db.get_eligible_markets = AsyncMock(return_value=list(markets))
        with patch("src.jobs.execute.reconcile_stale_failed_rows", AsyncMock()), \
                patch("src.utils.position_limits.PositionLimitsManager", return_value=self.limits), \
                patch("src.utils.cash_reserves.CashReservesManager", return_value=self.cash):
            return run(self.system.execute_unified_trading_strategy())


def test_cycle_runs_both_strategies_and_compiles(db):
    c = Cycle(db)
    results = c.run()
    assert c.mm.await_count == 1 and c.directional.await_count == 1
    assert results.market_making_orders == 2 and results.total_positions == 1
    assert c.limits.enforce_position_limits.await_count == 0


@pytest.mark.parametrize("cash", [
    {"status": "EMERGENCY", "reserve_percentage": 1.0, "emergency_status": True},
    {"status": "ERROR", "message": "unreadable"},  # no emergency_status key counts as an emergency
])
def test_cycle_halts_on_a_cash_emergency(db, cash):
    c = Cycle(db, cash=cash, emergency="halt_trading")
    assert c.run() == TradingSystemResults()
    assert c.mm.await_count == 0 and c.directional.await_count == 0


def test_cycle_continues_when_the_emergency_only_asks_for_closes(db):
    c = Cycle(db, cash={"status": "LOW", "reserve_percentage": 5.0, "emergency_status": True})
    c.run()
    assert c.directional.await_count == 1


@pytest.mark.parametrize("status", ["OVER_LIMIT", "WARNING"])
def test_cycle_enforces_position_limits_when_near_or_over(db, status):
    c = Cycle(db, limits_status=status)
    c.run()
    assert c.limits.enforce_position_limits.await_count == 1


def test_cycle_without_markets_runs_no_strategy(db):
    c = Cycle(db)
    assert c.run(markets=()) == TradingSystemResults()
    assert c.directional.await_count == 0


def test_cycle_error_returns_empty_results(db):
    c = Cycle(db)
    c.limits.get_position_limits_status = AsyncMock(side_effect=RuntimeError("boom"))
    assert c.run() == TradingSystemResults()
    assert c.directional.await_count == 0


# ------------------------------------------------- unified: each strategy

def test_market_making_stops_at_no_opportunities(db):
    s = make_system(db, FakeKalshi(None))
    s.market_maker = MagicMock(analyze_market_making_opportunities=AsyncMock(return_value=[]))
    assert run(s._execute_market_making_strategy(["m"])) == {"orders_placed": 0, "expected_profit": 0.0}
    assert s.market_maker.execute_market_making_strategy.call_count == 0


def test_market_making_takes_one_opportunity_per_hundred_dollars(db):
    s = make_system(db, FakeKalshi(None))
    s.market_making_capital = 250.0
    execute = AsyncMock(return_value={"orders_placed": 4, "expected_profit": 2.0})
    s.market_maker = MagicMock(analyze_market_making_opportunities=AsyncMock(return_value=[1, 2, 3, 4]),
                               execute_market_making_strategy=execute)
    assert run(s._execute_market_making_strategy(["m"]))["orders_placed"] == 4
    execute.assert_awaited_once_with([1, 2], capital=250.0)


def test_market_making_error_reports_nothing_placed(db):
    s = make_system(db, FakeKalshi(None))
    s.market_maker = MagicMock(analyze_market_making_opportunities=AsyncMock(side_effect=RuntimeError("x")))
    assert run(s._execute_market_making_strategy(["m"])) == {"orders_placed": 0, "expected_profit": 0.0}


def directional_system(db, allocation=None):
    s = make_system(db, FakeKalshi(None), directional=70.0)
    s.portfolio_optimizer = MagicMock(total_capital=999.0, _empty_allocation=MagicMock(return_value="EMPTY"))
    seen = []

    async def optimize(opps):
        seen.append(s.portfolio_optimizer.total_capital)
        return allocation

    s.portfolio_optimizer.optimize_portfolio = optimize
    s._execute_portfolio_allocations = AsyncMock(return_value={"positions_created": 1})
    return s, seen


def test_directional_without_opportunities_is_empty(db):
    s, _ = directional_system(db)
    with patch(f"{UTS}.create_market_opportunities_from_markets", AsyncMock(return_value=[])):
        assert run(s._execute_directional_trading_strategy(["m"])) == "EMPTY"


def test_directional_optimizes_on_its_own_capital_and_restores_the_optimizer(db):
    allocation = alloc()
    s, seen = directional_system(db, allocation)
    with patch(f"{UTS}.create_market_opportunities_from_markets", AsyncMock(return_value=["opp"])) as create:
        assert run(s._execute_directional_trading_strategy(["m"])) is allocation
    assert create.await_args.args[-1] == 70.0
    assert seen == [70.0] and s.portfolio_optimizer.total_capital == 999.0
    s._execute_portfolio_allocations.assert_awaited_once_with(allocation, ["opp"])


def test_directional_empty_allocation_executes_nothing(db):
    s, _ = directional_system(db, empty_allocation())
    with patch(f"{UTS}.create_market_opportunities_from_markets", AsyncMock(return_value=["opp"])):
        run(s._execute_directional_trading_strategy(["m"]))
    assert s._execute_portfolio_allocations.await_count == 0


def test_directional_error_is_empty(db):
    s, _ = directional_system(db)
    with patch(f"{UTS}.create_market_opportunities_from_markets", AsyncMock(side_effect=RuntimeError("x"))):
        assert run(s._execute_directional_trading_strategy(["m"])) == "EMPTY"


# ------------------------------------------------- unified: reporting

def test_compile_weights_and_counts(db):
    s = make_system(db, FakeKalshi(None))
    s.total_capital = 100.0
    a = alloc()
    a.total_capital_used = 30.0
    a.expected_portfolio_return = 0.2
    r = s._compile_unified_results({"orders_placed": 4, "total_exposure": 10.0, "expected_profit": 0.4}, a,
                                   {"arbitrage_trades": 1, "arbitrage_exposure": 10.0, "arbitrage_profit": 0.0})
    assert r.total_capital_used == 50.0
    assert r.capital_efficiency == pytest.approx(0.5)
    assert r.portfolio_expected_return == pytest.approx(0.2 * 0.4 + 0.6 * 0.2)
    assert r.expected_annual_return == pytest.approx(r.portfolio_expected_return * 365 / 30)
    assert r.total_positions == 2 + 1 + 1


def test_compile_with_no_capital_has_zero_efficiency(db):
    s = make_system(db, FakeKalshi(None))
    s.total_capital = 0
    assert s._compile_unified_results({}, empty_allocation(), {}).capital_efficiency == 0.0


def test_compile_error_is_empty(db):
    s = make_system(db, FakeKalshi(None))
    assert s._compile_unified_results({}, None, {}) == TradingSystemResults()


def test_rebalance_clock_resets_only_when_due(db):
    s = make_system(db, FakeKalshi(None))
    recent = datetime.now()
    s.last_rebalance = recent
    run(s._manage_risk_and_rebalance(TradingSystemResults()))
    assert s.last_rebalance == recent
    s.last_rebalance = recent - timedelta(hours=s.config.rebalance_frequency_hours + 1)
    run(s._manage_risk_and_rebalance(TradingSystemResults(portfolio_volatility=9, max_portfolio_drawdown=9,
                                                          correlation_score=9)))
    assert s.last_rebalance > recent - timedelta(minutes=1)


def test_risk_check_error_does_not_raise(db):
    s = make_system(db, FakeKalshi(None))
    s.last_rebalance = None
    run(s._manage_risk_and_rebalance(TradingSystemResults()))
    assert s.last_rebalance is None


def test_performance_summary_before_and_after_initialize(db):
    s = make_system(db, FakeKalshi(None))
    before = s.get_system_performance_summary()
    assert before["system_status"] == "pending_init"
    assert before["market_making_performance"]["status"] == "Uninitialized"
    s.market_maker = MagicMock(get_performance_summary=MagicMock(return_value={"net_profit": 3.0}))
    after = s.get_system_performance_summary()
    assert after["system_status"] == "active"
    assert after["market_making_performance"] == {"net_profit": 3.0}
    assert after["risk_limits"]["max_drawdown"] == s.config.max_drawdown_limit


def test_performance_summary_error_is_empty(db):
    s = make_system(db, FakeKalshi(None))
    s.last_rebalance = None
    assert s.get_system_performance_summary() == {}


def test_entry_point_initializes_then_runs(db):
    expected = TradingSystemResults(total_positions=3)
    with patch.object(UnifiedAdvancedTradingSystem, "async_initialize", AsyncMock()) as init, \
            patch.object(UnifiedAdvancedTradingSystem, "execute_unified_trading_strategy",
                         AsyncMock(return_value=expected)):
        assert run(run_unified_trading_system(db, FakeKalshi(None), MagicMock())) is expected
    assert init.await_count == 1


def test_entry_point_error_is_empty(db):
    with patch.object(UnifiedAdvancedTradingSystem, "async_initialize", AsyncMock(side_effect=RuntimeError("x"))):
        assert run(run_unified_trading_system(db, FakeKalshi(None), MagicMock())) == TradingSystemResults()
