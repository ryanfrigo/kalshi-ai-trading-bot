"""Pin the behaviour of src/utils/cash_reserves.py, the pre-trade cash gate.

No network, no credentials: the Kalshi client and DB are fakes. The fake client
records every call so we can prove the gate never sells, cancels or orders.

The five bugs found in the first pass (v2 positions, non-finite/<=0 trade value,
is_cash_emergency on ERROR, unified KeyError, stale 15% text) are fixed and covered.
"""
import math

import pytest

from src.utils import cash_reserves as cr
from src.utils.cash_reserves import (
    CashReservesManager,
    check_can_trade_with_cash_reserves,
    get_max_trade_size_for_reserves,
    is_cash_emergency,
)

READS = {"get_balance", "get_positions"}


class FakeClient:
    """Serves canned balance/positions; records every call; any other method
    (order, sell, cancel...) is recorded as a violation and raises."""

    def __init__(self, balance=10_000, positions=None, balance_exc=None, positions_exc=None):
        self.balance = balance
        self.positions = positions if positions is not None else {"market_positions": []}
        self.balance_exc = balance_exc
        self.positions_exc = positions_exc
        self.calls = []

    async def get_balance(self):
        self.calls.append("get_balance")
        if self.balance_exc:
            raise self.balance_exc
        return self.balance if isinstance(self.balance, dict) else {"balance": self.balance}

    async def get_positions(self, *a, **k):
        self.calls.append("get_positions")
        if self.positions_exc:
            raise self.positions_exc
        return self.positions

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        async def _forbidden(*a, **k):
            self.calls.append(name)
            raise AssertionError(f"cash gate called forbidden client method {name}")

        return _forbidden

    def forbidden_calls(self):
        return [c for c in self.calls if c not in READS]


class FakeDB:
    def __init__(self, positions=(), exc=None):
        self.positions = list(positions)
        self.exc = exc
        self.calls = []

    async def get_open_positions(self):
        self.calls.append("get_open_positions")
        if self.exc:
            raise self.exc
        return self.positions

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        async def _w(*a, **k):
            self.calls.append(name)
            raise AssertionError(f"cash gate wrote to db via {name}")

        return _w


def mgr(client=None, db=None):
    return CashReservesManager(db or FakeDB(), client or FakeClient())


# ---------------------------------------------------------------- thresholds

def test_thresholds_are_the_documented_ones():
    m = mgr()
    assert (m.minimum_reserve_pct, m.optimal_reserve_pct) == (0.5, 1.0)
    assert (m.emergency_threshold_pct, m.critical_threshold_pct) == (0.2, 0.05)
    assert m.max_single_trade_impact == 5.0
    assert m.buffer_for_opportunities == 0.5
    # ordering invariant the if/elif chain depends on
    assert m.critical_threshold_pct < m.emergency_threshold_pct < m.minimum_reserve_pct < m.optimal_reserve_pct


def test_settings_has_no_cash_reserve_knob():
    """Reserve levels are hard-coded in the manager, not read from settings.
    If someone adds a settings field, the manager must be wired to it."""
    from src.config.settings import settings
    names = {n.lower() for n in dir(settings.trading)} | {n.lower() for n in dir(settings)}
    assert not [n for n in names if "reserve" in n]


# --------------------------------------------------------- units: cents -> $

@pytest.mark.asyncio
async def test_balance_is_read_as_cents():
    m = mgr(FakeClient(balance=12_345))
    assert await m._get_available_cash() == pytest.approx(123.45)


@pytest.mark.asyncio
async def test_balance_dollars_field_is_not_used():
    # Kalshi's balance endpoint is cents; a *_dollars key must not be mistaken for it.
    m = mgr(FakeClient(balance={"balance": 500, "balance_dollars": "999.00"}))
    assert await m._get_available_cash() == pytest.approx(5.0)


# ------------------------------------------------------------- fail closed


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [None, "abc", "5000", []])
async def test_non_numeric_balance_blocks_trading(bad):
    ok, reason = await check_can_trade_with_cash_reserves(1.0, FakeDB(), FakeClient(balance=bad))
    assert ok is False
    assert reason


@pytest.mark.asyncio
async def test_missing_balance_field_blocks_trading():
    c = FakeClient(balance={"unrelated": 1})
    ok, _ = await check_can_trade_with_cash_reserves(1.0, FakeDB(), c)
    assert ok is False


@pytest.mark.asyncio
async def test_balance_read_error_blocks_trading_and_reports_emergency():
    c = FakeClient(balance_exc=RuntimeError("503"))
    r = await mgr(c).check_cash_reserves(1.0)
    assert r.can_trade is False
    assert r.emergency_status is True
    assert r.current_cash == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("bal", [0, -1, -50_000])
async def test_zero_or_negative_balance_blocks(bal):
    r = await mgr(FakeClient(balance=bal)).check_cash_reserves(0.01)
    assert r.can_trade is False


@pytest.mark.asyncio
async def test_zero_balance_with_zero_trade_still_blocks():
    r = await mgr(FakeClient(balance=0)).check_cash_reserves()
    assert r.can_trade is False
    assert r.emergency_status is True


@pytest.mark.asyncio
async def test_zero_or_negative_portfolio_value_blocks():
    for pv in (0.0, -10.0):
        r = await mgr(FakeClient(balance=10_000)).check_cash_reserves(1.0, portfolio_value=pv)
        assert r.can_trade is False


@pytest.mark.asyncio
async def test_positions_read_error_does_not_open_the_gate_for_a_big_trade():
    # portfolio value falls back to $100 when positions blow up; a $50 trade
    # against that must be refused (impact 50%).
    c = FakeClient(balance=1_000_000, positions_exc=RuntimeError("boom"))
    r = await mgr(c).check_cash_reserves(50.0)
    assert r.can_trade is False


@pytest.mark.asyncio
async def test_balance_error_only_on_first_read_still_fails_closed():
    """get_balance is called twice (portfolio value, then cash). If only the
    first fails, portfolio falls back to $100 and the second read succeeds."""
    class Flaky(FakeClient):
        n = 0
        async def get_balance(self):
            Flaky.n += 1
            if Flaky.n == 1:
                raise RuntimeError("blip")
            return {"balance": 10_000}
    r = await mgr(Flaky()).check_cash_reserves(10.0)
    # $10 of $100 fallback = 10% impact > 5% -> blocked
    assert r.can_trade is False


@pytest.mark.asyncio
async def test_gate_swallows_internal_exception_and_blocks():
    m = mgr()
    async def boom(*a, **k):
        raise ValueError("x")
    m._get_available_cash = boom
    r = await m.check_cash_reserves(1.0)
    assert (r.can_trade, r.emergency_status) == (False, True)
    assert "Error checking cash reserves" in r.reason


# ------------------------------------------------------------ boundary maths

PV = 10_000.0  # explicit portfolio value so cash is the only free variable


async def _check(cash_dollars, trade, pv=PV):
    c = FakeClient(balance=round(cash_dollars * 100))
    return await mgr(c).check_cash_reserves(trade, portfolio_value=pv)


@pytest.mark.asyncio
async def test_exactly_critical_threshold_is_emergency_not_critical():
    # 0.05% of 10k = $5.00 -> not < critical, but < emergency(0.2%)
    r = await _check(5.00, None)
    assert r.can_trade is False and r.emergency_status is True
    assert "EMERGENCY" in r.reason and "CRITICAL" not in r.reason


@pytest.mark.asyncio
async def test_just_below_critical_says_critical():
    r = await _check(4.99, None)
    assert r.can_trade is False and "CRITICAL" in r.reason


@pytest.mark.asyncio
async def test_exactly_emergency_threshold_falls_to_minimum_check():
    # 0.2% = $20.00: not emergency, but below minimum 0.5% -> blocked, non-emergency
    r = await _check(20.00, None)
    assert r.can_trade is False and r.emergency_status is False
    assert "below minimum" in r.reason


@pytest.mark.asyncio
async def test_just_below_emergency_threshold_is_emergency():
    r = await _check(19.99, None)
    assert r.emergency_status is True and "EMERGENCY" in r.reason


@pytest.mark.asyncio
async def test_exactly_minimum_reserve_with_zero_trade_is_allowed():
    # 0.5% = $50.00
    r = await _check(50.00, None)
    assert r.can_trade is True and r.emergency_status is False


@pytest.mark.asyncio
async def test_trade_leaving_exactly_minimum_is_allowed():
    r = await _check(100.00, 50.00)  # leaves $50 = 0.5%
    assert r.can_trade is True


@pytest.mark.asyncio
async def test_trade_leaving_one_cent_under_minimum_is_blocked():
    r = await _check(100.00, 50.01)
    assert r.can_trade is False
    assert "reduce reserves" in r.reason


@pytest.mark.asyncio
async def test_trade_that_spends_all_cash_is_blocked():
    r = await _check(100.00, 100.00)
    assert r.can_trade is False


@pytest.mark.asyncio
async def test_trade_larger_than_cash_is_blocked():
    r = await _check(100.00, 500.00)
    assert r.can_trade is False


@pytest.mark.asyncio
async def test_trade_impact_at_cap_is_allowed():
    r = await _check(5_000.00, 500.00)  # exactly 5%
    assert r.can_trade is True


@pytest.mark.asyncio
async def test_trade_impact_tolerance_is_one_hundredth_of_a_percent():
    assert (await _check(5_000.00, 500.99)).can_trade is True   # 5.0099%
    r = await _check(5_000.00, 501.01)                           # 5.0101%
    assert r.can_trade is False and "exceeds maximum" in r.reason


@pytest.mark.asyncio
async def test_impact_block_reports_max_trade_size():
    r = await _check(5_000.00, 600.00)
    assert any("$500.00" in a for a in r.recommended_actions)


@pytest.mark.asyncio
async def test_opportunity_buffer_warns_but_still_allows():
    # leave 0.8% (between min 0.5 and min+buffer 1.0)
    r = await _check(100.00, 20.00)
    assert r.can_trade is True
    assert any(a.startswith("Warning") for a in r.recommended_actions)


@pytest.mark.asyncio
async def test_no_warning_when_blocked():
    r = await _check(100.00, 90.00)  # blocked: leaves 0.1%
    assert r.can_trade is False
    assert not any(a.startswith("Warning") for a in r.recommended_actions)


@pytest.mark.asyncio
async def test_healthy_account_is_approved_with_positive_message():
    r = await _check(2_000.00, 100.00)
    assert r.can_trade is True and r.reason == "Cash reserves adequate"
    assert r.required_reserve_pct == 0.5
    assert r.cash_reserve_pct == pytest.approx(20.0)
    assert any("Excellent" in a for a in r.recommended_actions)


@pytest.mark.asyncio
async def test_explicit_portfolio_value_overrides_fetch():
    c = FakeClient(balance=10_000)
    await mgr(c).check_cash_reserves(1.0, portfolio_value=1_000.0)
    assert c.calls.count("get_positions") == 0


# ---------------------------------------------------- portfolio value maths

@pytest.mark.asyncio
async def test_portfolio_value_is_cash_plus_half_dollar_per_contract_without_exposure():
    c = FakeClient(balance=10_000, positions={"market_positions": [
        {"position_fp": "10.00"}, {"position_fp": "-4.00"}, {"position_fp": "0.00"}]})
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0 + 7.0)


@pytest.mark.asyncio
async def test_malformed_rows_are_skipped_not_zeroing_the_portfolio():
    rows = ["x", None, {"position_fp": "abc"}, {"position_fp": None}, {"position_fp": "nan"},
            {"position_fp": "inf", "market_exposure_dollars": "5"},  # non-finite contracts: skipped
            {"position_fp": "2.00"}]
    c = FakeClient(balance=10_000, positions={"market_positions": rows})
    assert await mgr(c)._get_portfolio_value() == pytest.approx(101.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", [[1, 2], {"positions": [{"quantity": 5}]}, {"market_positions": None}, {}])
async def test_unreadable_positions_shape_raises_not_cash_only(resp):
    # Missing/legacy shape must not be read as "no positions" (that would
    # overstate the reserve) nor replaced by an invented value: it raises.
    c = FakeClient(balance=500_000, positions=resp)
    with pytest.raises(ValueError, match="market_positions"):
        await mgr(c)._get_portfolio_value()


@pytest.mark.asyncio
async def test_exposure_dollars_preferred_over_estimate():
    rows = [{"position_fp": "-40.00", "market_exposure_dollars": "12.34"},
            {"position_fp": "10.00", "market_exposure_dollars": "garbage"},   # -> $0.50 estimate
            {"position_fp": "10.00", "market_exposure_dollars": "-3"},        # negative -> estimate
            {"position_fp": "10.00"}]                                          # absent -> estimate
    c = FakeClient(balance=10_000, positions={"market_positions": rows})
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0 + 12.34 + 5.0 * 3)


@pytest.mark.asyncio
async def test_flat_position_with_exposure_is_not_counted():
    rows = [{"position_fp": "0.00", "market_exposure_dollars": "9.99"}]
    c = FakeClient(balance=10_000, positions={"market_positions": rows})
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_portfolio_value_balance_error_raises_no_invented_value():
    c = FakeClient(balance_exc=RuntimeError("x"))
    with pytest.raises(RuntimeError):
        await mgr(c)._get_portfolio_value()


# -------------------------------------------------- v2 drift (#101): BUGS

V2_POSITIONS = {"market_positions": [
    {"ticker": "KXA", "position_fp": "-40.00"},
    {"ticker": "KXB", "position_fp": "+10.00"},
]}


@pytest.mark.asyncio
async def test_portfolio_value_counts_v2_market_positions():
    c = FakeClient(balance=10_000, positions=V2_POSITIONS)
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0 + 50 * 0.5)


@pytest.mark.asyncio
async def test_reserve_pct_reflects_v2_held_positions():
    c = FakeClient(balance=2_000, positions=V2_POSITIONS)
    r = await mgr(c).check_cash_reserves()
    assert r.cash_reserve_pct == pytest.approx(20 / 45 * 100, rel=1e-3)


@pytest.mark.asyncio
async def test_v2_reserve_tiers_now_bind():
    """$20 cash + 50 contracts held ($25 est.): 44% reserve, and a $10 trade
    (22% of the book) is refused by the 5% impact cap, which in the old
    cash-only reading it would have been judged against $20 instead of $45."""
    c = FakeClient(balance=2_000, positions=V2_POSITIONS)
    r = await mgr(c).check_cash_reserves(2.0)   # 4.4% of $45
    assert r.portfolio_value == pytest.approx(45.0) and r.can_trade is True
    r = await mgr(c).check_cash_reserves(3.0)   # 6.7% of $45
    assert r.can_trade is False and "exceeds maximum" in r.reason


@pytest.mark.asyncio
async def test_big_book_small_cash_trips_emergency_tier():
    """$4 cash vs 2,000 contracts ($1000 est.) = 0.4% -> below minimum and blocked.
    Before the fix portfolio==cash, reserve read 100% and this passed."""
    rows = {"market_positions": [{"position_fp": "2000.00"}]}
    r = await mgr(FakeClient(balance=400, positions=rows)).check_cash_reserves(0.01)
    assert r.can_trade is False
    assert r.cash_reserve_pct == pytest.approx(0.4, rel=1e-2)


# ------------------------------------------------- NaN / negative trade value

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -500.0, -0.01, 0.0, "abc", [1]])
async def test_invalid_trade_value_is_refused(bad):
    c = FakeClient(balance=1_000_000)
    r = await mgr(c).check_cash_reserves(bad, portfolio_value=10_000.0)
    assert r.can_trade is False
    assert "Invalid proposed trade value" in r.reason
    assert c.calls == []          # refused before touching the exchange


@pytest.mark.asyncio
async def test_invalid_trade_value_refused_through_convenience_fn():
    ok, reason = await check_can_trade_with_cash_reserves(math.nan, FakeDB(), FakeClient(balance=1_000_000))
    assert ok is False and "Invalid" in reason


@pytest.mark.asyncio
async def test_smallest_positive_trade_is_still_valid():
    r = await _check(2_000.00, 0.01)
    assert r.can_trade is True


@pytest.mark.asyncio
async def test_nan_portfolio_value_is_refused():
    r = await mgr(FakeClient(balance=10_000)).check_cash_reserves(1.0, portfolio_value=math.nan)
    assert r.can_trade is False


# ----------------------------------------------------------- convenience fns

@pytest.mark.asyncio
async def test_convenience_check_matches_manager_and_returns_tuple():
    ok, reason = await check_can_trade_with_cash_reserves(1.0, FakeDB(), FakeClient(balance=10_000))
    assert ok is True and reason == "Cash reserves adequate"


@pytest.mark.asyncio
async def test_convenience_check_blocks_oversized_trade():
    # cash-only portfolio $100 -> $6 is 6% > 5%
    ok, reason = await check_can_trade_with_cash_reserves(6.0, FakeDB(), FakeClient(balance=10_000))
    assert ok is False and "exceeds maximum" in reason


@pytest.mark.asyncio
async def test_max_trade_size_is_cash_above_minimum_reserve():
    # cash 100, portfolio 100, min 0.5% -> 99.50
    v = await get_max_trade_size_for_reserves(FakeDB(), FakeClient(balance=10_000))
    assert v == pytest.approx(99.5)


@pytest.mark.asyncio
async def test_max_trade_size_zero_when_broke_or_unreadable():
    assert await get_max_trade_size_for_reserves(FakeDB(), FakeClient(balance=0)) == 0
    assert await get_max_trade_size_for_reserves(FakeDB(), FakeClient(balance_exc=RuntimeError())) == 0


# --------------------------------------------------------------- status/tiers

@pytest.mark.asyncio
@pytest.mark.parametrize("cash,status,emergency,permitted", [
    (200.0, "EXCELLENT", False, True),    # 2%
    (100.0, "EXCELLENT", False, True),    # exactly optimal 1%
    (99.0, "GOOD", False, True),
    (50.0, "GOOD", False, True),          # exactly minimum
    (49.0, "WARNING", False, False),
    (20.0, "WARNING", False, False),      # exactly emergency threshold
    (19.0, "EMERGENCY", True, False),
    (5.0, "EMERGENCY", True, False),      # exactly critical threshold
    (4.0, "CRITICAL", True, False),
])
async def test_status_tiers(cash, status, emergency, permitted):
    # make portfolio exactly 10,000 via 9,800 worth... use mocked portfolio instead
    m = mgr(FakeClient(balance=round(cash * 100)))
    async def pv(): return 10_000.0
    m._get_portfolio_value = pv
    s = await m.get_cash_status()
    assert (s["status"], s["emergency_status"], s["trading_permitted"]) == (status, emergency, permitted)


@pytest.mark.asyncio
async def test_status_targets_and_shortfalls():
    m = mgr(FakeClient(balance=3_000))   # $30
    async def pv(): return 10_000.0
    m._get_portfolio_value = pv
    s = await m.get_cash_status()
    assert s["cash_shortfall"] == pytest.approx(20.0)      # 50 - 30
    assert s["cash_to_optimal"] == pytest.approx(70.0)     # 100 - 30
    assert s["max_trade_size"] == 0                        # below minimum -> clamped
    assert s["minimum_required"] == 0.5 and s["optimal_target"] == 1.0


@pytest.mark.asyncio
async def test_status_error_shape_has_no_emergency_key():
    m = mgr()
    async def boom(): raise ValueError("x")
    m._get_portfolio_value = boom
    s = await m.get_cash_status()
    assert s["status"] == "ERROR" and "emergency_status" not in s


# ----------------------------------------------------------- is_cash_emergency

@pytest.mark.asyncio
async def test_is_cash_emergency_true_when_broke_false_when_healthy():
    assert await is_cash_emergency(FakeDB(), FakeClient(balance=0)) is True
    assert await is_cash_emergency(FakeDB(), FakeClient(balance=10_000)) is False


@pytest.mark.asyncio
async def test_is_cash_emergency_true_on_unreadable_balance():
    assert await is_cash_emergency(FakeDB(), FakeClient(balance_exc=RuntimeError("x"))) is True


@pytest.mark.asyncio
async def test_is_cash_emergency_true_when_status_errors(monkeypatch):
    async def boom(self): raise ValueError("x")
    monkeypatch.setattr(CashReservesManager, "_get_portfolio_value", boom)
    assert await is_cash_emergency(FakeDB(), FakeClient()) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [{"status": "ERROR", "message": "x"}, {"status": "GOOD"}, {}])
async def test_is_cash_emergency_true_on_error_or_missing_key(monkeypatch, status):
    async def fake(self): return status
    monkeypatch.setattr(CashReservesManager, "get_cash_status", fake)
    assert await is_cash_emergency(FakeDB(), FakeClient()) is True


@pytest.mark.asyncio
async def test_is_cash_emergency_honours_explicit_false(monkeypatch):
    async def fake(self): return {"status": "GOOD", "emergency_status": False}
    monkeypatch.setattr(CashReservesManager, "get_cash_status", fake)
    assert await is_cash_emergency(FakeDB(), FakeClient()) is False


# ---- unified_trading_system consumes get_cash_status

def test_unified_system_halts_instead_of_keyerror_on_error_status():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch
    from src.strategies.unified_trading_system import UnifiedAdvancedTradingSystem

    db = MagicMock()
    db.get_eligible_markets = AsyncMock(return_value=[])
    sysm = UnifiedAdvancedTradingSystem(db, MagicMock(), MagicMock())
    limits = MagicMock()
    limits.get_position_limits_status = AsyncMock(return_value={"status": "OK", "position_utilization": "0/1"})
    cash = MagicMock()
    cash.get_cash_status = AsyncMock(return_value={"status": "ERROR", "message": "boom"})
    cash.handle_cash_emergency = AsyncMock(return_value=SimpleNamespace(
        action_type="halt_trading", reason="r", positions_to_close=0))
    with patch("src.jobs.execute.reconcile_stale_failed_rows", AsyncMock()), \
         patch("src.utils.position_limits.PositionLimitsManager", return_value=limits), \
         patch("src.utils.cash_reserves.CashReservesManager", return_value=cash):
        res = asyncio.run(sysm.execute_unified_trading_strategy())
    cash.handle_cash_emergency.assert_awaited_once()       # treated as an emergency
    db.get_eligible_markets.assert_not_awaited()           # and halted before scanning
    assert res.total_positions == 0


@pytest.mark.asyncio
async def test_is_cash_emergency_only_reads():
    c, d = FakeClient(balance=0), FakeDB(positions=[object()] * 3)
    await is_cash_emergency(d, c)
    assert c.forbidden_calls() == [] and d.calls == []


# --------------------------------------------------------- handle_cash_emergency

@pytest.mark.asyncio
async def test_emergency_no_action_when_reserves_adequate():
    a = await mgr(FakeClient(balance=10_000)).handle_cash_emergency()
    assert (a.action_type, a.urgency, a.positions_to_close, a.expected_cash_freed) == ("no_action", "info", 0, 0.0)


async def _emergency(cash, db_positions, pv=10_000.0):
    m = mgr(FakeClient(balance=round(cash * 100)), FakeDB(positions=db_positions))
    async def f(): return pv
    m._get_portfolio_value = f
    return await m.handle_cash_emergency(), m


@pytest.mark.asyncio
@pytest.mark.parametrize("cash,action,urgency", [
    (4.0, "halt_trading", "critical"),
    (19.0, "close_positions", "critical"),
    (30.0, "raise_alert", "warning"),
])
async def test_emergency_action_tiers(cash, action, urgency):
    a, _ = await _emergency(cash, [object()] * 3)
    assert (a.action_type, a.urgency) == (action, urgency)


@pytest.mark.asyncio
async def test_emergency_estimates_positions_to_close_from_shortfall():
    # shortfall = 50 - 0 = 50 -> 1 position at the $50 estimate
    a, _ = await _emergency(0.0, [object()] * 5)
    assert a.positions_to_close == 1 and a.expected_cash_freed == 50.0
    # shortfall 500 on a $100k book -> capped at positions held
    a, _ = await _emergency(0.0, [object()] * 3, pv=100_000.0)
    assert a.positions_to_close == 3 and a.expected_cash_freed == 150.0


@pytest.mark.asyncio
async def test_emergency_with_no_positions_closes_nothing():
    a, _ = await _emergency(0.0, [])
    assert a.positions_to_close == 0 and a.expected_cash_freed == 0.0


@pytest.mark.asyncio
async def test_emergency_db_error_halts_trading():
    m = mgr(FakeClient(balance=0), FakeDB(exc=RuntimeError("db down")))
    async def f(): return 10_000.0
    m._get_portfolio_value = f
    a = await m.handle_cash_emergency()
    assert a.action_type == "halt_trading" and a.urgency == "critical"


@pytest.mark.asyncio
async def test_emergency_balance_error_halts_trading():
    a = await mgr(FakeClient(balance_exc=RuntimeError("x"))).handle_cash_emergency()
    assert a.action_type == "halt_trading"


@pytest.mark.asyncio
async def test_emergency_handler_only_reports_never_trades():
    """handle_cash_emergency says 'close_positions' but must not do it: no
    sell/cancel/order on the client, no writes on the db."""
    a, m = await _emergency(19.0, [object()] * 4)
    assert a.action_type == "close_positions" and a.positions_to_close >= 1
    assert m.kalshi_client.forbidden_calls() == []
    assert m.db_manager.calls == ["get_open_positions"]


@pytest.mark.asyncio
async def test_full_gate_surface_never_touches_orders():
    c, d = FakeClient(balance=0), FakeDB(positions=[object()])
    m = mgr(c, d)
    await m.check_cash_reserves(10.0)
    await m.get_cash_status()
    await m.handle_cash_emergency()
    await check_can_trade_with_cash_reserves(1.0, d, c)
    await get_max_trade_size_for_reserves(d, c)
    await is_cash_emergency(d, c)
    assert c.forbidden_calls() == []
    assert set(d.calls) <= {"get_open_positions"}


# ------------------------------------------------------------ message drift

@pytest.mark.asyncio
async def test_emergency_messages_quote_the_real_threshold():
    r = await _check(10.00, None)  # emergency tier
    assert "Suspend new trading until above 0.5%" in r.recommended_actions
    text = " ".join(r.recommended_actions + mgr()._get_cash_recommendations(0.1)
                    + mgr()._get_cash_recommendations(0.3) + mgr()._get_cash_recommendations(0.7))
    assert "15%" not in text and "20%" not in text
    assert "above 0.5%" in text and "toward 1.0% optimal" in text


def test_module_docstring_matches_thresholds():
    assert "15%" not in cr.__doc__


# ------------------------------------------------------- positions pagination

class PagedClient(FakeClient):
    """get_positions serves `pages` keyed by cursor; records the kwargs it got."""

    def __init__(self, pages, fail_cursor=None, **kw):
        super().__init__(**kw)
        self.pages, self.fail_cursor, self.kwargs = pages, fail_cursor, []

    async def get_positions(self, *a, **k):
        self.calls.append("get_positions")
        self.kwargs.append(k)
        if k.get("cursor") is not None and k.get("cursor") == self.fail_cursor:
            raise RuntimeError("page 2 failed")
        return self.pages[k.get("cursor")]


def _row(n):
    return {"ticker": f"T{n}", "position_fp": "10.00"}  # $5 estimate each


@pytest.mark.asyncio
async def test_two_pages_are_summed_and_cursor_is_followed():
    c = PagedClient({None: {"market_positions": [_row(1), _row(2)], "cursor": "c2"},
                     "c2": {"market_positions": [_row(3)], "cursor": ""}}, balance=10_000)
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0 + 15.0)
    assert [k.get("cursor") for k in c.kwargs] == [None, "c2"]
    assert all(k.get("limit") == 200 for k in c.kwargs)


@pytest.mark.asyncio
async def test_second_page_changes_the_gate_decision():
    """Page 1 alone: $2 cash vs $5 held => 28% reserve. Page 2 adds $2000 of
    positions => 0.1% reserve => blocked. Reading only page 1 would approve."""
    big = {"position_fp": "4000.00"}
    c = PagedClient({None: {"market_positions": [_row(1)], "cursor": "c2"},
                     "c2": {"market_positions": [big]}}, balance=200)
    r = await mgr(c).check_cash_reserves(0.01)
    assert r.can_trade is False and r.portfolio_value == pytest.approx(2.0 + 5.0 + 2000.0)


@pytest.mark.asyncio
async def test_page_cap_raises_not_partial_sum():
    pages = {None: {"market_positions": [_row(0)], "cursor": "p1"}}
    for i in range(1, 40):
        pages[f"p{i}"] = {"market_positions": [_row(i)], "cursor": f"p{i + 1}"}
    c = PagedClient(pages, balance=10_000_000)
    with pytest.raises(ValueError, match="more than 20 pages"):
        await mgr(c)._fetch_all_positions()
    assert len(c.kwargs) == 20                              # exactly the cap, no more
    with pytest.raises(ValueError, match="more than 20 pages"):
        await mgr(c)._get_portfolio_value()                 # raises, not 20 pages' sum
    assert (await mgr(c).check_cash_reserves(4.0)).can_trade is False


@pytest.mark.asyncio
async def test_exactly_cap_pages_is_accepted():
    pages = {None: {"market_positions": [_row(0)], "cursor": "p1"}}
    for i in range(1, 20):
        nxt = {"cursor": f"p{i + 1}"} if i < 19 else {}
        pages[f"p{i}"] = {"market_positions": [_row(i)], **nxt}
    c = PagedClient(pages, balance=10_000)
    assert len(await mgr(c)._fetch_all_positions()) == 20


@pytest.mark.asyncio
async def test_failure_on_page_two_raises_not_partial_sum():
    c = PagedClient({None: {"market_positions": [_row(1)], "cursor": "c2"}},
                    fail_cursor="c2", balance=10_000_000)
    with pytest.raises(RuntimeError, match="page 2 failed"):
        await mgr(c)._get_portfolio_value()
    assert (await mgr(c).check_cash_reserves(4.0)).can_trade is False


@pytest.mark.asyncio
async def test_malformed_page_two_raises():
    c = PagedClient({None: {"market_positions": [_row(1)], "cursor": "c2"},
                     "c2": {"oops": 1}}, balance=10_000_000)
    with pytest.raises(ValueError, match="market_positions"):
        await mgr(c)._get_portfolio_value()


@pytest.mark.asyncio
async def test_repeated_cursor_raises():
    c = PagedClient({None: {"market_positions": [_row(1)], "cursor": "c2"},
                     "c2": {"market_positions": [_row(2)], "cursor": "c2"}}, balance=10_000)
    with pytest.raises(ValueError, match="repeated cursor"):
        await mgr(c)._fetch_all_positions()


# ------------------------------- positions unreadable => gate blocks (no $100 fallback)

def _positions_fail_clients():
    """Each: $500 cash, positions unreadable in a different way."""
    cap = {None: {"market_positions": [_row(0)], "cursor": "p1"}}
    for i in range(1, 40):
        cap[f"p{i}"] = {"market_positions": [_row(i)], "cursor": f"p{i + 1}"}
    return {
        "read_error": FakeClient(balance=50_000, positions_exc=RuntimeError("boom")),
        "missing_key": FakeClient(balance=50_000, positions={"positions": []}),
        "not_a_list": FakeClient(balance=50_000, positions={"market_positions": None}),
        "page2_failure": PagedClient({None: {"market_positions": [_row(1)], "cursor": "c2"}},
                                     fail_cursor="c2", balance=50_000),
        "cap_hit": PagedClient(cap, balance=50_000),
        "repeated_cursor": PagedClient({None: {"market_positions": [_row(1)], "cursor": "c2"},
                                        "c2": {"market_positions": [], "cursor": "c2"}}, balance=50_000),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(_positions_fail_clients()))
async def test_unreadable_positions_block_a_small_trade_with_ample_cash(case):
    """$500 cash, positions unreadable, $4 trade (0.8% of cash). The old $100
    fallback read this as a 500% reserve and a 4% impact => approved."""
    c = _positions_fail_clients()[case]
    r = await mgr(c).check_cash_reserves(4.0)
    assert r.can_trade is False
    assert r.emergency_status is True
    assert r.reason.startswith("Error checking cash reserves:")
    ok, reason = await check_can_trade_with_cash_reserves(4.0, FakeDB(), _positions_fail_clients()[case])
    assert ok is False and "Error checking cash reserves" in reason


@pytest.mark.asyncio
async def test_unreadable_positions_reason_carries_the_error():
    c = FakeClient(balance=50_000, positions_exc=RuntimeError("positions 503"))
    r = await mgr(c).check_cash_reserves(4.0)
    assert "positions 503" in r.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(_positions_fail_clients()))
async def test_unreadable_positions_make_status_error_and_emergency(case):
    clients = _positions_fail_clients()
    s = await mgr(clients[case]).get_cash_status()
    assert s["status"] == "ERROR" and "emergency_status" not in s
    assert await is_cash_emergency(FakeDB(), _positions_fail_clients()[case]) is True
    a = await mgr(_positions_fail_clients()[case]).handle_cash_emergency()
    assert (a.action_type, a.urgency) == ("halt_trading", "critical")
    assert await get_max_trade_size_for_reserves(FakeDB(), _positions_fail_clients()[case]) == 0


@pytest.mark.asyncio
async def test_readable_positions_still_approve_the_same_trade():
    """Control for the scenario above: same $500 cash and $4 trade is approved
    when positions read fine, so the block is the read failure, not the size."""
    c = FakeClient(balance=50_000)
    assert (await mgr(c).check_cash_reserves(4.0)).can_trade is True
