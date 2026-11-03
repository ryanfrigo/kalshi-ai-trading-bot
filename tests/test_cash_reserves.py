"""Pin the behaviour of src/utils/cash_reserves.py, the pre-trade cash gate.

No network, no credentials: the Kalshi client and DB are fakes. The fake client
records every call so we can prove the gate never sells, cancels or orders.

Known bugs are pinned with xfail(strict=True); fixing one turns it into an
XPASS failure so the marker gets removed.
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
    r = await mgr(FakeClient(balance=0)).check_cash_reserves(0.0)
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
    r = await _check(5.00, 0.0)
    assert r.can_trade is False and r.emergency_status is True
    assert "EMERGENCY" in r.reason and "CRITICAL" not in r.reason


@pytest.mark.asyncio
async def test_just_below_critical_says_critical():
    r = await _check(4.99, 0.0)
    assert r.can_trade is False and "CRITICAL" in r.reason


@pytest.mark.asyncio
async def test_exactly_emergency_threshold_falls_to_minimum_check():
    # 0.2% = $20.00: not emergency, but below minimum 0.5% -> blocked, non-emergency
    r = await _check(20.00, 0.0)
    assert r.can_trade is False and r.emergency_status is False
    assert "below minimum" in r.reason


@pytest.mark.asyncio
async def test_just_below_emergency_threshold_is_emergency():
    r = await _check(19.99, 0.0)
    assert r.emergency_status is True and "EMERGENCY" in r.reason


@pytest.mark.asyncio
async def test_exactly_minimum_reserve_with_zero_trade_is_allowed():
    # 0.5% = $50.00
    r = await _check(50.00, 0.0)
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
async def test_portfolio_value_is_cash_plus_half_dollar_per_contract():
    c = FakeClient(balance=10_000, positions={"positions": [{"quantity": 10}, {"quantity": -4}, {"quantity": 0}]})
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0 + 7.0)


@pytest.mark.asyncio
async def test_portfolio_value_ignores_non_dict_rows_and_response():
    c = FakeClient(balance=10_000, positions={"positions": ["x", None, {"quantity": 2}]})
    assert await mgr(c)._get_portfolio_value() == pytest.approx(101.0)
    c = FakeClient(balance=10_000, positions=[1, 2])
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_portfolio_value_error_falls_back_to_100_dollars():
    c = FakeClient(balance_exc=RuntimeError("x"))
    assert await mgr(c)._get_portfolio_value() == 100.0


# -------------------------------------------------- v2 drift (#101): BUGS

V2_POSITIONS = {"market_positions": [
    {"ticker": "KXA", "position_fp": "-40.00"},
    {"ticker": "KXB", "position_fp": "+10.00"},
]}


@pytest.mark.xfail(strict=True, reason=(
    "BUG src/utils/cash_reserves.py:289-298 — _get_portfolio_value reads response key "
    "'positions' (v2 returns 'market_positions') and row field 'quantity' (v2 rows carry "
    "signed string 'position_fp'), so held contracts are valued at $0 and portfolio == cash. "
    "Same drift as #101; see market_prices.signed_position."))
@pytest.mark.asyncio
async def test_portfolio_value_counts_v2_market_positions():
    c = FakeClient(balance=10_000, positions=V2_POSITIONS)
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0 + 50 * 0.5)


@pytest.mark.xfail(strict=True, reason=(
    "BUG cash_reserves.py:289-298 — consequence of v2 drift: with $20 cash and 50 contracts "
    "held, portfolio is read as $20 so reserve is 100%, not ~44%; the reserve tiers can "
    "never trigger on a live account because the denominator excludes every position."))
@pytest.mark.asyncio
async def test_reserve_pct_reflects_v2_held_positions():
    c = FakeClient(balance=2_000, positions=V2_POSITIONS)
    r = await mgr(c).check_cash_reserves(0.0)
    assert r.cash_reserve_pct == pytest.approx(20 / 45 * 100, rel=1e-3)


@pytest.mark.asyncio
async def test_v2_positions_currently_read_as_zero_value():
    """Characterisation of the bug above (passes today, flips when fixed)."""
    c = FakeClient(balance=10_000, positions=V2_POSITIONS)
    assert await mgr(c)._get_portfolio_value() == pytest.approx(100.0)


# ------------------------------------------------- NaN / negative trade value

@pytest.mark.xfail(strict=True, reason=(
    "BUG cash_reserves.py:72-158 — every check is a '<'/'>' comparison; a NaN proposed trade "
    "value makes them all False, so the gate returns can_trade=True. Fails OPEN."))
@pytest.mark.asyncio
async def test_nan_trade_value_is_refused():
    r = await mgr(FakeClient(balance=1_000_000)).check_cash_reserves(math.nan, portfolio_value=10_000.0)
    assert r.can_trade is False


@pytest.mark.xfail(strict=True, reason=(
    "BUG cash_reserves.py:72-158 — negative trade value is not rejected; it 'adds' cash "
    "and passes every check, including when current reserves are fine. A sign error upstream "
    "sails through."))
@pytest.mark.asyncio
async def test_negative_trade_value_is_refused():
    r = await mgr(FakeClient(balance=1_000_000)).check_cash_reserves(-500.0, portfolio_value=10_000.0)
    assert r.can_trade is False


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


@pytest.mark.xfail(strict=True, reason=(
    "BUG cash_reserves.py:363-370 — is_cash_emergency does status.get('emergency_status', False); "
    "when get_cash_status hits its outer except it returns {'status':'ERROR'} with no such key, "
    "so an error reads as 'no emergency'. Fails OPEN. (unified_trading_system.py:~226 indexes "
    "cash_status['emergency_status'] directly and would KeyError instead.)"))
@pytest.mark.asyncio
async def test_is_cash_emergency_true_when_status_errors(monkeypatch):
    async def boom(self): raise ValueError("x")
    monkeypatch.setattr(CashReservesManager, "_get_portfolio_value", boom)
    assert await is_cash_emergency(FakeDB(), FakeClient()) is True


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

@pytest.mark.xfail(strict=True, reason=(
    "BUG(doc, low) cash_reserves.py:4,8,117,327,330,333 — module docstring and operator-facing "
    "messages say 'above 15%' / '20% optimal', but minimum is 0.5% and optimal 1.0%. Misleads "
    "whoever reads the emergency alert about how much cash is needed."))
@pytest.mark.asyncio
async def test_emergency_messages_quote_the_real_threshold():
    r = await _check(10.00, 0.0)  # emergency tier
    text = " ".join(r.recommended_actions)
    assert "15%" not in text
