"""Pins the behaviour of the pre-trade position-limits gate.

No network, no credentials: the Kalshi client and the DB are fakes, and every
call the fake client receives is recorded so the enforcement path can be
proved not to place or modify an order.

Tests marked xfail(strict=True) pin REAL bugs found while writing this file;
they flip to failures (XPASS) the moment the bug is fixed, so the marker must
then be removed.
"""
import math
from datetime import datetime, timedelta

import pytest

from src.config.settings import settings
from src.utils.database import Position
from src.utils.position_limits import (
    PositionLimitsManager,
    check_can_add_position,
    enforce_limits_if_needed,
    get_max_position_size,
)


class FakeClient:
    """Records every call; any order-ish method would show up in .calls."""

    def __init__(self, balance_cents=100_000, positions=None, balance_exc=None,
                 positions_exc=None, balance_resp=None):
        self.balance_cents = balance_cents
        self.positions = positions or []
        self.balance_exc = balance_exc
        self.positions_exc = positions_exc
        self.balance_resp = balance_resp
        self.calls = []

    async def get_balance(self):
        self.calls.append("get_balance")
        if self.balance_exc:
            raise self.balance_exc
        if self.balance_resp is not None:
            return self.balance_resp
        return {"balance": self.balance_cents}

    async def get_positions(self, *a, **k):
        self.calls.append("get_positions")
        if self.positions_exc:
            raise self.positions_exc
        return self.positions_resp()

    def positions_resp(self):
        return {"market_positions": self.positions}

    def __getattr__(self, name):  # any other client method is a recorded call
        if name.startswith("_"):
            raise AttributeError(name)

        async def _rec(*a, **k):
            self.calls.append(name)
            return {}
        return _rec


class FakeDB:
    def __init__(self, positions=None, exc=None):
        self.positions = positions or []
        self.exc = exc
        self.status_updates = []

    async def get_open_positions(self):
        if self.exc:
            raise self.exc
        return list(self.positions)

    async def update_position_status(self, position_id, status):
        self.status_updates.append((position_id, status))


def _pos(i, confidence=0.8, age_h=1, qty=10, price=0.5, stop=0.4):
    return Position(
        market_id=f"KX-{i}", side="NO", entry_price=price, quantity=qty,
        timestamp=datetime.now() - timedelta(hours=age_h), confidence=confidence,
        id=i, stop_loss_price=stop,
    )


def _mgr(client=None, db=None):
    client = client or FakeClient()
    db = db or FakeDB()
    return PositionLimitsManager(db, client), client, db


# ---------------------------------------------------------------- boundaries

async def test_position_count_at_limit_blocks_one_below_allows():
    lim = m_cap()
    m, _, _ = _mgr(db=FakeDB([_pos(i) for i in range(lim - 1)]))
    assert (await m.check_position_limits(1.0, portfolio_value=1000.0)).can_trade is True
    m, _, _ = _mgr(db=FakeDB([_pos(i) for i in range(lim)]))
    r = await m.check_position_limits(1.0, portfolio_value=1000.0)
    assert r.can_trade is False and "limit" in r.reason


async def test_position_size_exactly_at_cap_allowed_one_cent_over_blocked():
    m, _, _ = _mgr(client=FakeClient(balance_cents=100_000))  # $1000 cash
    cap = 1000.0 * m.max_position_size_pct / 100  # $50
    assert (await m.check_position_limits(cap, portfolio_value=1000.0)).can_trade is True
    r = await m.check_position_limits(cap + 0.01, portfolio_value=1000.0)
    assert r.can_trade is False and "exceeds limit" in r.reason


async def test_cash_reserve_boundary():
    # cash $100, portfolio $1000 -> min reserve 0.5% = $5; trade of $95 leaves exactly $5.
    m, _, _ = _mgr(client=FakeClient(balance_cents=10_000))
    m.max_position_size_pct = 100.0  # isolate the cash-reserve check
    assert (await m.check_position_limits(95.0, portfolio_value=1000.0)).can_trade is True
    r = await m.check_position_limits(95.01, portfolio_value=1000.0)
    assert r.can_trade is False and "below minimum" in r.reason


async def test_cannot_spend_more_than_cash_even_when_size_cap_ok():
    m, _, _ = _mgr(client=FakeClient(balance_cents=1_000))  # $10 cash
    r = await m.check_position_limits(40.0, portfolio_value=1000.0)  # under $50 cap
    assert r.can_trade is False and "below minimum" in r.reason


async def test_warning_zone_still_trades_with_recommendation():
    m, _, _ = _mgr(db=FakeDB([_pos(i) for i in range(m_warn())]))
    r = await m.check_position_limits(1.0, portfolio_value=1000.0)
    assert r.can_trade is True
    assert any("Approaching" in a for a in r.recommended_actions)


def m_warn():
    return PositionLimitsManager(FakeDB(), FakeClient()).warning_threshold


async def test_result_reports_configured_numbers():
    m, _, _ = _mgr(db=FakeDB([_pos(1), _pos(2)]))
    r = await m.check_position_limits(10.0, portfolio_value=1000.0)
    assert (r.current_positions, r.max_positions, r.max_position_size) == (2, settings.trading.max_positions, 50.0)


# --------------------------------------------------------------- fail closed

async def test_balance_read_error_blocks():
    m, _, _ = _mgr(client=FakeClient(balance_exc=RuntimeError("boom")))
    r = await m.check_position_limits(1.0)
    assert r.can_trade is False


async def test_balance_error_blocks_even_with_explicit_portfolio_value():
    m, _, _ = _mgr(client=FakeClient(balance_exc=RuntimeError("boom")))
    r = await m.check_position_limits(1.0, portfolio_value=1000.0)
    assert r.can_trade is False


async def test_positions_read_error_still_blocks_a_trade_over_fallback_cap():
    # _get_portfolio_value falls back to $100 -> $5 cap; $20 must not pass.
    m, _, _ = _mgr(client=FakeClient(balance_cents=100_000, positions_exc=RuntimeError("x")))
    r = await m.check_position_limits(20.0)
    assert r.can_trade is False


async def test_db_error_blocks_and_reports_error():
    m, _, _ = _mgr(db=FakeDB(exc=RuntimeError("db down")))
    r = await m.check_position_limits(1.0, portfolio_value=1000.0)
    assert r.can_trade is False and "db down" in r.reason


@pytest.mark.parametrize("resp", [{}, {"balance": None}, {"balance": "n/a"}, {"balance": -500}])
async def test_missing_none_nonnumeric_negative_balance_blocks(resp):
    m, _, _ = _mgr(client=FakeClient(balance_resp=resp))
    r = await m.check_position_limits(1.0)
    assert r.can_trade is False


async def test_zero_balance_blocks_via_convenience_wrapper():
    ok, reason = await check_can_add_position(1.0, FakeDB(), FakeClient(balance_cents=0))
    assert ok is False and reason


async def test_non_dict_balance_response_blocks():
    m, _, _ = _mgr(client=FakeClient(balance_resp=["unexpected"]))
    assert (await m.check_position_limits(1.0)).can_trade is False


async def test_portfolio_value_fallback_is_100_dollars_on_error():
    m, _, _ = _mgr(client=FakeClient(balance_exc=RuntimeError("x")))
    assert await m._get_portfolio_value() == 100.0


async def test_available_cash_is_zero_on_error():
    m, _, _ = _mgr(client=FakeClient(balance_exc=RuntimeError("x")))
    assert await m._get_available_cash() == 0.0


@pytest.mark.parametrize("kw", [dict(balance_exc=RuntimeError("x")), dict(balance_resp={}),
                                 dict(balance_resp={"balance": "n/a"}), dict(balance_cents=0)])
async def test_status_reports_error_not_healthy_when_balance_unreadable(kw):
    m, _, _ = _mgr(client=FakeClient(**kw))
    st = await m.get_position_limits_status()
    assert st["status"] == "ERROR"
    assert st["position_utilization"]  # the unified system formats this key


async def test_status_levels():
    for n, want in [(0, "HEALTHY"), (m_warn(), "WARNING"), (m_cap(), "OVER_LIMIT")]:
        m, _, _ = _mgr(db=FakeDB([_pos(i) for i in range(n)]))
        assert (await m.get_position_limits_status())["status"] == want


# ---------------------------------------------------- invalid proposed size

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -500.0, None, "5"])
async def test_non_finite_or_non_positive_proposed_size_is_rejected(bad):
    m, _, _ = _mgr()
    r = await m.check_position_limits(bad, portfolio_value=1000.0)
    assert r.can_trade is False and "Invalid" in r.reason


async def test_tiny_positive_size_still_allowed():
    m, _, _ = _mgr()
    assert (await m.check_position_limits(0.01, portfolio_value=1000.0)).can_trade is True


# ------------------------------------------ v2 / legacy field drift (#101)

_FP = {"ticker": "KX-A", "position_fp": "-40.00", "market_exposure_dollars": "36.00"}


async def test_portfolio_value_counts_v2_market_positions_using_exposure_dollars():
    m, _, _ = _mgr(client=FakeClient(balance_cents=10_000, positions=[_FP]))
    assert await m._get_portfolio_value() == pytest.approx(136.0)


async def test_portfolio_value_estimates_from_position_fp_without_exposure():
    row = {"ticker": "KX-A", "position_fp": "-40.00"}
    m, _, _ = _mgr(client=FakeClient(balance_cents=10_000, positions=[row]))
    assert await m._get_portfolio_value() == pytest.approx(100.0 + 20.0)


async def test_portfolio_value_ignores_legacy_positions_key():
    class LegacyKeyClient(FakeClient):
        def positions_resp(self):
            return {"positions": [{"quantity": -10}]}
    m, _, _ = _mgr(client=LegacyKeyClient(balance_cents=10_000))
    assert await m._get_portfolio_value() == pytest.approx(100.0)


async def test_portfolio_value_skips_malformed_rows_and_keeps_real_cash():
    rows = [{"position_fp": None, "ticker": "A"}, "junk", {"position_fp": "abc"},
            {"position_fp": "nan"}, {"position_fp": "0.00"}, _FP]
    m, _, _ = _mgr(client=FakeClient(balance_cents=500_000, positions=rows))
    assert await m._get_portfolio_value() == pytest.approx(5036.0)


async def test_portfolio_value_bad_exposure_falls_back_to_estimate():
    row = {"position_fp": "10.00", "market_exposure_dollars": "oops"}
    m, _, _ = _mgr(client=FakeClient(balance_cents=10_000, positions=[row]))
    assert await m._get_portfolio_value() == pytest.approx(105.0)


async def test_held_positions_raise_the_size_cap():
    m, _, _ = _mgr(client=FakeClient(balance_cents=10_000, positions=[_FP]))
    r = await m.check_position_limits(6.0)  # cap = 5% of $136 = $6.80
    assert r.can_trade is True and r.max_position_size == pytest.approx(6.8)


async def test_balance_is_read_as_cents_matching_kalshi_balance_field():
    m, _, _ = _mgr(client=FakeClient(balance_cents=12_345))
    assert await m._get_available_cash() == pytest.approx(123.45)


# ------------------------------------------------ settings agreement

def test_max_positions_comes_from_settings(monkeypatch):
    monkeypatch.setattr(settings.trading, "max_positions", 4)
    m, _, _ = _mgr()
    assert m.max_positions == 4 and m.warning_threshold == 1


def test_size_pct_is_the_tighter_of_module_cap_and_settings(monkeypatch):
    monkeypatch.setattr(settings.trading, "max_position_size_pct", 2.0)
    assert _mgr()[0].max_position_size_pct == 2.0
    monkeypatch.setattr(settings.trading, "max_position_size_pct", 10.0)
    assert _mgr()[0].max_position_size_pct == 5.0


def test_manager_is_never_looser_than_settings():
    m, _, _ = _mgr()
    assert m.max_position_size_pct <= settings.trading.max_position_size_pct
    assert m.max_positions == settings.trading.max_positions


# ----------------------------------------------------- usage-reason text

@pytest.mark.xfail(strict=True, reason="Usage-limit message says 'limit: 97%' but the code tests "
                   "> 100 (position_limits.py:~113); the usage gate is effectively dead because "
                   "the cash check trips first.")
async def test_usage_limit_reason_matches_threshold():
    m, _, _ = _mgr(client=FakeClient(balance_cents=100_000))
    m.max_position_size_pct = 1000.0
    r = await m.check_position_limits(900.0, portfolio_value=1000.0)
    # projected usage = 0 + 90% -> under any stated limit; push over the stated 97%:
    r = await m.check_position_limits(980.0, portfolio_value=1000.0)
    assert r.can_trade is False and "usage" in r.reason


# ------------------------------------------------------------ enforcement

def m_cap():
    return PositionLimitsManager(FakeDB(), FakeClient()).max_positions


async def test_enforce_noop_within_limits_touches_nothing():
    m, client, db = _mgr(db=FakeDB([_pos(i) for i in range(m_cap())]))
    out = await m.enforce_position_limits()
    assert out["action"] == "no_action_needed"
    assert db.status_updates == [] and client.calls == []


async def test_enforce_over_limit_reports_candidates_and_writes_nothing():
    good = [_pos(i, confidence=0.9, age_h=1, qty=1, stop=0.4) for i in range(m_cap())]
    bad = _pos(99, confidence=0.4, age_h=100, qty=500, stop=None)  # riskiest
    m, client, db = _mgr(db=FakeDB(good + [bad]))
    out = await m.enforce_position_limits()
    assert out["action"] == "over_limit_manual_review"
    assert out["over_limit_count"] == 1
    assert out["candidates"] == ["KX-99"]
    assert "positions_closed" not in out and "closed_markets" not in out
    assert db.status_updates == []          # no DB row marked closed
    assert client.calls == []               # no order / cancel / sell


async def test_enforce_over_limit_logs_a_warning():
    m, _, _ = _mgr(db=FakeDB([_pos(i) for i in range(m_cap() + 1)]))
    warnings = []
    m.logger = type("L", (), {"warning": lambda self, msg, *a, **k: warnings.append(msg),
                              "error": lambda *a, **k: None, "info": lambda *a, **k: None})()
    await m.enforce_position_limits()
    assert warnings and "Manual review" in warnings[0]


async def test_force_closure_reviews_down_to_warning_threshold_without_writes():
    m, client, db = _mgr(db=FakeDB([_pos(i) for i in range(m_cap())]))
    out = await m.enforce_position_limits(force_closure=True)
    assert out["action"] == "over_limit_manual_review"
    assert len(out["candidates"]) == m.max_positions - m.warning_threshold
    assert out["over_limit_count"] == 0
    assert db.status_updates == [] and client.calls == []


async def test_enforce_db_error_returns_error_action():
    m, _, _ = _mgr(db=FakeDB(exc=RuntimeError("db")))
    out = await m.enforce_position_limits()
    assert out["action"] == "error"


def test_manager_has_no_db_closing_path():
    assert not hasattr(PositionLimitsManager, "_close_position")


async def test_closure_priority_ranking_factors():
    m, _, _ = _mgr()
    safe = await m._calculate_closure_priority(_pos(1, confidence=0.9, age_h=1, qty=1))
    assert safe == 0.0
    assert await m._calculate_closure_priority(_pos(2, confidence=0.5, age_h=1, qty=1)) == 3.0
    assert await m._calculate_closure_priority(_pos(3, confidence=0.65, age_h=1, qty=1)) == 1.0
    assert await m._calculate_closure_priority(_pos(4, confidence=0.9, age_h=25, qty=1)) == 1.0
    assert await m._calculate_closure_priority(_pos(5, confidence=0.9, age_h=80, qty=1)) == 2.0
    assert await m._calculate_closure_priority(_pos(6, confidence=0.9, age_h=1, qty=200, price=0.5)) == 1.0
    assert await m._calculate_closure_priority(_pos(7, confidence=0.9, age_h=1, qty=1, stop=None)) == 2.0


# ------------------------------------------------------ convenience wrappers

async def test_check_can_add_position_wrapper_matches_manager():
    ok, reason = await check_can_add_position(
        1.0, FakeDB(), FakeClient(balance_cents=100_000))
    assert ok is True and reason == "Position limits satisfied"
    ok, reason = await check_can_add_position(
        1.0, FakeDB([_pos(i) for i in range(m_cap())]), FakeClient(balance_cents=100_000))
    assert ok is False


async def test_enforce_limits_if_needed_never_writes_or_orders():
    client = FakeClient()
    db = FakeDB([_pos(i) for i in range(m_cap() + 1)])
    assert await enforce_limits_if_needed(db, client) is False  # nothing was enforced
    assert db.status_updates == [] and client.calls == []


async def test_get_max_position_size_is_pct_of_portfolio():
    c = FakeClient(balance_cents=100_000)
    assert await get_max_position_size(FakeDB(), c) == pytest.approx(50.0)
