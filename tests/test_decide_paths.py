"""src/jobs/decide.py make_decision_for_market: the buy path behind `cli.py run`.

Mocks only: DB, LLM client and Kalshi client are fakes, the position-limit and cash-reserve
checks are patched, nothing touches the network. The Kalshi fake raises if place_order is
ever called: this job must only *create* a Position; execute.py places the order.

Strict xfails mark real defects; each reason carries a dollar example.
"""
import asyncio
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.config.settings import settings
from src.jobs import decide
from src.jobs.decide import (
    _calculate_dynamic_quantity, estimate_market_volatility, make_decision_for_market)
from src.utils.database import Market, Position

T = settings.trading
LIMITS = "src.utils.position_limits.check_can_add_position"
CASH = "src.utils.cash_reserves.check_can_trade_with_cash_reserves"


def _market(yes=0.60, no=0.40, volume=600, days=3.0, category="Politics", mid="KXT-1"):
    # volume 600: above the AI (200) and edge (500) minimums, below the news-search
    # threshold (1000), so no sentiment/news client is ever constructed.
    return Market(market_id=mid, title="Will X happen?", yes_price=yes, no_price=no,
                  volume=volume, expiration_ts=int(time.time() + days * 86400),
                  category=category, status="active", last_updated=datetime.now())


def _decision(side="YES", conf=0.85, action="BUY"):
    return SimpleNamespace(action=action, side=side, confidence=conf, limit_price=60,
                           reasoning="because")


class Db:
    def __init__(self, daily_cost=0.0, recent=False, count=0):
        self.get_daily_ai_cost = AsyncMock(return_value=daily_cost)
        self.was_recently_analyzed = AsyncMock(return_value=recent)
        self.get_market_analysis_count_today = AsyncMock(return_value=count)
        self.record_market_analysis = AsyncMock()
        self.upsert_daily_cost = AsyncMock()


class Kalshi:
    def __init__(self, balance_cents=100_000):
        self.get_balance = AsyncMock(return_value={"balance": balance_cents})
        self.get_market = AsyncMock(return_value={"market": {"rules": "r"}})
        self.place_order = AsyncMock(side_effect=AssertionError("decide.py must not place orders"))
        self.cancel_order = AsyncMock(side_effect=AssertionError("decide.py must not cancel"))


def _xai(decision):
    return SimpleNamespace(get_trading_decision=AsyncMock(return_value=decision),
                           search=AsyncMock(return_value="news"))


class LimitsManager:
    """Stands in for PositionLimitsManager on the 'even the smallest size fails' branch."""
    open_positions, max_positions = 0, 10

    def __init__(self, *a):
        pass

    async def _get_position_count(self):
        return LimitsManager.open_positions


def _run(market, decision, db=None, kalshi=None, limits=(True, "ok"), cash=(True, "ok")):
    db, kalshi, xai = db or Db(), kalshi or Kalshi(), _xai(decision)
    LimitsManager.open_positions = 0
    with patch(LIMITS, AsyncMock(side_effect=limits if callable(limits) else None,
                                 return_value=None if callable(limits) else limits)), \
         patch(CASH, AsyncMock(return_value=cash)), \
         patch("src.utils.position_limits.PositionLimitsManager", LimitsManager):
        pos = asyncio.run(make_decision_for_market(market, db, xai, kalshi))
    kalshi.place_order.assert_not_awaited()
    kalshi.cancel_order.assert_not_awaited()
    return pos, db, xai, kalshi


def _actions(db):
    return [c.args[1] for c in db.record_market_analysis.await_args_list]


# ---- cost / dedupe / volume / category gates -------------------------------------
@pytest.mark.parametrize("db", [
    Db(daily_cost=T.daily_ai_budget),
    Db(daily_cost=T.daily_ai_budget + 1),
    Db(recent=True),
    Db(count=T.max_analyses_per_market_per_day),
])
def test_budget_and_dedupe_gates_stop_before_any_spend(db):
    pos, db, xai, kalshi = _run(_market(), _decision(), db=db)
    assert pos is None
    xai.get_trading_decision.assert_not_awaited()
    kalshi.get_balance.assert_not_awaited()


def test_just_under_budget_and_count_still_proceeds():
    db = Db(daily_cost=T.daily_ai_budget - 0.01, count=T.max_analyses_per_market_per_day - 1)
    pos, *_ = _run(_market(), _decision(), db=db)
    assert pos is not None


def test_volume_below_ai_threshold_is_skipped_at_the_boundary():
    below = _market(volume=int(T.min_volume_for_ai_analysis) - 1)
    pos, _, xai, _ = _run(below, _decision())
    assert pos is None and xai.get_trading_decision.await_count == 0


def test_position_rationale_is_the_models_reasoning():
    pos, *_ = _run(_market(), _decision())
    assert pos is not None and pos.rationale == "because"


def test_position_rationale_falls_back_when_the_model_gives_none():
    d = _decision()
    d.reasoning = None
    pos, *_ = _run(_market(), d)
    assert pos is not None and pos.rationale == "No reasoning provided by LLM."


def test_excluded_category_is_skipped_case_insensitively(monkeypatch):
    monkeypatch.setattr(T, "exclude_low_liquidity_categories", ["Sports"])
    pos, _, xai, _ = _run(_market(category="SPORTS"), _decision())
    assert pos is None and xai.get_trading_decision.await_count == 0


# ---- standard LLM path -----------------------------------------------------------
def test_buy_yes_creates_a_yes_position_at_the_yes_price_and_never_orders():
    m = _market(yes=0.60, no=0.40)
    pos, db, *_ = _run(m, _decision("YES", 0.85))
    assert isinstance(pos, Position)
    assert (pos.side, pos.entry_price, pos.live) == ("YES", 0.60, False)
    assert pos.quantity > 0 and pos.confidence == 0.85


def test_buy_no_is_priced_at_the_no_price_not_the_yes_price():
    m = _market(yes=0.60, no=0.40)
    pos, *_ = _run(m, _decision("NO", 0.85))
    assert (pos.side, pos.entry_price) == ("NO", 0.40)


@pytest.mark.parametrize("side,yes,no,conf", [
    ("YES", 0.60, 0.40, 0.85), ("NO", 0.60, 0.40, 0.85), ("NO", 0.15, 0.85, 0.95)])
def test_stop_and_take_profit_sit_around_the_held_contracts_own_price(side, yes, no, conf):
    # 66d565a: for BOTH sides entry is the held contract's own price, stop below, target above.
    pos, *_ = _run(_market(yes=yes, no=no), _decision(side, conf))
    assert 0.01 <= pos.stop_loss_price < pos.entry_price < pos.take_profit_price <= 0.99
    assert pos.max_hold_hours >= 6


def test_non_buy_and_missing_decisions_create_nothing():
    for d in (_decision(action="SKIP"), _decision(action="HOLD")):
        pos, *_ = _run(_market(), d)
        assert pos is None
    pos, db, *_ = _run(_market(), None)
    assert pos is None and "SKIP" in _actions(db)


def test_confidence_below_minimum_to_trade_creates_nothing():
    pos, *_ = _run(_market(yes=0.10, no=0.90), _decision("YES", T.min_confidence_to_trade - 0.01))
    assert pos is None


def test_edge_filter_rejects_thin_edge_low_volume_and_far_expiry():
    pos, db, *_ = _run(_market(yes=0.84, no=0.16), _decision("YES", 0.85))   # 1c edge
    assert pos is None and "EDGE_FILTERED" in _actions(db)
    pos, db, *_ = _run(_market(volume=300), _decision("YES", 0.85))          # < min_volume 500
    assert pos is None and "EDGE_FILTERED" in _actions(db)
    far = _market(days=T.max_time_to_expiry_days + 5)
    pos, db, *_ = _run(far, _decision("YES", 0.85))
    assert pos is None and "EDGE_FILTERED" in _actions(db)


def test_buy_is_refused_when_confidence_is_below_the_price_paid():
    pos, *_ = _run(_market(yes=0.80, no=0.20), _decision("YES", 0.50))
    assert pos is None


def test_no_buy_is_refused_when_confidence_is_below_the_no_price():
    pos, *_ = _run(_market(yes=0.30, no=0.70), _decision("NO", 0.45))
    assert pos is None


# ---- position limits and cash reserves -------------------------------------------
def test_limits_exceeded_at_every_size_records_and_returns_none():
    pos, db, *_ = _run(_market(), _decision(), limits=(False, "too big"), db=Db())
    assert pos is None and "POSITION_LIMITS" in _actions(db)
    reasons = [c.args[4] for c in db.record_market_analysis.await_args_list if c.args[1] == "POSITION_LIMITS"]
    assert reasons == ["Position size limit exceeded"]


def test_position_count_at_max_is_reported_as_such():
    db = Db()
    with patch.object(LimitsManager, "open_positions", 10):
        async def go():
            return await make_decision_for_market(_market(), db, _xai(_decision()), Kalshi())
        with patch(LIMITS, AsyncMock(return_value=(False, "n"))), \
             patch(CASH, AsyncMock(return_value=(True, "ok"))), \
             patch("src.utils.position_limits.PositionLimitsManager", LimitsManager):
            LimitsManager.open_positions = 10
            assert asyncio.run(go()) is None
    reasons = [c.args[4] for c in db.record_market_analysis.await_args_list if c.args[1] == "POSITION_LIMITS"]
    assert reasons == ["Position count limit reached"]


def test_limits_reduce_the_size_until_it_fits():
    calls = []

    async def fit_only_when_small(value, *_):
        calls.append(value)
        return (len(calls) >= 3, "x")          # 1st (full) and 2nd (80%) rejected, 3rd (60%) ok

    m = _market()
    full = _calculate_dynamic_quantity(1000.0, 0.60, 0.85 - 0.60)
    pos, *_ = _run(m, _decision(), limits=fit_only_when_small)
    assert 0 < pos.quantity < full
    assert pos.quantity == int(full * 0.60 * 0.60 / 0.60)    # 60% of the original value


def test_small_order_that_no_limit_check_accepts_is_not_created():
    kalshi = Kalshi(balance_cents=4_000)
    assert _calculate_dynamic_quantity(40.0, 0.60, 0.25) == 2           # precondition
    pos, db, *_ = _run(_market(), _decision("YES", 0.85), kalshi=kalshi, limits=(False, "max positions"))
    assert pos is None and "POSITION_LIMITS" in _actions(db)


def test_insufficient_cash_reserve_records_and_returns_none():
    pos, db, *_ = _run(_market(), _decision(), cash=(False, "reserve"))
    assert pos is None and "CASH_RESERVES" in _actions(db)


def test_exception_inside_the_job_is_swallowed_and_recorded_as_error():
    db = Db()
    db.get_daily_ai_cost = AsyncMock(side_effect=RuntimeError("db down"))
    pos, db, *_ = _run(_market(), _decision(), db=db)
    assert pos is None and _actions(db) == ["ERROR"]


# ---- high-confidence near-expiry branch ------------------------------------------
def _near(yes=0.92, no=0.08, hours=6.0):
    m = _market(yes=yes, no=no, days=hours / 24)
    return m


def test_high_confidence_yes_is_priced_at_the_yes_price_with_3_percent_sizing():
    m = _near(yes=0.92)
    pos, db, xai, _ = _run(m, _decision("YES", 0.96))
    assert (pos.side, pos.entry_price, pos.live) == ("YES", 0.92, False)
    # $1000 balance: 3% x (1 + 0.04) = $31.20 -> 33 contracts at 0.92
    assert pos.quantity == 33
    assert "BUY" in _actions(db)
    assert xai.get_trading_decision.await_count == 1        # no second LLM call


def test_high_confidence_branch_cannot_create_a_no_position_at_the_yes_price():
    m = _near(yes=0.92, no=0.08)
    pos, *_ = _run(m, _decision("NO", 0.96))               # LLM says NO with 96%
    # the branch only accepts YES; the standard path then prices NO at the NO price
    assert pos is None or (pos.side == "NO" and pos.entry_price == 0.08)


def test_high_confidence_below_threshold_falls_through_to_standard_path():
    m = _near(yes=0.92)
    pos, _, xai, _ = _run(m, _decision("YES", T.high_confidence_threshold - 0.01))
    assert xai.get_trading_decision.await_count == 2        # HC probe, then the standard call


def test_near_expiry_branch_needs_the_market_odds_and_the_window():
    pos, _, xai, _ = _run(_near(yes=T.high_confidence_market_odds - 0.05, no=0.15), _decision("YES", 0.96))
    assert xai.get_trading_decision.await_count == 1        # went straight to standard
    far = _market(yes=0.92, no=0.08, days=(T.high_confidence_expiry_hours + 24) / 24)
    _, _, xai, _ = _run(far, _decision("YES", 0.96))
    assert xai.get_trading_decision.await_count == 1


def test_high_confidence_branch_honours_position_limits():
    pos, db, xai, _ = _run(_near(yes=0.92), _decision("YES", 0.96), limits=(False, "max positions"))
    assert pos is None and "POSITION_LIMITS" in _actions(db)
    assert xai.get_trading_decision.await_count == 1


def test_high_confidence_branch_honours_cash_reserves():
    pos, db, *_ = _run(_near(yes=0.92), _decision("YES", 0.96), cash=(False, "reserve"))
    assert pos is None and "CASH_RESERVES" in _actions(db)


def test_high_confidence_branch_refuses_confidence_at_or_below_the_yes_price():
    pos, db, *_ = _run(_near(yes=0.97, no=0.03), _decision("YES", 0.96))
    assert pos is None and "EDGE_FILTERED" in _actions(db)


def test_expired_market_does_not_create_a_position():
    m = _market(yes=0.95, no=0.05, days=-2 / 24)
    pos, *_ = _run(m, _decision("YES", 0.97))
    assert pos is None


# ---- sizing ----------------------------------------------------------------------
def test_sizing_is_three_percent_scaled_by_confidence_delta():
    assert _calculate_dynamic_quantity(1000.0, 0.50, 0.0) == 60            # $30 / 0.50
    assert _calculate_dynamic_quantity(1000.0, 0.50, 0.30) == 78           # $39 / 0.50


def test_sizing_is_hard_capped_at_ten_percent_of_balance(monkeypatch):
    monkeypatch.setattr(T, "position_size_multiplier", 50.0)               # absurd scaler
    q = _calculate_dynamic_quantity(1000.0, 0.50, 0.40)
    assert q == 200 and q * 0.50 <= 1000 * T.max_position_size_pct / 100   # exactly $100


def test_sizing_degenerate_inputs_never_go_negative():
    assert _calculate_dynamic_quantity(1000.0, 0.0, 0.2) == 0
    assert _calculate_dynamic_quantity(1000.0, -0.5, 0.2) == 0
    assert _calculate_dynamic_quantity(1000.0, 0.50, -1.0) == 0           # scaler 0
    assert _calculate_dynamic_quantity(0.0, 0.50, 0.3) == 0


def test_zero_quantity_creates_no_position():
    pos, *_ = _run(_market(), _decision(), kalshi=Kalshi(balance_cents=0))
    assert pos is None


# ---- helpers ---------------------------------------------------------------------
def test_volatility_of_a_fifty_cent_market_is_high():
    m = _market(yes=0.50, no=0.50, volume=900, days=7.0)
    assert estimate_market_volatility(m) >= 0.4


def test_time_to_expiry_floor_and_default():
    assert decide.get_time_to_expiry_days(_market(days=-5)) == 0.1
    assert decide.get_time_to_expiry_days(_market(days=2)) == pytest.approx(2, abs=0.01)
