"""Market making: the refusals, the AI parse, order tracking and the entry point.

Everything is faked: no network, no credentials, no real orders. analyze is fed the flat
market shape because the real get_market envelope still yields nothing (a strict xfail in
test_strategies_unified_immediate.py, and the owner's decision).
"""
import json
import math
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import settings
from src.strategies import market_making
from src.strategies.market_making import (
    AdvancedMarketMaker,
    LimitOrder,
    run_market_making_strategy,
)

BOOK = {"status": "active", "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.44",
        "no_bid_dollars": "0.56", "no_ask_dollars": "0.60"}
GOOD_AI = json.dumps({"probability": 0.8, "confidence": 0.9})


def run(coro):
    import asyncio
    return asyncio.run(coro)


def market(market_id="T", expiration_ts=None):
    return SimpleNamespace(market_id=market_id, title="t", expiration_ts=expiration_ts)


def maker(book=BOOK, ai=GOOD_AI):
    kalshi = MagicMock()
    kalshi.get_market = AsyncMock(return_value=book)
    kalshi.place_order = AsyncMock(return_value={"order": {"order_id": "o1"}})
    xai = MagicMock()
    xai.get_completion = AsyncMock(return_value=ai)
    return AdvancedMarketMaker(MagicMock(), kalshi, xai)


def opportunities(mm, *markets):
    return run(mm.analyze_market_making_opportunities(list(markets) or [market()]))


# ---------------------------------------------------------------- analyze: refusals

def test_analyze_quotes_an_active_flat_book():
    opp = opportunities(maker())[0]
    assert opp.market_id == "T"
    assert opp.optimal_yes_bid == pytest.approx(0.43)
    assert opp.optimal_yes_size == 250


def test_analyze_quotes_an_open_market():
    assert len(opportunities(maker({**BOOK, "status": "open"}))) == 1


@pytest.mark.parametrize("status", ["closed", "settled", "finalized", "paused", "", None])
def test_analyze_refuses_a_market_that_is_not_active_or_open(status):
    mm = maker({**BOOK, "status": status})
    assert opportunities(mm) == []
    mm.xai_client.get_completion.assert_not_awaited()


def test_analyze_refuses_a_market_with_no_status():
    book = {k: v for k, v in BOOK.items() if k != "status"}
    assert opportunities(maker(book)) == []


@pytest.mark.parametrize("key, value", [
    ("yes_bid_dollars", "NaN"),
    ("no_bid_dollars", "NaN"),
    ("no_bid_dollars", "inf"),
    ("no_bid_dollars", "-0.5"),
])
def test_analyze_refuses_an_unusable_bid(key, value):
    mm = maker({**BOOK, key: value})
    assert opportunities(mm) == []
    mm.xai_client.get_completion.assert_not_awaited()


@pytest.mark.parametrize("bid", ["0.01", "0.99"])
def test_analyze_skips_extreme_yes_prices(bid):
    mm = maker({**BOOK, "yes_bid_dollars": bid})
    assert opportunities(mm) == []
    mm.xai_client.get_completion.assert_not_awaited()


def test_analyze_skips_when_the_market_read_is_empty():
    mm = maker({})
    assert opportunities(mm) == []
    mm.xai_client.get_completion.assert_not_awaited()


def test_analyze_skips_when_there_is_no_analysis():
    mm = maker()
    mm._get_ai_analysis = AsyncMock(return_value=None)
    assert opportunities(mm) == []


def test_analyze_filters_a_market_with_no_edge_on_either_side():
    # .50 estimate against a 48c YES bid and a 48c NO bid: 2c each way, below the 4% floor.
    book = {**BOOK, "yes_bid_dollars": "0.48", "no_bid_dollars": "0.48"}
    assert opportunities(maker(book, json.dumps({"probability": 0.5, "confidence": 0.9}))) == []


def test_analyze_one_bad_market_does_not_stop_the_next():
    mm = maker()
    mm.kalshi_client.get_market = AsyncMock(side_effect=[RuntimeError("down"), BOOK])
    found = opportunities(mm, market("BAD"), market("GOOD"))
    assert [o.market_id for o in found] == ["GOOD"]


def test_analyze_ranks_by_expected_profit():
    mm = maker()
    # A 41c YES ask clamps that bid down to 40c, so the quoted spread is wider and ranks first.
    mm.kalshi_client.get_market = AsyncMock(side_effect=[BOOK, {**BOOK, "yes_ask_dollars": "0.41"}])
    found = opportunities(mm, market("UNCLAMPED"), market("CLAMPED"))
    assert [o.market_id for o in found] == ["CLAMPED", "UNCLAMPED"]
    assert found[0].total_expected_profit > found[1].total_expected_profit


@pytest.mark.parametrize("ai", [None, "not json", "{broken", '{"no": "probability"}'])
def test_ai_fallback_estimate_never_gets_a_quote(ai):
    # The fallbacks invent a .50 estimate at confidence .2 or .3; the edge filter wants .35.
    assert opportunities(maker(ai=ai)) == []


def test_ai_call_raising_never_gets_a_quote():
    mm = maker()
    mm.xai_client.get_completion = AsyncMock(side_effect=RuntimeError("boom"))
    assert opportunities(mm) == []


# ---------------------------------------------------------------- the AI parse

@pytest.mark.parametrize("probability, confidence", [
    (True, 0.9), (0.8, True), (False, 0.9), (0.8, False),
    (1.2, 0.9), (-0.1, 0.9), (0.8, 1.2), (0.8, -0.1),
    ("0.8", 0.9), (0.8, "0.9"), (None, 0.9), (0.8, None),
])
def test_ai_parse_falls_back_on_invalid_figures(probability, confidence):
    ai = json.dumps({"probability": probability, "confidence": confidence})
    got = run(maker(ai=ai)._get_ai_analysis(market()))
    assert (got["probability"], got["confidence"]) == (0.5, 0.3)


@pytest.mark.parametrize("ai", ['{"probability": NaN, "confidence": 0.9}',
                                '{"probability": 0.8, "confidence": NaN}'])
def test_ai_parse_falls_back_on_nan(ai):
    got = run(maker(ai=ai)._get_ai_analysis(market()))
    assert (got["probability"], got["confidence"]) == (0.5, 0.3)


def test_ai_parse_keeps_valid_figures_and_extra_fields():
    ai = 'Sure:\n{"probability": 0.8, "confidence": 0.9, "stability": 0.4}\nDone.'
    got = run(maker(ai=ai)._get_ai_analysis(market()))
    assert got == {"probability": 0.8, "confidence": 0.9, "stability": 0.4}


@pytest.mark.parametrize("probability, confidence", [(0, 0), (1, 1), (0.0, 1.0)])
def test_ai_parse_accepts_the_range_ends(probability, confidence):
    ai = json.dumps({"probability": probability, "confidence": confidence})
    got = run(maker(ai=ai)._get_ai_analysis(market()))
    assert (got["probability"], got["confidence"]) == (probability, confidence)


def test_ai_unavailable_gets_the_lowest_confidence():
    got = run(maker(ai=None)._get_ai_analysis(market()))
    assert (got["probability"], got["confidence"]) == (0.5, 0.2)


def test_ai_exception_is_conservative():
    mm = maker()
    mm.xai_client.get_completion = AsyncMock(side_effect=RuntimeError("boom"))
    got = run(mm._get_ai_analysis(market()))
    assert (got["probability"], got["confidence"]) == (0.5, 0.3)


def test_ai_prompt_carries_the_title_and_low_temperature():
    mm = maker()
    run(mm._get_ai_analysis(SimpleNamespace(market_id="T", title="Will it rain?")))
    args, kwargs = mm.xai_client.get_completion.await_args
    assert "Will it rain?" in args[0]
    assert kwargs == {"max_tokens": 3000, "temperature": 0.1}


# ---------------------------------------------------------------- capital

@pytest.mark.parametrize("capital", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_capital_places_nothing(capital):
    mm = maker()
    opp = opportunities(mm)[0]
    res = run(mm.execute_market_making_strategy([opp], capital=capital))
    assert res == {"orders_placed": 0, "total_exposure": 0.0, "expected_profit": 0.0, "markets_count": 0}
    assert mm.active_orders == {}


@pytest.mark.parametrize("capital", [0.0, -5.0])
def test_zero_or_negative_capital_places_nothing(capital):
    mm = maker()
    opp = opportunities(mm)[0]
    res = run(mm.execute_market_making_strategy([opp], capital=capital))
    assert res["orders_placed"] == 0 and res["markets_count"] == 0


def test_capital_is_split_across_the_markets_it_actually_trades(monkeypatch):
    monkeypatch.setattr(settings.trading, "max_concurrent_markets", 1, raising=False)
    mm = maker()
    opp = opportunities(mm)[0]
    second = opportunities(mm, market("B"))[0]
    res = run(mm.execute_market_making_strategy([opp, second], capital=50.0))
    assert res["markets_count"] == 1
    assert res["total_exposure"] == pytest.approx(50.0, abs=1.0)
    assert list(mm.active_orders) == ["T"]


def test_capital_is_shared_between_two_markets():
    mm = maker()
    first = opportunities(mm)[0]
    second = opportunities(mm, market("B"))[0]
    res = run(mm.execute_market_making_strategy([first, second], capital=50.0))
    assert res["markets_count"] == 2
    assert res["total_exposure"] <= 50.0
    assert res["total_exposure"] == pytest.approx(50.0, abs=2.0)


def test_analyze_still_quotes_a_market_with_a_zero_no_bid():
    # Zero means no bid on that side, which is a real book, unlike a negative or NaN price.
    assert len(opportunities(maker({**BOOK, "no_bid_dollars": "0"}))) == 1


def test_budget_below_cost_scales_both_legs_down():
    mm = maker()
    opp = opportunities(mm)[0]
    orders = run(mm._place_market_making_orders(opp, budget=10.0))
    assert sum(o.quantity * o.price / 100 for o in orders) <= 10.0
    assert all(o.quantity < 250 for o in orders)


def test_budget_above_cost_leaves_sizes_alone():
    mm = maker()
    opp = opportunities(mm)[0]
    orders = run(mm._place_market_making_orders(opp, budget=1000.0))
    assert [o.quantity for o in orders] == [opp.optimal_yes_size, opp.optimal_no_size]


def test_execute_survives_a_placement_error():
    mm = maker()
    opp = opportunities(mm)[0]
    mm._place_market_making_orders = AsyncMock(side_effect=RuntimeError("boom"))
    assert run(mm.execute_market_making_strategy([opp]))["orders_placed"] == 0


# ---------------------------------------------------------------- placing an order

def test_paper_mode_never_calls_the_exchange(monkeypatch):
    monkeypatch.setattr(settings.trading, "live_trading_enabled", False)
    mm = maker()
    order = LimitOrder("T", "YES", 43.0, 5)
    run(mm._place_limit_order(order))
    mm.kalshi_client.place_order.assert_not_awaited()
    assert order.status == "placed"
    assert order.order_id.startswith("sim_T_YES_")


@pytest.mark.parametrize("response", [None, {}, {"error": "x"}])
def test_live_reply_without_an_order_is_failed(response):
    mm = maker()
    mm.kalshi_client.place_order = AsyncMock(return_value=response)
    order = LimitOrder("T", "YES", 43.0, 5)
    with patch.object(settings.trading, "live_trading_enabled", True):
        run(mm._place_limit_order(order))
    assert order.status == "failed" and order.order_id is None


def test_live_order_records_the_exchange_id_and_floors_the_price():
    mm = maker()
    order = LimitOrder("T", "NO", 42.9, 5)
    with patch.object(settings.trading, "live_trading_enabled", True):
        run(mm._place_limit_order(order))
    kwargs = mm.kalshi_client.place_order.await_args.kwargs
    assert kwargs["no_price"] == 42 and "yes_price" not in kwargs
    assert (kwargs["side"], kwargs["action"], kwargs["count"]) == ("no", "buy", 5)
    assert order.status == "placed" and order.order_id == "o1"


# ---------------------------------------------------------------- volatility and sizes

def test_volatility_uses_the_time_to_expiry():
    mm = maker()
    soon = mm._estimate_volatility(0.5, market(expiration_ts=int(time.time()) + 86400))
    far = mm._estimate_volatility(0.5, market(expiration_ts=int(time.time()) + 30 * 86400))
    assert soon == pytest.approx(0.20)  # sqrt(.25 / 1) = .5, capped at .20
    assert far == pytest.approx(math.sqrt(0.25 / 30), rel=0.01)


def test_volatility_defaults_to_seven_days():
    assert maker()._estimate_volatility(0.5, market()) == pytest.approx(math.sqrt(0.25 / 7))


def test_volatility_expired_market_uses_the_tenth_of_a_day_floor():
    assert maker()._estimate_volatility(0.5, market(expiration_ts=1)) == pytest.approx(0.20)


def test_volatility_tenth_of_a_day_floor_shows_on_a_low_price():
    # sqrt(.001 * .999 / .1) is about .0999, well under the .20 cap, so the floor is what binds.
    got = maker()._estimate_volatility(0.001, market(expiration_ts=1))
    assert got == pytest.approx(math.sqrt(0.001 * 0.999 / 0.1))


def test_volatility_floors_at_one_percent():
    assert maker()._estimate_volatility(0.0, market()) == 0.01


def test_volatility_bad_timestamp_returns_the_default():
    assert maker()._estimate_volatility(0.5, market(expiration_ts="soon")) == 0.05


def test_sizes_have_a_floor_of_ten_contracts():
    assert maker()._calculate_optimal_sizes(0.001, 0.001, 0.05, 0.1) == (10, 10)


def test_sizes_on_bad_input_return_the_default():
    assert maker()._calculate_optimal_sizes(0.1, 0.1, 0.05, "high") == (50, 50)


def test_calculation_error_returns_none():
    mm = maker()
    mm._estimate_volatility = MagicMock(side_effect=RuntimeError("boom"))
    assert run(mm._calculate_market_making_opportunity(market(), 0.4, 0.56, 0.8, 0.9)) is None


# ---------------------------------------------------------------- tracked orders

def test_monitor_only_checks_placed_orders():
    mm = maker({"last_price_dollars": "0.90"})
    placed = LimitOrder("T", "YES", 40.0, 1, status="placed")
    failed = LimitOrder("T", "YES", 40.0, 1, status="failed")
    mm.active_orders["T"] = [placed, failed]
    run(mm.monitor_and_update_orders())
    assert (placed.status, failed.status) == ("cancelled", "failed")


def test_monitor_leaves_an_order_that_has_not_moved():
    mm = maker({"last_price_dollars": "0.42"})
    placed = LimitOrder("T", "YES", 40.0, 1, status="placed")
    mm.active_orders["T"] = [placed]
    run(mm.monitor_and_update_orders())
    assert placed.status == "placed"


def test_monitor_survives_an_error_on_one_market():
    mm = maker({"last_price_dollars": "0.90"})
    mm._should_update_order = AsyncMock(side_effect=[RuntimeError("boom"), True])
    second = LimitOrder("B", "YES", 40.0, 1, status="placed")
    mm.active_orders = {"A": [LimitOrder("A", "YES", 40.0, 1, status="placed")], "B": [second]}
    run(mm.monitor_and_update_orders())
    assert second.status == "cancelled"


def test_should_update_is_false_on_an_empty_read():
    mm = maker({})
    assert run(mm._should_update_order(LimitOrder("T", "YES", 40.0, 1))) is False


def test_should_update_is_false_on_a_read_error():
    mm = maker()
    mm.kalshi_client.get_market = AsyncMock(side_effect=RuntimeError("down"))
    assert run(mm._should_update_order(LimitOrder("T", "YES", 40.0, 1))) is False


@pytest.mark.xfail(strict=True, reason=(
    "DEAD CODE, nothing calls monitor_and_update_orders. _should_update_order compares every "
    "order with the YES price, so a NO bid at 40c on a market whose YES last is 60c reads as "
    "20c away when it sits exactly at the market."))
def test_should_update_compares_a_no_order_with_the_no_price():
    mm = maker({"last_price_dollars": "0.60"})
    assert run(mm._should_update_order(LimitOrder("T", "NO", 40.0, 1))) is False


@pytest.mark.xfail(strict=True, reason=(
    "DEAD CODE, nothing calls it. _update_order marks a live order cancelled and never tells "
    "the exchange, so the order keeps resting while the bot believes it is gone."))
def test_update_order_cancels_on_the_exchange():
    mm = maker()
    mm.kalshi_client.cancel_order = AsyncMock(return_value={})
    order = LimitOrder("T", "YES", 40.0, 1, status="placed", order_id="o1")
    run(mm._update_order(order))
    mm.kalshi_client.cancel_order.assert_awaited_once()


@pytest.mark.xfail(strict=True, reason=(
    "get_performance_summary counts failed and cancelled orders as active, so a run where the "
    "exchange refused everything still reports active orders."))
def test_summary_counts_only_live_orders_as_active():
    mm = maker()
    mm.active_orders["T"] = [LimitOrder("T", "YES", 40.0, 1, status="failed"),
                             LimitOrder("T", "NO", 40.0, 1, status="cancelled")]
    assert mm.get_performance_summary()["active_orders"] == 0


def test_summary_reports_the_counters():
    mm = maker()
    mm.active_orders["T"] = [LimitOrder("T", "YES", 40.0, 1, status="placed")]
    mm.filled_orders.append(LimitOrder("T", "NO", 40.0, 1))
    mm.total_pnl, mm.markets_traded, mm.win_rate, mm.total_volume = 1.5, 2, 0.5, 9
    assert mm.get_performance_summary() == {
        "total_pnl": 1.5, "active_orders": 1, "filled_orders": 1,
        "markets_traded": 2, "win_rate": 0.5, "total_volume": 9}


def test_summary_error_returns_empty():
    mm = maker()
    mm.active_orders = None
    assert mm.get_performance_summary() == {}


# ---------------------------------------------------------------- entry point

def entry(markets, book=BOOK, ai=GOOD_AI):
    db = MagicMock()
    db.get_eligible_markets = AsyncMock(return_value=markets)
    mm = maker(book, ai)
    return db, run(run_market_making_strategy(db, mm.kalshi_client, mm.xai_client))


def test_entry_asks_for_liquid_markets():
    db, _ = entry([])
    db.get_eligible_markets.assert_awaited_once_with(volume_min=30000, max_days_to_expiry=365)


def test_entry_with_no_markets():
    assert entry([])[1] == {"error": "No markets available"}


def test_entry_with_no_opportunities():
    assert entry([market()], book={**BOOK, "status": "closed"})[1] == {"opportunities": 0}


def test_entry_places_and_reports():
    res = entry([market()])[1]
    assert res["orders_placed"] == 2 and res["markets_count"] == 1
    assert res["performance"]["active_orders"] == 2


def test_entry_returns_the_error():
    db = MagicMock()
    db.get_eligible_markets = AsyncMock(side_effect=RuntimeError("db down"))
    assert run(run_market_making_strategy(db, MagicMock(), MagicMock())) == {"error": "db down"}
