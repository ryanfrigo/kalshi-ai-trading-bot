"""Second layer of SafeCompounder tests (the owner's scheduled live path: `cli.py daily`).

Adds what tests/test_compounder_live_path.py and test_safe_compounder_helpers.py do not
cover: error branches, mid-run API failures, sizing boundaries, the real client's risk
gate and kill switch, v2-vs-v1 field reads, rerun idempotency, market paging, check_fills.

Rules this file keeps: no network, no credentials, no real files (KALSHI_RUNTIME_DIR is a
tmp dir), and no order ever reaches Kalshi: every client is a recording fake. Bugs are
pinned with strict xfails whose reason starts with file:line; nothing in src/ is changed.
"""
import ast
import asyncio
import inspect
import logging
import re
from datetime import datetime, timezone

import pytest

from src.clients.kalshi_client import KalshiClient
from src.strategies import safe_compounder as sc
from src.strategies.safe_compounder import (
    SafeCompounder, _fetch_all_pages, estimate_true_no_prob, market_confidence_score,
)
from tests.test_compounder_live_path import FakeClient, _compounder, _iso, _market, _opp

SRC = "src/strategies/safe_compounder.py"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(*a, **k):
        return None
    monkeypatch.setattr("src.strategies.safe_compounder.asyncio.sleep", _instant)


def _run(coro):
    return asyncio.run(coro)


def _place(client, opps, portfolio=100_000, cash=100_000, dry_run=False, **kw):
    return _run(_compounder(client, dry_run=dry_run, **kw)._place_resting_orders(
        opps, portfolio=portfolio, cash=cash))


# ---- a recording exchange: remembers what it was asked, never cancels --------------
class Exchange(FakeClient):
    """Stateful fake. place_order rests the order (or fills it / raises, as scripted);
    any cancel/amend/other mutating call is recorded in .forbidden instead of raised
    (the compounder swallows exceptions, so raising would hide the call)."""

    MUTATORS = ("cancel", "amend", "decrease", "batch", "delete", "create", "close", "sell")

    def __init__(self, fail=None, fill=None, **kw):
        super().__init__(**kw)
        self.fail = {k: list(v) for k, v in (fail or {}).items()}   # ticker -> [exc, ...]
        self.fill = fill or {}                                       # ticker -> filled count
        self.attempts = []
        self.forbidden = []

    async def place_order(self, **kw):
        self.attempts.append(kw)
        if self.fail.get(kw["ticker"]):
            raise self.fail[kw["ticker"]].pop(0)
        self.placed.append(kw)
        n = self.fill.get(kw["ticker"], 0)
        if n >= kw["count"]:
            self.positions.append({"ticker": kw["ticker"], "position_fp": f"-{kw['count']}.00"})
        else:
            self.orders.append({"ticker": kw["ticker"], "order_id": kw["client_order_id"]})
        return {"order": {"status": "executed" if n else "resting", "fill_count_fp": f"{n}.00"}}

    def __getattr__(self, name):
        if name.startswith("_") or not any(m in name for m in self.MUTATORS):
            raise AttributeError(name)

        async def record(*a, **k):
            self.forbidden.append(name)
            return {}
        return record


# =====================================================================================
# 1. Pure math: estimate_true_no_prob, every branch
# =====================================================================================
@pytest.mark.parametrize("yes,hours,expected", [
    # <= 24h: complement + 4/3/2/1c bump, capped
    (0.05, 10, 0.99), (0.01, 10, 0.99), (0.10, 10, 0.93), (0.15, 10, 0.87), (0.20, 10, 0.81),
    # 24h < h <= 72h
    (0.05, 48, 0.98), (0.10, 48, 0.92), (0.20, 48, 0.81),
    # 72h < h <= 168h
    (0.05, 100, 0.97), (0.10, 100, 0.91), (0.20, 100, 0.80),
    # > 168h
    (0.03, 400, 0.97), (0.04, 400, 0.96), (0.20, 400, 0.80),
])
def test_estimate_true_no_prob_every_branch(yes, hours, expected):
    assert estimate_true_no_prob(yes, hours) == pytest.approx(expected)


def test_expired_horizon_returns_the_bare_complement():
    assert estimate_true_no_prob(0.07, 0) == pytest.approx(0.93)
    assert estimate_true_no_prob(0.07, -5) == pytest.approx(0.93)


def test_estimate_never_exceeds_99c():
    for yes in [i / 100 for i in range(1, 21)]:
        for h in (1, 24, 25, 72, 73, 168, 169, 1000):
            assert estimate_true_no_prob(yes, h) <= 0.99 + 1e-9


@pytest.mark.xfail(strict=True, reason=(
    f"{SRC}:107-108,114-115 min(cap, base+bump) can land BELOW the bare complement: YES last 1c "
    "with 400h left -> est 0.97 although 1-yes = 0.99 (72-168h: 0.98). The model is more "
    "pessimistic than the market's own price at the extreme tail, so it rejects the safest "
    "longshots (conservative; costs missed trades, not losses)."))
def test_estimate_never_undercuts_the_bare_complement():
    for yes in (0.01, 0.02, 0.03):
        for h in (1, 48, 100, 400):
            assert estimate_true_no_prob(yes, h) >= 1 - yes - 1e-9


# =====================================================================================
# 2. market_confidence_score: branches, v2 vs v1 equivalence, malformed rows
# =====================================================================================
_MKT = {"volume_fp": "5000", "last_price_dollars": "0.05", "_days_to_expiry": 10}


def test_confidence_v1_cent_book_scores_the_same_as_the_v2_dollar_book():
    v2 = {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.85", "100"]]}
    v1 = {"yes": [[10, 100]], "no": [[85, 100]]}          # v1 read: prices in cents
    assert market_confidence_score("T", v1, _MKT) == market_confidence_score("T", v2, _MKT)


def test_confidence_v1_cent_last_price_is_normalised():
    v2 = {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.85", "100"]]}
    cents = {**_MKT, "last_price_dollars": 0, "last_price": 5}    # v1 read: 5 cents
    assert market_confidence_score("T", v2, cents) == market_confidence_score("T", v2, _MKT)


def test_confidence_flags_wide_spread_thin_volume_and_price_gap():
    book = {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.70", "100"]]}   # ask .90 bid .70
    score, reason = market_confidence_score(
        "T", book, {"volume_fp": "5", "last_price_dollars": "0.30", "_days_to_expiry": 10})
    assert "wide spread" in reason and "thin volume" in reason and "price gap" in reason
    assert score < 0.4


def test_confidence_healthy_book_is_ok_and_high():
    book = {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.89", "100"]]}
    score, reason = market_confidence_score("T", book, _MKT)
    assert reason == "ok" and score > 0.9


def test_confidence_empty_book_says_no_book():
    score, reason = market_confidence_score("T", {"yes_dollars": [], "no_dollars": []}, _MKT)
    assert "no book" in reason and score < 0.4


def test_confidence_with_one_unparseable_level_degrades_but_never_crashes():
    # the garbage level is dropped from depth, but max() over the raw side raises, so the
    # spread is "unclear": a lower score than the clean book (fails toward skipping)
    book = {"yes_dollars": [["x", "1"], ["0.10", "100"]], "no_dollars": [["y", "1"], ["0.85", "100"]]}
    clean = {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.85", "100"]]}
    bad, ok = market_confidence_score("T", book, _MKT), market_confidence_score("T", clean, _MKT)
    assert bad[0] < ok[0] and bad[1] == "unclear spread"


def test_confidence_all_garbage_prices_degrade_to_unclear_not_a_crash():
    score, reason = market_confidence_score(
        "T", {"yes_dollars": [["x", "1"]], "no_dollars": [["y", "1"]]}, _MKT)
    assert "no book" in reason


def test_confidence_falls_back_to_v1_volume_key_and_zero_day_horizon():
    book = {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.89", "100"]]}
    a = market_confidence_score("T", book, {"volume": 5000, "last_price_dollars": "0.05",
                                            "_days_to_expiry": 0})
    assert a[1] == "ok"       # vol/day = 5000 / max(0,1) -> full volume score


# =====================================================================================
# 3. _find_no_candidates: unparseable close_time, the 200 cap
# =====================================================================================
def test_unparseable_close_time_is_treated_as_thirty_days_out():
    (c,) = _compounder(FakeClient())._find_no_candidates([{**_market(), "close_time": "soon-ish"}])
    assert c["_hours_to_expiry"] == 720 and c["_days_to_expiry"] == 30.0


def test_naive_close_time_falls_back_to_thirty_days_not_a_crash():
    naive = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    (c,) = _compounder(FakeClient())._find_no_candidates([{**_market(), "close_time": naive}])
    assert c["_hours_to_expiry"] == 720      # tz-naive minus aware raises, swallowed -> 30d


def test_missing_close_time_is_thirty_days_out():
    m = _market()
    del m["close_time"]
    (c,) = _compounder(FakeClient())._find_no_candidates([m])
    assert c["_days_to_expiry"] == 30.0


def test_candidate_list_is_capped_at_200_keeping_the_highest_no_probability():
    # 150 short-dated 5c markets (est .99) and 150 long-dated 20c markets (est .80)
    good = [_market(ticker=f"KXG{i}", last="0.05", hours=10, volume="500") for i in range(150)]
    weak = [_market(ticker=f"KXW{i}", last="0.20", hours=400, volume="500") for i in range(150)]
    out = _compounder(FakeClient())._find_no_candidates(weak + good)
    assert len(out) == 200
    assert {c["ticker"] for c in out} >= {m["ticker"] for m in good}
    assert sum(c["ticker"].startswith("KXW") for c in out) == 50


def test_cap_tiebreaks_on_volume_then_soonest_expiry():
    rows = [_market(ticker=f"KXV{i}", last="0.05", hours=10, volume=str(100 + i)) for i in range(201)]
    out = _compounder(FakeClient())._find_no_candidates(rows)
    assert len(out) == 200
    assert "KXV0" not in {c["ticker"] for c in out}          # lowest volume is the one dropped


# =====================================================================================
# 4. _check_orderbook_and_price: boundaries and malformed books
# =====================================================================================
def _scan(market, book=None, yes_bid="0.10", **kw):
    c = _compounder(FakeClient(yes_bid=yes_bid), **kw)
    if book is not None:
        async def ob(ticker, depth=10):
            return book
        c.client.get_orderbook = ob
    return _run(c._check_orderbook_and_price(c._find_no_candidates([market])))


def test_edge_of_exactly_three_cents_is_accepted():
    # yes .05 / 10h -> est .99; yes bid .04 -> ask .96 -> edge 3c (the code accepts ==, the docstring says >)
    assert len(_scan(_market(last="0.05", hours=10), yes_bid="0.04")) == 1


@pytest.mark.xfail(strict=True, reason=(
    f"{SRC}:601-602 edge = true - ask is float subtraction compared with `< 0.03`: a true 3c "
    "edge computes as 0.0299999 and is rejected (14 yes_last/horizon/bid combos). "
    "Example: yes_last .05, 400h, YES bid .08 -> est .95, ask .92, edge 2.9999c. Misses a valid trade."))
def test_float_drift_does_not_reject_an_exact_three_cent_edge():
    assert len(_scan(_market(last="0.05", hours=400), yes_bid="0.08")) == 1


def test_v1_cent_orderbook_keys_are_still_read():
    book = {"orderbook": {"yes": [[10, 100]], "no": [[80, 100]]}}       # v1 read
    (opp,) = _scan(_market(), book=book)
    assert opp["lowest_no_ask"] == pytest.approx(0.90) and opp["best_no_bid"] == pytest.approx(0.80)


def test_orderbook_fp_takes_precedence_over_the_v1_orderbook_key():
    book = {"orderbook_fp": {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.80", "100"]]},
            "orderbook": {"yes": [[30, 100]], "no": [[60, 100]]}}
    (opp,) = _scan(_market(), book=book)
    assert opp["lowest_no_ask"] == pytest.approx(0.90)


def test_v1_one_cent_yes_bid_fails_safe_as_a_dollar_so_no_order_results():
    # ambiguity: a v1 bid of 1 (cent) is read as $1.00 -> NO ask $0.00 -> below the 80c floor
    assert _scan(_market(), book={"orderbook": {"yes": [[1, 100]], "no": [[80, 100]]}}) == []


def test_best_ask_uses_the_highest_yes_bid_and_best_bid_the_highest_no_bid():
    book = {"orderbook_fp": {"yes_dollars": [["0.02", "9"], ["0.08", "9"], ["0.05", "9"]],
                             "no_dollars": [["0.70", "9"], ["0.88", "9"]]}}
    (opp,) = _scan(_market(), book=book)
    assert opp["lowest_no_ask"] == pytest.approx(0.92) and opp["best_no_bid"] == pytest.approx(0.88)


def test_unparseable_yes_bid_means_no_ask_and_no_order():
    book = {"orderbook_fp": {"yes_dollars": [["??", "100"]], "no_dollars": [["0.80", "100"]]}}
    assert _scan(_market(), book=book) == []


def test_unparseable_no_bid_does_not_block_a_valid_ask():
    book = {"orderbook_fp": {"yes_dollars": [["0.10", "100"]], "no_dollars": [["??", "100"]]}}
    (opp,) = _scan(_market(), book=book)
    assert opp["best_no_bid"] == 0


def test_opportunity_carries_v2_volume_and_close_date():
    (opp,) = _scan(_market(volume="321"))
    assert opp["volume"] == 321 and opp["close_time"] == _market()["close_time"][:10]
    assert opp["roi_pct"] == pytest.approx(0.11 / 0.89 * 100)


def test_null_orderbook_sides_skip_the_market_instead_of_aborting_the_scan():
    book = {"orderbook_fp": {"yes_dollars": None, "no_dollars": None}}
    assert _scan(_market(), book=book) == []


def test_null_orderbook_payload_skips_the_market():
    assert _scan(_market(), book={"orderbook_fp": None}) == []


def test_malformed_volume_skips_that_market_only():
    out = _compounder(FakeClient())._find_no_candidates(
        [_market(ticker="KXBAD", volume="n/a"), _market(ticker="KXOK")])
    assert [c["ticker"] for c in out] == ["KXOK"]


def test_null_title_does_not_abort_candidate_filtering():
    out = _compounder(FakeClient())._find_no_candidates([{**_market(), "title": None}])
    assert out == []  # skipped, not aborted: the title exclusion filter can't vouch for it


def test_null_close_time_does_not_abort_the_opportunity_scan():
    assert len(_scan({**_market(), "close_time": None})) == 1


def test_bad_rows_are_skipped_and_good_rows_kept_in_one_batch():
    out = _compounder(FakeClient())._find_no_candidates([
        {**_market(ticker="KXT"), "title": None},
        _market(ticker="KXV", volume="n/a"),
        _market(ticker="KXOK"),
    ])
    assert [c["ticker"] for c in out] == ["KXOK"]


def test_confidence_gate_boundary_is_inclusive_at_the_threshold():
    c = _compounder(FakeClient())
    m = c._find_no_candidates([_market()])
    book = {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.80", "100"]]}
    score, _ = market_confidence_score("KXCPI-26NOV-T3", book, m[0])
    for thr, kept in ((score, 1), (score + 0.001, 0)):
        c2 = _compounder(FakeClient(), min_confidence=thr)
        assert len(_run(c2._check_orderbook_and_price(m))) == kept


def test_one_failed_orderbook_does_not_stop_the_other_markets():
    c = _compounder(FakeClient())
    real = c.client.get_orderbook

    async def flaky(ticker, depth=10):
        if ticker == "KXBAD":
            raise RuntimeError("503")
        return await real(ticker, depth)
    c.client.get_orderbook = flaky
    cands = c._find_no_candidates([_market(ticker="KXBAD"), _market(ticker="KXOK")])
    assert [o["ticker"] for o in _run(c._check_orderbook_and_price(cands))] == ["KXOK"]


def test_progress_logging_paths_run_at_50_and_25_market_marks(caplog):
    caplog.set_level(logging.INFO, logger="src.strategies.safe_compounder")
    rows = [_market(ticker=f"KXP{i}") for i in range(50)]
    c = _compounder(FakeClient())
    assert len(_run(c._check_orderbook_and_price(c._find_no_candidates(rows)))) == 50
    assert "Orderbook progress: 50/50" in caplog.text and "Checked 25/50" in caplog.text


# =====================================================================================
# 5. Sizing boundaries
# =====================================================================================
def _size(portfolio, price=0.89, true_no=0.99, **kw):
    c = _compounder(FakeClient(), **kw)
    return c._calculate_position_size(_opp(price=price, true_no=true_no), portfolio, portfolio)


def test_one_contract_exactly_at_the_cap_is_allowed_and_one_cent_short_is_not():
    assert _size(890, use_kelly=False) == 1       # cap 89c == 1 contract @ 89c
    assert _size(889, use_kelly=False) == 0       # int(88.9) = 88c < 89c


def test_below_one_contract_is_skipped_and_counted_not_ordered():
    client = FakeClient()
    stats = _place(client, [_opp("KXA")], portfolio=800, cash=800)
    assert client.placed == [] and stats["skipped_size"] == 1 and stats["placed"] == 0


def test_zero_or_negative_portfolio_sizes_to_nothing():
    assert _size(0) == 0 and _size(-5_000, use_kelly=False) <= 0


def test_kelly_binds_below_the_ten_percent_cap_when_edge_is_thin():
    # p=.90 at 89c: full kelly .0910, half .0455 -> 455c of 10_000 -> 5 contracts; cap would allow 11
    assert _size(10_000, true_no=0.90) == 5
    assert _size(10_000, true_no=0.90, use_kelly=False) == 11


def test_negative_edge_gives_zero_kelly_size():
    assert _size(100_000, true_no=0.85) == 0


def test_cap_of_200_contracts_applies_with_kelly_too():
    assert _size(50_000_000) == 200


def test_custom_max_position_pct_scales_the_cap():
    assert _size(100_000, use_kelly=False, max_position_pct=0.05) == 56      # 5000 // 89


def test_negative_max_position_pct_never_orders():
    client = FakeClient()
    stats = _place(client, [_opp("KXA")], max_position_pct=-0.1)
    assert client.placed == [] and stats["skipped_size"] == 1


@pytest.mark.xfail(strict=True, reason=(
    f"{SRC}:324,775 sizing uses portfolio_value (open-position value, cash excluded) while the "
    "governor and the printed Total use cash+portfolio. Cash $1000, no positions -> the "
    "compounder sizes 10% of $0 = 0 contracts; with $100 of positions it sizes 10% of $100 = $10 "
    "instead of $110. Under-deploys capital; never over-orders."))
def test_position_cap_is_a_share_of_total_equity_not_of_positions_alone():
    client = Exchange(cash=100_000, pv=0)
    stats = _place(client, [_opp("KXA")], portfolio=0, cash=100_000)
    assert stats["placed"] == 1


def test_dry_run_reports_the_same_size_and_cost_as_a_live_run():
    opps = [_opp("KXA"), _opp("KXB", price=0.93)]
    dry = _place(Exchange(), opps, dry_run=True)
    live_client = Exchange()
    live = _place(live_client, opps)
    assert [o["count"] for o in live_client.placed] == [112, 107]
    for k in ("placed", "total_deployed", "total_potential_profit"):
        assert dry[k] == pytest.approx(live[k])
    assert live["total_deployed"] == pytest.approx(112 * 89 + 107 * 93)


# =====================================================================================
# 6. API errors mid-run: partial orders, no cancels, no in-process retries
# =====================================================================================
def test_one_rejected_order_does_not_stop_later_tickers_and_is_not_retried():
    client = Exchange(fail={"KXA": [RuntimeError("insufficient funds")]})
    stats = _place(client, [_opp("KXA"), _opp("KXB"), _opp("KXC")])
    assert [a["ticker"] for a in client.attempts] == ["KXA", "KXB", "KXC"]   # KXA tried exactly once
    assert [o["ticker"] for o in client.placed] == ["KXB", "KXC"]
    assert stats["errors"] == 1 and stats["placed"] == 2


def test_failed_order_is_excluded_from_deployed_and_profit_totals():
    client = Exchange(fail={"KXA": [RuntimeError("boom")]})
    stats = _place(client, [_opp("KXA"), _opp("KXB")])
    assert stats["total_deployed"] == pytest.approx(112 * 89)


def test_every_attempt_uses_a_fresh_client_order_id():
    client = Exchange(fail={"KXA": [RuntimeError("timeout")]})
    _place(client, [_opp("KXA"), _opp("KXB")])
    ids = [a["client_order_id"] for a in client.attempts]
    assert len(set(ids)) == len(ids) == 2
    assert all(re.fullmatch(r"[0-9a-f-]{36}", i) for i in ids)


def test_only_place_order_is_ever_called_to_mutate_no_cancel_no_amend():
    client = Exchange(fail={"KXA": [RuntimeError("500")]})
    _place(client, [_opp("KXA"), _opp("KXB")])
    assert client.forbidden == []


def test_every_order_is_a_no_side_buy_limit():
    client = Exchange()
    _place(client, [_opp(f"KX{i}") for i in range(4)])
    assert {(o["side"], o["action"], o["type_"]) for o in client.placed} == {("no", "buy", "limit")}
    assert all("yes_price" not in o and 80 <= o["no_price"] <= 99 for o in client.placed)


def test_source_has_one_mutating_call_site_and_it_is_a_no_buy():
    tree = ast.parse(inspect.getsource(sc))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Attribute)
             and n.func.value.attr == "client"]
    names = [n.func.attr for n in calls]
    mutating = [n for n in names if re.search(r"place|cancel|amend|decrease|batch|delete|create|close", n)]
    assert mutating == ["place_order"]
    (call,) = [n for n in calls if n.func.attr == "place_order"]
    kw = {k.arg: k.value.value for k in call.keywords if isinstance(k.value, ast.Constant)}
    assert kw["side"] == "no" and kw["action"] == "buy" and kw["type_"] == "limit"


def test_fractional_or_missing_fill_fields_never_overcount_fills():
    class Odd(Exchange):
        async def place_order(self, **kw):
            self.placed.append(kw)
            return {"order": {"status": "resting", "fill_count_fp": "0.60"}}
    stats = _place(Odd(), [_opp("KXA")])
    assert stats["filled"] == 0 and stats["placed"] == 1


def test_instant_partial_fill_is_counted_and_the_full_order_is_still_booked():
    client = Exchange(fill={"KXA": 40})
    stats = _place(client, [_opp("KXA")])
    assert stats["filled"] == 40 and stats["placed"] == 1
    assert stats["total_deployed"] == pytest.approx(112 * 89)


def test_v2_fill_count_fp_is_preferred_over_the_v1_fill_count():
    class Both(Exchange):
        async def place_order(self, **kw):
            self.placed.append(kw)
            return {"order": {"status": "executed", "fill_count_fp": "7.00", "fill_count": 99}}
    assert _place(Both(), [_opp("KXA")])["filled"] == 7


def test_flat_response_without_an_order_wrapper_is_booked_as_resting():
    class Flat(Exchange):
        async def place_order(self, **kw):
            self.placed.append(kw)
            return {}
    stats = _place(Flat(), [_opp("KXA")])
    assert stats["placed"] == 1 and stats["filled"] == 0


def test_balance_failure_at_start_places_nothing_and_raises():
    class NoBal(Exchange):
        async def get_balance(self):
            raise RuntimeError("401")
    client = NoBal()
    with pytest.raises(RuntimeError):
        _run(_compounder(client).run())
    assert client.placed == []


@pytest.mark.xfail(strict=True, reason=(
    f"{SRC}:370 the post-order get_balance() is unguarded: if it raises after orders were "
    "placed, run() raises and the stats are lost (cli sees a crash although N orders are live). "
    "No money moves wrongly; the audit trail does."))
def test_run_still_returns_stats_when_the_final_balance_read_fails():
    class Flaky(Exchange):
        calls = 0

        async def get_balance(self):
            Flaky.calls += 1
            if Flaky.calls > 1:
                raise RuntimeError("503")
            return await super().get_balance()

        async def _make_authenticated_request(self, *a, **k):
            return {"events": [{"markets": [_market()]}]} if a[1].endswith("/events") else {"markets": []}
    client = Flaky()
    stats = _run(_compounder(client).run())
    assert stats["placed"] == 1


# =====================================================================================
# 7. Rerun idempotency
# =====================================================================================
def test_second_run_over_the_same_opportunities_places_nothing():
    client = Exchange()
    opps = [_opp("KXA"), _opp("KXB")]
    first = _place(client, opps)
    second = _place(client, opps)
    assert first["placed"] == 2 and second["placed"] == 0
    assert second["skipped_existing"] == 2 and len(client.placed) == 2


def test_rerun_after_an_instant_full_fill_is_blocked_by_the_position():
    client = Exchange(fill={"KXA": 112})
    _place(client, [_opp("KXA")])
    assert client.orders == [] and len(client.positions) == 1
    assert _place(client, [_opp("KXA")])["placed"] == 0 and len(client.placed) == 1


def test_rerun_retries_only_the_ticker_that_failed_the_first_time():
    client = Exchange(fail={"KXA": [RuntimeError("503")]})
    opps = [_opp("KXA"), _opp("KXB")]
    _place(client, opps)
    again = _place(client, opps)
    assert [o["ticker"] for o in client.placed] == ["KXB", "KXA"]
    assert again["placed"] == 1 and again["skipped_existing"] == 1


def test_rerun_reorders_once_a_resting_order_was_cancelled_unfilled():
    client = Exchange()
    _place(client, [_opp("KXA")])
    client.orders.clear()                       # expired / cancelled by the operator
    assert _place(client, [_opp("KXA")])["placed"] == 1


def test_dry_run_leaves_no_state_so_a_live_rerun_still_orders():
    client = Exchange()
    _place(client, [_opp("KXA")], dry_run=True)
    assert _place(client, [_opp("KXA")])["placed"] == 1


def test_dedupe_lookup_failure_on_rerun_places_nothing_and_counts_one_error():
    client = Exchange()
    _place(client, [_opp("KXA")])
    client.orders_error = RuntimeError("429")
    stats = _place(client, [_opp("KXA"), _opp("KXB")])
    assert stats["errors"] == 1 and len(client.placed) == 1


def test_position_row_with_no_ticker_fails_closed():
    client = Exchange(positions=[{"position_fp": "3"}])
    stats = _place(client, [_opp("KXA")])
    assert client.placed == [] and stats["errors"] == 1


def test_position_row_with_garbage_size_fails_closed():
    client = Exchange(positions=[{"ticker": "KXZ", "position_fp": "lots"}])
    assert _place(client, [_opp("KXA")])["errors"] == 1 and client.placed == []


def test_positive_and_negative_positions_both_block():
    client = Exchange(positions=[{"ticker": "KXA", "position_fp": "4.00"},
                                 {"ticker": "KXB", "position_fp": "-4.00"}])
    _place(client, [_opp("KXA"), _opp("KXB"), _opp("KXC")])
    assert [o["ticker"] for o in client.placed] == ["KXC"]


def test_legacy_v1_position_key_alone_is_not_read():
    # v1 read FLAGGED: only position_fp is consulted. A row carrying just v1 `position`
    # is treated as flat (documented here; the live API sends position_fp).
    client = Exchange(positions=[{"ticker": "KXA", "position": 5}])
    _place(client, [_opp("KXA")])
    assert len(client.placed) == 1


def test_resting_order_row_with_no_ticker_fails_closed():
    client = Exchange(orders=[{"order_id": "x"}])
    stats = _place(client, [_opp("KXA")])
    assert client.placed == [] and stats["errors"] == 1


def test_fetch_all_pages_returns_every_row_across_pages_and_stops_on_empty_cursor():
    pages = {None: {"orders": [{"i": 1}], "cursor": "a"}, "a": {"orders": [{"i": 2}], "cursor": ""}}

    async def fetch(limit, cursor, **kw):
        assert limit == 200
        return pages[cursor]
    assert _run(_fetch_all_pages(fetch, "orders")) == [{"i": 1}, {"i": 2}]


# =====================================================================================
# 8. Real KalshiClient + real RiskGovernor + KALSHI_RUNTIME_DIR kill switch
# =====================================================================================
def _stack(tmp_path, monkeypatch, markets=None, cash=100_000, pv=0, halt_after_posts=None):
    """Real KalshiClient and RiskGovernor; only the HTTP layer is faked and every POST is
    recorded (nothing leaves the process). The kill switch is the real env-var path."""
    runtime = tmp_path / "rt"
    runtime.mkdir()
    monkeypatch.setenv("KALSHI_RUNTIME_DIR", str(runtime))
    client = KalshiClient.__new__(KalshiClient)
    client.posts, client.gets, client.rest = [], [], []
    markets = markets if markets is not None else [_market(ticker="KXCPI-26NOV-T3")]

    async def fake(method, path, params=None, json_data=None, **kw):
        if method == "POST":
            client.posts.append(json_data)
            client.rest.append({"ticker": json_data["ticker"], "order_id": "x"})
            if halt_after_posts is not None and len(client.posts) >= halt_after_posts:
                (runtime / "TRADING_HALTED").write_text("operator stop mid-run")
            return {"order": {"order_id": "x", "status": "resting", "fill_count": "0.00"}}
        assert method == "GET", f"unexpected {method} {path}"       # never DELETE/PUT
        client.gets.append(path)
        if path.endswith("/portfolio/balance"):
            return {"balance": cash, "portfolio_value": pv}
        if "orderbook" in path:
            return {"orderbook_fp": {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.80", "100"]]}}
        if path.endswith("/events"):
            return {"events": [{"category": "econ", "title": "t", "markets": markets}]}
        if path.endswith("/portfolio/positions"):
            return {"market_positions": []}
        if path.endswith("/portfolio/orders"):
            return {"orders": list(client.rest)}
        return {"markets": []}

    client._make_authenticated_request = fake
    return client, runtime


def _many(n):
    return [_market(ticker=f"KXCPI-26NOV-T{i}") for i in range(n)]


def test_kill_switch_file_stops_every_order_end_to_end(tmp_path, monkeypatch):
    client, runtime = _stack(tmp_path, monkeypatch, markets=_many(3), cash=100_000, pv=100_000)
    (runtime / "TRADING_HALTED").write_text("stop")
    stats = _run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())
    assert client.posts == [] and stats["placed"] == 0 and stats["errors"] == 3


def test_kill_switch_dropped_mid_run_blocks_the_remaining_orders(tmp_path, monkeypatch):
    client, runtime = _stack(tmp_path, monkeypatch, markets=_many(4), cash=100_000, pv=100_000,
                             halt_after_posts=1)
    stats = _run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())
    assert len(client.posts) == 1 and stats["placed"] == 1 and stats["errors"] == 3


def test_relative_runtime_dir_fails_closed(tmp_path, monkeypatch):
    client, _ = _stack(tmp_path, monkeypatch, cash=100_000, pv=100_000)
    monkeypatch.setenv("KALSHI_RUNTIME_DIR", "relative/dir")
    stats = _run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())
    assert client.posts == [] and stats["errors"] == 1


def test_dry_run_with_a_clear_kill_switch_still_posts_nothing(tmp_path, monkeypatch):
    client, _ = _stack(tmp_path, monkeypatch, cash=100_000, pv=100_000)
    stats = _run(SafeCompounder(client, dry_run=True, min_confidence=0.0).run())
    assert client.posts == [] and stats["placed"] == 1


def test_clearing_the_kill_switch_resumes_ordering(tmp_path, monkeypatch):
    client, runtime = _stack(tmp_path, monkeypatch, cash=100_000, pv=100_000)
    flag = runtime / "TRADING_HALTED"
    flag.write_text("stop")
    assert _run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())["placed"] == 0
    flag.unlink()
    assert _run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())["placed"] == 1
    assert len(client.posts) == 1


@pytest.mark.xfail(strict=True, reason=(
    f"{SRC}:303-370 run() never reads TRADING_HALTED itself; the kill switch is enforced only "
    "inside KalshiClient.place_order. Any other client object (or a future place_order path) "
    "bypasses it: with the flag set, a plain client still receives the order."))
def test_run_itself_honours_the_kill_switch_regardless_of_the_client(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "TRADING_HALTED").write_text("stop")

    class Plain(Exchange):
        async def _make_authenticated_request(self, *a, **k):
            return {"events": [{"markets": [_market()]}]} if a[1].endswith("/events") else {"markets": []}
    client = Plain()
    _run(_compounder(client).run())
    assert client.placed == []


def test_governor_cap_boundary_one_contract_costing_exactly_the_cap_is_posted(tmp_path, monkeypatch):
    # equity 890c -> cap $0.89 -> one 89c contract costs exactly the cap; cost > cap is the block
    client, _ = _stack(tmp_path, monkeypatch, cash=890, pv=0)
    c = SafeCompounder(client, dry_run=False, min_confidence=0.0)
    stats = _run(c._place_resting_orders([_opp("KXA")], portfolio=890, cash=890))
    assert stats["placed"] == 1 and len(client.posts) == 1 and client.posts[0]["count"] == "1.00"


def test_governor_position_cap_below_the_compounder_cap_rejects_orders_cleanly(
        tmp_path, monkeypatch):
    from src.config import settings as cfg
    monkeypatch.setattr(cfg.settings.trading, "max_position_size_pct", 5.0)
    client, _ = _stack(tmp_path, monkeypatch, cash=100_000, pv=0)
    c = SafeCompounder(client, dry_run=False, min_confidence=0.0)
    stats = _run(c._place_resting_orders([_opp("KXA")], portfolio=100_000, cash=100_000))
    # compounder sizes 112 x 89c = $99.68 (10%); the governor's 5% cap ($50) blocks it, no POST
    assert client.posts == [] and stats["errors"] == 1 and stats["placed"] == 0


def test_governor_infrastructure_error_fails_closed_for_the_compounder(tmp_path, monkeypatch):
    client, _ = _stack(tmp_path, monkeypatch)
    monkeypatch.setattr("src.risk.risk_governor.RiskGovernor.check",
                        lambda self: (_ for _ in ()).throw(RuntimeError("state unreadable")))
    stats = _run(SafeCompounder(client, dry_run=False, min_confidence=0.0)._place_resting_orders(
        [_opp("KXA")], portfolio=100_000, cash=100_000))
    assert client.posts == [] and stats["errors"] == 1


def test_no_delete_or_put_request_is_ever_made_in_a_full_live_run(tmp_path, monkeypatch):
    client, _ = _stack(tmp_path, monkeypatch, markets=_many(3), cash=100_000, pv=100_000)
    _run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())   # asserts on non-GET/POST
    assert len(client.posts) == 3
    assert all(p["side"] == "ask" and p["time_in_force"] == "good_till_canceled" for p in client.posts)


def test_full_run_rerun_is_idempotent_through_the_real_client(tmp_path, monkeypatch):
    client, _ = _stack(tmp_path, monkeypatch, markets=_many(2), cash=100_000, pv=100_000)
    c = SafeCompounder(client, dry_run=False, min_confidence=0.0)
    assert _run(c.run())["placed"] == 2
    again = _run(c.run())
    assert again["placed"] == 0 and again["skipped_existing"] == 2 and len(client.posts) == 2


def test_v2_wire_prices_are_the_yes_leg_complement(tmp_path, monkeypatch):
    client, _ = _stack(tmp_path, monkeypatch, cash=100_000, pv=100_000)
    _run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())
    (post,) = client.posts
    assert post["price"] == "0.1100" and re.fullmatch(r"\d+\.00", post["count"])


# =====================================================================================
# 9. _fetch_all_markets: paging, caps, fallback, errors
# =====================================================================================
class Paged:
    def __init__(self, events_pages=None, market_pages=None, events_error=None,
                 markets_error_at=None, endless=False):
        self.events_pages, self.market_pages = events_pages or [], market_pages or []
        self.events_error, self.markets_error_at, self.endless = events_error, markets_error_at, endless
        self.event_calls, self.market_calls = [], []

    async def _make_authenticated_request(self, method, path, params=None):
        self.event_calls.append(dict(params))
        if self.events_error and len(self.event_calls) > self.events_error[0]:
            raise self.events_error[1]
        i = len(self.event_calls) - 1
        if self.endless:
            return {"events": [{"category": "c", "title": "t", "markets": [{"ticker": f"E{i}"}]}],
                    "cursor": "same"}
        if i >= len(self.events_pages):
            return {"events": []}
        return {"events": self.events_pages[i],
                "cursor": f"c{i + 1}" if i + 1 < len(self.events_pages) else ""}

    async def get_markets(self, **params):
        self.market_calls.append(dict(params))
        i = len(self.market_calls) - 1
        if self.markets_error_at is not None and i == self.markets_error_at:
            raise RuntimeError("markets 500")
        if self.endless:
            return {"markets": [{"ticker": f"M{i}"}], "cursor": "same"}
        if i >= len(self.market_pages):
            return {"markets": []}
        return {"markets": self.market_pages[i],
                "cursor": f"m{i + 1}" if i + 1 < len(self.market_pages) else ""}


def _events(n, start=0, cat="econ"):
    return [{"category": cat, "title": f"ev{start}",
             "markets": [{"ticker": f"T{start + i}"} for i in range(n)]}]


def _fetch(client):
    return _run(SafeCompounder(client)._fetch_all_markets())


def test_enough_event_markets_skips_the_markets_fallback():
    client = Paged(events_pages=[_events(120)])
    out = _fetch(client)
    assert len(out) == 120 and client.market_calls == []


def test_events_are_paged_with_their_cursor_and_carry_category_and_title():
    client = Paged(events_pages=[_events(60, 0, "econ"), _events(60, 100, "politics")])
    out = _fetch(client)
    assert [c.get("cursor") for c in client.event_calls] == [None, "c1"]
    assert client.event_calls[0] == {"status": "open", "limit": 100, "with_nested_markets": "true"}
    by = {m["ticker"]: m for m in out}
    assert by["T0"]["_event_category"] == "econ" and by["T100"]["_event_category"] == "politics"


def test_duplicate_and_blank_tickers_are_dropped():
    dup = [{"category": "c", "title": "t", "markets": [{"ticker": "A"}, {"ticker": "A"}, {"ticker": ""}, {}]}]
    client = Paged(events_pages=[dup], market_pages=[[{"ticker": "A"}, {"ticker": "B"}, {"ticker": ""}]])
    assert [m["ticker"] for m in _fetch(client)] == ["A", "B"]


def test_few_event_markets_trigger_the_markets_fallback_and_merge_without_duplicates():
    client = Paged(events_pages=[_events(5)], market_pages=[[{"ticker": "T0"}, {"ticker": "NEW"}]])
    out = _fetch(client)
    assert len(out) == 6 and client.market_calls[0] == {"status": "open", "limit": 200}
    assert "_event_category" not in next(m for m in out if m["ticker"] == "NEW")


def test_events_api_failure_falls_back_to_markets(caplog):
    client = Paged(events_error=(0, RuntimeError("events 500")), market_pages=[[{"ticker": "M1"}]])
    caplog.set_level(logging.WARNING, logger="src.strategies.safe_compounder")
    assert [m["ticker"] for m in _fetch(client)] == ["M1"]
    assert "falling back to /markets" in caplog.text


def test_events_failure_mid_paging_keeps_what_was_already_fetched():
    client = Paged(events_pages=[_events(120), _events(60, 500)], events_error=(1, RuntimeError("503")))
    assert len(_fetch(client)) == 120


def test_markets_fallback_error_is_logged_and_ends_paging_with_partial_results(caplog):
    client = Paged(market_pages=[[{"ticker": "A"}], [{"ticker": "B"}]], markets_error_at=1)
    caplog.set_level(logging.ERROR, logger="src.strategies.safe_compounder")
    assert [m["ticker"] for m in _fetch(client)] == ["A"]
    assert "Error fetching markets page" in caplog.text


def test_events_paging_stops_after_101_pages_even_if_the_cursor_never_ends():
    client = Paged(endless=True)
    out = _fetch(client)
    assert len(client.event_calls) == 101 and len(out) >= 101


def test_markets_fallback_paging_stops_after_51_pages():
    client = Paged(market_pages=[[{"ticker": f"X{i}"}] for i in range(80)])
    _fetch(client)
    assert len(client.market_calls) == 51


def test_mve_parlay_tickers_from_the_fallback_are_filtered_by_the_skip_list():
    c = _compounder(FakeClient())
    mve = _market(ticker="KXMVECROSSCATEGORY-S2026ABC")
    assert c._find_no_candidates([mve]) == []


# =====================================================================================
# 10. check_fills
# =====================================================================================
class FillsClient(FakeClient):
    def __init__(self, fills=None, resting=None, fills_error=None, orders_error=None, **kw):
        super().__init__(**kw)
        self.fills, self.resting = fills or [], resting or []
        self.fills_error, self.orders_error_cf = fills_error, orders_error

    async def get_orders(self, **kw):
        if self.orders_error_cf:
            raise self.orders_error_cf
        return {"orders": self.resting}

    async def get_fills(self, limit=100, **kw):
        assert limit == 20
        if self.fills_error:
            raise self.fills_error
        return {"fills": self.fills}


def _cf(client, capsys):
    _run(SafeCompounder(client).check_fills())
    return capsys.readouterr().out


def test_check_fills_prints_balance_and_resting_split_by_side(capsys):
    out = _cf(FillsClient(cash=12_345, pv=6_700, resting=[
        {"side": "no"}, {"side": "no"}, {"side": "yes"}]), capsys)
    assert "Cash: $123.45" in out and "Portfolio: $67.00" in out and "Total: $190.45" in out
    assert "Resting: 2 NO, 1 YES" in out


def test_check_fills_formats_legacy_cent_and_dollar_prices(capsys):
    out = _cf(FillsClient(fills=[
        {"ticker": "KXA", "side": "no", "count": 5, "yes_price": 11, "created_time": "2026-11-02T10:11:12Z"},
        {"ticker": "KXB", "side": "yes", "count": 2, "yes_price": 0.5, "created_time": "2026-11-02T10:11:12Z"},
    ]), capsys)
    assert "2026-11-02T10:11 | no x5 @ $0.11 | KXA" in out
    assert "yes x2 @ $0.50 | KXB" in out


def test_check_fills_string_price_is_shown_with_a_cent_sign(capsys):
    out = _cf(FillsClient(fills=[{"ticker": "KXA", "side": "no", "count": 1, "yes_price": "n/a"}]), capsys)
    assert "n/a¢" in out


def test_check_fills_swallows_orders_and_fills_errors_and_still_prints_balance(capsys):
    out = _cf(FillsClient(orders_error=RuntimeError("500"), fills_error=RuntimeError("500")), capsys)
    assert "Cash:" in out and "Resting" not in out and "Last 20 fills" not in out


def test_check_fills_balance_failure_propagates():
    class NoBal(FillsClient):
        async def get_balance(self):
            raise RuntimeError("401")
    with pytest.raises(RuntimeError):
        _run(SafeCompounder(NoBal()).check_fills())


def test_check_fills_reads_v2_fill_fields(capsys):
    out = _cf(FillsClient(fills=[{"ticker": "KXA", "side": "no", "count_fp": "5.00",
                                  "no_price_dollars": "0.8900", "yes_price_dollars": "0.1100",
                                  "created_time": "2026-11-02T10:11:12Z"}]), capsys)
    assert "no x5.00 @ $0.89" in out or "no x5 @ $0.89" in out


def test_check_fills_shows_the_no_price_for_a_no_fill(capsys):
    out = _cf(FillsClient(fills=[{"ticker": "KXA", "side": "no", "count": 5, "yes_price": 11,
                                  "no_price": 89, "created_time": "2026-11-02T10:11:12Z"}]), capsys)
    assert "no x5 @ $0.89" in out


def test_check_fills_v2_yes_fill_uses_the_yes_leg_price(capsys):
    out = _cf(FillsClient(fills=[{"ticker": "KXB", "side": "yes", "count_fp": "2.50",
                                  "yes_price_dollars": "0.0700", "no_price_dollars": "0.9300",
                                  "created_time": "2026-11-02T10:11:12Z"}]), capsys)
    assert "yes x2.5 @ $0.07 | KXB" in out


def test_v1_cent_last_price_is_normalised_in_the_reported_opportunity():
    m = {"ticker": "KXA", "title": "t", "volume": 50, "last_price": 5, "close_time": _iso(10)}  # v1: 5 cents
    (opp,) = _scan(m)
    assert opp["yes_last"] == pytest.approx(0.05) and opp["volume"] == 50
