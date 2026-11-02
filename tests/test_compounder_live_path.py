"""The owner's scheduled live path: `cli.py daily` -> SafeCompounder.run, and the agent's
place_guarded_order. Filters, sizing, dedupe, dry-run and halt behaviour.

Mocks only: no network, no credentials, no real files (governor state lives in tmp_path).
Strict xfails mark real defects; each carries a dollar example in its reason.
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.kalshi_client import KalshiClient
from src.risk.risk_governor import GovernorDecision, RiskGovernor
from src.strategies.safe_compounder import SafeCompounder

import cli


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(*a, **k):
        return None
    monkeypatch.setattr("src.strategies.safe_compounder.asyncio.sleep", _instant)


def _iso(hours):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def _market(ticker="KXCPI-26NOV-T3", last="0.05", volume="500", hours=20.0, title="Will CPI exceed 3%?"):
    m = {"ticker": ticker, "title": title, "volume_fp": volume, "close_time": _iso(hours)}
    if last is not None:
        m["last_price_dollars"] = last
    return m


def _opp(ticker="KXA", price=0.89, true_no=0.99):
    return {"ticker": ticker, "title": "t", "our_price": price, "true_no_prob": true_no,
            "lowest_no_ask": round(price + 0.01, 2), "edge": round(true_no - price - 0.01, 2),
            "profit": 1.0 - price, "roi_pct": 12.0, "annualized_roi": 500.0,
            "days_to_expiry": 1.0, "yes_last": 0.05}


class FakeClient:
    """Compounder-facing client double. Records every order it is asked to place."""

    def __init__(self, cash=100_000, pv=100_000, positions=None, orders=None,
                 orders_error=None, positions_error=None, yes_bid="0.10",
                 orders_cursor=None, positions_cursor=None):
        self.cash, self.pv = cash, pv
        self.positions, self.orders = positions or [], orders or []
        self.orders_error, self.positions_error = orders_error, positions_error
        self.yes_bid = yes_bid
        self.orders_cursor, self.positions_cursor = orders_cursor, positions_cursor
        self.placed = []

    async def get_balance(self):
        return {"balance": self.cash, "portfolio_value": self.pv}

    async def get_positions(self, **kw):
        if self.positions_error:
            raise self.positions_error
        return {"market_positions": self.positions, "cursor": self.positions_cursor}

    async def get_orders(self, **kw):
        if self.orders_error:
            raise self.orders_error
        return {"orders": self.orders, "cursor": self.orders_cursor}

    async def get_orderbook(self, ticker, depth=10):
        return {"orderbook_fp": {"yes_dollars": [[self.yes_bid, "100"]],
                                 "no_dollars": [["0.80", "100"]]}}

    async def place_order(self, **kw):
        self.placed.append(kw)
        return {"order": {"status": "resting", "fill_count": "0.00"}}


def _compounder(client, dry_run=False, **kw):
    kw.setdefault("min_confidence", 0.0)
    return SafeCompounder(client, db_path="unused.db", dry_run=dry_run, **kw)


def _scan(market, yes_bid):
    """Candidate filter + orderbook/edge filter for one market; returns opportunities."""
    c = _compounder(FakeClient(yes_bid=yes_bid))
    return asyncio.run(c._check_orderbook_and_price(c._find_no_candidates([market])))


# ---- candidate filter: YES last <= 20c, volume, skips, expiry --------------------
@pytest.mark.parametrize("last,kept", [("0.20", True), ("0.21", False), ("0.05", True)])
def test_yes_last_must_be_at_most_20c(last, kept):
    c = _compounder(FakeClient())
    assert bool(c._find_no_candidates([_market(last=last)])) is kept


def test_old_cent_format_last_price_is_normalised():
    c = _compounder(FakeClient())
    keep = {"ticker": "KXA", "title": "t", "volume": 50, "last_price": 15, "close_time": _iso(10)}
    drop = {**keep, "ticker": "KXB", "last_price": 25}
    assert [x["ticker"] for x in c._find_no_candidates([keep, drop])] == ["KXA"]


@pytest.mark.parametrize("last", [None, "", "0", "0.0000", 0, 0.0, "-0.01", "garbage"])
def test_missing_or_nonpositive_last_price_is_not_a_candidate(last):
    m = _market(last=None)
    if last is not None:
        m["last_price_dollars"] = last
    assert _compounder(FakeClient())._find_no_candidates([m]) == []


@pytest.mark.parametrize("volume,kept", [("9", False), ("10", True)])
def test_minimum_volume_boundary(volume, kept):
    c = _compounder(FakeClient())
    assert bool(c._find_no_candidates([_market(volume=volume)])) is kept


def test_sports_prefix_mention_titles_and_expired_markets_are_skipped():
    c = _compounder(FakeClient())
    assert c._find_no_candidates([_market(ticker="KXNBAGAME-26NOV-LAL")]) == []
    assert c._find_no_candidates([_market(title="Will Trump mention tariffs?")]) == []
    assert c._find_no_candidates([_market(hours=-1)]) == []


def test_market_without_last_price_is_not_a_candidate():
    c = _compounder(FakeClient())
    assert c._find_no_candidates([_market(last=None)]) == []


# ---- orderbook filters: NO ask > 80c, edge > 3c ----------------------------------
def test_viable_market_becomes_a_no_buy_one_cent_under_the_ask():
    (opp,) = _scan(_market(), yes_bid="0.10")        # NO ask 0.90, est 0.99
    assert opp["side"] == "no"
    assert opp["lowest_no_ask"] == pytest.approx(0.90)
    assert opp["our_price"] == pytest.approx(0.89)
    assert opp["edge"] == pytest.approx(0.09)


@pytest.mark.parametrize("yes_bid", ["0.20", "0.21", "0.30"])   # NO ask 0.80, 0.79, 0.70
def test_no_ask_at_or_below_80c_is_rejected(yes_bid):
    assert _scan(_market(), yes_bid=yes_bid) == []


def test_no_ask_just_over_80c_is_accepted_and_our_bid_never_drops_below_80c():
    (opp,) = _scan(_market(), yes_bid="0.19")   # NO ask 0.81 (edge is wide: est 0.99)
    assert opp["our_price"] >= 0.80 - 1e-9


def test_edge_below_3c_is_rejected_and_4c_is_accepted():
    assert _scan(_market(), yes_bid="0.03") == []         # ask 0.97, est 0.99 -> edge 2c
    assert len(_scan(_market(), yes_bid="0.05")) == 1      # ask 0.95 -> edge 4c


def test_market_with_no_book_at_all_is_skipped():
    c = _compounder(FakeClient())

    async def empty(ticker, depth=10):
        return {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
    c.client.get_orderbook = empty
    assert asyncio.run(c._check_orderbook_and_price(c._find_no_candidates([_market()]))) == []


def test_low_confidence_book_is_skipped():
    c = _compounder(FakeClient(), min_confidence=0.99)
    assert asyncio.run(c._check_orderbook_and_price(c._find_no_candidates([_market()]))) == []


def test_orderbook_failure_skips_the_market_instead_of_crashing():
    c = _compounder(FakeClient())

    async def boom(ticker, depth=10):
        raise RuntimeError("429")
    c.client.get_orderbook = boom
    assert asyncio.run(c._check_orderbook_and_price(c._find_no_candidates([_market()]))) == []


# ---- sizing -----------------------------------------------------------------------
@pytest.mark.parametrize("use_kelly", [True, False])
def test_position_never_exceeds_ten_percent_of_portfolio(use_kelly):
    c = _compounder(FakeClient(), use_kelly=use_kelly)
    n = c._calculate_position_size(_opp(price=0.89), portfolio=100_000, cash=100_000)
    assert 1 <= n and n * 89 <= 100_000 * 0.10        # 112 x 89c = $99.68 <= $100


def test_contract_count_is_hard_capped_at_200():
    c = _compounder(FakeClient(), use_kelly=False)
    assert c._calculate_position_size(_opp(price=0.89), portfolio=10_000_000, cash=10_000_000) == 200


def test_floor_of_one_contract_does_not_break_the_ten_percent_cap():
    c = _compounder(FakeClient(), use_kelly=False)
    n = c._calculate_position_size(_opp(price=0.89), portfolio=500, cash=500)
    assert n * 89 <= 500 * 0.10


@pytest.mark.xfail(strict=True, reason=(
    "OWNER DECISION: bid nets may exceed cash by design (resting buys do not reserve cash). "
    "Cash $50, one 112-contract order = $99.68 is posted."))
def test_single_order_is_limited_by_cash_on_hand():
    client = FakeClient(cash=5_000, pv=100_000)
    asyncio.run(_compounder(client)._place_resting_orders([_opp("KXA")], portfolio=100_000, cash=5_000))
    assert sum(o["count"] * o["no_price"] for o in client.placed) <= 5_000


@pytest.mark.xfail(strict=True, reason=(
    "OWNER DECISION: bid nets may exceed cash by design (resting buys do not reserve cash). "
    "Cash $100, five $99.68 bids = $498 posted."))
def test_resting_bids_in_one_run_do_not_commit_more_than_cash():
    client = FakeClient(cash=10_000, pv=100_000)
    opps = [_opp(f"KX{i}") for i in range(5)]
    asyncio.run(_compounder(client)._place_resting_orders(opps, portfolio=100_000, cash=10_000))
    assert sum(o["count"] * o["no_price"] for o in client.placed) <= 10_000


# ---- dedupe of its own resting orders / positions ---------------------------------
def _place(client, opps, dry_run=False):
    return asyncio.run(_compounder(client, dry_run=dry_run)._place_resting_orders(
        opps, portfolio=100_000, cash=100_000))


def test_resting_order_on_a_ticker_blocks_a_second_buy():
    client = FakeClient(orders=[{"ticker": "KXA", "order_id": "o1"}])
    stats = _place(client, [_opp("KXA"), _opp("KXB")])
    assert [o["ticker"] for o in client.placed] == ["KXB"]
    assert stats["skipped_existing"] == 1


def test_open_position_blocks_a_second_buy_but_a_flat_one_does_not():
    client = FakeClient(positions=[{"ticker": "KXA", "position_fp": "-12.00"},
                                   {"ticker": "KXB", "position_fp": "0.00"}])
    _place(client, [_opp("KXA"), _opp("KXB")])
    assert [o["ticker"] for o in client.placed] == ["KXB"]


def test_same_ticker_twice_in_one_run_is_bought_once():
    client = FakeClient()
    _place(client, [_opp("KXA"), _opp("KXA")])
    assert len(client.placed) == 1


@pytest.mark.parametrize("which", ["orders_error", "positions_error"])
def test_failed_dedupe_lookup_fails_closed(which):
    # the failing endpoint is the only one that knows about KXA
    held = {"orders": [{"ticker": "KXA"}]} if which == "orders_error" else \
           {"positions": [{"ticker": "KXA", "position_fp": "5"}]}
    client = FakeClient(**held, **{which: RuntimeError("429")})
    _place(client, [_opp("KXA")])
    assert client.placed == []


@pytest.mark.parametrize("which", ["orders_cursor", "positions_cursor"])
def test_paged_dedupe_lookup_fails_closed(which):
    # a cursor means page 1 is not the whole book: KXA may be on page 2
    client = FakeClient(**{which: "next-page-token"})
    stats = _place(client, [_opp("KXA")])
    assert client.placed == [] and stats["errors"] == 1


def test_empty_cursor_is_a_complete_lookup():
    client = FakeClient(orders_cursor="", positions_cursor=None)
    _place(client, [_opp("KXA")])
    assert len(client.placed) == 1


# ---- order shape ------------------------------------------------------------------
def test_live_order_is_a_resting_limit_buy_on_the_no_side_in_cents():
    client = FakeClient()
    stats = _place(client, [_opp("KXA", price=0.89)])
    (o,) = client.placed
    assert (o["side"], o["action"], o["type_"], o["no_price"]) == ("no", "buy", "limit", 89)
    assert "yes_price" not in o
    assert stats["placed"] == 1 and stats["errors"] == 0


def test_bid_price_is_rounded_not_truncated_to_the_priced_cent():
    assert int(0.29 * 100) == 28          # the float trap this guards against
    client = FakeClient()
    _place(client, [_opp("KXA", price=0.29)])
    assert client.placed[0]["no_price"] == 29


def test_rejected_order_counts_as_error_not_placed():
    client = FakeClient()

    async def reject(**kw):
        raise RuntimeError("insufficient funds")
    client.place_order = reject
    stats = _place(client, [_opp("KXA")])
    assert stats["errors"] == 1 and stats["placed"] == 0 and stats["total_deployed"] == 0


# ---- dry run never posts ----------------------------------------------------------
def test_dry_run_never_calls_place_order_but_reports_the_plan():
    client = FakeClient()
    stats = _place(client, [_opp("KXA"), _opp("KXB")], dry_run=True)
    assert client.placed == []
    assert stats["placed"] == 2 and stats["total_deployed"] > 0


def _live_stack(tmp_path, equity=100_000, halt=False):
    """Real KalshiClient + real SafeCompounder + real governor, HTTP faked, POSTs recorded."""
    if halt:
        (tmp_path / "HALT").write_text("operator stop")
    client = KalshiClient.__new__(KalshiClient)
    client.posts = []
    market = _market()

    async def fake(method, path, params=None, json_data=None, **kw):
        if method == "POST":
            client.posts.append(json_data)
            return {"order": {"order_id": "x", "status": "resting", "fill_count": "0.00"}}
        if path.endswith("/portfolio/balance"):
            return {"balance": equity, "portfolio_value": equity}
        if "orderbook" in path:
            return {"orderbook_fp": {"yes_dollars": [["0.10", "100"]], "no_dollars": [["0.80", "100"]]}}
        if path.endswith("/events"):
            return {"events": [{"category": "econ", "title": "CPI", "markets": [market]}]}
        if path.endswith("/portfolio/positions"):
            return {"market_positions": []}
        if path.endswith("/portfolio/orders"):
            return {"orders": []}
        return {"markets": []}

    client._make_authenticated_request = fake
    real = RiskGovernor

    def factory(**kwargs):
        kwargs.update(state_path=str(tmp_path / "state.json"), halt_flag_path=str(tmp_path / "HALT"))
        return real(**kwargs)

    return client, patch("src.risk.risk_governor.RiskGovernor", side_effect=factory)


def test_run_live_end_to_end_posts_exactly_one_capped_no_buy(tmp_path):
    client, gov = _live_stack(tmp_path)
    with gov:
        stats = asyncio.run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())
    (post,) = client.posts
    # v2 payload: buying NO = the ask side of the YES leg; NO 89c -> YES-leg price 0.11
    assert post["ticker"] == "KXCPI-26NOV-T3"
    assert (post["side"], post["price"], post["time_in_force"]) == ("ask", "0.1100", "good_till_canceled")
    assert float(post["count"]) * 0.89 <= 0.10 * 200_000 / 100  # <= 10% of $2000 equity
    assert stats["placed"] == 1


def test_run_dry_argument_overrides_a_live_constructor_and_posts_nothing(tmp_path):
    client, gov = _live_stack(tmp_path)
    with gov:
        stats = asyncio.run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run(dry_run=True))
    assert client.posts == []
    assert stats["placed"] == 1                                  # planned, never sent


def test_run_with_halt_flag_posts_zero_orders(tmp_path):
    client, gov = _live_stack(tmp_path, halt=True)
    with gov:
        stats = asyncio.run(SafeCompounder(client, dry_run=False, min_confidence=0.0).run())
    assert client.posts == []
    assert stats["placed"] == 0 and stats["errors"] == 1


# ---- cli daily: halted / dry -> compounder is constructed dry ---------------------
def _daily(live, halted):
    decision = GovernorDecision(halted, ["DAILY_LOSS: x"] if halted else [], 100_000, 100_000, 100_000)

    class Gov:
        def __init__(self, **kw):
            pass

        async def check(self):
            return decision

    client = MagicMock()
    client.close = AsyncMock()
    compounder = MagicMock()
    compounder.run = AsyncMock(return_value={})
    made = MagicMock(return_value=compounder)
    with patch("src.risk.risk_governor.RiskGovernor", Gov), \
         patch("src.clients.kalshi_client.KalshiClient", return_value=client), \
         patch("src.strategies.safe_compounder.SafeCompounder", made), \
         patch("src.data.collector.snapshot_account", AsyncMock(return_value={"equity_cents": 100_000})), \
         patch("src.utils.logging_setup.setup_logging"):
        cli.cmd_daily(argparse.Namespace(live=live, log_level="INFO"))
    return made.call_args.kwargs["dry_run"]


@pytest.mark.parametrize("live,halted,dry", [
    (False, False, True),    # default is dry
    (False, True, True),
    (True, True, True),      # halted: never live
    (True, False, False),    # the only live combination
])
def test_cmd_daily_goes_live_only_when_asked_and_not_halted(live, halted, dry):
    assert _daily(live, halted) is dry


# ---- toolbelt.place_guarded_order -------------------------------------------------
class _Decision:
    def __init__(self, halted=False, equity=80_000):
        self.halted, self.current_equity_cents = halted, equity

    def to_dict(self):
        return {"halted": self.halted}


class _Gov:
    def __init__(self, **kw):
        self.d = _Decision(**kw)

    async def check(self):
        return self.d


class TClient:
    def __init__(self, status="active", cash=100_000, positions=None, resting=None, read_error=None):
        self.status, self.cash, self.positions = status, cash, positions or []
        self.resting, self.read_error = resting or [], read_error
        self.placed, self.market_calls = [], 0

    async def get_market(self, ticker):
        self.market_calls += 1
        return {"market": {"status": self.status, "yes_bid_dollars": 0.03, "yes_ask_dollars": 0.10,
                           "no_bid_dollars": 0.88, "no_ask_dollars": 0.95}}

    async def get_balance(self):
        return {"balance": self.cash, "portfolio_value": 0}

    async def get_positions(self, ticker=None):
        if self.read_error == "positions":
            raise RuntimeError("429")
        return {"market_positions": self.positions}

    async def get_orders(self, ticker=None, status=None):
        if self.read_error == "orders":
            raise RuntimeError("429")
        return {"orders": self.resting}

    async def place_order(self, **kw):
        self.placed.append(kw)
        return {"order": {"order_id": "oid", "fill_count": 0}}


def _guarded(tmp_path, client, gov=None, **kw):
    from src.agent.toolbelt import place_guarded_order
    args = dict(ticker="KXT", side="no", count=10, price=0.90, est_prob=0.95, governor=gov or _Gov(),
                journal_path=str(tmp_path / "j.jsonl"), policy_path=str(tmp_path / "none.json"))
    args.update(kw)
    return asyncio.run(place_guarded_order(client, **args))


def test_halted_governor_returns_before_touching_the_market(tmp_path):
    c = TClient()
    r = _guarded(tmp_path, c, gov=_Gov(halted=True))
    assert r["ok"] is False and r["reason"] == "governor_halted"
    assert c.market_calls == 0 and c.placed == []


def test_inactive_market_is_refused(tmp_path):
    c = TClient(status="closed")
    r = _guarded(tmp_path, c)
    assert r["ok"] is False and r["reason"].startswith("market_not_tradeable")
    assert c.placed == []


@pytest.mark.parametrize("price", [0.0, 0.004, 1.0, 1.5])
def test_out_of_range_price_is_refused(tmp_path, price):
    c = TClient()
    r = _guarded(tmp_path, c, price=price)
    assert r["ok"] is False and "out of range" in r["reason"]
    assert c.placed == []


def test_size_is_capped_to_ten_percent_of_governor_equity(tmp_path):
    c = TClient()                                   # equity $800 -> cap $80 -> 88 x 90c = $79.20
    r = _guarded(tmp_path, c, count=1000)
    assert r["count"] == 88 and "position cap" in r["cap_reason"]
    assert c.placed[0]["count"] == 88


def test_size_is_capped_to_cash_and_zero_cash_refuses(tmp_path):
    c = TClient(cash=3 * 90)
    assert _guarded(tmp_path, c, count=50)["count"] == 3
    c0 = TClient(cash=89)
    r = _guarded(tmp_path, c0, count=5)
    assert r["ok"] is False and r["reason"].startswith("size capped to 0")
    assert c0.placed == []


def test_dry_run_never_posts_or_journals(tmp_path):
    c = TClient()
    r = _guarded(tmp_path, c, dry=True)
    assert r["ok"] is True and r["dry"] is True and r["journaled"] is False
    assert c.placed == [] and not (tmp_path / "j.jsonl").exists()


def test_live_no_buy_uses_no_price_and_is_journaled(tmp_path):
    c = TClient()
    r = _guarded(tmp_path, c, count=5)
    (o,) = c.placed
    assert (o["side"], o["action"], o["no_price"], o["count"]) == ("no", "buy", 90, 5)
    assert "yes_price" not in o and r["journaled"] is True
    assert (tmp_path / "j.jsonl").exists()


def test_live_yes_buy_uses_yes_price(tmp_path):
    c = TClient()
    _guarded(tmp_path, c, side="yes", price=0.10, count=5, est_prob=0.2)
    (o,) = c.placed
    assert o["yes_price"] == 10 and "no_price" not in o


def test_existing_position_counts_against_the_ten_percent_cap(tmp_path):
    c = TClient(positions=[{"ticker": "KXT", "position_fp": "-88.00"}])
    _guarded(tmp_path, c, count=88)
    assert c.placed == []


def test_resting_order_counts_against_the_cap_too(tmp_path):
    # cap $80 = 88 x 90c; 50 already resting leaves room for 38
    c = TClient(resting=[{"ticker": "KXT", "remaining_count_fp": "50.00"}])
    r = _guarded(tmp_path, c, count=88)
    assert r["count"] == 38 and c.placed[0]["count"] == 38


def test_position_at_the_cap_leaves_no_room(tmp_path):
    c = TClient(positions=[{"ticker": "KXT", "position_fp": "-88.00"}])
    r = _guarded(tmp_path, c, count=1)
    assert r["ok"] is False and r["reason"].startswith("size capped to 0")


def test_other_tickers_and_flat_positions_do_not_count(tmp_path):
    c = TClient(positions=[{"ticker": "KXOTHER", "position_fp": "500"},
                           {"ticker": "KXT", "position_fp": "0.00"}])
    assert _guarded(tmp_path, c, count=88)["count"] == 88


@pytest.mark.parametrize("which", ["positions", "orders"])
def test_unreadable_exposure_fails_closed(tmp_path, which):
    c = TClient(read_error=which)
    r = _guarded(tmp_path, c)
    assert r["ok"] is False and r["reason"].startswith("exposure_unreadable")
    assert c.placed == []
