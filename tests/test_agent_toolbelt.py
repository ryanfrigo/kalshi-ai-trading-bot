"""Tests for the agent toolbelt order-sizing guard.

cap_count caps an intended contract count so the order never exceeds the
per-position equity cap or available cash — a hard backstop independent of
whatever size the agent (or strategy) asks for.
"""
from src.agent.toolbelt import cap_count


def test_cap_count_allows_within_limits():
    # equity 80000c -> 10% cap 8000c; price 95c -> up to 84 contracts; cash ample.
    n, reason = cap_count(16, price_cents=95, equity_cents=80000, cash_cents=60000)
    assert n == 16
    assert reason == ""


def test_cap_count_caps_to_position_pct():
    n, reason = cap_count(1000, price_cents=95, equity_cents=80000, cash_cents=60000)
    assert n == 84  # floor(8000 / 95)
    assert "position cap" in reason.lower()


def test_cap_count_caps_to_cash():
    n, reason = cap_count(1000, price_cents=95, equity_cents=80000, cash_cents=1000)
    assert n == 10  # floor(1000 / 95)
    assert "cash" in reason.lower()


def test_cap_count_zero_when_insufficient_cash():
    n, reason = cap_count(16, price_cents=95, equity_cents=80000, cash_cents=50)
    assert n == 0


def test_cap_count_custom_position_pct():
    # 5% of 80000 = 4000c -> floor(4000/95)=42
    n, reason = cap_count(1000, price_cents=95, equity_cents=80000,
                          cash_cents=60000, max_position_pct=0.05)
    assert n == 42


# ---------------------------------------------------------------------------
# Edge Policy gate — place_guarded_order consults the active policy
# ---------------------------------------------------------------------------
import asyncio
import json

import pytest

from src.agent.toolbelt import place_guarded_order


class _Decision:
    halted = False
    current_equity_cents = 80000
    def to_dict(self):
        return {"halted": False}


class _Governor:
    async def check(self):
        return _Decision()


class _ReachedMarket(Exception):
    """Sentinel: proves execution got PAST the policy gate to the market call."""


class _NoMarketClient:
    async def get_market(self, ticker):
        raise _ReachedMarket(ticker)


def _write_policy(tmp_path, blocks):
    p = tmp_path / "edge_policy.json"
    p.write_text(json.dumps({"meta": {}, "blocks": blocks,
                             "warnings": [], "haircuts": []}))
    return str(p)


def test_policy_blocks_a_trade_in_a_blocked_category(tmp_path):
    # Policy blocks the KXCPI series; a KXCPI trade must be refused BEFORE any
    # market call (the fake client would raise if reached).
    policy_path = _write_policy(tmp_path, [
        {"dimension": "category", "label": "KXCPI", "reason": "loses", "n": 6, "pnl": -66.0}])
    res = asyncio.run(place_guarded_order(
        _NoMarketClient(), ticker="KXCPI-26JUN-3", side="no", count=10,
        price=0.95, est_prob=0.97, governor=_Governor(), dry=True,
        policy_path=policy_path))
    assert res["ok"] is False
    assert res["reason"] == "blocked_by_policy"
    assert res["policy_verdict"]["verdict"] == "BLOCK"


def test_override_policy_bypasses_the_block(tmp_path):
    # With override, the gate is skipped and execution proceeds to the market
    # (proven by the sentinel the fake client raises).
    policy_path = _write_policy(tmp_path, [
        {"dimension": "category", "label": "KXCPI", "reason": "loses", "n": 6, "pnl": -66.0}])
    with pytest.raises(_ReachedMarket):
        asyncio.run(place_guarded_order(
            _NoMarketClient(), ticker="KXCPI-26JUN-3", side="no", count=10,
            price=0.95, est_prob=0.97, governor=_Governor(), dry=True,
            policy_path=policy_path, override_policy=True))


def test_no_policy_file_allows_trade_through_to_market(tmp_path):
    # Backward-compatible: no saved policy -> no gate -> proceeds to the market.
    missing = str(tmp_path / "does_not_exist.json")
    with pytest.raises(_ReachedMarket):
        asyncio.run(place_guarded_order(
            _NoMarketClient(), ticker="KXCPI-26JUN-3", side="no", count=10,
            price=0.95, est_prob=0.97, governor=_Governor(), dry=True,
            policy_path=missing))


# ---------------------------------------------------------------------------
# H1: a HAIRCUT that erases edge must abort (not silently place a -EV order)
# ---------------------------------------------------------------------------

class _MarketClient:
    """Minimal happy-path fake: tradeable market + ample cash, dry-run friendly."""
    def __init__(self, no_ask=0.95, yes_ask=0.10, cash=100000):
        self._no_ask, self._yes_ask, self._cash = no_ask, yes_ask, cash
        self.placed = []
    async def get_market(self, ticker):
        return {"market": {"status": "active",
                           "yes_bid_dollars": 0.03, "yes_ask_dollars": self._yes_ask,
                           "no_bid_dollars": round(1 - self._yes_ask - 0.02, 2),
                           "no_ask_dollars": self._no_ask}}
    async def get_balance(self):
        return {"balance": self._cash, "portfolio_value": 0}
    async def get_positions(self, ticker=None):
        return {"market_positions": []}
    async def get_orders(self, ticker=None, status=None):
        return {"orders": []}
    async def place_order(self, **kw):
        self.placed.append(kw)
        return {"order": {"order_id": "oid", "fill_count": kw.get("count")}}


def _write_policy_full(tmp_path, blocks=None, warnings=None, haircuts=None):
    p = tmp_path / "edge_policy.json"
    p.write_text(json.dumps({"meta": {}, "blocks": blocks or [],
                             "warnings": warnings or [], "haircuts": haircuts or []}))
    return str(p)


_HAIRCUT_9599 = {"band": "[0.95,0.99]", "lo": 0.95, "hi": 0.99,
                 "shrink_to": 0.72, "n": 8, "gap": 0.25}


def test_haircut_that_erases_edge_aborts(tmp_path):
    # est_prob 0.97 shrinks to 0.72; NO ask is 0.95 -> edge 0.72-0.95 < 0 -> abort.
    policy_path = _write_policy_full(tmp_path, haircuts=[_HAIRCUT_9599])
    client = _MarketClient(no_ask=0.95)
    res = asyncio.run(place_guarded_order(
        client, ticker="KXX-1", side="no", count=10, price=None, est_prob=0.97,
        governor=_Governor(), dry=True, policy_path=policy_path))
    assert res["ok"] is False
    assert res["reason"] == "haircut_erased_edge"
    assert client.placed == []  # nothing placed


def test_haircut_that_preserves_edge_proceeds(tmp_path):
    # est_prob 0.97 shrinks to 0.72; buy at 0.60 -> edge 0.72-0.60 > 0 -> proceeds.
    policy_path = _write_policy_full(tmp_path, haircuts=[_HAIRCUT_9599])
    client = _MarketClient(no_ask=0.95)
    res = asyncio.run(place_guarded_order(
        client, ticker="KXX-1", side="no", count=10, price=0.60, est_prob=0.97,
        governor=_Governor(), dry=True, policy_path=policy_path))
    assert res["ok"] is True
    assert res["policy_note"]["est_prob_shrunk_to"] == 0.72
    assert res["edge"] == round(0.72 - 0.60, 4)


def test_override_lets_an_edge_erasing_haircut_through(tmp_path):
    policy_path = _write_policy_full(tmp_path, haircuts=[_HAIRCUT_9599])
    client = _MarketClient(no_ask=0.95)
    res = asyncio.run(place_guarded_order(
        client, ticker="KXX-1", side="no", count=10, price=None, est_prob=0.97,
        governor=_Governor(), dry=True, policy_path=policy_path, override_policy=True))
    assert res["ok"] is True
    assert res["edge"] < 0  # -EV, but the agent explicitly overrode


def test_resting_order_expiry_is_passed_to_the_exchange(tmp_path):
    # Resting-bid nets must expire so stale quotes can't be picked off.
    client = _MarketClient(no_ask=0.95)
    res = asyncio.run(place_guarded_order(
        client, ticker="KXX-1", side="no", count=2, price=0.88, est_prob=0.95,
        governor=_Governor(), dry=False, journal_path=str(tmp_path / "j.jsonl"),
        policy_path=str(tmp_path / "none.json"), expiration_ts=1790500000))
    assert res["ok"] is True
    assert client.placed[0]["expiration_ts"] == 1790500000


def test_no_expiry_by_default(tmp_path):
    client = _MarketClient(no_ask=0.95)
    asyncio.run(place_guarded_order(
        client, ticker="KXX-1", side="no", count=2, price=0.88, est_prob=0.95,
        governor=_Governor(), dry=False, journal_path=str(tmp_path / "j.jsonl"),
        policy_path=str(tmp_path / "none.json")))
    assert "expiration_ts" not in client.placed[0]


# ---------------------------------------------------------------------------
# Aggregate exposure: held + resting on a ticker must be read fail-closed
# ---------------------------------------------------------------------------

class _ExposureClient(_MarketClient):
    """Fake whose position/order pages are scripted per cursor (None = first page)."""
    def __init__(self, positions=None, orders=None, **kw):
        super().__init__(**kw)
        self._positions = positions if positions is not None else {None: {"market_positions": []}}
        self._orders = orders if orders is not None else {None: {"orders": []}}
        self.position_cursors, self.order_cursors = [], []
    async def get_positions(self, ticker=None, cursor=None, **kw):
        self.position_cursors.append(cursor)
        return self._positions[cursor]
    async def get_orders(self, ticker=None, status=None, cursor=None, **kw):
        self.order_cursors.append(cursor)
        return self._orders[cursor]


def _buy(client, tmp_path):
    # $800 equity -> 10% cap = 8000c; at 90c one ticker may hold at most 88 contracts.
    return asyncio.run(place_guarded_order(
        client, ticker="KXT", side="no", count=88, price=0.90, est_prob=0.99,
        governor=_Governor(), dry=True, journal_path=str(tmp_path / "j.jsonl"),
        policy_path=str(tmp_path / "none.json")))


def test_legacy_position_field_counts_as_held(tmp_path):
    client = _ExposureClient(positions={None: {"market_positions": [
        {"ticker": "KXT", "position": -88}]}})
    res = _buy(client, tmp_path)
    assert res["ok"] is False
    assert "capped to 0" in res["reason"]


def test_legacy_remaining_count_counts_as_resting(tmp_path):
    client = _ExposureClient(orders={None: {"orders": [
        {"ticker": "KXT", "remaining_count": 88}]}})
    res = _buy(client, tmp_path)
    assert res["ok"] is False
    assert "capped to 0" in res["reason"]


def test_v2_fp_fields_still_count(tmp_path):
    client = _ExposureClient(
        positions={None: {"market_positions": [{"ticker": "KXT", "position_fp": "-44.00"}]}},
        orders={None: {"orders": [{"ticker": "KXT", "remaining_count_fp": "44.00"}]}})
    assert _buy(client, tmp_path)["ok"] is False


def test_matching_row_without_any_count_field_fails_closed(tmp_path):
    client = _ExposureClient(positions={None: {"market_positions": [{"ticker": "KXT"}]}})
    res = _buy(client, tmp_path)
    assert res["ok"] is False and res["reason"].startswith("exposure_unreadable")


def test_unparseable_count_fails_closed(tmp_path):
    client = _ExposureClient(orders={None: {"orders": [
        {"ticker": "KXT", "remaining_count_fp": "lots"}]}})
    res = _buy(client, tmp_path)
    assert res["ok"] is False and res["reason"].startswith("exposure_unreadable")


@pytest.mark.parametrize("bad", [{}, {"market_positions": None}, {"market_positions": {}}])
def test_missing_or_odd_positions_list_fails_closed(tmp_path, bad):
    res = _buy(_ExposureClient(positions={None: bad}), tmp_path)
    assert res["ok"] is False and res["reason"].startswith("exposure_unreadable")


@pytest.mark.parametrize("bad", [{}, {"orders": None}, {"orders": "x"}])
def test_missing_or_odd_orders_list_fails_closed(tmp_path, bad):
    res = _buy(_ExposureClient(orders={None: bad}), tmp_path)
    assert res["ok"] is False and res["reason"].startswith("exposure_unreadable")


def test_second_page_of_positions_is_consumed(tmp_path):
    client = _ExposureClient(positions={
        None: {"market_positions": [{"ticker": "OTHER", "position_fp": "5"}], "cursor": "p2"},
        "p2": {"market_positions": [{"ticker": "KXT", "position_fp": "-88"}], "cursor": ""},
    })
    res = _buy(client, tmp_path)
    assert client.position_cursors == [None, "p2"]
    assert res["ok"] is False and "capped to 0" in res["reason"]


def test_second_page_of_orders_is_consumed(tmp_path):
    client = _ExposureClient(orders={
        None: {"orders": [], "cursor": "o2"},
        "o2": {"orders": [{"ticker": "KXT", "remaining_count_fp": "88"}]},
    })
    res = _buy(client, tmp_path)
    assert client.order_cursors == [None, "o2"]
    assert res["ok"] is False and "capped to 0" in res["reason"]


def test_endless_cursor_fails_closed(tmp_path):
    class Loop(_ExposureClient):
        async def get_positions(self, ticker=None, cursor=None, **kw):
            return {"market_positions": [], "cursor": "again"}
    res = _buy(Loop(), tmp_path)
    assert res["ok"] is False and res["reason"].startswith("exposure_unreadable")


def test_empty_exposure_still_allows_the_buy(tmp_path):
    res = _buy(_ExposureClient(), tmp_path)
    assert res["ok"] is True and res["count"] == 88


@pytest.mark.parametrize("kind,row", [
    ("positions", {"ticker": "KXT", "position_fp": "nan"}),
    ("positions", {"ticker": "KXT", "position_fp": "inf"}),
    ("orders", {"ticker": "KXT", "remaining_count_fp": "nan"}),
])
def test_non_finite_count_fails_closed(tmp_path, kind, row):
    if kind == "positions":
        client = _ExposureClient(positions={None: {"market_positions": [row]}})
    else:
        client = _ExposureClient(orders={None: {"orders": [row]}})
    res = _buy(client, tmp_path)
    assert res["ok"] is False and "exposure_unreadable" in res["reason"]


@pytest.mark.parametrize("kind,resp", [
    ("positions", {"market_positions": [None]}),
    ("positions", {"market_positions": ["KXT"]}),
    ("positions", {"market_positions": [{"position_fp": "-88"}]}),
    ("positions", {"market_positions": [{"ticker": "", "position_fp": "-88"}]}),
    ("orders", {"orders": [None]}),
    ("orders", {"orders": [{"remaining_count_fp": "88"}]}),
])
def test_malformed_row_fails_closed(tmp_path, kind, resp):
    # A row we cannot attribute to a ticker must not count as zero exposure.
    client = (_ExposureClient(positions={None: resp}) if kind == "positions"
              else _ExposureClient(orders={None: resp}))
    res = _buy(client, tmp_path)
    assert res["ok"] is False and res["reason"].startswith("exposure_unreadable")


def test_malformed_row_on_second_positions_page_fails_closed(tmp_path):
    client = _ExposureClient(positions={
        None: {"market_positions": [{"ticker": "OTHER", "position_fp": "5"}], "cursor": "p2"},
        "p2": {"market_positions": [None]},
    })
    res = _buy(client, tmp_path)
    assert client.position_cursors == [None, "p2"]
    assert res["ok"] is False and res["reason"].startswith("exposure_unreadable")


def test_well_formed_row_for_another_ticker_is_ignored(tmp_path):
    client = _ExposureClient(positions={None: {"market_positions": [
        {"ticker": "OTHER", "position_fp": "5"}]}})
    res = _buy(client, tmp_path)
    assert res["ok"] is True and res["count"] == 88


def test_position_fp_wins_over_legacy_position(tmp_path):
    # fp says 1 contract held, legacy says 88: at 90c the cap leaves room for 87.
    client = _ExposureClient(positions={None: {"market_positions": [
        {"ticker": "KXT", "position_fp": "-1.00", "position": -88}]}})
    res = _buy(client, tmp_path)
    assert res["ok"] is True and res["count"] == 87


# ---------------------------------------------------------------------------
# fill_count: v2 fill_count_fp, legacy fill_count fallback, including an order
# recovered by client_order_id after an ambiguous POST
# ---------------------------------------------------------------------------
from src.agent.toolbelt import close_position


class _FillClient(_MarketClient):
    def __init__(self, order):
        super().__init__(no_ask=0.60)
        self.order = order
    async def get_positions(self, ticker=None):
        return {"market_positions": [{"ticker": ticker, "position_fp": "5.00"}]}
    async def _held_on_side(self, ticker, side):
        return 5.0
    async def place_order(self, **kw):
        self.placed.append(kw)
        return {"order": self.order, **self.order}


_RECOVERED = {"order_id": "oid", "fill_count_fp": "3.00", "recovered_by_client_order_id": True}


@pytest.mark.parametrize("order,expected", [
    (_RECOVERED, 3.0),
    ({"order_id": "oid", "fill_count_fp": "2.50", "fill_count": 9}, 2.5),
    ({"order_id": "oid", "fill_count": 4}, 4.0),
    ({"order_id": "oid"}, None),
    ({"order_id": "oid", "fill_count_fp": "junk"}, None),
])
def test_fill_count_reads_fp_then_legacy(tmp_path, order, expected):
    journal = str(tmp_path / "j.jsonl")
    buy = asyncio.run(place_guarded_order(
        _FillClient(order), ticker="KXX-1", side="no", count=5, price=0.60, est_prob=0.80,
        governor=_Governor(), dry=False, journal_path=journal,
        policy_path=str(tmp_path / "none.json")))
    sell = asyncio.run(close_position(_FillClient(order), ticker="KXX-1", dry=False,
                                      journal_path=journal))
    assert buy["ok"] is True and buy["order_id"] == "oid" and buy["fill_count"] == expected
    assert sell["ok"] is True and sell["order_id"] == "oid" and sell["fill_count"] == expected


class _UnknownClient(_FillClient):
    async def place_order(self, **kw):
        from src.clients.kalshi_client import KalshiOrderOutcomeUnknown
        self.placed.append(kw)
        raise KalshiOrderOutcomeUnknown("outcome unknown; not resent")


def test_outcome_unknown_is_reported_not_raised(tmp_path):
    journal = tmp_path / "j.jsonl"
    buyer, seller = _UnknownClient({}), _UnknownClient({})
    buy = asyncio.run(place_guarded_order(
        buyer, ticker="KXX-1", side="no", count=5, price=0.60, est_prob=0.80,
        governor=_Governor(), dry=False, journal_path=str(journal),
        policy_path=str(tmp_path / "none.json")))
    sell = asyncio.run(close_position(seller, ticker="KXX-1", dry=False,
                                      journal_path=str(journal)))
    for res, client in ((buy, buyer), (sell, seller)):
        assert res["ok"] is False and res["reason"] == "outcome_unknown"
        assert len(client.placed) == 1  # sent once, never resent
        assert res["client_order_id"] == client.placed[0]["client_order_id"]
    assert sell["action"] == "sell"
    # journaled as pending, so the order is never invisible to the record
    recs = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(recs) == 2
    for rec, res, action in zip(recs, (buy, sell), ("buy", "sell")):
        assert rec["status"] == "pending" and rec["order_outcome"] == "outcome_unknown"
        assert rec["client_order_id"] == res["client_order_id"]
        assert rec["action"] == action and rec["ticker"] == "KXX-1"
        assert rec["order_id"] is None and rec["outcome"] is None
    assert recs[0]["count"] == 5 and recs[0]["price"] == 0.60 and recs[0]["est_prob"] == 0.80
    assert recs[1]["count"] == sell["count"] and recs[1]["price"] == sell["price_cents"] / 100


def _raising_journal(monkeypatch):
    import src.agent.journal as journal_mod

    def boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(journal_mod, "append_decision", boom)


def test_outcome_unknown_buy_survives_journal_failure(tmp_path, monkeypatch, caplog):
    _raising_journal(monkeypatch)
    client = _UnknownClient({})
    with caplog.at_level("WARNING"):
        res = asyncio.run(place_guarded_order(
            client, ticker="KXX-1", side="no", count=5, price=0.60, est_prob=0.80,
            governor=_Governor(), dry=False, journal_path=str(tmp_path / "j.jsonl"),
            policy_path=str(tmp_path / "none.json")))
    assert res["ok"] is False and res["reason"] == "outcome_unknown"
    assert res["client_order_id"] == client.placed[0]["client_order_id"]
    assert res["journal_error"] == "disk full"
    assert res["journaled"] is False
    assert res["client_order_id"] in caplog.text


def test_outcome_unknown_sell_survives_journal_failure(tmp_path, monkeypatch, caplog):
    _raising_journal(monkeypatch)
    client = _UnknownClient({})
    with caplog.at_level("WARNING"):
        res = asyncio.run(close_position(client, ticker="KXX-1", dry=False,
                                         journal_path=str(tmp_path / "j.jsonl")))
    assert res["ok"] is False and res["reason"] == "outcome_unknown"
    assert res["client_order_id"] == client.placed[0]["client_order_id"]
    assert res["journal_error"] == "disk full"
    assert res["journaled"] is False
    assert res["client_order_id"] in caplog.text


@pytest.mark.parametrize("content", ['{"blocks": [{"dimens', '[1, 2]', '"text"', ''])
def test_an_unreadable_policy_file_refuses_before_any_market_call(tmp_path, content):
    # The file exists but can't be read as a policy, so its blocks are unknown.
    # Trading on would silently switch the gate off.
    p = tmp_path / "edge_policy.json"
    p.write_text(content)
    res = asyncio.run(place_guarded_order(
        _NoMarketClient(), ticker="KXCPI-26JUN-3", side="no", count=10,
        price=0.95, est_prob=0.97, governor=_Governor(), dry=True,
        policy_path=str(p)))
    assert res["ok"] is False
    assert res["reason"] == "policy_unreadable"


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_override_policy_passes_an_unreadable_policy_file_and_records_it(tmp_path, content):
    p = tmp_path / "edge_policy.json"
    p.write_text(content)
    res = asyncio.run(place_guarded_order(
        _MarketClient(no_ask=0.95), ticker="KXX-1", side="no", count=10, price=0.60,
        est_prob=0.97, governor=_Governor(), dry=True, policy_path=str(p),
        override_policy=True, journal_path=str(tmp_path / "journal.jsonl")))
    assert res["ok"] is True
    assert res["policy_note"] == {"overridden_unreadable_policy": str(p)}


def test_a_missing_policy_file_still_means_no_opinion(tmp_path):
    with pytest.raises(_ReachedMarket):
        asyncio.run(place_guarded_order(
            _NoMarketClient(), ticker="KXCPI-26JUN-3", side="no", count=10,
            price=0.95, est_prob=0.97, governor=_Governor(), dry=True,
            policy_path=str(tmp_path / "absent.json")))
