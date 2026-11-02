"""End-to-end audit of the MCP server's two mutating tools, ``trade`` and ``close``.

Every test calls the real MCP tool function (``src.mcp_server.trade`` / ``close`` /
the read-only tools) with a REAL KalshiClient whose httpx client talks to an
in-memory fake exchange (``httpx.MockTransport``). Nothing leaves the process.
The full stack under test is:

    MCP tool -> toolbelt.place_guarded_order / close_position (governor, policy,
    price, cap) -> KalshiClient.place_order (risk gate) -> wire

Risk-governor state, the manual halt flag, the policy file and the decision
journal are all relative paths under ``data/runtime``; each test chdirs into
tmp_path so they live there.

Defects D1-D5 from the audit are fixed; each test that pinned one says which.
"""
from __future__ import annotations

import contextvars
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs

import httpx
import pytest

import src.mcp_server as server
from src.clients import kalshi_client as kc_mod
from src.clients.kalshi_client import KalshiAPIError, KalshiClient

pytestmark = pytest.mark.asyncio

ORDERS = "/trade-api/v2/portfolio/events/orders"
_IN_PLACE_ORDER = contextvars.ContextVar("in_place_order", default=False)


class Exchange:
    """A fake Kalshi: balance, markets, positions, resting orders. Records every
    request. A POST that does not come from inside KalshiClient.place_order is
    recorded as a bypass."""

    def __init__(self):
        self.balance = 100_000          # cents: $1,000 cash
        self.portfolio_value = 0
        self.markets = {
            "T-A": {"status": "active", "yes_bid_dollars": 0.48, "yes_ask_dollars": 0.50,
                    "no_bid_dollars": 0.50, "no_ask_dollars": 0.52},
        }
        self.positions = {}             # ticker -> position_fp string
        self.resting = []
        self.posts = []
        self.bypass = []
        self.log = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        q = parse_qs(request.url.query.decode())
        self.log.append((request.method, path, request.url.query.decode()))
        if request.method != "GET":
            if not _IN_PLACE_ORDER.get():
                self.bypass.append((request.method, path))
            if request.method == "POST" and path == ORDERS:
                body = json.loads(request.content)
                self.posts.append(body)
                oid = f"O-{len(self.posts)}"
                self.resting.append({"order_id": oid, "ticker": body["ticker"],
                                     "book_side": body["side"], "status": "resting",
                                     "remaining_count_fp": body["count"]})
                return httpx.Response(201, json={"order_id": oid, "fill_count": "0.00",
                                                 "status": "resting"})
            return httpx.Response(405, json={"error": "fake exchange: mutation refused"})
        if path == "/trade-api/v2/portfolio/balance":
            return httpx.Response(200, json={"balance": self.balance,
                                             "portfolio_value": self.portfolio_value})
        if path == "/trade-api/v2/portfolio/positions":
            t = q.get("ticker", [None])[0]
            rows = [{"ticker": k, "position_fp": v} for k, v in self.positions.items()
                    if t is None or k == t]
            return httpx.Response(200, json={"market_positions": rows, "cursor": ""})
        if path.startswith("/trade-api/v2/markets/"):
            t = path.rsplit("/", 1)[-1]
            if t not in self.markets:
                return httpx.Response(404, json={"error": "market not found"})
            return httpx.Response(200, json={"market": {"ticker": t, **self.markets[t]}})
        if path == "/trade-api/v2/portfolio/orders":
            t = q.get("ticker", [None])[0]
            return httpx.Response(200, json={"cursor": "", "orders": [
                o for o in self.resting if t is None or o["ticker"] == t]})
        if path == "/trade-api/v2/portfolio/settlements":
            return httpx.Response(200, json={"settlements": [], "cursor": ""})
        if path == "/trade-api/v2/events":
            return httpx.Response(200, json={"events": [], "cursor": ""})
        return httpx.Response(404, json={"error": f"unexpected {path}"})

    def gets_only(self):
        return all(m == "GET" for m, _, _ in self.log)


@pytest.fixture
def ex(tmp_path, monkeypatch):
    """chdir to tmp_path, then make every KalshiClient() the MCP layer builds a
    real client wired to the fake exchange."""
    monkeypatch.chdir(tmp_path)
    exchange = Exchange()
    real_place = KalshiClient.place_order
    real_cls = KalshiClient

    async def traced_place(self, *a, **k):
        tok = _IN_PLACE_ORDER.set(True)
        try:
            return await real_place(self, *a, **k)
        finally:
            _IN_PLACE_ORDER.reset(tok)

    def factory(*a, **k):
        c = real_cls(*a, **k)
        c.client = httpx.AsyncClient(transport=httpx.MockTransport(exchange.handle))
        return c

    with patch.object(KalshiClient, "_load_private_key", lambda self: None), \
         patch.object(KalshiClient, "_sign_request", lambda self, *a: "sig"), \
         patch.object(KalshiClient, "place_order", traced_place), \
         patch.object(KalshiClient, "cancel_order", AsyncMock(side_effect=AssertionError("cancel"))), \
         patch.object(kc_mod, "KalshiClient", factory), \
         patch("src.clients.kalshi_client.asyncio.sleep", AsyncMock()):
        yield exchange
    assert exchange.bypass == [], f"non-GET sent outside KalshiClient.place_order: {exchange.bypass}"


def _halt_flag(reason="operator stop"):
    p = Path("data/runtime/TRADING_HALTED")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(reason)


def _governor_state(sod_cents, peak_cents=None):
    from src.risk.risk_governor import DEFAULT_STATE_PATH
    p = Path(DEFAULT_STATE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sod_equity_cents": sod_cents,
        "peak_equity_cents": peak_cents if peak_cents is not None else sod_cents,
    }))


def _journal_lines():
    p = Path("data/runtime/decision_journal.jsonl")
    return p.read_text().splitlines() if p.exists() else []


# ---------------------------------------------------------------------------
# trade: dry-run default, explicit live flag
# ---------------------------------------------------------------------------

async def test_trade_without_confirm_sends_no_order_and_writes_no_journal(ex):
    out = await server.trade(ticker="T-A", side="yes", count=10)
    assert out["ok"] is True and out["dry"] is True and out["count"] == 10
    assert out["price_cents"] == 50 and out["journaled"] is False
    assert ex.posts == [] and ex.gets_only()
    assert _journal_lines() == []


async def test_trade_confirm_false_explicitly_is_still_dry(ex):
    out = await server.trade(ticker="T-A", side="no", count=3, confirm=False)
    assert out["dry"] is True and ex.posts == []


async def test_trade_confirm_true_places_exactly_one_gated_order(ex):
    out = await server.trade(ticker="T-A", side="yes", count=10, confirm=True)
    assert out["ok"] is True and out["dry"] is False and out["order_id"] == "O-1"
    assert len(ex.posts) == 1
    p = ex.posts[0]
    assert (p["ticker"], p["side"], p["count"], p["price"]) == ("T-A", "bid", "10.00", "0.5000")
    assert len(_journal_lines()) == 1


async def test_trade_buy_no_maps_to_yes_leg_ask(ex):
    await server.trade(ticker="T-A", side="no", count=4, confirm=True)
    p = ex.posts[0]
    assert (p["side"], p["price"]) == ("ask", "0.4800")  # NO at 52c = YES leg at 48c


# ---------------------------------------------------------------------------
# trade: governor halt blocks, dry or live
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confirm", [False, True])
async def test_manual_halt_blocks_trade(ex, confirm):
    _halt_flag()
    out = await server.trade(ticker="T-A", side="yes", count=1, confirm=confirm)
    assert out == {"ok": False, "reason": "governor_halted", "governor": out["governor"]}
    assert any(r.startswith("MANUAL_HALT") for r in out["governor"]["reasons"])
    assert ex.posts == []


@pytest.mark.parametrize("confirm", [False, True])
async def test_drawdown_halt_blocks_trade(ex, confirm):
    _governor_state(sod_cents=100_000, peak_cents=200_000)  # $1,000 now vs $2,000 peak: -50%
    out = await server.trade(ticker="T-A", side="yes", count=1, confirm=confirm)
    assert out["reason"] == "governor_halted" and ex.posts == []


async def test_daily_loss_between_client_and_toolbelt_limits_is_blocked_by_client_gate(ex):
    """The toolbelt's governor uses the RiskGovernor defaults (10% daily loss); the
    client gate uses settings (5%). At -7% the toolbelt lets the order through and
    KalshiClient.place_order refuses it: the client gate is the backstop."""
    _governor_state(sod_cents=107_527)  # $1,000 is -7.0% from $1,075.27
    dry = await server.trade(ticker="T-A", side="yes", count=1)
    assert dry["ok"] is True  # the preview does not know
    with pytest.raises(KalshiAPIError, match="RISK_HALT: DAILY_LOSS"):
        await server.trade(ticker="T-A", side="yes", count=1, confirm=True)
    assert ex.posts == []


async def test_corrupt_governor_state_blocks_trade(ex):
    p = Path("data/runtime")
    p.mkdir(parents=True, exist_ok=True)
    from src.risk.risk_governor import DEFAULT_STATE_PATH
    Path(DEFAULT_STATE_PATH).write_text("{not json")
    out = await server.trade(ticker="T-A", side="yes", count=1, confirm=True)
    assert out["reason"] == "governor_halted" and ex.posts == []


async def test_unreadable_equity_blocks_trade(ex):
    ex.balance = 0
    out = await server.trade(ticker="T-A", side="yes", count=1, confirm=True)
    assert out["reason"] == "governor_halted" and ex.posts == []


# ---------------------------------------------------------------------------
# trade: 10% position cap
# ---------------------------------------------------------------------------

async def test_trade_caps_a_huge_count_to_ten_percent_of_equity(ex):
    out = await server.trade(ticker="T-A", side="yes", count=10**9, confirm=True)
    assert out["count"] == 200 and "position cap" in out["cap_reason"]  # $100 of $1,000 at 50c
    assert ex.posts[0]["count"] == "200.00"


async def test_trade_caps_to_cash_when_cash_is_the_tighter_limit(ex):
    ex.balance, ex.portfolio_value = 2_000, 98_000   # $20 cash, $1,000 equity
    out = await server.trade(ticker="T-A", side="yes", count=100, confirm=True)
    assert out["count"] == 40 and "cash" in out["cap_reason"]


async def test_raising_max_position_pct_is_clamped_live(ex):
    """max_position_pct is an MCP argument; asking for 100% still sends only the
    10% size, which the client gate accepts."""
    out = await server.trade(ticker="T-A", side="yes", count=1000, max_position_pct=1.0,
                             confirm=True)
    assert out["count"] == 200 and [p["count"] for p in ex.posts] == ["200.00"]


@pytest.mark.parametrize("pct", [1.0, 0.5, 0.11, math.inf, math.nan])
async def test_dry_run_does_not_report_a_size_above_the_ten_percent_cap(ex, pct):
    """D2: max_position_pct can only lower the configured 10% cap."""
    out = await server.trade(ticker="T-A", side="yes", count=1000, max_position_pct=pct)
    assert out["ok"] and out["count"] == 200  # $100 of $1,000 at 50c


async def test_max_position_pct_can_lower_the_cap(ex):
    out = await server.trade(ticker="T-A", side="yes", count=1000, max_position_pct=0.05,
                             confirm=True)
    assert out["count"] == 100 and ex.posts[0]["count"] == "100.00"


async def test_cap_counts_what_is_already_held(ex):
    """D1 (fixed in ac49b26): $1,000 equity already holding $100 of T-A buys nothing more."""
    ex.balance, ex.portfolio_value = 90_000, 10_000      # $900 cash + $100 in T-A
    ex.positions["T-A"] = "200.00"
    out = await server.trade(ticker="T-A", side="yes", count=200, confirm=True)
    assert ex.posts == [] and out["ok"] is False


async def test_cap_counts_resting_buys(ex):
    """D1b (fixed in ac49b26): a resting $100 buy uses up the cap for the next call."""
    await server.trade(ticker="T-A", side="yes", count=200, confirm=True)
    await server.trade(ticker="T-A", side="yes", count=200, confirm=True)
    total = sum(float(p["count"]) * float(p["price"]) for p in ex.posts)
    assert total <= 100.0


# ---------------------------------------------------------------------------
# trade: price bounds 1..99c
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("price", [0.0, 0.004, -0.5, 0.995, 1.0, 1.5, 50.0])
@pytest.mark.parametrize("confirm", [False, True])
async def test_trade_price_outside_1_to_99_cents_is_refused(ex, price, confirm):
    out = await server.trade(ticker="T-A", side="yes", count=1, price=price, confirm=confirm)
    assert out["ok"] is False and "out of range 1..99" in out["reason"]
    assert ex.posts == []


@pytest.mark.parametrize("price, cents", [(0.01, 1), (0.99, 99), (0.505, 50)])
async def test_trade_price_on_the_bounds_is_accepted(ex, price, cents):
    out = await server.trade(ticker="T-A", side="yes", count=1, price=price, confirm=True)
    assert out["price_cents"] == cents and ex.posts[0]["price"] == f"{cents / 100:.4f}"


@pytest.mark.parametrize("price", [math.nan, math.inf, -math.inf])
async def test_trade_non_finite_price_raises_before_any_order(ex, price):
    with pytest.raises((ValueError, OverflowError)):
        await server.trade(ticker="T-A", side="yes", count=1, price=price, confirm=True)
    assert ex.posts == []


async def test_trade_with_empty_book_is_refused(ex):
    ex.markets["T-A"] = {"status": "active", "yes_bid_dollars": 0, "yes_ask_dollars": 0,
                         "no_bid_dollars": 0, "no_ask_dollars": 0}
    out = await server.trade(ticker="T-A", side="yes", count=1, confirm=True)
    assert out["ok"] is False and ex.posts == []


@pytest.mark.parametrize("market", [
    {"status": "closed", "yes_bid_dollars": 0.4, "yes_ask_dollars": 0.5, "no_ask_dollars": 0.5},
    {"status": "active", "yes_bid_dollars": 0.0, "yes_ask_dollars": 1.0,
     "no_ask_dollars": 1.0},  # collection ticker
])
async def test_trade_on_untradeable_market_is_refused(ex, market):
    ex.markets["T-A"] = market
    out = await server.trade(ticker="T-A", side="yes", count=1, confirm=True)
    assert "market_not_tradeable" in out["reason"] and ex.posts == []


# ---------------------------------------------------------------------------
# trade: input validation (counts, side strings, ticker injection)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("count", [0, -1, -10**9])
@pytest.mark.parametrize("confirm", [False, True])
async def test_trade_non_positive_count_is_refused(ex, count, confirm):
    out = await server.trade(ticker="T-A", side="yes", count=count, confirm=confirm)
    assert out["ok"] is False and "size capped to 0" in out["reason"]
    assert ex.posts == []


@pytest.mark.parametrize("side", ["YES", "Yes", "NO", "nO"])
async def test_trade_side_is_case_insensitive(ex, side):
    out = await server.trade(ticker="T-A", side=side, count=1, confirm=True)
    assert out["ok"] is True and out["side"] == side.lower() and len(ex.posts) == 1


@pytest.mark.parametrize("side", ["maybe", "", "yes ", "buy", "y", "sell"])
async def test_trade_bad_side_never_reaches_the_wire_live(ex, side):
    out = await server.trade(ticker="T-A", side=side, count=1, confirm=True)
    assert out["ok"] is False and ex.posts == []


@pytest.mark.parametrize("side", ["maybe", "", "yes ", "buy", "y", "sell"])
async def test_trade_bad_side_is_refused_in_dry_run(ex, side):
    """D4: refused up front, before the governor or any market read."""
    out = await server.trade(ticker="T-A", side=side, count=1)
    assert out == {"ok": False, "reason": "side must be 'yes' or 'no'"} and ex.log == []


BAD_TICKERS = [
    "T-A/../../portfolio/balance", "", "  ", "T-A%2F..", "'; DROP TABLE positions;--",
    "T-A?status=resting", "t-a?x=1", "T-A#frag", "../markets/T-A", "T A", "-T-A", "A" * 101,
]


@pytest.mark.parametrize("ticker", BAD_TICKERS)
@pytest.mark.parametrize("confirm", [False, True])
async def test_trade_injected_ticker_is_refused_before_any_request(ex, ticker, confirm):
    """D5: validated at the toolbelt boundary, so nothing reaches the URL."""
    out = await server.trade(ticker=ticker, side="yes", count=1, confirm=confirm)
    assert out == {"ok": False, "reason": f"invalid ticker {ticker!r}"} and ex.log == []


@pytest.mark.parametrize("ticker", BAD_TICKERS)
async def test_close_injected_ticker_is_refused_before_any_request(ex, ticker):
    ex.positions["T-A"] = "10.00"
    out = await server.close(ticker=ticker, confirm=True)
    assert out["ok"] is False and out["reason"].startswith("invalid ticker") and ex.log == []


async def test_trade_unknown_ticker_sends_no_order(ex):
    with pytest.raises(KalshiAPIError, match="404"):
        await server.trade(ticker="T-B", side="yes", count=1, confirm=True)
    assert ex.posts == []


@pytest.mark.parametrize("ticker", ["t-a", " T-A", "T-A\n", "  t-A\t"])
async def test_trade_lowercase_or_padded_ticker_is_sent_uppercase(ex, ticker):
    out = await server.trade(ticker=ticker, side="yes", count=1, confirm=True)
    assert out["ok"] is True and out["ticker"] == "T-A"
    assert [p["ticker"] for p in ex.posts] == ["T-A"]
    assert all(path in ("/trade-api/v2/markets/T-A",) or "/markets/" not in path
               for _, path, _ in ex.log)


@pytest.mark.parametrize("ticker", ["t-a", " t-a\n"])
async def test_close_lowercase_or_padded_ticker_is_sent_uppercase(ex, ticker):
    ex.positions["T-A"] = "3.00"
    out = await server.close(ticker=ticker, confirm=True)
    assert out["ok"] is True and out["ticker"] == "T-A"
    assert [(p["ticker"], p["count"]) for p in ex.posts] == [("T-A", "3.00")]


@pytest.mark.parametrize("ticker", ["KXHIGHNY-26NOV02-B58.5", "A", "A" * 100])
async def test_valid_ticker_shapes_pass_validation(ex, ticker):
    ex.markets[ticker] = ex.markets["T-A"]
    out = await server.trade(ticker=ticker, side="yes", count=1)
    assert out["ok"] is True


# ---------------------------------------------------------------------------
# close: dry-run default, held side, never more than held
# ---------------------------------------------------------------------------

async def test_close_without_confirm_sends_no_order(ex):
    ex.positions["T-A"] = "10.00"
    out = await server.close(ticker="T-A")
    assert out["ok"] is True and out["dry"] is True and out["count"] == 10
    assert (out["side"], out["price_cents"]) == ("yes", 48)  # at the YES bid
    assert ex.posts == [] and ex.gets_only() and _journal_lines() == []


async def test_close_confirm_sells_the_held_side_at_the_bid(ex):
    ex.positions["T-A"] = "-6.00"   # 6 NO
    out = await server.close(ticker="T-A", confirm=True)
    assert out["ok"] is True and out["side"] == "no" and out["count"] == 6
    p = ex.posts[0]
    # sell NO at the 50c NO bid = bid the YES leg at 50c
    assert (p["side"], p["price"], p["count"]) == ("bid", "0.5000", "6.00")
    assert len(_journal_lines()) == 1


@pytest.mark.parametrize("count, sent", [(10**9, "10.00"), (11, "10.00"), (3, "3.00")])
async def test_close_never_sells_more_than_held(ex, count, sent):
    ex.positions["T-A"] = "10.00"
    await server.close(ticker="T-A", count=count, confirm=True)
    assert [p["count"] for p in ex.posts] == [sent]


@pytest.mark.parametrize("count", [0, -1, -10**9])
async def test_close_non_positive_count_is_refused(ex, count):
    ex.positions["T-A"] = "10.00"
    out = await server.close(ticker="T-A", count=count, confirm=True)
    assert out["ok"] is False and ex.posts == []


@pytest.mark.parametrize("fp", [None, "0.00", "0.40"])
async def test_close_with_no_position_is_refused(ex, fp):
    if fp is not None:
        ex.positions["T-A"] = fp
    out = await server.close(ticker="T-A", confirm=True)
    assert out == {"ok": False, "reason": "no position to close", "ticker": "T-A"}
    assert ex.posts == []


@pytest.mark.parametrize("price, cents", [(5.0, 99), (0.99, 99), (0.01, 1)])
async def test_close_explicit_price_is_clamped_to_99(ex, price, cents):
    ex.positions["T-A"] = "2.00"
    out = await server.close(ticker="T-A", price=price, confirm=True)
    assert out["price_cents"] == cents and ex.posts[0]["price"] == f"{cents / 100:.4f}"


@pytest.mark.parametrize("price", [0.0, 0.004, -1.0])
async def test_close_price_below_one_cent_is_refused(ex, price):
    ex.positions["T-A"] = "2.00"
    out = await server.close(ticker="T-A", price=price, confirm=True)
    assert out["ok"] is False and ex.posts == []


async def test_close_with_no_bid_is_refused(ex):
    ex.positions["T-A"] = "2.00"
    ex.markets["T-A"]["yes_bid_dollars"] = 0
    out = await server.close(ticker="T-A", confirm=True)
    assert out["ok"] is False and "no bid" in out["reason"] and ex.posts == []


async def test_close_nets_resting_sells_against_the_holding(ex):
    """D3: holding 10 YES with a 10-lot sell resting, a second close sends nothing
    (before the fix it sent another 10 and could flip the account to 10 NO)."""
    ex.positions["T-A"] = "10.00"
    await server.close(ticker="T-A", confirm=True)    # rests (fake exchange never fills)
    await server.close(ticker="T-A", confirm=True)
    assert sum(float(p["count"]) for p in ex.posts) <= 10


async def test_close_sells_only_the_part_not_already_resting(ex):
    ex.positions["T-A"] = "10.00"
    ex.resting.append({"order_id": "R-1", "ticker": "T-A", "outcome_side": "no",
                       "remaining_count_fp": "4.00", "status": "resting"})
    out = await server.close(ticker="T-A", confirm=True)
    assert out["count"] == 6 and [p["count"] for p in ex.posts] == ["6.00"]


@pytest.mark.parametrize("bad", [
    {"order_id": "R-1", "ticker": "T-A", "remaining_count_fp": "4.00"},       # no side
    {"order_id": "R-1", "ticker": "T-A", "outcome_side": "no"},               # no count
])
@pytest.mark.parametrize("confirm", [False, True])
async def test_close_fails_closed_on_an_unreadable_resting_order(ex, bad, confirm):
    ex.positions["T-A"] = "10.00"
    ex.resting.append(bad)
    out = await server.close(ticker="T-A", confirm=confirm)
    assert out["ok"] is False and out["reason"] == "sellable_unreadable" and ex.posts == []


# ---------------------------------------------------------------------------
# close under a halt
# ---------------------------------------------------------------------------

async def test_close_under_manual_halt_is_blocked_live(ex):
    """The client gate refuses every sell under MANUAL_HALT."""
    _halt_flag()
    ex.positions["T-A"] = "10.00"
    with pytest.raises(KalshiAPIError, match="RISK_HALT: MANUAL_HALT"):
        await server.close(ticker="T-A", confirm=True)
    assert ex.posts == []


async def test_close_under_drawdown_halt_goes_out_reduce_only_ioc(ex):
    _governor_state(sod_cents=100_000, peak_cents=200_000)
    ex.positions["T-A"] = "10.00"
    out = await server.close(ticker="T-A", confirm=True)
    assert out["ok"] is True
    p = ex.posts[0]
    assert p["reduce_only"] is True and p["time_in_force"] == "immediate_or_cancel"
    assert p["count"] == "10.00"


async def test_trade_under_drawdown_halt_cannot_buy_via_close(ex):
    """close only ever sells what is held: with no holding it cannot open anything,
    halt or not."""
    _governor_state(sod_cents=100_000, peak_cents=200_000)
    out = await server.close(ticker="T-A", confirm=True)
    assert out["ok"] is False and ex.posts == []


# ---------------------------------------------------------------------------
# no bypass of KalshiClient.place_order's gate
# ---------------------------------------------------------------------------

def test_mcp_server_source_never_reaches_the_order_path_directly():
    """The MCP module itself names no order method and no raw request: every order
    goes server -> toolbelt -> KalshiClient.place_order."""
    import ast
    tree = ast.parse(Path(server.__file__).read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    forbidden = {"place_order", "cancel_order", "amend_order", "batch_create_orders",
                 "batch_cancel_orders", "_make_authenticated_request"}
    assert not (attrs | names) & forbidden, (attrs | names) & forbidden
    assert "client" not in attrs  # never reaches the raw httpx client either


def test_toolbelt_mutating_calls_are_place_order_only():
    """In the toolbelt, the only mutating client call is place_order, and there is
    no POST/DELETE/PUT through the raw request helper."""
    import ast
    import src.agent.toolbelt as tb
    src_text = Path(tb.__file__).read_text()
    tree = ast.parse(src_text)
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "cancel_order" not in calls and "amend_order" not in calls
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_make_authenticated_request"):
            assert isinstance(n.args[0], ast.Constant) and n.args[0].value == "GET"


async def test_every_live_post_passes_through_the_client_gate(ex):
    """The fixture records any non-GET sent outside KalshiClient.place_order as a
    bypass (asserted at teardown). Drive both tools live and check the count."""
    ex.positions["T-A"] = "5.00"
    await server.trade(ticker="T-A", side="no", count=2, confirm=True)
    await server.close(ticker="T-A", count=1, confirm=True)
    assert len(ex.posts) == 2 and ex.bypass == []


# ---------------------------------------------------------------------------
# read-only tools never mutate the account
# ---------------------------------------------------------------------------

def _snapshot(root: Path):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


async def test_read_only_tools_send_only_gets_and_never_touch_the_order_path(ex, tmp_path):
    import sqlite3
    ex.positions["T-A"] = "3.00"
    # history reads <repo>/trading_system.db relative to the module file: point it at tmp
    fake_src = tmp_path / "repo" / "src"
    fake_src.mkdir(parents=True)
    db = tmp_path / "repo" / "trading_system.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trade_logs (market_id, side, entry_price, exit_price, quantity, "
                "pnl, entry_timestamp, strategy)")
    con.execute("INSERT INTO trade_logs VALUES ('T-A','YES',0.4,0.5,1,0.1,'2026-01-01','x')")
    con.commit()
    con.close()
    before_db = db.read_bytes()

    with patch.object(server, "__file__", str(fake_src / "mcp_server.py")), \
         patch.object(KalshiClient, "place_order", AsyncMock(side_effect=AssertionError("order"))):
        before = _snapshot(tmp_path)
        out = {
            "brief": await server.brief(),
            "status": await server.status(),
            "settle": await server.settle(),
            "learnings": await server.learnings(),
            "edge": await server.edge(),
            "policy": await server.policy(),
            "scores": await server.scores(),
            "history": await server.history(limit=5),
            "hunt": await server.hunt(top=3),
        }
        after = _snapshot(tmp_path)

    assert ex.gets_only() and ex.posts == [], [r for r in ex.log if r[0] != "GET"]
    assert out["brief"]["positions"][0]["ticker"] == "T-A"
    assert out["history"]["total_trades"] == 1
    assert db.read_bytes() == before_db
    # Local files those tools write (none is the account, none is an order):
    # brief/trade's governor check persists its baseline, and scores creates the
    # category-score tables in ./trading_system.db. settle appends to the
    # settlements log only when there are new settlements (none here).
    changed = {k for k in after if before.get(k) != after[k]}
    assert changed == {"data/runtime/risk_governor_state.json", "trading_system.db"}, changed


async def test_learnings_default_dry_writes_no_journal_or_learnings(ex, tmp_path):
    await server.learnings()
    assert not Path("data/runtime/decision_journal.jsonl").exists()
    assert not Path("data/runtime/learnings.jsonl").exists()


# ---------------------------------------------------------------------------
# Edge Policy gate: a BLOCK refuses; override_policy lifts only the policy
# ---------------------------------------------------------------------------

def _policy(**kw):
    p = Path("data/runtime/edge_policy.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"blocks": [], "warnings": [], "haircuts": [], **kw}))


@pytest.mark.parametrize("confirm", [False, True])
async def test_policy_block_refuses_trade(ex, confirm):
    _policy(blocks=[{"dimension": "category", "label": "sports", "reason": "loses"}])
    out = await server.trade(ticker="T-A", side="yes", count=1, category="sports",
                             confirm=confirm)
    assert out["reason"] == "blocked_by_policy" and ex.posts == []


async def test_override_policy_lifts_the_block_but_not_the_cap_or_halt(ex):
    _policy(blocks=[{"dimension": "category", "label": "sports", "reason": "loses"}])
    out = await server.trade(ticker="T-A", side="yes", count=10**6, category="sports",
                             override_policy=True, confirm=True)
    assert out["count"] == 200 and out["policy_note"]["overridden_block"]
    _halt_flag()
    out = await server.trade(ticker="T-A", side="yes", count=1, category="sports",
                             override_policy=True, confirm=True)
    assert out["reason"] == "governor_halted" and len(ex.posts) == 1


# ---------------------------------------------------------------------------
# toolbelt defaults: a caller that forgets `dry` must not send an order
# ---------------------------------------------------------------------------

async def test_place_guarded_order_with_no_dry_argument_sends_zero_posts(ex):
    from src.agent.toolbelt import place_guarded_order
    out = await server._with_client(lambda c: place_guarded_order(
        c, ticker="T-A", side="yes", count=10))
    assert out["ok"] is True and out["dry"] is True and out["journaled"] is False
    assert ex.posts == [] and ex.gets_only() and ex.bypass == []


async def test_close_position_with_no_dry_argument_sends_zero_posts(ex):
    from src.agent.toolbelt import close_position
    ex.positions["T-A"] = "10.00"
    out = await server._with_client(lambda c: close_position(c, ticker="T-A"))
    assert out["ok"] is True and out["dry"] is True and out["journaled"] is False
    assert ex.posts == [] and ex.gets_only() and ex.bypass == []
