"""close-all on the replies a stop has to survive. Fakes only: no network, no orders."""

from argparse import Namespace

import pytest

import cli
from src.clients import kalshi_client as kc

V2_BOOK = {"orderbook_fp": {"yes_dollars": [["0.40", "9"], ["0.48", "5"]],
                            "no_dollars": [["0.50", "5"]]}}


class Fake:
    pages: list = []      # replies to get_positions, in order, or an exception to raise
    books: dict = {}      # ticker -> reply, or an exception to raise
    default_book = V2_BOOK
    place_reply = {"order": {"order_id": "o"}}
    resting: dict = {}    # ticker -> rows, or a whole reply when it isn't a list
    held_override = None  # a value for _held_on_side to return, or an exception to raise
    position_calls: list = []
    placed: list = []
    closed = 0

    def __init__(self, *a, **k):
        pass

    async def get_positions(self, **kw):
        Fake.position_calls.append(kw)
        reply = Fake.pages[min(len(Fake.position_calls), len(Fake.pages)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def _held_on_side(self, ticker, side):
        if isinstance(Fake.held_override, Exception):
            raise Fake.held_override
        if Fake.held_override is not None:
            return Fake.held_override
        for page in Fake.pages:
            for row in (page.get("market_positions") or []) if isinstance(page, dict) else []:
                if isinstance(row, dict) and row.get("ticker") == ticker:
                    signed = float(row["position_fp"])
                    return abs(signed) if (signed > 0) == (side == "yes") else 0.0
        return 0.0

    async def get_orders(self, ticker=None, status=None, **kw):
        assert status == "resting"
        reply = Fake.resting.get(ticker, [])
        if isinstance(reply, Exception):
            raise reply
        return {"orders": reply, "cursor": ""} if isinstance(reply, list) else reply

    async def get_orderbook(self, ticker, depth=1):
        reply = Fake.books.get(ticker, Fake.default_book)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def place_order(self, **kw):
        Fake.placed.append(kw)
        if isinstance(Fake.place_reply, Exception):
            raise Fake.place_reply
        return Fake.place_reply

    async def close(self):
        Fake.closed += 1


@pytest.fixture
def run(monkeypatch, capsys):
    Fake.pages, Fake.books, Fake.position_calls, Fake.placed = [], {}, [], []
    Fake.resting, Fake.held_override = {}, None
    Fake.default_book, Fake.place_reply, Fake.closed = V2_BOOK, {"order": {"order_id": "o"}}, 0
    monkeypatch.setattr(kc, "KalshiClient", Fake)

    def _run(*pages, live=True):
        """Returns (exit code, stdout)."""
        Fake.pages = list(pages)
        capsys.readouterr()
        code = 0
        try:
            cli.cmd_close_all(Namespace(yes=True, live=live))
        except SystemExit as e:
            code = e.code
        return code, capsys.readouterr().out
    return _run


def _rows(*pairs):
    return {"market_positions": [{"ticker": t, "position_fp": p} for t, p in pairs]}


# ---- the v2 book: every position used to read as "no bids" and nothing sold ----

def test_v2_book_sells_yes_at_the_best_yes_bid(run):
    code, _ = run(_rows(("A", "5.00")))
    assert code == 0
    assert [(o["ticker"], o["side"], o["action"], o["count"], o["yes_price"])
            for o in Fake.placed] == [("A", "yes", "sell", 5.0, 48)]
    assert "no_price" not in Fake.placed[0]


def test_v2_book_sells_no_at_the_best_no_bid(run):
    code, _ = run(_rows(("A", "-3.00")))
    assert code == 0
    assert [(o["side"], o["count"], o["no_price"]) for o in Fake.placed] == [("no", 3.0, 50)]
    assert "yes_price" not in Fake.placed[0]


def test_old_book_shape_still_sells(run):
    Fake.default_book = {"orderbook": {"yes": [[30, 1], [47, 1]], "no": []}}
    code, _ = run(_rows(("A", "1.00")))
    assert code == 0 and Fake.placed[0]["yes_price"] == 47


def test_v2_book_wins_when_a_reply_has_both(run):
    Fake.default_book = {**V2_BOOK, "orderbook": {"yes": [[11, 1]]}}
    run(_rows(("A", "1.00")))
    assert Fake.placed[0]["yes_price"] == 48


def test_old_book_is_used_when_the_v2_side_is_empty(run):
    Fake.default_book = {"orderbook_fp": {"yes_dollars": []}, "orderbook": {"yes": [[12, 1]]}}
    run(_rows(("A", "1.00")))
    assert Fake.placed[0]["yes_price"] == 12


@pytest.mark.parametrize("bid,cents", [("0.487", 48), ("0.99", 99), ("0.01", 1), ("0.29", 29)])
def test_v2_bid_rounds_down_to_whole_cents_so_the_sell_crosses(run, bid, cents):
    Fake.default_book = {"orderbook_fp": {"yes_dollars": [[bid, "1"]]}}
    run(_rows(("A", "1.00")))
    assert Fake.placed[0]["yes_price"] == cents


@pytest.mark.parametrize("book", [
    {}, {"orderbook_fp": None}, {"orderbook_fp": {}}, {"orderbook": None},
    {"orderbook_fp": {"yes_dollars": []}}, {"orderbook_fp": {"yes_dollars": None}},
    {"orderbook_fp": {"no_dollars": [["0.50", "5"]]}},
])
def test_no_bid_on_our_side_sends_nothing_and_exits_1(run, book):
    Fake.default_book = book
    code, out = run(_rows(("A", "1.00")))
    assert code == 1 and Fake.placed == []
    assert "no YES bids" in out and "1 position(s) not closed" in out


@pytest.mark.parametrize("book", [
    {"orderbook_fp": {"yes_dollars": [["abc", "1"]]}},
    {"orderbook_fp": {"yes_dollars": [["NaN", "1"]]}},
    {"orderbook_fp": {"yes_dollars": [["Infinity", "1"]]}},
    {"orderbook_fp": {"yes_dollars": [[]]}},
    {"orderbook_fp": {"yes_dollars": 5}},
    {"orderbook_fp": {"yes_dollars": [["1.00", "1"]]}},
    {"orderbook_fp": {"yes_dollars": [["0.004", "1"]]}},
    {"orderbook_fp": {"yes_dollars": [["-0.10", "1"]]}},
    {"orderbook": {"yes": [[100, 1]]}},
    {"orderbook": {"yes": [[0, 1]]}},
    "not a dict",
    RuntimeError("503"),
])
def test_a_bid_it_cant_read_or_send_fails_that_position_and_sells_the_rest(run, book):
    Fake.books = {"BAD": book}
    code, out = run(_rows(("BAD", "1.00"), ("GOOD", "1.00")))
    assert code == 1
    assert [o["ticker"] for o in Fake.placed] == ["GOOD"]
    assert "BAD: orderbook fetch failed" in out


# ---- every page, or a loud failure ----

def test_positions_past_the_first_page_are_sold(run):
    p1 = {**_rows(*((f"P{i}", "1.00") for i in range(100))), "cursor": "c1"}
    p2 = _rows(*((f"Q{i}", "1.00") for i in range(50)))
    code, out = run(p1, p2)
    assert code == 0
    assert len(Fake.placed) == 150 and Fake.placed[-1]["ticker"] == "Q49"
    assert [c.get("cursor") for c in Fake.position_calls] == [None, "c1"]
    assert "Found 150 open position(s)" in out


@pytest.mark.parametrize("reply", [
    {}, {"market_positions": None}, {"market_positions": "x"}, {"event_positions": []},
    [], None, "oops", RuntimeError("503"),
])
def test_a_bad_positions_reply_is_never_reported_as_no_positions(run, reply):
    code, out = run(reply)
    assert code == 1 and Fake.placed == []
    assert "No open positions" not in out and "Error:" in out


@pytest.mark.parametrize("page2", [{}, {"market_positions": None}, RuntimeError("503")])
def test_a_bad_later_page_stops_with_exit_1_and_sells_nothing(run, page2):
    code, out = run({**_rows(("A", "1.00")), "cursor": "c1"}, page2)
    assert code == 1 and Fake.placed == [] and "Error:" in out


def test_a_repeated_cursor_stops_with_exit_1(run):
    page = {**_rows(("A", "1.00")), "cursor": "same"}
    code, out = run(page, page, page)
    assert code == 1 and Fake.placed == [] and "Error:" in out


def test_a_real_empty_account_says_so_and_exits_0(run):
    code, out = run({"market_positions": []})
    assert code == 0 and "No open positions" in out and Fake.placed == []


def test_all_zero_rows_read_as_no_positions(run):
    code, out = run(_rows(("A", "0.00")))
    assert code == 0 and "No open positions" in out and Fake.placed == []


# ---- a bad row can't pass as flat, and can't stop the others ----

@pytest.mark.parametrize("row", [
    {"ticker": "BAD"}, {"ticker": "BAD", "position_fp": None}, {"ticker": "BAD", "position_fp": ""},
    {"ticker": "BAD", "position_fp": "abc"}, {"ticker": "BAD", "position_fp": "nan"},
    {"ticker": "BAD", "position_fp": "inf"}, {"ticker": "BAD", "position_fp": "-inf"},
    {"ticker": "BAD", "position_fp": []}, {"position_fp": "5.00"}, {"ticker": "", "position_fp": "5.00"},
    "not a row", None,
])
def test_an_unreadable_row_fails_and_the_good_row_still_sells(run, row):
    code, out = run({"market_positions": [row, {"ticker": "GOOD", "position_fp": "1.00"}]})
    assert code == 1
    assert [o["ticker"] for o in Fake.placed] == ["GOOD"]
    assert "unreadable position row" in out and "Failed: 1" in out


def test_only_unreadable_rows_is_not_no_positions(run):
    code, out = run({"market_positions": [{"ticker": "BAD"}]})
    assert code == 1 and "No open positions" not in out and "Failed: 1" in out


@pytest.mark.parametrize("row", [{"ticker": "X", "position_fp": "nan"}, {"ticker": "X"}, {"position": 1}])
def test_helper_raises_on_a_row_that_may_be_a_position(row):
    with pytest.raises(ValueError):
        cli.open_positions_from_v2([row])


# ---- the exit code tells the truth ----

def test_a_rejected_order_exits_1_and_the_rest_are_tried(run):
    Fake.place_reply = RuntimeError("refused")
    code, out = run(_rows(("A", "1.00"), ("B", "1.00")))
    assert code == 1 and len(Fake.placed) == 2
    assert "Placed: 0 | Failed: 2" in out and "2 position(s) not closed" in out


@pytest.mark.parametrize("reply", [None, {}, {"error": "no"}])
def test_an_unexpected_order_reply_exits_1(run, reply):
    Fake.place_reply = reply
    code, out = run(_rows(("A", "1.00")))
    assert code == 1 and "unexpected response" in out


def test_all_sold_exits_0_and_names_no_failure(run):
    code, out = run(_rows(("A", "1.00"), ("B", "-1.00")))
    assert code == 0 and "Placed: 2 | Failed: 0" in out and "not closed" not in out


def test_the_client_closes_on_every_path(run):
    run(_rows(("A", "1.00")))
    run(RuntimeError("503"))
    Fake.default_book = {}
    run(_rows(("A", "1.00")))
    assert Fake.closed == 3


# ---- never past the holding: a second run can't open the other side ----

def _sell_yes(n, ticker="A"):
    return {"ticker": ticker, "outcome_side": "no", "remaining_count_fp": n}


def test_resting_sells_shrink_the_sell_to_what_is_left(run):
    Fake.resting = {"A": [_sell_yes("3.00")]}
    code, _ = run(_rows(("A", "5.00")))
    assert code == 0 and Fake.placed[0]["count"] == 2.0


def test_fractional_remainder_rounds_down_never_up(run):
    Fake.resting = {"A": [_sell_yes("0.504")]}
    run(_rows(("A", "1.00")))
    assert Fake.placed[0]["count"] == 0.49


def test_a_second_run_sends_nothing_and_says_the_position_is_open(run):
    Fake.resting = {"A": [_sell_yes("5.00")]}
    code, out = run(_rows(("A", "5.00"), ("B", "1.00")))
    assert code == 1
    assert [o["ticker"] for o in Fake.placed] == ["B"]
    assert "A: resting orders cover the holding" in out and "Failed: 1" in out


def test_resting_orders_past_the_holding_send_nothing(run):
    Fake.resting = {"A": [_sell_yes("9.00")]}
    code, _ = run(_rows(("A", "5.00")))
    assert code == 1 and Fake.placed == []


def test_a_resting_order_on_our_own_side_takes_nothing_off(run):
    Fake.resting = {"A": [{"ticker": "A", "outcome_side": "yes", "remaining_count_fp": "3.00"}]}
    run(_rows(("A", "5.00")))
    assert Fake.placed[0]["count"] == 5.0


def test_a_short_side_holding_nets_the_opposite_orders(run):
    Fake.resting = {"A": [{"ticker": "A", "outcome_side": "yes", "remaining_count_fp": "1.00"}]}
    run(_rows(("A", "-3.00")))
    assert (Fake.placed[0]["side"], Fake.placed[0]["count"]) == ("no", 2.0)


def test_held_smaller_than_the_list_said_caps_the_sell(run):
    Fake.held_override = 2.0
    run(_rows(("A", "5.00")))
    assert Fake.placed[0]["count"] == 2.0


@pytest.mark.parametrize("reply", [
    {}, {"orders": None}, RuntimeError("503"),
    [{"ticker": "BAD", "remaining_count_fp": "1.00"}],
    [{"ticker": "BAD", "outcome_side": "no", "remaining_count_fp": "abc"}],
    {"orders": [_sell_yes("0.00", "BAD")] * 100, "cursor": "more"},
])
def test_resting_orders_it_cant_read_send_nothing_for_that_position(run, reply):
    Fake.resting = {"BAD": reply}
    code, out = run(_rows(("BAD", "1.00"), ("GOOD", "1.00")))
    assert code == 1
    assert [o["ticker"] for o in Fake.placed] == ["GOOD"]
    assert "BAD: can't read what's held or resting" in out


@pytest.mark.parametrize("held", [RuntimeError("503"), float("nan")])
def test_a_held_read_that_fails_sends_nothing(run, held):
    Fake.held_override = held
    code, out = run(_rows(("A", "1.00")))
    assert code == 1 and Fake.placed == [] and "A:" in out


# ---- dry run sends nothing, and reports what a live run would do ----

def test_dry_run_on_a_v2_book_previews_and_sends_nothing(run):
    code, out = run(_rows(("A", "2.00")), live=False)
    assert code == 0 and Fake.placed == []
    assert "would sell 2.0 YES of A at 48¢ (~$0.96)" in out
    assert "Would place: 1 | Failed: 0" in out and "Placed:" not in out


def test_dry_run_exits_1_when_a_live_run_would_leave_something_open(run):
    Fake.default_book = {}
    code, out = run(_rows(("A", "2.00")), live=False)
    assert code == 1 and Fake.placed == [] and "Would place: 0 | Failed: 1" in out
