"""account_brief over a fake client: paging, and a failed read is reported, never shown as nothing."""
import pytest

from src.agent.toolbelt import account_brief


class FakeDecision:
    def to_dict(self):
        return {"allowed": True}


class FakeGovernor:
    def __init__(self):
        self.equity = None

    async def check(self, current_equity_cents=None):
        self.equity = current_equity_cents
        return FakeDecision()


class FakeClient:
    def __init__(self, pages=(), orders=(), orders_fail=False, orders_cursor=None):
        self.pages, self.orders, self.orders_fail, self.cursors = list(pages), list(orders), orders_fail, []
        self.orders_cursor = orders_cursor

    async def get_balance(self):
        return {"balance": 1000, "portfolio_value": 250}

    async def _make_authenticated_request(self, method, path, params=None):
        assert method == "GET"
        self.cursors.append(params.get("cursor"))
        return self.pages.pop(0)

    async def get_orders(self, status=None):
        if self.orders_fail:
            raise RuntimeError("503")
        return {"orders": self.orders, "cursor": self.orders_cursor}


def _pos(ticker, fp="3.00"):
    return {"ticker": ticker, "position_fp": fp, "market_exposure_dollars": "2.70", "total_traded_dollars": "2.70"}


async def _brief(client):
    return await account_brief(client, governor=FakeGovernor())


async def test_a_clean_read_reports_no_errors_and_converts_cents():
    b = await _brief(FakeClient(pages=[{"market_positions": [_pos("A"), _pos("B", "-2.00")]}]))
    assert b["read_errors"] == []
    assert (b["cash"], b["portfolio"], b["equity"]) == (10.0, 2.5, 12.5)
    assert [(p["ticker"], p["side"], p["count"]) for p in b["positions"]] == [("A", "YES", 3), ("B", "NO", 2)]


async def test_positions_follow_the_cursor_across_pages():
    c = FakeClient(pages=[{"market_positions": [_pos("A")], "cursor": "next"}, {"market_positions": [_pos("B")]}])
    b = await _brief(c)
    assert c.cursors == [None, "next"]
    assert [p["ticker"] for p in b["positions"]] == ["A", "B"]


async def test_a_failed_resting_read_is_reported_not_shown_as_none():
    b = await _brief(FakeClient(pages=[{"market_positions": []}], orders_fail=True))
    assert b["resting_orders"] == []
    assert b["read_errors"] == ["resting_orders"]


async def test_an_unreadable_position_is_reported_not_silently_dropped():
    b = await _brief(FakeClient(pages=[{"market_positions": [_pos("BAD", "n/a"), _pos("A")]}]))
    assert [p["ticker"] for p in b["positions"]] == ["A"]
    assert b["read_errors"] == ["position BAD"]


@pytest.mark.parametrize("fp", ["nan", "inf", "-inf", "1e400", 10 ** 400, None, ["3"], {"a": 1}, ""])
async def test_a_position_fp_that_is_no_finite_number_is_reported(fp):
    b = await _brief(FakeClient(pages=[{"market_positions": [_pos("BAD", fp), _pos("A")]}]))
    assert [p["ticker"] for p in b["positions"]] == ["A"]
    assert b["read_errors"] == ["position BAD"]


async def test_a_position_with_no_position_fp_is_reported():
    b = await _brief(FakeClient(pages=[{"market_positions": [{"ticker": "BAD"}]}]))
    assert b["positions"] == [] and b["read_errors"] == ["position BAD"]


async def test_hitting_the_position_page_cap_is_reported():
    c = FakeClient(pages=[{"market_positions": [], "cursor": "more"}] * 30)
    assert (await _brief(c))["read_errors"] == ["positions: page cap"]
    assert c.pages == []


async def test_a_last_position_page_inside_the_cap_is_not_an_error():
    c = FakeClient(pages=[{"market_positions": [], "cursor": "more"}] * 29 + [{"market_positions": []}])
    assert (await _brief(c))["read_errors"] == []


async def test_more_resting_pages_than_were_read_is_reported():
    b = await _brief(FakeClient(pages=[{"market_positions": []}], orders=[{"ticker": "A"}], orders_cursor="more"))
    assert len(b["resting_orders"]) == 1
    assert b["read_errors"] == ["resting_orders: first page only"]


async def test_a_flat_position_is_not_an_error():
    b = await _brief(FakeClient(pages=[{"market_positions": [_pos("FLAT", "0.00")]}]))
    assert b["positions"] == [] and b["read_errors"] == []


class NoBalanceClient(FakeClient):
    def __init__(self, bal, **kw):
        super().__init__(**kw)
        self.bal = bal

    async def get_balance(self):
        return self.bal


@pytest.mark.parametrize("bal", [
    {"status_code": 200, "empty_body": True},
    {"balance": 1000},
    {"portfolio_value": 250},
    {"balance": None, "portfolio_value": 250},
    {"balance": 1000, "portfolio_value": None},
])
async def test_a_balance_reply_missing_a_figure_is_reported(bal):
    b = await _brief(NoBalanceClient(bal, pages=[{"market_positions": []}]))
    assert b["read_errors"] == ["balance"]


async def test_a_real_zero_balance_is_not_an_error():
    b = await _brief(NoBalanceClient({"balance": 0, "portfolio_value": 0}, pages=[{"market_positions": []}]))
    assert b["read_errors"] == []


@pytest.mark.parametrize("page", [
    {},
    {"status_code": 200, "empty_body": True},
    {"market_positions": None},
    {"market_positions": "A"},
    {"market_positions": {"ticker": "A"}},
])
async def test_a_positions_reply_with_no_list_is_reported(page):
    b = await _brief(FakeClient(pages=[page]))
    assert b["positions"] == [] and b["read_errors"] == ["positions: no list"]


async def test_a_page_with_no_list_still_follows_the_cursor():
    c = FakeClient(pages=[{"cursor": "next"}, {"market_positions": [_pos("A")]}])
    b = await _brief(c)
    assert [p["ticker"] for p in b["positions"]] == ["A"]
    assert b["read_errors"] == ["positions: no list"]


@pytest.mark.parametrize("row", [None, "A", 7, ["A"]])
async def test_a_position_row_that_is_no_dict_is_reported(row):
    b = await _brief(FakeClient(pages=[{"market_positions": [row, _pos("A")]}]))
    assert [p["ticker"] for p in b["positions"]] == ["A"]
    assert b["read_errors"] == ["positions: bad row"]


@pytest.mark.parametrize("fp", [True, False])
async def test_a_boolean_position_fp_is_reported(fp):
    b = await _brief(FakeClient(pages=[{"market_positions": [_pos("BAD", fp)]}]))
    assert b["positions"] == [] and b["read_errors"] == ["position BAD"]


class OrdersReplyClient(FakeClient):
    def __init__(self, reply, **kw):
        super().__init__(**kw)
        self.reply = reply

    async def get_orders(self, status=None):
        return self.reply


@pytest.mark.parametrize("reply", [
    {},
    {"status_code": 200, "empty_body": True},
    {"orders": None},
    {"orders": "A", "cursor": "more"},
    None,
])
async def test_a_resting_reply_with_no_list_is_reported(reply):
    b = await _brief(OrdersReplyClient(reply, pages=[{"market_positions": []}]))
    assert b["resting_orders"] == [] and b["read_errors"] == ["resting_orders"]


async def test_a_resting_row_that_is_no_dict_is_reported():
    b = await _brief(OrdersReplyClient({"orders": [None, {"ticker": "A"}]}, pages=[{"market_positions": []}]))
    assert [o["ticker"] for o in b["resting_orders"]] == ["A"]
    assert b["read_errors"] == ["resting_orders: bad row"]


class CountingClient(FakeClient):
    def __init__(self, bals, **kw):
        super().__init__(**kw)
        self.bals = list(bals)

    async def get_balance(self):
        return self.bals.pop(0)


async def test_the_governor_gets_the_same_balance_reply_as_the_brief():
    gov = FakeGovernor()
    c = CountingClient([{"balance": 1000, "portfolio_value": 250}], pages=[{"market_positions": []}])
    b = await account_brief(c, governor=gov)
    assert gov.equity == 1250 and b["equity"] == 12.5 and c.bals == []


async def test_an_empty_balance_reply_cannot_slip_past_with_a_real_governor(tmp_path):
    from src.risk.risk_governor import RiskGovernor

    state = tmp_path / "state.json"
    gov = RiskGovernor(state_path=str(state), halt_flag_path=str(tmp_path / "halt"))
    await gov.check(current_equity_cents=50000)
    before = state.read_text()
    # A second reply is here only to show it never gets read.
    c = CountingClient(
        [{"status_code": 200, "empty_body": True}, {"balance": 50000, "portfolio_value": 0}],
        pages=[{"market_positions": []}],
    )
    b = await account_brief(c, governor=gov)
    assert b["read_errors"] == ["balance"] and len(c.bals) == 1
    assert b["governor"]["halted"] is True
    assert b["governor"]["reasons"][0].startswith("EQUITY_UNREADABLE")
    assert state.read_text() == before


async def test_a_partial_balance_reaches_the_governor_as_unreadable():
    gov = FakeGovernor()
    c = CountingClient([{"balance": 1000}], pages=[{"market_positions": []}])
    b = await account_brief(c, governor=gov)
    assert gov.equity == 0 and b["read_errors"] == ["balance"]
