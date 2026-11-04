"""account_brief over a fake client: paging, and a failed read is reported, never shown as nothing."""
import pytest

from src.agent.toolbelt import account_brief


class FakeDecision:
    def to_dict(self):
        return {"allowed": True}


class FakeGovernor:
    async def check(self):
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
