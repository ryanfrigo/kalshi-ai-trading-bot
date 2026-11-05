"""execute._sellable_contracts: held on a side minus our resting orders against it.

It decides how many contracts an exit may sell. Any unreadable input returns
None, so the caller sends nothing. The client is a MagicMock: no network, no keys.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.jobs.execute import _sellable_contracts

_UNSET = object()


def client(held=10.0, body=_UNSET, orders_exc=None):
    c = MagicMock()
    c._held_on_side = AsyncMock(return_value=held)
    if orders_exc is not None:
        c.get_orders = AsyncMock(side_effect=orders_exc)
    else:
        c.get_orders = AsyncMock(return_value={"orders": []} if body is _UNSET else body)
    return c


def sellable(c, side="yes", ticker="MKT-1"):
    return asyncio.run(_sellable_contracts(c, ticker, side))


def order(**kw):
    o = {"ticker": "MKT-1", "order_id": "o1", "remaining_count_fp": "3.00"}
    o.update(kw)
    return o


def test_no_resting_orders_sells_what_is_held():
    c = client(held=10.0)
    assert sellable(c) == 10.0
    c._held_on_side.assert_awaited_once_with("MKT-1", "yes")
    c.get_orders.assert_awaited_once_with(ticker="MKT-1", status="resting")


@pytest.mark.parametrize("held", [None, True, "10", float("nan"), float("inf")])
def test_unreadable_holding_blocks_the_exit(held):
    c = client(held=held)
    assert sellable(c) is None
    c.get_orders.assert_not_awaited()


def test_orders_read_failure_blocks_the_exit():
    assert sellable(client(orders_exc=RuntimeError("503"))) is None


@pytest.mark.parametrize("body", [None, [], {}, {"orders": None}, {"orders": {}}])
def test_unreadable_orders_response_blocks_the_exit(body):
    assert sellable(client(body=body)) is None


def test_a_full_page_with_a_cursor_blocks_the_exit():
    body = {"orders": [order(ticker="OTHER")] * 100, "cursor": "next"}
    assert sellable(client(body=body)) is None


def test_a_short_page_with_a_cursor_is_read():
    body = {"orders": [order(ticker="OTHER")] * 99, "cursor": "next"}
    assert sellable(client(body=body)) == 10.0


@pytest.mark.parametrize("fields", [
    {"outcome_side": "no"},
    {"book_side": "ask"},
    {"side": "yes", "action": "sell"},
    {"side": "no", "action": "buy"},
])
def test_resting_orders_against_a_yes_holding_are_subtracted(fields):
    assert sellable(client(held=10.0, body={"orders": [order(**fields)]})) == 7.0


@pytest.mark.parametrize("fields", [
    {"outcome_side": "yes"},
    {"book_side": "bid"},
    {"side": "yes", "action": "buy"},
    {"side": "no", "action": "sell"},
])
def test_orders_on_the_held_side_are_not_subtracted(fields):
    assert sellable(client(held=10.0, body={"orders": [order(**fields)]})) == 10.0


def test_a_no_holding_subtracts_orders_positioned_for_yes():
    body = {"orders": [order(outcome_side="yes"),
                       order(outcome_side="no", remaining_count_fp="5.00")]}
    assert sellable(client(held=10.0, body=body), side="NO") == 7.0


def test_orders_on_other_tickers_and_non_dict_entries_are_ignored():
    body = {"orders": [order(ticker="OTHER", outcome_side="no"), "garbage", None]}
    assert sellable(client(body=body)) == 10.0


def test_an_order_with_no_readable_outcome_blocks_the_exit():
    body = {"orders": [order(outcome_side="no"), order(side="maybe", action="sell")]}
    assert sellable(client(body=body)) is None


@pytest.mark.parametrize("remaining, expected", [
    ({"remaining_count_fp": "2.50"}, 7.5),
    ({"remaining_count_fp": "junk", "remaining_count": 4}, 6.0),
    ({"remaining_count_fp": None, "remaining_count": 1}, 9.0),
])
def test_remaining_count_is_read_with_fallback(remaining, expected):
    o = {"ticker": "MKT-1", "order_id": "o1", "outcome_side": "no", **remaining}
    assert sellable(client(body={"orders": [o]})) == expected


@pytest.mark.parametrize("remaining", [
    {}, {"remaining_count_fp": "nan"}, {"remaining_count_fp": "-1"}, {"remaining_count": "x"},
])
def test_unreadable_or_negative_remaining_blocks_the_exit(remaining):
    o = {"ticker": "MKT-1", "order_id": "o1", "outcome_side": "no", **remaining}
    assert sellable(client(body={"orders": [o]})) is None


def test_resting_orders_can_leave_nothing_sellable():
    body = {"orders": [order(outcome_side="no", remaining_count_fp="6"),
                       order(book_side="ask", remaining_count_fp="4")]}
    assert sellable(client(held=10.0, body=body)) == 0.0


def test_an_uppercase_yes_side_from_the_db_still_nets_no_orders():
    # Position rows store side as "YES"/"NO"; the guard must lower-case it.
    body = {"orders": [order(outcome_side="no"),
                       order(outcome_side="yes", remaining_count_fp="5.00")]}
    assert sellable(client(held=10.0, body=body), side="YES") == 7.0


def test_a_full_page_without_a_cursor_is_the_last_page_and_is_read():
    body = {"orders": [order(ticker="OTHER")] * 100, "cursor": ""}
    assert sellable(client(body=body)) == 10.0
