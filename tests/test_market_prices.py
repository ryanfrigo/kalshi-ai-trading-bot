"""Orderbook parsing must work for both Kalshi shapes.

`cli.py close-all` read the v2-named key incorrectly and assumed integer cents, so it concluded
every position had no bid and refused to flatten anything — while reporting the book as empty.
"""
from src.utils.market_prices import orderbook_levels


def test_v2_shape_returns_dollar_floats():
    resp = {"orderbook_fp": {"yes_dollars": [["0.9500", "100.00"], ["0.9400", "50.00"]],
                             "no_dollars": [["0.0500", "10.00"]]}}
    assert orderbook_levels(resp, "yes") == [(0.95, 100.0), (0.94, 50.0)]
    assert orderbook_levels(resp, "no") == [(0.05, 10.0)]


def test_legacy_shape_converts_cents():
    resp = {"orderbook": {"yes": [[95, 100], [94, 50]], "no": [[5, 10]]}}
    assert orderbook_levels(resp, "yes") == [(0.95, 100.0), (0.94, 50.0)]
    assert orderbook_levels(resp, "no") == [(0.05, 10.0)]


def test_missing_side_is_empty_not_an_error():
    assert orderbook_levels({}, "yes") == []
    assert orderbook_levels({"orderbook_fp": {"yes_dollars": []}}, "yes") == []


def test_malformed_levels_are_skipped():
    resp = {"orderbook_fp": {"yes_dollars": [["0.95"], ["0.94", "10"]]}}
    assert orderbook_levels(resp, "yes") == [(0.94, 10.0)]
