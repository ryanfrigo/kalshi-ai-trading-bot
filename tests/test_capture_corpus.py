"""Tests for the corpus snapshot flattener — the price-history capture that
will feed an honest backtest.

Pure-function tests with in-memory fixtures (no network, no live Kalshi, no IO).
``snapshot_rows`` is the only pure logic in the capture script; the IO (fetch,
gzip write, idempotency) lives in ``main`` and is exercised operationally.

Covers: one row per market with the fixed schema; the passed-in timestamp is
stamped verbatim (the script never reads the clock in the pure path);
category inherited from the parent event; markets without a ticker skipped;
missing price/volume fields degrade to 0.0 via ``_f`` rather than raising;
empty input yields no rows.
"""
from scripts.capture_corpus import snapshot_rows

TS = "2026-07-04T12:00:00+00:00"


def _event(cat="Politics", markets=None):
    return {"category": cat, "markets": markets or []}


def _market(ticker="KXFOO-27", **over):
    m = {
        "ticker": ticker,
        "yes_ask_dollars": "0.12",
        "yes_bid_dollars": "0.10",
        "no_ask_dollars": "0.90",
        "no_bid_dollars": "0.88",
        "last_price_dollars": "0.11",
        "volume_24h_fp": "1500.00",
        "volume_fp": "9000.00",
        "open_interest_fp": "300.00",
        "close_time": "2026-12-31T00:00:00Z",
    }
    m.update(over)
    return m


def test_one_row_per_market_with_full_schema():
    rows = snapshot_rows([_event(markets=[_market()])], TS)
    assert len(rows) == 1
    r = rows[0]
    assert r == {
        "ts": TS, "ticker": "KXFOO-27", "cat": "Politics",
        "yes_ask": 0.12, "yes_bid": 0.10, "no_ask": 0.90, "no_bid": 0.88,
        "last": 0.11, "vol24": 1500.0, "vol": 9000.0, "oi": 300.0,
        "close": "2026-12-31T00:00:00Z",
    }


def test_timestamp_is_stamped_verbatim():
    # The pure path must never read the clock — the caller owns the timestamp.
    rows = snapshot_rows([_event(markets=[_market()])], "STAMP")
    assert rows[0]["ts"] == "STAMP"


def test_category_inherited_from_parent_event():
    rows = snapshot_rows([_event(cat="Crypto", markets=[_market()])], TS)
    assert rows[0]["cat"] == "Crypto"


def test_market_without_ticker_skipped():
    rows = snapshot_rows([_event(markets=[_market(ticker=""), _market("KXBAR-27")])], TS)
    assert [r["ticker"] for r in rows] == ["KXBAR-27"]


def test_missing_price_fields_degrade_to_zero():
    bare = {"ticker": "KXBARE-27"}  # no price/volume fields at all
    rows = snapshot_rows([_event(markets=[bare])], TS)
    r = rows[0]
    assert r["yes_ask"] == 0.0 and r["no_ask"] == 0.0 and r["oi"] == 0.0
    assert r["close"] == ""


def test_multiple_events_flattened():
    events = [
        _event(cat="A", markets=[_market("KXA1"), _market("KXA2")]),
        _event(cat="B", markets=[_market("KXB1")]),
    ]
    rows = snapshot_rows(events, TS)
    assert [r["ticker"] for r in rows] == ["KXA1", "KXA2", "KXB1"]
    assert [r["cat"] for r in rows] == ["A", "A", "B"]


def test_empty_input_yields_no_rows():
    assert snapshot_rows([], TS) == []
    assert snapshot_rows([_event(markets=[])], TS) == []
