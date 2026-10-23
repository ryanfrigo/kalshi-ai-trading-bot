"""The live-book capture must never write a stale price as if it were executable.

``capture_corpus.py`` stores events-API ``*_dollars`` prices, which lag the live book.
``capture_books.py`` exists to capture the real thing, so its row shaping drops any
candidate that failed live enrichment rather than falling back to the snapshot price.
"""
import gzip
import json

import scripts.capture_books as cb


def _cand(ticker, with_book=True):
    r = {"ticker": ticker, "cat": "Sports", "q": "Q", "snap_yes_ask": 0.08,
         "dtc": 3.0, "vol24": 1000, "oi": 5000}
    if with_book:
        r.update({"yes_bid": 0.07, "yes_ask": 0.09, "no_bid": 0.91, "no_ask": 0.93})
    return r


def test_rows_without_a_live_book_are_dropped():
    rows = cb.book_rows([_cand("A"), _cand("B", with_book=False)], "2026-01-01T00:00:00+00:00")
    assert [r["ticker"] for r in rows] == ["A"]
    assert rows[0]["yes_ask"] == 0.09 and rows[0]["no_ask"] == 0.93
    assert rows[0]["ts"] == "2026-01-01T00:00:00+00:00"


def test_append_day_writes_readable_gzipped_jsonl(tmp_path):
    path = tmp_path / "books_20260101.jsonl.gz"
    n = cb.append_day(cb.book_rows([_cand("A"), _cand("B")], "t"), path)
    assert n == 2
    with gzip.open(path, "rt", encoding="utf-8") as f:
        got = [json.loads(line) for line in f if line.strip()]
    assert [r["ticker"] for r in got] == ["A", "B"]
