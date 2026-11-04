"""End-to-end journal lifecycle: pending order -> fills -> settlement -> summary.

One fixture journal drives the real bot's four stages in order, persisted to
disk between each: resolve_pending -> reconcile_fills -> reconcile_outcomes ->
summarize_journal. No network: the Kalshi client is a local fake.
"""
import asyncio
import json

import pytest

from src.agent.journal import (
    load_journal,
    make_decision_record,
    mark_outcome_unknown,
    reconcile_fills,
    summarize_journal,
    write_journal,
)
from src.agent.learnings import reconcile_outcomes
from src.agent.pending import resolve_pending


class FakeKalshi:
    def __init__(self, orders):
        self.orders = orders

    async def get_orders(self, ticker=None, limit=None, cursor=None):
        return {"orders": [o for o in self.orders if o.get("ticker") == ticker], "cursor": None}

    async def get_historical_orders(self, ticker, limit=None, cursor=None):
        return {"orders": [], "cursor": None}


ORDERS = [
    {"ticker": "KXA", "client_order_id": "c1", "order_id": "o1", "status": "executed",
     "fill_count_fp": "3.00", "taker_fill_cost_dollars": "0.9000",
     "maker_fill_cost_dollars": "0.2700"},
]

FILLS = [
    {"order_id": "o1", "count_fp": "3.00"},
    {"order_id": "o2", "count_fp": "1.50"},
    {"order_id": "o2", "count_fp": "1.00"},
    {"order_id": "o5", "count_fp": "1.00"},
    {"order_id": "o6", "count_fp": "1.00"},
]

RESTING_ORDER_IDS = {"o6"}

SETTLEMENTS = [
    {"ticker": "KXA", "market_result": "yes", "yes_count_fp": "3.00", "no_count_fp": "0",
     "yes_total_cost_dollars": "1.17", "no_total_cost_dollars": "0", "fee_cost": "0.03",
     "revenue": 300, "settled_time": "2026-11-02T00:00:00Z"},
    {"ticker": "KXB", "market_result": "no", "yes_count_fp": "1.00", "no_count_fp": "0",
     "yes_total_cost_dollars": "0.60", "no_total_cost_dollars": "0", "fee_cost": "0.01",
     "revenue": 0, "settled_time": "2026-11-02T00:00:00Z"},
    {"ticker": "KXC", "market_result": "yes", "yes_count_fp": "2.00", "no_count_fp": "0",
     "yes_total_cost_dollars": "1.00", "no_total_cost_dollars": "0", "fee_cost": "0",
     "revenue": 200, "settled_time": "2026-11-02T00:00:00Z"},
    {"ticker": "KXD", "market_result": "yes", "yes_count_fp": "1.00", "no_count_fp": "0",
     "yes_total_cost_dollars": "0.10", "no_total_cost_dollars": "0", "fee_cost": "0",
     "revenue": 100, "settled_time": "2026-11-02T00:00:00Z"},
    {"ticker": "KXE", "market_result": "no", "yes_count_fp": "1.00", "no_count_fp": "0",
     "yes_total_cost_dollars": "0.20", "no_total_cost_dollars": "0", "fee_cost": "0.01",
     "revenue": 0, "settled_time": "2026-11-02T00:00:00Z"},
]

TS = "2026-11-01T00:00:00+00:00"


def _build_records():
    a = make_decision_record("KXA", "yes", 5, 0.40, ts=TS)
    mark_outcome_unknown(a, "c1")
    b = make_decision_record("KXB", "no", 4, 0.30, order_id="o2", ts=TS)
    c = make_decision_record("KXC", "yes", 2, 0.50, order_id="o3", ts=TS)
    d = make_decision_record("KXD", "yes", 1, 0.10, ts=TS)
    mark_outcome_unknown(d, "c4")
    e = make_decision_record("KXE", "yes", 1, 0.20, order_id="o5", ts=TS)
    f = make_decision_record("KXF", "yes", 3, 0.25, order_id="o6", ts=TS)
    return [a, b, c, d, e, f]


def _journal_path(tmp_path):
    return str(tmp_path / "decision_journal.jsonl")


def _run(tmp_path):
    """Run all four stages in order, persisting between each."""
    path = _journal_path(tmp_path)
    write_journal(_build_records(), path)

    records = load_journal(path)
    records, _ = asyncio.run(resolve_pending(records, FakeKalshi(ORDERS)))
    write_journal(records, path)

    records = load_journal(path)
    records, _ = reconcile_fills(records, FILLS, RESTING_ORDER_IDS)
    write_journal(records, path)

    records = load_journal(path)
    records, _ = reconcile_outcomes(records, SETTLEMENTS)
    write_journal(records, path)

    records = load_journal(path)
    return records, summarize_journal(records)


def _by_ticker(records):
    return {r["ticker"]: r for r in records}


def test_a_pending_resolved_from_order_and_settled(tmp_path):
    a = _by_ticker(_run(tmp_path)[0])["KXA"]
    assert "status" not in a
    assert a["order_id"] == "o1"
    assert a["count"] == 3
    assert a["original_count"] == 5
    assert a["price"] == pytest.approx(0.39)
    assert a["outcome"]["won"] is True
    assert a["outcome"]["pnl"] == pytest.approx(1.80)


def test_b_partial_fill_kept_fractional_and_reconstructed_pnl(tmp_path):
    b = _by_ticker(_run(tmp_path)[0])["KXB"]
    assert b["count"] == 2.5
    assert isinstance(b["count"], float)
    assert b["original_count"] == 4
    assert b["outcome"]["won"] is True
    assert b["outcome"]["pnl"] == pytest.approx(1.75)


def test_c_zero_fill_voided_despite_settlement(tmp_path):
    c = _by_ticker(_run(tmp_path)[0])["KXC"]
    assert c["voided"] is True
    assert c["outcome"] is None


def test_d_order_never_reached_book_voided_despite_settlement(tmp_path):
    d = _by_ticker(_run(tmp_path)[0])["KXD"]
    assert d["voided"] is True
    assert d["voided_reason"] == "order never reached the book"
    assert d["outcome"] is None


def test_e_fully_filled_settled_loss(tmp_path):
    e = _by_ticker(_run(tmp_path)[0])["KXE"]
    assert e["count"] == 1
    assert "original_count" not in e
    assert e["outcome"]["won"] is False
    assert e["outcome"]["pnl"] == pytest.approx(-0.21)


def test_f_resting_order_untouched_and_unsettled(tmp_path):
    f = _by_ticker(_run(tmp_path)[0])["KXF"]
    assert f["count"] == 3
    assert not f.get("voided")
    assert "original_count" not in f
    assert f["outcome"] is None


def test_summary_counts_only_real_settlements(tmp_path):
    _, summary = _run(tmp_path)
    assert summary["total"] == 6
    assert summary["settled"] == 3
    assert summary["wins"] == 2
    assert summary["pnl"] == pytest.approx(3.34)
    assert summary["win_rate"] == pytest.approx(2 / 3)


def test_reconcile_outcomes_is_idempotent(tmp_path):
    records, _ = _run(tmp_path)
    again, newly = reconcile_outcomes(records, SETTLEMENTS)
    assert newly == 0
    assert again == records


class RaisingKalshi:
    async def get_orders(self, ticker=None, limit=None, cursor=None):
        raise RuntimeError("lookup failed")

    async def get_historical_orders(self, ticker, limit=None, cursor=None):
        return {"orders": [], "cursor": None}


def test_failed_lookup_stays_pending_and_gets_no_outcome():
    rec = make_decision_record("KXP", "yes", 1, 0.20, ts=TS)
    mark_outcome_unknown(rec, "cx")
    records, _ = asyncio.run(resolve_pending([rec], RaisingKalshi()))
    assert records[0]["status"] == "pending"
    settled, _ = reconcile_outcomes(records, [
        {"ticker": "KXP", "market_result": "yes", "yes_count_fp": "1.00",
         "no_count_fp": "0", "yes_total_cost_dollars": "0.20",
         "no_total_cost_dollars": "0", "fee_cost": "0", "revenue": 100,
         "settled_time": "2026-11-02T00:00:00Z"},
    ])
    assert settled[0]["outcome"] is None


def test_journal_file_has_one_json_line_per_record(tmp_path):
    _run(tmp_path)
    lines = [line for line in open(_journal_path(tmp_path)).read().splitlines() if line.strip()]
    assert len(lines) == 6
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_load_journal_skips_a_corrupt_line_and_keeps_the_rest(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text('{"a": 1}\n{"truncated": \n\n{"b": 2}\n')
    assert load_journal(str(p)) == [{"a": 1}, {"b": 2}]


def test_a_decision_record_carries_method_only_when_given():
    assert make_decision_record("KXA", "no", 1, 0.95, method="workflow")["method"] == "workflow"
    assert "method" not in make_decision_record("KXA", "no", 1, 0.95)
