"""`cli.py fills`: a record is voided only on real evidence that the order never filled.

Fakes only. The Kalshi fake has read methods only, so an order attempt would raise.
"""
import argparse
import json

import pytest

import cli
import src.clients.kalshi_client as kc
from src.agent.journal import load_journal, write_journal


class FakeKalshi:
    """Pages get_orders with a cursor, the way the API does."""

    def __init__(self, resting=(), fills=None, orders_reply=None, fills_reply=None, fills_error=None):
        self.resting = list(resting)
        self.fills = fills or {}
        self.orders_reply = orders_reply
        self.fills_reply = fills_reply
        self.fills_error = fills_error
        self.fills_asked = []
        self.closed = 0

    async def get_orders(self, status=None, limit=None, cursor=None, ticker=None):
        assert status == "resting"
        if self.orders_reply is not None:
            return self.orders_reply
        size = min(limit or 100, 100)
        start = int(cursor or 0)
        rows = [{"order_id": oid} for oid in self.resting[start:start + size]]
        return {"orders": rows, "cursor": str(start + size) if start + size < len(self.resting) else ""}

    async def get_fills(self, order_id=None, limit=100, ticker=None):
        self.fills_asked.append(order_id)
        if self.fills_error:
            raise self.fills_error
        if self.fills_reply is not None:
            return self.fills_reply
        return {"fills": [{"order_id": order_id, "count_fp": c} for c in self.fills.get(order_id, [])]}

    async def close(self):
        self.closed += 1


def _record(oid, **extra):
    return {"ticker": f"T-{oid}", "side": "no", "count": 10, "price": 0.5, "order_id": oid, **extra}


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def _run(records, client, dry=False):
        write_journal(records)
        monkeypatch.setattr(kc, "KalshiClient", lambda: client)
        cli.cmd_fills(argparse.Namespace(dry=dry))
        return json.loads(capsys.readouterr().out)

    return _run


def test_a_resting_order_past_the_first_page_is_not_voided(run):
    client = FakeKalshi(resting=[f"o{i}" for i in range(150)])
    out = run([_record("o149")], client)
    assert out["changes"] == [] and out["written"] is False
    assert not load_journal()[0].get("voided")
    assert client.fills_asked == []


def test_an_order_gone_from_every_page_with_no_fills_is_voided(run):
    client = FakeKalshi(resting=[f"o{i}" for i in range(150)])
    out = run([_record("gone")], client)
    assert out["changes"][0]["change"] == "voided (0 fills)" and out["written"] is True
    assert load_journal()[0]["voided"] is True


def test_a_partly_filled_order_shrinks_to_its_fills(run):
    out = run([_record("p")], FakeKalshi(fills={"p": ["3.00", "1.00"]}))
    assert out["written"] is True
    rec = load_journal()[0]
    assert rec["count"] == 4 and rec["original_count"] == 10 and not rec.get("voided")


def test_a_fully_filled_order_is_untouched(run):
    out = run([_record("f")], FakeKalshi(fills={"f": ["10.00"]}))
    assert out["changes"] == [] and out["checked"] == 1
    assert load_journal() == [_record("f")]


@pytest.mark.parametrize("reply", [{}, {"orders": "nope"}, ["orders"]])
def test_a_bad_resting_reply_raises_and_writes_nothing(run, reply):
    client = FakeKalshi(orders_reply=reply)
    with pytest.raises(RuntimeError, match="malformed orders page"):
        run([_record("o1")], client)
    assert load_journal() == [_record("o1")]
    assert client.fills_asked == [] and client.closed == 1


@pytest.mark.parametrize("reply", [
    {},
    {"fills": None},
    {"fills": "nope"},
    {"fills": [{"order_id": "o1", "count_fp": "1.00"}] * 200, "cursor": "next"},
])
def test_no_complete_fills_list_is_no_evidence(run, reply):
    out = run([_record("o1")], FakeKalshi(fills_reply=reply))
    assert out["changes"] == [] and out["written"] is False
    assert out["skipped_no_evidence"] == [{"ticker": "T-o1", "order_id": "o1"}]
    assert load_journal() == [_record("o1")]


def test_a_cursor_on_a_short_fills_page_is_still_evidence(run):
    reply = {"fills": [], "cursor": "stale"}
    out = run([_record("o1")], FakeKalshi(fills_reply=reply))
    assert out["changes"][0]["change"] == "voided (0 fills)"


def test_a_failed_fills_read_is_no_evidence(run):
    out = run([_record("o1")], FakeKalshi(fills_error=RuntimeError("boom")))
    assert out["skipped_no_evidence"] == [{"ticker": "T-o1", "order_id": "o1"}]
    assert not load_journal()[0].get("voided")


def test_dry_reports_the_void_and_writes_nothing(run):
    out = run([_record("gone")], FakeKalshi(), dry=True)
    assert out["dry"] is True and out["written"] is False
    assert out["changes"][0]["change"] == "voided (0 fills)"
    assert load_journal() == [_record("gone")]


def test_settled_voided_and_id_less_records_are_not_asked_about(run):
    records = [
        _record("s", outcome={"won": True, "pnl": 1.0}),
        _record("v", voided=True),
        {"ticker": "L", "side": "no", "count": 1, "price": 0.5},
    ]
    client = FakeKalshi()
    out = run(records, client)
    assert out["checked"] == 0 and out["changes"] == []
    assert client.fills_asked == [] and client.closed == 1
