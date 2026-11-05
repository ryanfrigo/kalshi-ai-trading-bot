"""`cli.py fills`: a record is voided or shrunk only when the order itself proves it ended.

Fakes only. The Kalshi fake has read methods only, so an order attempt would raise.
"""
import argparse
import json

import pytest

import cli
import src.clients.kalshi_client as kc
from src.agent.journal import load_journal, write_journal


def _order(oid="o1", status="canceled", filled="0.00"):
    return {"order_id": oid, "status": status, "fill_count_fp": filled}


class FakeKalshi:
    """Pages get_orders with a cursor, the way the API does."""

    def __init__(self, resting=(), orders=None, orders_reply=None, order_reply=None, order_error=None):
        self.resting = list(resting)
        self.orders = orders or {}
        self.orders_reply = orders_reply
        self.order_reply = order_reply
        self.order_error = order_error
        self.order_asked = []
        self.closed = 0

    async def get_orders(self, status=None, limit=None, cursor=None, ticker=None):
        assert status == "resting"
        if self.orders_reply is not None:
            return self.orders_reply
        size = min(limit or 100, 100)
        start = int(cursor or 0)
        rows = [{"order_id": oid} for oid in self.resting[start:start + size]]
        return {"orders": rows, "cursor": str(start + size) if start + size < len(self.resting) else ""}

    async def get_order(self, order_id):
        self.order_asked.append(order_id)
        if self.order_error:
            raise self.order_error
        if self.order_reply is not None:
            return self.order_reply
        return {"order": self.orders[order_id]}

    async def close(self):
        self.closed += 1


def _record(oid="o1", **extra):
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


def test_a_resting_order_past_the_first_page_is_not_asked_about(run):
    client = FakeKalshi(resting=[f"o{i}" for i in range(150)])
    out = run([_record("o149")], client)
    assert out["changes"] == [] and out["written"] is False
    assert load_journal() == [_record("o149")]
    assert client.order_asked == []


@pytest.mark.parametrize("status", ["canceled", "expired"])
def test_an_order_that_ended_unfilled_is_voided(run, status):
    out = run([_record()], FakeKalshi(orders={"o1": _order(status=status)}))
    assert out["changes"][0]["change"] == "voided (0 fills)" and out["written"] is True
    assert load_journal()[0]["voided"] is True


@pytest.mark.parametrize("reply", [{"orders": None}, {"orders": []}])
def test_an_empty_resting_list_cannot_void_a_live_order(run, reply):
    client = FakeKalshi(orders_reply=reply, orders={"o1": _order(status="resting")})
    out = run([_record()], client)
    assert out["changes"] == [] and out["written"] is False
    assert out["skipped_no_evidence"] == [{"ticker": "T-o1", "order_id": "o1"}]
    assert load_journal() == [_record()]


def test_an_empty_resting_list_still_lets_an_ended_order_void(run):
    client = FakeKalshi(orders_reply={"orders": None}, orders={"o1": _order()})
    assert run([_record()], client)["changes"][0]["change"] == "voided (0 fills)"


def test_a_canceled_order_with_some_fills_shrinks_to_them(run):
    out = run([_record()], FakeKalshi(orders={"o1": _order(filled="4.00")}))
    assert out["written"] is True
    rec = load_journal()[0]
    assert rec["count"] == 4 and rec["original_count"] == 10 and not rec.get("voided")


def test_a_fully_filled_order_is_untouched(run):
    out = run([_record()], FakeKalshi(orders={"o1": _order(status="executed", filled="10.00")}))
    assert out["changes"] == [] and out["checked"] == 1
    assert load_journal() == [_record()]


@pytest.mark.parametrize("reply", [{}, {"orders": "nope"}, ["orders"]])
def test_a_bad_resting_reply_raises_and_writes_nothing(run, reply):
    client = FakeKalshi(orders_reply=reply)
    with pytest.raises(RuntimeError, match="malformed orders page"):
        run([_record()], client)
    assert load_journal() == [_record()]
    assert client.order_asked == [] and client.closed == 1


@pytest.mark.parametrize("reply", [
    {},
    {"order": None},
    {"order": "nope"},
    {"order": {"status": "canceled", "order_id": "o1"}},
    {"order": {"fill_count_fp": "0.00", "order_id": "o1"}},
    {"order": {"status": "canceled", "fill_count_fp": "0.00"}},
    {"order": _order(oid="other")},
    {"order": _order(status="resting")},
    {"order": _order(status="pending")},
    {"order": _order(status="Canceled")},
    {"order": _order(status="executed")},
    {"order": _order(filled="nan")},
    {"order": _order(filled="-1.00")},
    {"order": _order(filled="lots")},
])
def test_a_reply_that_proves_nothing_leaves_the_record_alone(run, reply):
    out = run([_record()], FakeKalshi(order_reply=reply))
    assert out["changes"] == [] and out["written"] is False
    assert out["skipped_no_evidence"] == [{"ticker": "T-o1", "order_id": "o1"}]
    assert load_journal() == [_record()]


def test_a_failed_order_read_is_no_evidence(run):
    out = run([_record()], FakeKalshi(order_error=RuntimeError("404")))
    assert out["skipped_no_evidence"] == [{"ticker": "T-o1", "order_id": "o1"}]
    assert not load_journal()[0].get("voided")


def test_one_bad_order_does_not_stop_the_next(run):
    class OneFails(FakeKalshi):
        async def get_order(self, order_id):
            if order_id == "bad":
                raise RuntimeError("boom")
            return await super().get_order(order_id)

    out = run([_record("bad"), _record("o1")], OneFails(orders={"o1": _order()}))
    assert [c["order_id"] for c in out["changes"]] == ["o1"]
    assert [s["order_id"] for s in out["skipped_no_evidence"]] == ["bad"]
    assert [bool(r.get("voided")) for r in load_journal()] == [False, True]


def test_dry_reports_the_void_and_writes_nothing(run):
    out = run([_record()], FakeKalshi(orders={"o1": _order()}), dry=True)
    assert out["dry"] is True and out["written"] is False
    assert out["changes"][0]["change"] == "voided (0 fills)"
    assert load_journal() == [_record()]


def test_settled_voided_and_id_less_records_are_not_asked_about(run):
    records = [
        _record("s", outcome={"won": True, "pnl": 1.0}),
        _record("v", voided=True),
        {"ticker": "L", "side": "no", "count": 1, "price": 0.5},
    ]
    client = FakeKalshi()
    out = run(records, client)
    assert out["checked"] == 0 and out["changes"] == []
    assert client.order_asked == [] and client.closed == 1
