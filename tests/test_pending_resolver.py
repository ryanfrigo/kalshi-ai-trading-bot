"""resolve_pending: fake read-only client, one test per branch (each fails if its branch is wrong)."""
import asyncio

from src.agent.journal import is_pending
from src.agent.learnings import reconcile_outcomes
from src.agent.pending import resolve_pending


def _rec(coid="c-1"):
    return {"ts": "2026-10-01T00:00:00+00:00", "ticker": "KXP-1", "side": "no", "action": "buy",
            "count": 10, "price": 0.90, "order_id": None, "outcome": None, "status": "pending",
            "order_outcome": "outcome_unknown", "client_order_id": coid}


class FakeClient:
    """Read-only: only get_orders / get_historical_orders exist, so any write would AttributeError."""
    def __init__(self, live=(), archived=(), fail=False, live_cursor=None):
        self.live, self.archived, self.fail, self.live_cursor = list(live), list(archived), fail, live_cursor

    async def get_orders(self, ticker=None, limit=None, cursor=None, status=None):
        if self.fail:
            raise RuntimeError("503")
        return {"orders": self.live, "cursor": self.live_cursor}

    async def get_historical_orders(self, ticker, limit=None, cursor=None):
        if self.fail:
            raise RuntimeError("503")
        return {"orders": self.archived, "cursor": None}


def _order(fill, status, coid="c-1", price="0.88"):
    return {"order_id": "oid-9", "client_order_id": coid, "status": status,
            "fill_count_fp": fill, "no_price_dollars": price,
            "taker_fill_cost_dollars": "0.0000", "maker_fill_cost_dollars": "0.0000"}


def _run(recs, client):
    return asyncio.run(resolve_pending(recs, client))


def test_filled_records_real_order_and_becomes_open_record():
    (r,), ch = _run([_rec()], FakeClient(live=[_order("7.00", "canceled")]))
    assert r["order_id"] == "oid-9" and r["count"] == 7 and r["price"] == 0.88
    assert r["original_count"] == 10 and not is_pending(r) and not r.get("voided")
    assert "order_outcome" not in r and len(ch) == 1
    # existing settlement logic now settles it
    out, newly = reconcile_outcomes([r], [{"ticker": "KXP-1", "event": "KXP-1", "held_side": "no",
        "count": 7, "won": True, "result": "no", "cost": 6.2, "revenue": 7.0, "fee": 0.0,
        "pnl": 0.8, "settled_time": "2026-10-02T00:00:00+00:00"}])
    assert newly == 1


def test_filled_found_in_archive():
    (r,), _ = _run([_rec()], FakeClient(archived=[_order("10.00", "executed")]))
    assert r["order_id"] == "oid-9" and r["count"] == 10 and "original_count" not in r


def test_zero_fill_canceled_is_voided():
    (r,), ch = _run([_rec()], FakeClient(live=[_order("0.00", "canceled")]))
    assert r["voided"] is True and r["order_id"] is None and ch[0]["change"] == "voided (canceled)"


def test_zero_fill_expired_is_voided():
    (r,), _ = _run([_rec()], FakeClient(archived=[_order("0.00", "expired")]))
    assert r["voided"] is True


def test_zero_fill_resting_stays_pending():
    (r,), ch = _run([_rec()], FakeClient(live=[_order("0.00", "resting")]))
    assert is_pending(r) and not r.get("voided") and ch == []


def test_absent_after_all_pages_is_voided():
    (r,), ch = _run([_rec()], FakeClient(live=[_order("5.00", "executed", coid="other")]))
    assert r["voided"] is True and ch[0]["change"] == "voided (absent)"


def test_read_failure_stays_pending():
    (r,), ch = _run([_rec()], FakeClient(fail=True))
    assert is_pending(r) and not r.get("voided") and ch == []


def test_page_cap_stays_pending():
    (r,), ch = _run([_rec()], FakeClient(live=[], live_cursor="more"))  # cursor never ends
    assert is_pending(r) and not r.get("voided") and ch == []


def test_unreadable_fill_stays_pending():
    bad = _order("0.00", "canceled"); del bad["fill_count_fp"]
    (r,), ch = _run([_rec()], FakeClient(live=[bad]))
    assert is_pending(r) and ch == []


def test_non_pending_and_input_untouched():
    normal = {"ticker": "KXC-1", "order_id": "o", "outcome": None, "count": 1, "price": 0.5}
    pend = _rec()
    snapshot = dict(pend)
    out, ch = _run([normal, pend], FakeClient(live=[_order("10.00", "executed")]))
    assert out[0] is normal and pend == snapshot and len(ch) == 1


def test_resting_partial_stays_pending():
    (r,), ch = _run([_rec()], FakeClient(live=[_order("4.00", "resting")]))
    assert is_pending(r) and r["order_id"] is None and ch == []


def test_resting_full_fill_resolves():
    (r,), _ = _run([_rec()], FakeClient(live=[_order("10.00", "resting")]))
    assert r["count"] == 10 and not is_pending(r)


def test_canceled_after_partial_resolves_with_partial_count():
    (r,), _ = _run([_rec()], FakeClient(live=[_order("4.00", "canceled")]))
    assert r["count"] == 4 and r["original_count"] == 10 and not is_pending(r)


def test_average_fill_price_from_fill_costs():
    o = _order("10.00", "executed", price="0.95")
    o.update(taker_fill_cost_dollars="3.00", maker_fill_cost_dollars="6.00")
    (r,), _ = _run([_rec()], FakeClient(live=[o]))
    assert abs(r["price"] - 0.90) < 1e-9  # (3+6)/10, not the 0.95 limit


def test_limit_price_fallback_when_costs_missing_or_not_finite():
    o = _order("10.00", "executed", price="0.95")
    (r,), _ = _run([_rec()], FakeClient(live=[o]))
    assert r["price"] == 0.95
    o.update(taker_fill_cost_dollars="nan", maker_fill_cost_dollars="1.0")
    (r,), _ = _run([_rec()], FakeClient(live=[o]))
    assert r["price"] == 0.95


def test_cli_fills_resolves_pending_and_persists(tmp_path, monkeypatch, capsys):
    import argparse
    import json

    import cli
    import src.clients.kalshi_client as kc
    from src.agent.journal import load_journal, write_journal

    monkeypatch.chdir(tmp_path)
    write_journal([_rec()])

    class CliClient(FakeClient):
        async def get_fills(self, **kw):
            return {"fills": []}

        async def close(self):
            pass

    live = _order("10.00", "executed")
    monkeypatch.setattr(kc, "KalshiClient", lambda: CliClient(live=[live]))
    cli.cmd_fills(argparse.Namespace(dry=False))
    (r,) = load_journal()
    assert r["order_id"] == "oid-9" and not is_pending(r)
    assert json.loads(capsys.readouterr().out)["written"] is True


def test_fractional_fill_keeps_the_float_count():
    (r,), _ = _run([_rec()], FakeClient(live=[_order("2.50", "canceled")]))
    assert r["count"] == 2.5 and r["original_count"] == 10
    (r,), _ = _run([_rec()], FakeClient(live=[_order("10.00", "executed")]))
    assert r["count"] == 10 and isinstance(r["count"], int)


def test_zero_fill_canceled_with_cost_fields_is_voided_not_a_crash():
    o = _order("0.00", "canceled")
    assert o["taker_fill_cost_dollars"] == "0.0000"  # as the real API sends it
    (r,), ch = _run([_rec()], FakeClient(live=[o]))
    assert r["voided"] is True and ch[0]["change"] == "voided (canceled)"


def test_zero_fill_resting_with_cost_fields_stays_pending_not_a_crash():
    (r,), ch = _run([_rec()], FakeClient(live=[_order("0.00", "resting")]))
    assert is_pending(r) and not r.get("voided") and ch == []


def test_missing_side_price_and_bad_costs_fall_back_to_record_price_exactly():
    o = _order("10.00", "executed")
    del o["no_price_dollars"]
    o.update(taker_fill_cost_dollars="oops", maker_fill_cost_dollars="1.0")
    (r,), _ = _run([_rec()], FakeClient(live=[o]))
    assert r["price"] == 0.90 and r["order_id"] == "oid-9"


def test_zero_costs_with_fill_fall_back_to_limit_price_not_zero():
    o = _order("10.00", "executed", price="0.88")  # costs default to "0.0000"
    (r,), _ = _run([_rec()], FakeClient(live=[o]))
    assert r["price"] == 0.88


def test_executed_below_intended_still_resolves():
    (r,), _ = _run([_rec()], FakeClient(live=[_order("4.00", "executed")]))
    assert r["count"] == 4 and r["original_count"] == 10 and not is_pending(r)


def test_avg_fill_price_with_zero_fill_returns_zero_not_a_crash():
    from src.agent.pending import _avg_fill_price
    assert _avg_fill_price(_order("0.00", "canceled"), 0.0) == 0.0


def test_a_negative_or_infinite_fill_stays_pending():
    for fill in ("-1.00", "inf", "nan"):
        rec = _rec()
        recs, changes = _run([rec], FakeClient(live=[_order(fill, "executed")]))
        assert recs == [rec] and changes == [], fill


def test_a_fill_with_no_order_id_stays_pending():
    order = _order("10.00", "executed")
    del order["order_id"]
    rec = _rec()
    assert _run([rec], FakeClient(live=[order])) == ([rec], [])


def test_a_fill_with_no_readable_price_stays_pending():
    order = _order("10.00", "executed", price=None)
    del order["taker_fill_cost_dollars"]
    rec = _rec()
    del rec["price"]
    assert _run([rec], FakeClient(live=[order])) == ([rec], [])


def test_a_fill_whose_price_is_not_finite_stays_pending():
    order = _order("10.00", "executed", price="inf")
    rec = _rec()
    assert _run([rec], FakeClient(live=[order])) == ([rec], [])
