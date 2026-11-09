"""A journal record for an order whose POST outcome is unknown (status "pending")
is never counted as a filled trade or a settlement by any journal reader.

The record exists so the order is not invisible, but it may never have reached
the book: it must not earn an outcome, enter calibration, edge or the policy
corpus, or hide a real settlement on its ticker. Pure, in-memory fixtures.
"""
from src.agent.edge import edge_report
from src.agent.journal import reconcile_fills
from src.agent.learnings import calibration_table, edge_breakdown, reconcile_outcomes
from src.agent.policy import build_settled_records


def _rec(ticker, pending=False):
    rec = {"ts": "2026-10-01T00:00:00+00:00", "strategy": "claude", "ticker": ticker,
           "side": "no", "action": "buy", "count": 10, "price": 0.90, "est_prob": 0.97,
           "edge": 0.07, "rationale": "", "category": "cat", "order_id": "oid",
           "outcome": None}
    if pending:
        rec.update(order_id=None, status="pending", order_outcome="outcome_unknown",
                   client_order_id="c-1")
    return rec


def _settlement(ticker):
    return {"ticker": ticker, "event": ticker, "held_side": "no", "count": 10, "won": True,
            "result": "no", "cost": 9.0, "revenue": 10.0, "fee": 0.0, "pnl": 1.0,
            "settled_time": "2026-10-02T00:00:00+00:00"}


def test_reconcile_outcomes_never_settles_a_pending_record():
    journal = [_rec("KXP-1", pending=True), _rec("KXC-1")]
    out, newly = reconcile_outcomes(journal, [_settlement("KXP-1"), _settlement("KXC-1")])
    assert newly == 1
    assert out[0]["outcome"] is None and out[0]["status"] == "pending"
    assert out[1]["outcome"]["won"] is True  # control: a normal record settles


def test_pending_record_stays_out_of_edge_calibration_and_breakdown():
    journal = [_rec("KXP-1", pending=True), _rec("KXC-1")]
    out, _ = reconcile_outcomes(journal, [_settlement("KXP-1"), _settlement("KXC-1")])
    assert edge_report(out, "2026-10-03")["n_settled"] == 1
    assert sum(b["n"] for b in calibration_table(out)) == 1
    assert sum(b["n"] for b in edge_breakdown(out)["by_category"].values()) == 1


def test_pending_record_does_not_hide_a_real_settlement_from_the_policy():
    recs = build_settled_records([_rec("KXP-1", pending=True)], [_settlement("KXP-1")])
    settled = [r for r in recs if r.get("outcome")]
    assert len(settled) == 1
    assert settled[0].get("status") != "pending"  # the settlement's own record, counted once
    assert settled[0]["outcome"]["won"] is True


def test_reconcile_fills_leaves_a_pending_record_untouched():
    rec = _rec("KXP-1", pending=True)
    out, changes = reconcile_fills([rec], [], set())
    assert out == [rec] and changes == []
