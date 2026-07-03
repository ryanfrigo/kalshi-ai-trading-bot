"""Tests for fill reconciliation — journaled decisions must reflect what executed.

Pure-function tests with in-memory fixtures (no network, no live Kalshi, no IO).
The scenario that motivated this: a maker order journaled at placement, then
cancelled unfilled — a phantom prediction that must never earn an outcome or
count toward calibration / the Edge Policy.

Covers:
  - reconcile_fills: void on zero fills, shrink on partial fill, untouched when
    fully filled / still resting / already settled / already voided / legacy
    (no order_id); no in-place mutation; idempotency.
  - reconcile_outcomes: a voided record is skipped even when its market settles.
"""
from src.agent.journal import make_decision_record, reconcile_fills
from src.agent.learnings import reconcile_outcomes


def _rec(order_id="oid-1", count=70, **kw):
    return make_decision_record(
        ticker=kw.pop("ticker", "KXALIENS-27"),
        side="no",
        count=count,
        price=0.92,
        est_prob=0.98,
        order_id=order_id,
        ts="2026-07-03T05:44:37+00:00",
        **kw,
    )


def _fill(order_id, count):
    # Mirror the real Kalshi fill shape: count arrives as fixed-point string
    # ``count_fp`` ("45.00"), not an integer ``count``.
    return {"order_id": order_id, "count_fp": f"{count}.00"}


def test_legacy_integer_count_field_also_accepted():
    updated, _ = reconcile_fills(
        [_rec(count=70)],
        fills=[{"order_id": "oid-1", "count": 70}],
        resting_order_ids=set(),
    )
    assert "voided" not in updated[0]


# ---------------------------------------------------------------------------
# reconcile_fills
# ---------------------------------------------------------------------------

def test_zero_fills_and_not_resting_voids_record():
    updated, changes = reconcile_fills([_rec()], fills=[], resting_order_ids=set())
    assert updated[0]["voided"] is True
    assert "cancelled" in updated[0]["voided_reason"]
    assert changes == [{"ticker": "KXALIENS-27", "order_id": "oid-1",
                        "change": "voided (0 fills)"}]


def test_partial_fill_shrinks_count_and_keeps_original():
    updated, changes = reconcile_fills(
        [_rec(count=70)], fills=[_fill("oid-1", 30)], resting_order_ids=set()
    )
    assert updated[0]["count"] == 30
    assert updated[0]["original_count"] == 70
    assert "voided" not in updated[0]
    assert "partial fill" in changes[0]["change"]


def test_partial_fills_sum_across_multiple_fills():
    updated, _ = reconcile_fills(
        [_rec(count=70)],
        fills=[_fill("oid-1", 30), _fill("oid-1", 40)],
        resting_order_ids=set(),
    )
    assert "voided" not in updated[0]  # 30 + 40 = 70 = fully filled
    assert updated[0]["count"] == 70


def test_still_resting_is_untouched():
    rec = _rec()
    updated, changes = reconcile_fills([rec], fills=[], resting_order_ids={"oid-1"})
    assert updated[0] is rec
    assert changes == []


def test_fully_filled_is_untouched():
    rec = _rec(count=70)
    updated, changes = reconcile_fills(
        [rec], fills=[_fill("oid-1", 70)], resting_order_ids=set()
    )
    assert updated[0] is rec
    assert changes == []


def test_legacy_record_without_order_id_is_untouched():
    rec = _rec(order_id=None)
    updated, changes = reconcile_fills([rec], fills=[], resting_order_ids=set())
    assert updated[0] is rec
    assert changes == []


def test_settled_record_is_untouched():
    rec = _rec()
    rec["outcome"] = {"won": True, "pnl": 5.6}
    updated, changes = reconcile_fills([rec], fills=[], resting_order_ids=set())
    assert updated[0] is rec
    assert changes == []


def test_idempotent_on_already_voided():
    once, _ = reconcile_fills([_rec()], fills=[], resting_order_ids=set())
    twice, changes = reconcile_fills(once, fills=[], resting_order_ids=set())
    assert twice[0] == once[0]
    assert changes == []


def test_input_records_are_not_mutated():
    rec = _rec()
    reconcile_fills([rec], fills=[], resting_order_ids=set())
    assert "voided" not in rec  # changed record was a copy


# ---------------------------------------------------------------------------
# reconcile_outcomes must skip voided records
# ---------------------------------------------------------------------------

def test_voided_record_never_earns_an_outcome():
    voided, _ = reconcile_fills([_rec()], fills=[], resting_order_ids=set())
    settlement = {
        "ticker": "KXALIENS-27",
        "market_result": "no",
        "revenue": 70.0,
        "pnl": 5.6,
    }
    reconciled, newly = reconcile_outcomes(voided, [settlement])
    assert newly == 0
    assert reconciled[0].get("outcome") is None
