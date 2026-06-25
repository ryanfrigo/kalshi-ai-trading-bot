"""Tests for the edge-measurement harness — the repo's headline feature.

Pure-function tests with in-memory fixtures (no network, no live Kalshi). They
cover the proper scoring rules on hand-computed fixtures (``brier_score``,
``log_loss``), the headline ``edge_vs_book`` (both a case where I beat the book
and one where the book beat me, plus sign/magnitude), the ``forward_only``
out-of-sample split (forward vs. suspect vs. unknown), and ``edge_report``'s
honest verdict gating (refuses to claim edge at n<10 and on non-forward data).
"""
import math

from src.agent.edge import (
    brier_score,
    log_loss,
    edge_vs_book,
    forward_only,
    edge_report,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _rec(ticker="T", side="no", est_prob=None, price=0.95, count=10,
         category="", won=None, ts="2026-06-20T00:00:00+00:00",
         settled_time="2026-06-21T00:00:00+00:00", outcome="__auto__", **extra):
    """A reconciled journal record (``outcome`` filled) for edge scoring.

    ``won`` drives the auto-built outcome; pass ``outcome=None`` for an
    unsettled record, or an explicit dict to override. ``settled_time`` lands on
    the outcome (where reconciliation copies it) so ``forward_only`` can read it.
    """
    if outcome == "__auto__":
        outcome = None if won is None else {"won": won, "pnl": 0.0,
                                            "settled_time": settled_time}
    rec = {
        "ts": ts, "strategy": "claude", "ticker": ticker, "side": side,
        "action": "buy", "count": count, "price": price, "est_prob": est_prob,
        "edge": None, "rationale": "", "category": category, "order_id": "oid",
        "outcome": outcome,
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# brier_score — hand-computed
# ---------------------------------------------------------------------------

def test_brier_score_hand_computed():
    # p=0.9 won  -> (0.9-1)^2 = 0.01
    # p=0.8 lost -> (0.8-0)^2 = 0.64
    # mean = 0.325
    recs = [
        _rec("A", est_prob=0.9, won=True),
        _rec("B", est_prob=0.8, won=False),
    ]
    res = brier_score(recs)
    assert res["n"] == 2
    assert abs(res["value"] - 0.325) < 1e-12


def test_brier_score_perfect_is_zero():
    recs = [
        _rec("A", est_prob=1.0, won=True),
        _rec("B", est_prob=0.0, won=False),
    ]
    assert brier_score(recs)["value"] == 0.0


def test_brier_score_coin_flip_is_quarter():
    # est_prob 0.5 on every trade -> (0.5)^2 = 0.25 regardless of outcome
    recs = [_rec("A", est_prob=0.5, won=True), _rec("B", est_prob=0.5, won=False)]
    assert abs(brier_score(recs)["value"] - 0.25) < 1e-12


def test_brier_score_skips_unsettled_and_none_estprob():
    recs = [
        _rec("A", est_prob=0.9, won=None),          # unsettled
        _rec("B", est_prob=None, won=True),         # no est_prob
        _rec("C", est_prob=0.9, won=True),          # the only scorable one
    ]
    res = brier_score(recs)
    assert res["n"] == 1
    assert abs(res["value"] - 0.01) < 1e-12


def test_brier_score_none_when_empty():
    assert brier_score([]) is None
    assert brier_score([_rec("A", est_prob=0.9, won=None)]) is None


# ---------------------------------------------------------------------------
# log_loss — hand-computed
# ---------------------------------------------------------------------------

def test_log_loss_hand_computed():
    # p=0.9 won  -> -ln(0.9)
    # p=0.8 lost -> -ln(1-0.8) = -ln(0.2)
    recs = [
        _rec("A", est_prob=0.9, won=True),
        _rec("B", est_prob=0.8, won=False),
    ]
    expected = (-math.log(0.9) + -math.log(0.2)) / 2
    res = log_loss(recs)
    assert res["n"] == 2
    assert abs(res["value"] - expected) < 1e-12


def test_log_loss_clips_confident_wrong_to_finite():
    # p=1.0 but LOST -> ln(1-1)=ln(0) would be -inf; eps-clip keeps it finite.
    recs = [_rec("A", est_prob=1.0, won=False)]
    res = log_loss(recs)
    assert math.isfinite(res["value"])
    assert res["value"] > 30  # ~ -ln(1e-15) ≈ 34.5, large but finite


def test_log_loss_none_when_empty():
    assert log_loss([]) is None


# ---------------------------------------------------------------------------
# edge_vs_book — sign & magnitude, beat-the-book vs. not
# ---------------------------------------------------------------------------

def test_edge_vs_book_i_beat_the_book():
    # Paid 0.90 implied each; actually won 3 of 4 (75%). I beat the book:
    # edge = 0.75 - 0.90 = -0.15  ... wait — beating means winning MORE than
    # implied. Here implied 0.90 but only 75% won -> book beat me. Construct a
    # genuine beat: paid 0.50 implied, won 4 of 4 -> edge +0.50.
    recs = [_rec(f"T{i}", price=0.50, won=True) for i in range(4)]
    res = edge_vs_book(recs)["overall"]
    assert res["n"] == 4
    assert res["realized_winrate"] == 1.0
    assert res["mean_implied"] == 0.50
    assert abs(res["edge_vs_book"] - 0.50) < 1e-9   # positive => beat the book
    assert abs(res["pnl_per_contract"] - 0.50) < 1e-9  # mean(1 - 0.50)


def test_edge_vs_book_book_beat_me():
    # Paid 0.90 implied each but only won 2 of 4 (50%) -> book beat me.
    recs = [
        _rec("A", price=0.90, won=True),
        _rec("B", price=0.90, won=True),
        _rec("C", price=0.90, won=False),
        _rec("D", price=0.90, won=False),
    ]
    res = edge_vs_book(recs)["overall"]
    assert res["realized_winrate"] == 0.5
    assert res["mean_implied"] == 0.90
    assert abs(res["edge_vs_book"] - (-0.40)) < 1e-9  # negative => no edge
    assert abs(res["pnl_per_contract"] - (0.5 - 0.9)) < 1e-9


def test_edge_vs_book_per_category_and_side():
    recs = [
        _rec("A", side="no", category="longshot", price=0.95, won=True),
        _rec("B", side="no", category="longshot", price=0.95, won=True),
        _rec("C", side="yes", category="sports", price=0.40, won=False),
    ]
    ev = edge_vs_book(recs)
    assert ev["by_category"]["longshot"]["n"] == 2
    assert ev["by_category"]["longshot"]["realized_winrate"] == 1.0
    assert ev["by_side"]["no"]["n"] == 2
    assert ev["by_side"]["yes"]["n"] == 1
    # yes/sports: paid 0.40, lost -> edge = 0 - 0.40 = -0.40
    assert abs(ev["by_side"]["yes"]["edge_vs_book"] - (-0.40)) < 1e-9


def test_edge_vs_book_overall_none_when_no_settled():
    recs = [_rec("A", price=0.9, won=None)]  # unsettled
    ev = edge_vs_book(recs)
    assert ev["overall"] is None
    assert ev["by_category"] == {}
    assert ev["by_side"] == {}


# ---------------------------------------------------------------------------
# forward_only — out-of-sample split
# ---------------------------------------------------------------------------

def test_forward_only_classifies_forward():
    # settled AFTER the trade -> forward (legitimate, out-of-sample).
    rec = _rec("A", ts="2026-06-20T00:00:00+00:00",
               settled_time="2026-06-21T00:00:00+00:00", won=True)
    split = forward_only([rec])
    assert split["forward"] == [rec]
    assert split["suspect"] == []
    assert split["unknown"] == []


def test_forward_only_classifies_suspect_when_resolved_before_trade():
    # resolution BEFORE the trade ts -> suspect (cannot be a forward prediction).
    rec = _rec("A", ts="2026-06-20T00:00:00+00:00",
               settled_time="2026-06-19T00:00:00+00:00", won=True)
    split = forward_only([rec])
    assert split["suspect"] == [rec]
    assert split["forward"] == []


def test_forward_only_equal_timestamp_is_suspect_not_forward():
    # resolution == trade ts -> NOT strictly after -> conservative: suspect.
    rec = _rec("A", ts="2026-06-20T00:00:00+00:00",
               settled_time="2026-06-20T00:00:00+00:00", won=True)
    split = forward_only([rec])
    assert split["suspect"] == [rec]
    assert split["forward"] == []


def test_forward_only_unknown_when_missing_timestamps():
    # missing settled_time -> unknown (NOT silently counted as forward).
    rec = _rec("A", ts="2026-06-20T00:00:00+00:00",
               outcome={"won": True, "pnl": 0.0})  # no settled_time on outcome
    split = forward_only([rec])
    assert split["unknown"] == [rec]
    assert split["forward"] == []
    # missing trade ts -> also unknown
    rec2 = _rec("B", ts=None, settled_time="2026-06-21T00:00:00+00:00", won=True)
    split2 = forward_only([rec2])
    assert split2["unknown"] == [rec2]


def test_forward_only_handles_z_and_offset_iso_shapes():
    # Real data mixes a trailing 'Z' and an explicit '+00:00' offset — both must
    # parse and compare correctly. Trade with offset, settled with Z, after.
    rec = _rec("A", ts="2026-06-20T00:00:00+00:00",
               settled_time="2026-06-21T00:00:00Z", won=True)
    assert forward_only([rec])["forward"] == [rec]


def test_forward_only_excludes_unsettled():
    rec = _rec("A", won=None)  # outcome is None
    split = forward_only([rec])
    assert split == {"forward": [], "suspect": [], "unknown": []}


def test_forward_only_reads_settled_time_on_record_when_not_on_outcome():
    # Fallback: settled_time on the record itself (not the outcome dict).
    rec = _rec("A", ts="2026-06-20T00:00:00+00:00",
               outcome={"won": True, "pnl": 0.0})
    rec["settled_time"] = "2026-06-22T00:00:00+00:00"  # record-level, not on outcome
    assert forward_only([rec])["forward"] == [rec]


# ---------------------------------------------------------------------------
# edge_report — honest verdict gating
# ---------------------------------------------------------------------------

def _forward_winners(n, price=0.50):
    """n forward-settled winning trades that genuinely beat the book at `price`."""
    return [
        _rec(f"T{i}", price=price, est_prob=0.60, won=True,
             ts="2026-06-20T00:00:00+00:00",
             settled_time="2026-06-21T00:00:00+00:00")
        for i in range(n)
    ]


def test_edge_report_insufficient_below_n10():
    report = edge_report(_forward_winners(9), date="2026-06-25")
    assert "INSUFFICIENT DATA" in report["verdict"]
    assert report["n_forward"] == 9
    assert report["date"] == "2026-06-25"


def test_edge_report_claims_edge_when_forward_and_beating_book():
    report = edge_report(_forward_winners(12, price=0.50), date="2026-06-25")
    assert report["n_forward"] == 12
    assert "BEATING THE BOOK" in report["verdict"]
    # won 100% vs 50% implied -> +50 pts
    assert "50.0 pts" in report["verdict"]


def test_edge_report_no_edge_when_book_wins():
    # 12 forward trades, all paid 0.90 implied but all lost -> edge -0.90.
    recs = [
        _rec(f"T{i}", price=0.90, est_prob=0.90, won=False,
             ts="2026-06-20T00:00:00+00:00",
             settled_time="2026-06-21T00:00:00+00:00")
        for i in range(12)
    ]
    report = edge_report(recs, date="2026-06-25")
    assert "NO MEASURED EDGE" in report["verdict"]


def test_edge_report_refuses_edge_on_non_forward_data():
    # 20 winning trades that BEAT the book on price — but every one is SUSPECT
    # (resolved before the trade). The verdict must NOT claim edge: forward n=0.
    recs = [
        _rec(f"T{i}", price=0.40, est_prob=0.60, won=True,
             ts="2026-06-20T00:00:00+00:00",
             settled_time="2026-06-19T00:00:00+00:00")  # resolved BEFORE trade
        for i in range(20)
    ]
    report = edge_report(recs, date="2026-06-25")
    assert report["n_settled"] == 20
    assert report["n_forward"] == 0
    assert report["n_suspect"] == 20
    assert "INSUFFICIENT DATA" in report["verdict"]
    assert "BEATING THE BOOK" not in report["verdict"]
    # Brier/edge are computed over the forward subset only -> empty.
    assert report["brier"] is None
    assert report["edge_vs_book"]["overall"] is None


def test_edge_report_only_forward_subset_scored():
    # Mix: 11 forward winners + 5 suspect winners. Only the 11 forward should
    # drive brier/edge/verdict; suspect counted but excluded from scoring.
    forward = _forward_winners(11, price=0.50)
    suspect = [
        _rec(f"S{i}", price=0.50, est_prob=0.60, won=True,
             ts="2026-06-20T00:00:00+00:00",
             settled_time="2026-06-19T00:00:00+00:00")
        for i in range(5)
    ]
    report = edge_report(forward + suspect, date="2026-06-25")
    assert report["n_settled"] == 16
    assert report["n_forward"] == 11
    assert report["n_suspect"] == 5
    assert report["brier"]["n"] == 11           # forward only
    assert report["edge_vs_book"]["overall"]["n"] == 11
    assert "BEATING THE BOOK" in report["verdict"]


def test_edge_report_empty_journal_is_insufficient():
    report = edge_report([], date="2026-06-25")
    assert report["n_settled"] == 0
    assert report["n_forward"] == 0
    assert "INSUFFICIENT DATA" in report["verdict"]
    assert report["brier"] is None
    assert report["log_loss"] is None
