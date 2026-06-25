"""Tests for the learnings system — the bridge from journaled predictions to
measured reality.

These are pure-function tests with in-memory fixtures (no network, no live
Kalshi). They cover the spine (``reconcile_outcomes``: join + idempotency +
won/pnl for both sides), calibration bucketing, the flag-rules (both rule types
fire above the n threshold and stay silent below it), and the append-only
learnings store's dedup.
"""
from src.agent.learnings import (
    reconcile_outcomes,
    calibration_table,
    edge_breakdown,
    flag_rules,
    append_learnings,
    load_learnings,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _journal(ticker, side, est_prob=None, edge=None, price=0.95, count=10,
             category="", outcome=None, **extra):
    rec = {
        "ts": "2026-06-20T00:00:00Z", "strategy": "claude", "ticker": ticker,
        "side": side, "action": "buy", "count": count, "price": price,
        "est_prob": est_prob, "edge": edge, "rationale": "", "category": category,
        "order_id": "oid", "outcome": outcome,
    }
    rec.update(extra)
    return rec


def _settlement(ticker, result, held_side=None, pnl=0.0):
    """A settlement_pnl-shaped record (already normalized)."""
    return {
        "ticker": ticker, "event": ticker, "held_side": held_side or result,
        "count": 10, "won": (held_side or result) == result, "result": result,
        "cost": 0.0, "revenue": 0.0, "fee": 0.0, "pnl": pnl,
        "settled_time": "2026-06-20T01:00:00Z",
    }


# ---------------------------------------------------------------------------
# reconcile_outcomes — join, both-side correctness, idempotency
# ---------------------------------------------------------------------------

def test_reconcile_no_side_won_uses_settlement_pnl():
    # I bought NO; the market resolved NO -> I won. Settlement's held side
    # matches mine, so its dollar pnl is trusted verbatim.
    journal = [_journal("KXALIENS-27", "no", est_prob=0.99, edge=0.09)]
    settlements = [_settlement("KXALIENS-27", result="no", held_side="no", pnl=0.80)]
    updated, newly = reconcile_outcomes(journal, settlements)
    assert newly == 1
    assert updated[0]["outcome"]["won"] is True
    assert abs(updated[0]["outcome"]["pnl"] - 0.80) < 1e-9


def test_reconcile_no_side_lost():
    # I bought NO; the market resolved YES -> I lost.
    journal = [_journal("KX-LOSE", "no", est_prob=0.97)]
    settlements = [_settlement("KX-LOSE", result="yes", held_side="no", pnl=-9.5)]
    updated, newly = reconcile_outcomes(journal, settlements)
    assert newly == 1
    assert updated[0]["outcome"]["won"] is False
    assert abs(updated[0]["outcome"]["pnl"] - (-9.5)) < 1e-9


def test_reconcile_yes_side_won():
    # I bought YES; the market resolved YES -> I won.
    journal = [_journal("KX-YES", "yes", est_prob=0.60, price=0.55)]
    settlements = [_settlement("KX-YES", result="yes", held_side="yes", pnl=4.5)]
    updated, newly = reconcile_outcomes(journal, settlements)
    assert newly == 1
    assert updated[0]["outcome"]["won"] is True
    assert abs(updated[0]["outcome"]["pnl"] - 4.5) < 1e-9


def test_reconcile_won_judged_against_my_side_not_settlement_held_side():
    # The settlement's *net* held side is NO and that side won (result=no), but
    # MY journal entry bought YES on this ticker -> I lost. won must reflect my
    # side, and pnl is reconstructed from my entry (held sides differ).
    journal = [_journal("KX-SPLIT", "yes", price=0.40, count=10)]
    settlements = [_settlement("KX-SPLIT", result="no", held_side="no", pnl=0.50)]
    updated, newly = reconcile_outcomes(journal, settlements)
    assert newly == 1
    assert updated[0]["outcome"]["won"] is False
    # reconstructed: revenue 0 (lost) - cost (0.40*10=4.0) = -4.0
    assert abs(updated[0]["outcome"]["pnl"] - (-4.0)) < 1e-9


def test_reconcile_idempotent_skips_filled():
    filled = _journal("KX-DONE", "no", outcome={"won": True, "pnl": 0.8})
    settlements = [_settlement("KX-DONE", result="yes", held_side="no", pnl=-99.0)]
    updated, newly = reconcile_outcomes([filled], settlements)
    assert newly == 0
    # untouched: outcome unchanged, NOT overwritten by the contradictory settlement
    assert updated[0]["outcome"] == {"won": True, "pnl": 0.8}


def test_reconcile_idempotent_on_rerun():
    journal = [_journal("KX-A", "no"), _journal("KX-B", "no")]
    settlements = [_settlement("KX-A", result="no", pnl=0.5),
                   _settlement("KX-B", result="yes", pnl=-9.0)]
    first, n1 = reconcile_outcomes(journal, settlements)
    assert n1 == 2
    second, n2 = reconcile_outcomes(first, settlements)  # re-run
    assert n2 == 0  # nothing new to reconcile
    assert second == first


def test_reconcile_unmatched_ticker_left_unsettled():
    journal = [_journal("KX-UNSETTLED", "no")]
    updated, newly = reconcile_outcomes(journal, [_settlement("OTHER", result="no")])
    assert newly == 0
    assert updated[0]["outcome"] is None


def test_reconcile_does_not_mutate_input():
    journal = [_journal("KX-M", "no")]
    settlements = [_settlement("KX-M", result="no", pnl=0.5)]
    reconcile_outcomes(journal, settlements)
    assert journal[0]["outcome"] is None  # original untouched


def test_reconcile_accepts_raw_kalshi_settlements():
    # Raw rows (no "result" key) are normalized via settlement_pnl.
    journal = [_journal("KX-RAW", "no", price=0.95, count=16)]
    raw = [{
        "ticker": "KX-RAW", "event_ticker": "E", "market_result": "no",
        "no_count_fp": "16.00", "yes_count_fp": "0.00",
        "no_total_cost_dollars": "15.20", "yes_total_cost_dollars": "0.0",
        "revenue": 1600, "fee_cost": "0.0",
    }]
    updated, newly = reconcile_outcomes(journal, raw)
    assert newly == 1
    assert updated[0]["outcome"]["won"] is True
    assert abs(updated[0]["outcome"]["pnl"] - 0.80) < 1e-9


# ---------------------------------------------------------------------------
# calibration_table — bucketing predicted vs. actual
# ---------------------------------------------------------------------------

def test_calibration_buckets_and_winrate():
    recs = [
        # 0.95+ band: 3 settled, 2 won -> actual 0.667
        _journal("A", "no", est_prob=0.96, outcome={"won": True, "pnl": 0.5}),
        _journal("B", "no", est_prob=0.97, outcome={"won": True, "pnl": 0.5}),
        _journal("C", "no", est_prob=0.98, outcome={"won": False, "pnl": -9.0}),
        # 0.90-0.95 band: 1 settled, won
        _journal("D", "no", est_prob=0.92, outcome={"won": True, "pnl": 0.4}),
    ]
    table = calibration_table(recs)
    by_bucket = {b["bucket"]: b for b in table}
    assert by_bucket["[0.95,0.99]"]["n"] == 3
    # actual/predicted are rounded to 4 dp by the implementation
    assert abs(by_bucket["[0.95,0.99]"]["actual"] - (2 / 3)) < 1e-4
    assert abs(by_bucket["[0.95,0.99]"]["predicted"] - (0.96 + 0.97 + 0.98) / 3) < 1e-4
    assert by_bucket["[0.90,0.95]"]["n"] == 1
    assert by_bucket["[0.90,0.95]"]["actual"] == 1.0


def test_calibration_skips_unsettled_and_none_estprob():
    recs = [
        _journal("A", "no", est_prob=0.97, outcome=None),                       # unsettled
        _journal("B", "no", est_prob=None, outcome={"won": True, "pnl": 0.5}),  # no est_prob
        _journal("C", "no", est_prob=0.96, outcome={"won": True, "pnl": 0.5}),  # counts
    ]
    table = calibration_table(recs)
    assert sum(b["n"] for b in table) == 1


def test_calibration_perfect_prediction_lands_in_last_bucket():
    recs = [_journal("A", "no", est_prob=1.0, outcome={"won": True, "pnl": 0.5})]
    table = calibration_table(recs)
    assert any(b["hi"] == 1.0 and b["n"] == 1 for b in table)


def test_calibration_empty_when_no_settled():
    assert calibration_table([_journal("A", "no", est_prob=0.9, outcome=None)]) == []


# ---------------------------------------------------------------------------
# edge_breakdown — per category / side / method
# ---------------------------------------------------------------------------

def test_edge_breakdown_category_and_side():
    recs = [
        _journal("A", "no", edge=0.09, price=0.90, category="longshot",
                 outcome={"won": True, "pnl": 1.0}),
        _journal("B", "no", edge=0.05, price=0.95, category="longshot",
                 outcome={"won": False, "pnl": -9.0}),
        _journal("C", "yes", edge=0.02, price=0.50, category="sports",
                 outcome={"won": True, "pnl": 0.5}),
    ]
    eb = edge_breakdown(recs)
    assert eb["by_category"]["longshot"]["n"] == 2
    assert abs(eb["by_category"]["longshot"]["pnl"] - (1.0 - 9.0)) < 1e-9
    assert eb["by_side"]["no"]["n"] == 2
    assert eb["by_side"]["yes"]["n"] == 1
    # method table empty when no record carries a method field
    assert eb["by_method"] == {}


def test_edge_breakdown_method_when_present():
    recs = [
        _journal("A", "no", category="x", method="workflow",
                 outcome={"won": True, "pnl": 1.0}),
        _journal("B", "no", category="x", method="manual",
                 outcome={"won": False, "pnl": -2.0}),
    ]
    eb = edge_breakdown(recs)
    assert eb["by_method"]["workflow"]["n"] == 1
    assert eb["by_method"]["manual"]["n"] == 1


# ---------------------------------------------------------------------------
# flag_rules — both rule types, threshold gating, determinism
# ---------------------------------------------------------------------------

def _losing_category_records(n):
    return [
        _journal(f"T{i}", "no", est_prob=0.95, edge=0.05, price=0.90,
                 category="econ", outcome={"won": False, "pnl": -1.0})
        for i in range(n)
    ]


def test_flag_rules_stop_fires_for_losing_category_at_n5():
    eb = edge_breakdown(_losing_category_records(5))
    flags = flag_rules([], eb, date="2026-06-24")
    stops = [f for f in flags if f["kind"] == "stop"]
    assert any("category=econ" in f["claim"] for f in stops)
    f = next(f for f in stops if "category=econ" in f["claim"])
    assert f["evidence"]["n"] == 5
    assert f["evidence"]["value"] < 0
    assert f["confidence"] == "med"      # 5 <= n < 15
    assert f["status"] == "candidate"
    assert f["supersedes"] is None
    assert f["date"] == "2026-06-24"     # date is passed in, not now()


def test_flag_rules_stop_does_not_fire_below_n5():
    eb = edge_breakdown(_losing_category_records(4))  # only 4 settled
    flags = flag_rules([], eb, date="2026-06-24")
    assert not [f for f in flags if f["kind"] == "stop" and "category=econ" in f["claim"]]


def test_flag_rules_stop_does_not_fire_when_profitable():
    recs = _losing_category_records(5)
    for r in recs:
        r["outcome"] = {"won": True, "pnl": 1.0}  # profitable now
    eb = edge_breakdown(recs)
    flags = flag_rules([], eb, date="2026-06-24")
    assert not [f for f in flags if f["kind"] == "stop"]


def test_flag_rules_overconfidence_fires():
    # predicted ~0.97, actual 0.60 -> gap 0.37 >= 0.10, n=5
    calibration = [{
        "bucket": "[0.95,0.99]", "lo": 0.95, "hi": 0.99, "n": 5,
        "predicted": 0.97, "actual": 0.60,
    }]
    flags = flag_rules(calibration, {}, date="2026-06-24")
    over = [f for f in flags if f["kind"] == "overconfidence"]
    assert len(over) == 1
    assert over[0]["evidence"]["n"] == 5
    assert abs(over[0]["evidence"]["value"] - 0.37) < 1e-9
    assert over[0]["confidence"] == "med"
    assert over[0]["status"] == "candidate"


def test_flag_rules_overconfidence_high_confidence_at_n15():
    calibration = [{
        "bucket": "[0.95,0.99]", "lo": 0.95, "hi": 0.99, "n": 15,
        "predicted": 0.97, "actual": 0.80,
    }]
    flags = flag_rules(calibration, {}, date="2026-06-24")
    assert flags[0]["confidence"] == "high"  # n >= 15


def test_flag_rules_overconfidence_silent_when_calibrated():
    # gap only 0.05 (< 0.10) -> no flag, even with plenty of samples
    calibration = [{
        "bucket": "[0.95,0.99]", "lo": 0.95, "hi": 0.99, "n": 20,
        "predicted": 0.97, "actual": 0.92,
    }]
    assert flag_rules(calibration, {}, date="2026-06-24") == []


def test_flag_rules_overconfidence_silent_below_n5():
    calibration = [{
        "bucket": "[0.95,0.99]", "lo": 0.95, "hi": 0.99, "n": 4,
        "predicted": 0.97, "actual": 0.30,  # huge gap but too few samples
    }]
    assert flag_rules(calibration, {}, date="2026-06-24") == []


# ---------------------------------------------------------------------------
# learnings store — append-only dedup
# ---------------------------------------------------------------------------

def _candidate(kind="stop", claim="econ is losing"):
    return {
        "date": "2026-06-24", "kind": kind, "claim": claim,
        "evidence": {"metric": "category.pnl", "n": 5, "value": -5.0},
        "confidence": "med", "status": "candidate", "supersedes": None,
    }


def test_append_learnings_writes_new(tmp_path):
    path = str(tmp_path / "learnings.jsonl")
    new = append_learnings([_candidate()], path)
    assert len(new) == 1
    assert len(load_learnings(path)) == 1


def test_append_learnings_dedups_against_existing_file(tmp_path):
    path = str(tmp_path / "learnings.jsonl")
    append_learnings([_candidate()], path)
    # second call with the SAME (kind, claim) writes nothing
    new = append_learnings([_candidate()], path)
    assert new == []
    assert len(load_learnings(path)) == 1


def test_append_learnings_dedups_within_batch(tmp_path):
    path = str(tmp_path / "learnings.jsonl")
    new = append_learnings([_candidate(), _candidate()], path)  # identical twice
    assert len(new) == 1
    assert len(load_learnings(path)) == 1


def test_append_learnings_distinguishes_by_kind_and_claim(tmp_path):
    path = str(tmp_path / "learnings.jsonl")
    append_learnings([_candidate(kind="stop", claim="econ losing")], path)
    new = append_learnings([
        _candidate(kind="overconfidence", claim="econ losing"),  # same claim, diff kind
        _candidate(kind="stop", claim="sports losing"),          # same kind, diff claim
    ], path)
    assert len(new) == 2
    assert len(load_learnings(path)) == 3


def test_load_learnings_missing_file_is_empty(tmp_path):
    assert load_learnings(str(tmp_path / "nope.jsonl")) == []
