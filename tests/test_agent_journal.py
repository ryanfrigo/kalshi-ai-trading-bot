"""Tests for the agent decision journal — the heart of 'profitable, measurable'.

Every trade I (the agent) make records my prediction (est_prob, edge, rationale).
When markets settle, outcomes attach. summarize_journal then measures my realized
win-rate, P&L, and calibration per category — so 'profitable' is provable, not
asserted.
"""
from src.agent.journal import make_decision_record, summarize_journal


def test_make_decision_record_core_fields():
    r = make_decision_record(
        ticker="X", side="no", count=16, price=0.95, est_prob=0.97,
        edge=0.03, rationale="favorite-longshot bias", category="sports",
        strategy="claude", order_id="oid", ts="2026-06-18T00:00:00Z",
    )
    assert r["ticker"] == "X" and r["side"] == "no" and r["count"] == 16
    assert r["price"] == 0.95 and r["est_prob"] == 0.97 and r["edge"] == 0.03
    assert r["rationale"] == "favorite-longshot bias"
    assert r["category"] == "sports" and r["order_id"] == "oid"
    assert r["outcome"] is None  # unsettled until settle() attaches it


def test_policy_note_recorded_when_present_absent_otherwise():
    # A policy override/haircut must land in the DURABLE journal, not just the
    # ephemeral response — the "override is recorded" claim depends on this.
    without = make_decision_record(ticker="X", side="no", count=1, price=0.9)
    assert "policy_note" not in without
    with_note = make_decision_record(
        ticker="X", side="no", count=1, price=0.9,
        policy_note={"overridden_block": ["category=KXCPI loses money"]})
    assert with_note["policy_note"] == {"overridden_block": ["category=KXCPI loses money"]}


def test_summarize_empty():
    s = summarize_journal([])
    assert s["total"] == 0 and s["settled"] == 0 and s["wins"] == 0
    assert s["pnl"] == 0.0


def test_summarize_winrate_pnl_and_category():
    recs = [
        {"ticker": "A", "side": "no", "category": "sports",
         "outcome": {"won": True, "pnl": 0.50}},
        {"ticker": "B", "side": "no", "category": "sports",
         "outcome": {"won": False, "pnl": -9.0}},
        {"ticker": "C", "side": "no", "category": "crypto",
         "outcome": {"won": True, "pnl": 0.40}},
        {"ticker": "D", "side": "no", "category": "crypto",
         "outcome": None},  # unsettled — excluded from settled stats
    ]
    s = summarize_journal(recs)
    assert s["total"] == 4
    assert s["settled"] == 3
    assert s["wins"] == 2
    assert abs(s["pnl"] - (0.50 - 9.0 + 0.40)) < 1e-9
    assert abs(s["win_rate"] - (2 / 3)) < 1e-9
    # per-category breakdown
    assert s["by_category"]["sports"]["settled"] == 2
    assert s["by_category"]["sports"]["wins"] == 1
    assert abs(s["by_category"]["sports"]["pnl"] - (0.50 - 9.0)) < 1e-9
    assert s["by_category"]["crypto"]["settled"] == 1
    assert abs(s["by_category"]["crypto"]["pnl"] - 0.40) < 1e-9


def test_summarize_ignores_unsettled_in_winrate():
    recs = [{"category": "x", "outcome": None}, {"category": "x", "outcome": None}]
    s = summarize_journal(recs)
    assert s["settled"] == 0
    assert s["win_rate"] is None  # undefined, not a crash
