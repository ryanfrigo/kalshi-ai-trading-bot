"""Tests for the Edge Policy — the component that closes the self-improvement
loop by turning the settled track record into a pre-trade gate.

Pure-function tests with in-memory fixtures (no network, no live Kalshi, no IO).
They cover:
  - derive_policy: losing category/method -> block; losing side -> warning (never
    a hard block); overconfident est_prob band -> haircut; thin data -> silence.
  - apply_policy: BLOCK on a matching category/method; HAIRCUT (with a shrunk
    est_prob) inside a haircut band; ALLOW otherwise; side warnings annotate but
    never flip the verdict.
  - diff_policy: what the newest settlements added / removed / changed.

The inputs mirror the real shapes emitted by ``learnings.edge_breakdown`` and
``learnings.calibration_table`` so the policy stays a thin, honest consumer.
"""
from src.agent.policy import derive_policy, apply_policy, diff_policy


# ---------------------------------------------------------------------------
# Fixtures — shaped exactly like learnings.edge_breakdown / calibration_table
# ---------------------------------------------------------------------------

def _edge_breakdown(by_category=None, by_side=None, by_method=None):
    return {
        "by_category": by_category or {},
        "by_side": by_side or {},
        "by_method": by_method or {},
    }


def _grp(n, pnl, win_rate=None, realized_edge=None, avg_entry_edge=None):
    return {
        "n": n, "win_rate": win_rate, "pnl": pnl,
        "avg_entry_edge": avg_entry_edge, "realized_edge": realized_edge,
    }


def _calib(bucket, lo, hi, n, predicted, actual):
    return {"bucket": bucket, "lo": lo, "hi": hi, "n": n,
            "predicted": predicted, "actual": actual}


DATE = "2026-06-30"


# ---------------------------------------------------------------------------
# derive_policy — blocks
# ---------------------------------------------------------------------------

def test_losing_category_with_enough_data_becomes_a_block():
    eb = _edge_breakdown(by_category={"economics": _grp(n=8, pnl=-12.30)})
    policy = derive_policy(eb, [], date=DATE)
    blocks = policy["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["dimension"] == "category"
    assert blocks[0]["label"] == "economics"
    assert blocks[0]["n"] == 8
    assert blocks[0]["pnl"] == -12.30


def test_losing_method_also_becomes_a_block():
    eb = _edge_breakdown(by_method={"workflow": _grp(n=6, pnl=-4.0)})
    policy = derive_policy(eb, [], date=DATE)
    assert any(b["dimension"] == "method" and b["label"] == "workflow"
               for b in policy["blocks"])


def test_thin_category_earns_no_block():
    # n below the min_n gate -> no opinion, silence. The honesty rule.
    eb = _edge_breakdown(by_category={"aliens": _grp(n=4, pnl=-99.0)})
    policy = derive_policy(eb, [], date=DATE)
    assert policy["blocks"] == []


def test_profitable_category_earns_no_block():
    eb = _edge_breakdown(by_category={"aliens": _grp(n=20, pnl=15.0)})
    policy = derive_policy(eb, [], date=DATE)
    assert policy["blocks"] == []


# ---------------------------------------------------------------------------
# derive_policy — side is a warning, never a hard block
# ---------------------------------------------------------------------------

def test_losing_side_becomes_a_warning_not_a_block():
    # NO is net-negative but that must NOT hard-block the whole strategy.
    eb = _edge_breakdown(by_side={"no": _grp(n=40, pnl=-59.0, win_rate=0.79)})
    policy = derive_policy(eb, [], date=DATE)
    assert policy["blocks"] == []
    assert len(policy["warnings"]) == 1
    assert policy["warnings"][0]["dimension"] == "side"
    assert policy["warnings"][0]["label"] == "no"


# ---------------------------------------------------------------------------
# derive_policy — haircuts from overconfident calibration bands
# ---------------------------------------------------------------------------

def test_overconfident_band_becomes_a_haircut():
    calib = [_calib("[0.90,0.95]", 0.90, 0.95, n=6, predicted=0.92, actual=0.72)]
    policy = derive_policy(_edge_breakdown(), calib, date=DATE)
    hc = policy["haircuts"]
    assert len(hc) == 1
    assert hc[0]["band"] == "[0.90,0.95]"
    assert hc[0]["shrink_to"] == 0.72
    assert abs(hc[0]["gap"] - 0.20) < 1e-9


def test_well_calibrated_band_earns_no_haircut():
    calib = [_calib("[0.95,0.99]", 0.95, 0.99, n=10, predicted=0.97, actual=0.96)]
    policy = derive_policy(_edge_breakdown(), calib, date=DATE)
    assert policy["haircuts"] == []


def test_thin_band_earns_no_haircut():
    calib = [_calib("[0.90,0.95]", 0.90, 0.95, n=3, predicted=0.92, actual=0.10)]
    policy = derive_policy(_edge_breakdown(), calib, date=DATE)
    assert policy["haircuts"] == []


def test_meta_records_settled_n_from_side_totals():
    eb = _edge_breakdown(by_side={"no": _grp(n=40, pnl=-59.0),
                                  "yes": _grp(n=12, pnl=3.0)})
    policy = derive_policy(eb, [], date=DATE)
    assert policy["meta"]["settled_n"] == 52
    assert policy["meta"]["generated_date"] == DATE


# ---------------------------------------------------------------------------
# apply_policy — the gate
# ---------------------------------------------------------------------------

def _policy(blocks=None, warnings=None, haircuts=None):
    return {
        "meta": {"version": 1, "generated_date": DATE, "settled_n": 0},
        "blocks": blocks or [],
        "warnings": warnings or [],
        "haircuts": haircuts or [],
    }


def test_apply_blocks_a_decision_in_a_blocked_category():
    policy = _policy(blocks=[{"dimension": "category", "label": "economics",
                              "reason": "losing", "n": 8, "pnl": -12.3}])
    verdict = apply_policy(policy, {"ticker": "KXCPI-1", "side": "no",
                                    "category": "economics", "est_prob": 0.97})
    assert verdict["verdict"] == "BLOCK"
    assert verdict["reasons"]


def test_apply_allows_a_decision_in_an_unblocked_category():
    policy = _policy(blocks=[{"dimension": "category", "label": "economics",
                              "reason": "losing", "n": 8, "pnl": -12.3}])
    verdict = apply_policy(policy, {"ticker": "KXALIENS-27", "side": "no",
                                    "category": "aliens", "est_prob": 0.97})
    assert verdict["verdict"] == "ALLOW"


def test_apply_haircuts_an_overconfident_estimate():
    policy = _policy(haircuts=[{"band": "[0.90,0.95]", "lo": 0.90, "hi": 0.95,
                                "shrink_to": 0.72, "n": 6, "gap": 0.20}])
    verdict = apply_policy(policy, {"ticker": "T", "side": "no",
                                    "category": "x", "est_prob": 0.93})
    assert verdict["verdict"] == "HAIRCUT"
    assert verdict["adjusted_est_prob"] == 0.72


def test_apply_block_wins_over_haircut():
    policy = _policy(
        blocks=[{"dimension": "category", "label": "economics", "reason": "x",
                 "n": 8, "pnl": -1.0}],
        haircuts=[{"band": "[0.90,0.95]", "lo": 0.90, "hi": 0.95,
                   "shrink_to": 0.72, "n": 6, "gap": 0.20}],
    )
    verdict = apply_policy(policy, {"ticker": "T", "side": "no",
                                    "category": "economics", "est_prob": 0.93})
    assert verdict["verdict"] == "BLOCK"


def test_apply_side_warning_annotates_but_does_not_block():
    policy = _policy(warnings=[{"dimension": "side", "label": "no",
                                "reason": "net-negative", "n": 40, "pnl": -59.0}])
    verdict = apply_policy(policy, {"ticker": "T", "side": "no",
                                    "category": "aliens", "est_prob": 0.97})
    assert verdict["verdict"] == "ALLOW"
    assert any("no" in r for r in verdict["reasons"])


def test_apply_on_empty_policy_allows():
    verdict = apply_policy(_policy(), {"ticker": "T", "side": "no",
                                       "category": "x", "est_prob": 0.9})
    assert verdict["verdict"] == "ALLOW"


# ---------------------------------------------------------------------------
# diff_policy — what the newest settlements changed
# ---------------------------------------------------------------------------

def test_diff_detects_added_and_removed_blocks():
    old = _policy(blocks=[{"dimension": "category", "label": "sports",
                           "reason": "x", "n": 6, "pnl": -3.0}])
    new = _policy(blocks=[{"dimension": "category", "label": "economics",
                           "reason": "x", "n": 8, "pnl": -12.0}])
    d = diff_policy(old, new)
    assert [b["label"] for b in d["added_blocks"]] == ["economics"]
    assert [b["label"] for b in d["removed_blocks"]] == ["sports"]


def test_diff_of_identical_policies_is_empty():
    p = _policy(blocks=[{"dimension": "category", "label": "economics",
                         "reason": "x", "n": 8, "pnl": -12.0}])
    d = diff_policy(p, p)
    assert d["added_blocks"] == []
    assert d["removed_blocks"] == []
    assert d["changed_haircuts"] == []
