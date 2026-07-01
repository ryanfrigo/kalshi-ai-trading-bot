"""Edge Policy — closes the self-improvement loop.

The measurement half of the toolkit (``journal -> settle -> learnings -> edge``)
proves what your settled trades did. Nothing consumed that proof, so the system
*measured* but never *learned*. This module is the missing feedback edge: it
turns the settled track record into a machine-readable **policy** that gates the
next decision.

  1. ``derive_policy`` — from ``learnings.edge_breakdown`` + ``learnings.
     calibration_table``, emit ``blocks`` (category/method groups your record
     proves lose money), ``warnings`` (a net-negative *side* — surfaced, never
     hard-blocked, because side is the strategy axis, not a selection axis), and
     ``haircuts`` (``est_prob`` bands you're overconfident in).
  2. ``apply_policy`` — the pre-trade gate. Given a candidate decision, return
     ALLOW / BLOCK / HAIRCUT with reasons (and a shrunk ``est_prob`` on a haircut).
  3. ``diff_policy`` — what the newest settlements added / removed / changed, for
     the ``cli improve`` report.

Design contract (identical to ``edge.py`` / ``learnings.py``): these functions
are PURE — no ``datetime.now``, no file IO. ``date`` is injected. The honesty
rule is absolute: a group with fewer than ``min_n`` settled trades earns **no
opinion** — the policy never asserts a constraint the sample can't support.

IO (load/save ``data/runtime/edge_policy.json``) lives in the CLI layer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.learnings import reconcile_outcomes
from src.agent.settle import settlement_to_record

DEFAULT_POLICY_PATH = "data/runtime/edge_policy.json"

POLICY_VERSION = 1

# Sample-size gate (mirrors learnings._MIN_N) and the overconfidence threshold
# (mirrors learnings.flag_rules rule (b)): actual >= 10pp below predicted.
DEFAULT_MIN_N = 5
_OVERCONFIDENCE_GAP = 0.10


# ---------------------------------------------------------------------------
# 1. derive_policy — settled tables -> gate-ready rules
# ---------------------------------------------------------------------------

def _losing_blocks(
    table: Dict[str, Dict[str, Any]], dimension: str, min_n: int,
) -> List[Dict[str, Any]]:
    """Every group in ``table`` with n>=min_n and realized pnl<0."""
    out: List[Dict[str, Any]] = []
    for label, stats in table.items():
        n = stats.get("n", 0)
        pnl = float(stats.get("pnl", 0.0) or 0.0)
        if n >= min_n and pnl < 0:
            out.append({
                "dimension": dimension,
                "label": label,
                "reason": (
                    f"{dimension}={label} is net-negative "
                    f"(${pnl:.2f} over {n} settled trades)"
                ),
                "n": n,
                "pnl": round(pnl, 4),
            })
    return out


def derive_policy(
    edge_breakdown: Dict[str, Dict[str, Any]],
    calibration: List[Dict[str, Any]],
    *,
    date: str,
    min_n: int = DEFAULT_MIN_N,
) -> Dict[str, Any]:
    """Derive a gate-ready Edge Policy from the settled-outcome tables. PURE.

    ``edge_breakdown`` is the ``{by_category, by_side, by_method}`` shape emitted
    by ``learnings.edge_breakdown``; ``calibration`` is the list emitted by
    ``learnings.calibration_table``. ``date`` MUST be passed in (never call
    ``datetime.now``) so the output stays deterministic and testable.
    """
    by_category = edge_breakdown.get("by_category", {})
    by_side = edge_breakdown.get("by_side", {})
    by_method = edge_breakdown.get("by_method", {})

    # Hard blocks: category + method losers. NOT side — a net-negative side must
    # not disable the whole strategy (see warnings below).
    blocks = (
        _losing_blocks(by_category, "category", min_n)
        + _losing_blocks(by_method, "method", min_n)
    )

    # Side losers are advisory warnings only.
    warnings = _losing_blocks(by_side, "side", min_n)
    for w in warnings:
        w["reason"] = (
            f"side={w['label']} is net-negative "
            f"(${w['pnl']:.2f} over {w['n']} settled) — check category mix, "
            f"do not disable the side wholesale"
        )

    # Haircuts: overconfident calibration bands.
    haircuts: List[Dict[str, Any]] = []
    for b in calibration:
        n = b.get("n", 0)
        predicted = b.get("predicted")
        actual = b.get("actual")
        if n < min_n or predicted is None or actual is None:
            continue
        gap = float(predicted) - float(actual)
        if gap >= _OVERCONFIDENCE_GAP:
            haircuts.append({
                "band": b.get("bucket"),
                "lo": b.get("lo"),
                "hi": b.get("hi"),
                "shrink_to": round(float(actual), 4),
                "n": n,
                "gap": round(gap, 4),
            })

    settled_n = sum(int(g.get("n", 0)) for g in by_side.values())

    return {
        "meta": {
            "version": POLICY_VERSION,
            "generated_date": date,
            "settled_n": settled_n,
        },
        "blocks": blocks,
        "warnings": warnings,
        "haircuts": haircuts,
    }


# ---------------------------------------------------------------------------
# 2. apply_policy — the pre-trade gate
# ---------------------------------------------------------------------------

def _in_band(p: float, lo: Any, hi: Any) -> bool:
    """Band membership mirrors calibration_table: [lo, hi) with the top edge
    closed so a perfect 1.0 estimate is not dropped."""
    if lo is None or hi is None:
        return False
    lo, hi = float(lo), float(hi)
    return (lo <= p < hi) or (p == hi and hi >= 1.0)


def apply_policy(policy: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    """Gate one candidate decision against the policy. PURE.

    ``decision``: ``{ticker, side, category, est_prob, method?}``. Returns
    ``{verdict: ALLOW|BLOCK|HAIRCUT, reasons: [...], adjusted_est_prob?}``.
    BLOCK (category/method) takes precedence over HAIRCUT; a matching side
    warning is appended to ``reasons`` but never changes the verdict.
    """
    reasons: List[str] = []
    category = decision.get("category") or ""
    method = decision.get("method") or ""
    side = decision.get("side") or ""

    # Advisory side warnings — collected first, never decisive.
    for w in policy.get("warnings", []):
        if w.get("dimension") == "side" and w.get("label") == side:
            reasons.append(
                f"warning: side={side} — {w.get('reason', 'net-negative')}"
            )

    # Hard blocks win.
    for b in policy.get("blocks", []):
        dim, label = b.get("dimension"), b.get("label")
        if (dim == "category" and label == category) or (dim == "method" and label == method):
            reasons.append(f"blocked: {b.get('reason', f'{dim}={label} loses money')}")
            return {"verdict": "BLOCK", "reasons": reasons}

    # Haircut an overconfident estimate.
    est_prob = decision.get("est_prob")
    if est_prob is not None:
        p = float(est_prob)
        for h in policy.get("haircuts", []):
            if _in_band(p, h.get("lo"), h.get("hi")):
                shrink_to = h.get("shrink_to")
                reasons.append(
                    f"haircut: est_prob {p:.2f} in overconfident band "
                    f"{h.get('band')} — shrink to {shrink_to}"
                )
                return {
                    "verdict": "HAIRCUT",
                    "reasons": reasons,
                    "adjusted_est_prob": shrink_to,
                }

    return {"verdict": "ALLOW", "reasons": reasons}


# ---------------------------------------------------------------------------
# 3. diff_policy — what the newest settlements changed
# ---------------------------------------------------------------------------

def _block_key(b: Dict[str, Any]) -> Any:
    return (b.get("dimension"), b.get("label"))


def diff_policy(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Structural diff of two policies for the ``improve`` report. PURE.

    Returns ``{added_blocks, removed_blocks, changed_haircuts}``.
    ``changed_haircuts`` reports bands whose ``shrink_to`` moved (or that
    appeared / disappeared).
    """
    old_blocks = {_block_key(b): b for b in old.get("blocks", [])}
    new_blocks = {_block_key(b): b for b in new.get("blocks", [])}
    added_blocks = [b for k, b in new_blocks.items() if k not in old_blocks]
    removed_blocks = [b for k, b in old_blocks.items() if k not in new_blocks]

    old_hc = {h.get("band"): h for h in old.get("haircuts", [])}
    new_hc = {h.get("band"): h for h in new.get("haircuts", [])}
    changed_haircuts: List[Dict[str, Any]] = []
    for band in set(old_hc) | set(new_hc):
        o = old_hc.get(band)
        n = new_hc.get(band)
        o_shrink = o.get("shrink_to") if o else None
        n_shrink = n.get("shrink_to") if n else None
        if o_shrink != n_shrink:
            changed_haircuts.append({
                "band": band,
                "was": o_shrink,
                "now": n_shrink,
            })

    return {
        "added_blocks": added_blocks,
        "removed_blocks": removed_blocks,
        "changed_haircuts": changed_haircuts,
    }


# ---------------------------------------------------------------------------
# 4. build_settled_records — compose the settled corpus the policy learns from
# ---------------------------------------------------------------------------

def build_settled_records(
    journal_records: List[Dict[str, Any]],
    settlements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union the two ground-truth sources into one settled-records list.

    The reconciled journal is authoritative for tickers I actually journaled (it
    carries my ``est_prob`` and my ``category``, so it feeds both blocks and
    calibration/haircuts). Authoritative settlements fill in every *other* market
    I held — real outcomes with a derived series category, feeding blocks and
    side-warnings. A ticker present in the journal is NOT re-added from
    settlements, so nothing is double-counted. Delegates to already-tested pure
    functions; ``settlements`` must be in ``settlement_pnl`` (normalized) shape.
    """
    reconciled, _ = reconcile_outcomes(journal_records, settlements)
    journaled = {r.get("ticker") for r in reconciled}
    extra: List[Dict[str, Any]] = []
    for s in settlements:
        rec = settlement_to_record(s)
        if rec and rec.get("ticker") not in journaled:
            extra.append(rec)
    return reconciled + extra


# ---------------------------------------------------------------------------
# 5. Policy store — IO layer (the ONLY non-pure functions here)
# ---------------------------------------------------------------------------

def load_policy(path: str = DEFAULT_POLICY_PATH) -> Optional[Dict[str, Any]]:
    """Load the active gate policy, or None if none has been saved yet."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def save_policy(policy: Dict[str, Any], path: str = DEFAULT_POLICY_PATH) -> None:
    """Persist the policy atomically (temp file + rename) as the active gate."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(policy, indent=2))
    tmp.replace(p)
