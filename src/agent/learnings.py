"""Learnings system — turns measured outcomes into evidenced, evolving strategy.

This module is the missing link between the decision journal (my *predictions*:
est_prob, edge, side) and Kalshi's authoritative settlements (the *reality*). It:

  1. ``reconcile_outcomes`` — joins settlements back into journal entries, filling
     each trade's ``outcome = {won, pnl}`` from *my* journaled side + the
     settlement result. Idempotent: already-reconciled entries are left alone.
  2. ``calibration_table`` — buckets my settled trades by ``est_prob`` and shows
     predicted vs. actual win-rate. Answers "when I say 97%, do ~97% win?".
  3. ``edge_breakdown`` — per-category / per-side (/ per-method, if present)
     realized performance: n, win-rate, P&L, edge-at-entry vs. realized edge.
  4. ``flag_rules`` — a deterministic rule pass over those tables that emits
     *candidate* learnings (regressions I'd miss eyeballing numbers). Every flag
     is status='candidate' until a human confirms it — a safety net, not a
     decision-maker.
  5. ``append_learnings`` / ``load_learnings`` — the append-only audit trail at
     ``data/runtime/learnings.jsonl``, deduped by (kind, claim).

Design contract: the calculation functions (reconcile/calibration/edge/flag) are
PURE — no ``datetime.now``, no file IO. ``date`` is passed into ``flag_rules`` so
the output is deterministic and unit-testable. IO lives only in the
load/append/CLI layer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agent.settle import settlement_pnl

DEFAULT_LEARNINGS_PATH = "data/runtime/learnings.jsonl"

# Calibration bucket edges. A record with est_prob p lands in the bucket whose
# half-open range [lo, hi) contains p; the final bucket is closed [.., 1.0] so a
# perfect 1.0 prediction is not dropped.
DEFAULT_CALIBRATION_EDGES: Tuple[float, ...] = (0.5, 0.85, 0.9, 0.95, 0.99, 1.0)

# Sample-size thresholds for both confidence scaling and the n>=5 emit gate.
_MIN_N = 5
_HIGH_N = 15


# ---------------------------------------------------------------------------
# 1. Reconciliation — join settlements back into the journal
# ---------------------------------------------------------------------------

def _index_settlements(settlements: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index settlement records by ticker.

    Accepts either *raw* Kalshi settlement rows (with ``yes_count_fp`` etc.) or
    rows already passed through ``settlement_pnl`` (with ``result``/``pnl``).
    Raw rows are normalized via ``settlement_pnl`` so callers can hand us either
    shape. If a ticker appears more than once, the last one wins (settlements are
    terminal, so duplicates should agree).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for s in settlements:
        if s is None:
            continue
        # Already-normalized rows carry a "result" key; raw rows do not.
        norm = s if "result" in s else settlement_pnl(s)
        if not norm:
            continue
        ticker = norm.get("ticker")
        if ticker:
            out[ticker] = norm
    return out


def _outcome_for_record(record: Dict[str, Any], settlement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute ``{won, pnl}`` for ONE journal record from its settlement.

    ``won`` is judged against MY journaled side (the side I bought), not the
    settlement's net held side — they can differ if I traded both sides or the
    legacy bot also held the market. pnl uses ``settlement_pnl`` semantics; we
    only trust the settlement's pnl directly when its held side matches mine,
    otherwise the settlement's dollars describe a different position than my
    journaled one and we fall back to a per-contract reconstruction.
    """
    result = settlement.get("result")
    side = record.get("side")
    if result is None or side is None:
        return None
    won = side == result
    return {"won": won, "pnl": _pnl_from_my_side(record, settlement, won)}


def _pnl_from_my_side(record: Dict[str, Any], settlement: Dict[str, Any], won: bool) -> float:
    """Realized P&L on MY journaled position.

    When the settlement's held side matches my journaled side, the settlement's
    own ``pnl`` already describes my position — trust it (it includes real fees).
    Otherwise reconstruct from my journaled entry price and contract count:
    payout is $1/contract on a win, $0 on a loss; cost is price * count.
    """
    if settlement.get("held_side") == record.get("side"):
        try:
            return round(float(settlement.get("pnl") or 0.0), 4)
        except (TypeError, ValueError):
            return 0.0
    # Reconstruct from the journal: my entry price and size.
    try:
        count = int(record.get("count") or 0)
        price = float(record.get("price") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    cost = price * count
    revenue = float(count) if won else 0.0
    return round(revenue - cost, 4)


def reconcile_outcomes(
    journal_records: List[Dict[str, Any]],
    settlements: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Fill each matching journal entry's ``outcome`` from settlements.

    Match by ticker. Idempotent: records that already have a non-null
    ``outcome`` are passed through untouched. Records are NOT mutated in place —
    a shallow copy is returned for any record that changes.

    Returns ``(updated_records, newly_reconciled_count)``. ``updated_records`` is
    parallel to the input (same order, same length).
    """
    index = _index_settlements(settlements)
    out: List[Dict[str, Any]] = []
    newly = 0
    for rec in journal_records:
        if rec.get("outcome"):
            out.append(rec)  # already reconciled — idempotent skip
            continue
        if rec.get("voided"):
            out.append(rec)  # order never filled — must never earn an outcome
            continue
        settlement = index.get(rec.get("ticker"))
        if not settlement:
            out.append(rec)  # no matching settlement yet
            continue
        outcome = _outcome_for_record(rec, settlement)
        if outcome is None:
            out.append(rec)
            continue
        updated = dict(rec)
        updated["outcome"] = outcome
        out.append(updated)
        newly += 1
    return out, newly


# ---------------------------------------------------------------------------
# 2. Calibration table — predicted vs. actual win-rate by est_prob bucket
# ---------------------------------------------------------------------------

def _bucket_label(lo: float, hi: float) -> str:
    return f"[{lo:.2f},{hi:.2f}]"


def calibration_table(
    records: List[Dict[str, Any]],
    edges: Tuple[float, ...] = DEFAULT_CALIBRATION_EDGES,
) -> List[Dict[str, Any]]:
    """Bucket settled records by ``est_prob`` and compare predicted vs. actual.

    Only settled records (``outcome`` filled) with a non-None ``est_prob`` are
    counted. Each returned bucket: ``{bucket, lo, hi, n, predicted, actual}``
    where ``predicted`` is the mean est_prob in the bucket and ``actual`` is the
    realized win-rate. Empty buckets are omitted. Pure / deterministic.
    """
    settled = [
        r for r in records
        if r.get("outcome") and r.get("est_prob") is not None
    ]
    buckets: List[Dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        is_last = i == len(edges) - 2
        members = [
            r for r in settled
            if lo <= float(r["est_prob"]) < hi
            or (is_last and float(r["est_prob"]) == hi)
        ]
        if not members:
            continue
        n = len(members)
        predicted = sum(float(r["est_prob"]) for r in members) / n
        wins = sum(1 for r in members if r["outcome"].get("won"))
        buckets.append({
            "bucket": _bucket_label(lo, hi),
            "lo": lo,
            "hi": hi,
            "n": n,
            "predicted": round(predicted, 4),
            "actual": round(wins / n, 4),
        })
    return buckets


# ---------------------------------------------------------------------------
# 3. Edge breakdown — realized performance per category / side / method
# ---------------------------------------------------------------------------

def _empty_bucket() -> Dict[str, Any]:
    return {"n": 0, "wins": 0, "pnl": 0.0, "_edge_sum": 0.0, "_edge_n": 0}


def _finalize_bucket(b: Dict[str, Any]) -> Dict[str, Any]:
    """Turn an accumulator into the public per-group record."""
    n = b["n"]
    avg_entry_edge = (b["_edge_sum"] / b["_edge_n"]) if b["_edge_n"] else None
    # Realized edge = realized win-rate minus the price paid (implied prob).
    # We approximate "price paid" via avg_entry_edge: edge-at-entry was
    # est_prob - price, so realized_edge = actual_win_rate - (predicted - edge)
    # collapses to (actual_win_rate - mean_price). We expose realized_edge as
    # realized win-rate minus mean implied probability when we can derive it,
    # else None. Here we report realized_edge as win_rate - mean_price.
    win_rate = (b["wins"] / n) if n else None
    realized_edge = None
    if n and b.get("_price_n"):
        mean_price = b["_price_sum"] / b["_price_n"]
        realized_edge = round(win_rate - mean_price, 4)
    return {
        "n": n,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "pnl": round(b["pnl"], 4),
        "avg_entry_edge": round(avg_entry_edge, 4) if avg_entry_edge is not None else None,
        "realized_edge": realized_edge,
    }


def _accumulate(buckets: Dict[str, Dict[str, Any]], key: str, rec: Dict[str, Any]) -> None:
    b = buckets.setdefault(key, _empty_bucket())
    b.setdefault("_price_sum", 0.0)
    b.setdefault("_price_n", 0)
    b["n"] += 1
    b["wins"] += 1 if rec["outcome"].get("won") else 0
    b["pnl"] += float(rec["outcome"].get("pnl", 0.0))
    edge = rec.get("edge")
    if edge is not None:
        b["_edge_sum"] += float(edge)
        b["_edge_n"] += 1
    price = rec.get("price")
    if price is not None:
        b["_price_sum"] += float(price)
        b["_price_n"] += 1


def edge_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Realized performance grouped by category, side, and (if present) method.

    Only settled records (``outcome`` filled) contribute. Returns
    ``{"by_category": {...}, "by_side": {...}, "by_method": {...}}`` where each
    inner value maps a group label to ``{n, win_rate, pnl, avg_entry_edge,
    realized_edge}``. ``by_method`` is empty when no record carries a ``method``
    field — its absence is tolerated, never an error. Pure / deterministic.
    """
    settled = [r for r in records if r.get("outcome")]
    by_category: Dict[str, Dict[str, Any]] = {}
    by_side: Dict[str, Dict[str, Any]] = {}
    by_method: Dict[str, Dict[str, Any]] = {}

    for r in settled:
        _accumulate(by_category, r.get("category") or "unknown", r)
        _accumulate(by_side, r.get("side") or "unknown", r)
        method = r.get("method")
        if method:
            _accumulate(by_method, method, r)

    return {
        "by_category": {k: _finalize_bucket(v) for k, v in by_category.items()},
        "by_side": {k: _finalize_bucket(v) for k, v in by_side.items()},
        "by_method": {k: _finalize_bucket(v) for k, v in by_method.items()},
    }


# ---------------------------------------------------------------------------
# 4. Flag rules — emit candidate learnings from the tables
# ---------------------------------------------------------------------------

def _confidence(n: int) -> str:
    """low (n<5 — shouldn't emit), med (5<=n<15), high (n>=15)."""
    if n < _MIN_N:
        return "low"
    if n < _HIGH_N:
        return "med"
    return "high"


def flag_rules(
    calibration: List[Dict[str, Any]],
    edges: Dict[str, Dict[str, Any]],
    date: str,
) -> List[Dict[str, Any]]:
    """Derive candidate learnings from calibration + edge tables. PURE.

    ``date`` MUST be passed in (the CLI passes today's date string) — this
    function never calls ``datetime.now`` so its output stays deterministic.

    Rules:
      (a) STOP-candidate: a category or side group with n>=5 and realized
          pnl < 0 → "stop / re-examine; it's losing money".
      (b) overconfidence-candidate: a calibration bucket with n>=5 where the
          actual win-rate is >=10 percentage points below the predicted prob
          → "I'm overconfident in this band; haircut est_prob".

    Each candidate:
      ``{date, kind, claim, evidence:{metric,n,value}, confidence, status, supersedes}``
    with ``status='candidate'`` and ``supersedes=None``.
    """
    out: List[Dict[str, Any]] = []

    # (a) Losing groups → STOP-candidate. Categories and sides.
    for dimension, table in (("category", edges.get("by_category", {})),
                             ("side", edges.get("by_side", {})),
                             ("method", edges.get("by_method", {}))):
        for label, stats in table.items():
            n = stats.get("n", 0)
            pnl = stats.get("pnl", 0.0)
            if n >= _MIN_N and pnl < 0:
                out.append({
                    "date": date,
                    "kind": "stop",
                    "claim": (
                        f"{dimension}={label} is losing money "
                        f"(realized pnl ${pnl:.2f} over {n} settled trades) "
                        f"— stop or re-examine entries here"
                    ),
                    "evidence": {"metric": f"{dimension}.pnl", "n": n, "value": round(pnl, 4)},
                    "confidence": _confidence(n),
                    "status": "candidate",
                    "supersedes": None,
                })

    # (b) Overconfident calibration buckets → overconfidence-candidate.
    for b in calibration:
        n = b.get("n", 0)
        predicted = b.get("predicted")
        actual = b.get("actual")
        if n < _MIN_N or predicted is None or actual is None:
            continue
        gap = predicted - actual  # positive => actual below predicted
        if gap >= 0.10:
            out.append({
                "date": date,
                "kind": "overconfidence",
                "claim": (
                    f"est_prob bucket {b.get('bucket')} is overconfident: "
                    f"predicted {predicted:.0%} but only {actual:.0%} won "
                    f"({n} settled) — haircut est_prob in this band"
                ),
                "evidence": {
                    "metric": f"calibration.{b.get('bucket')}.gap",
                    "n": n,
                    "value": round(gap, 4),
                },
                "confidence": _confidence(n),
                "status": "candidate",
                "supersedes": None,
            })

    return out


# ---------------------------------------------------------------------------
# 5. Learnings store — append-only audit trail (IO layer)
# ---------------------------------------------------------------------------

def load_learnings(path: str = DEFAULT_LEARNINGS_PATH) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _dedup_key(rec: Dict[str, Any]) -> Tuple[Any, Any]:
    return (rec.get("kind"), rec.get("claim"))


def append_learnings(
    candidates: List[Dict[str, Any]],
    path: str = DEFAULT_LEARNINGS_PATH,
) -> List[Dict[str, Any]]:
    """Append only NEW candidate learnings, deduped by (kind, claim).

    Dedup is against both the existing file and within this batch, so a single
    call never writes the same learning twice. Returns the records actually
    written (the genuinely new ones).
    """
    existing = {_dedup_key(r) for r in load_learnings(path)}
    new: List[Dict[str, Any]] = []
    for c in candidates:
        key = _dedup_key(c)
        if key in existing:
            continue
        existing.add(key)
        new.append(c)
    if new:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for c in new:
                f.write(json.dumps(c) + "\n")
    return new
