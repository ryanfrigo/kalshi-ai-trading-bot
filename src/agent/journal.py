"""Decision journal — records the agent's predictions and measures realized edge.

Each trade the agent makes appends a record with its *prediction* (est_prob,
edge, rationale). When the underlying market settles, `settle` attaches an
`outcome` ({won, pnl}). `summarize_journal` then computes realized win-rate,
P&L, and per-category performance — turning 'profitable' into a measurement.

Stored as append-only JSONL under the git-ignored runtime dir (never committed).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_JOURNAL_PATH = "data/runtime/decision_journal.jsonl"


def make_decision_record(
    ticker: str,
    side: str,
    count: int,
    price: float,
    est_prob: Optional[float] = None,
    edge: Optional[float] = None,
    rationale: str = "",
    category: str = "",
    strategy: str = "claude",
    order_id: Optional[str] = None,
    ts: Optional[str] = None,
    action: str = "buy",
    method: Optional[str] = None,
    policy_note: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one journal record. ``outcome`` is None until the market settles.

    ``method`` (e.g. ``"manual"`` | ``"workflow"``) is an OPTIONAL tag for which
    research approach produced the trade, so the learnings system can later learn
    which one actually pays. ``policy_note`` is an OPTIONAL durable record of any
    Edge Policy interaction (an overridden block, or an applied haircut) so the
    audit trail reflects it — not just the ephemeral tool response. Both are
    backward-compatible: when omitted the key is simply absent, so existing
    records and callers are unaffected.
    """
    record = {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "ticker": ticker,
        "side": side,
        "action": action,
        "count": int(count),
        "price": float(price),
        "est_prob": est_prob,
        "edge": edge,
        "rationale": rationale,
        "category": category,
        "order_id": order_id,
        "outcome": None,
    }
    if method is not None:
        record["method"] = method
    if policy_note is not None:
        record["policy_note"] = policy_note
    return record


def append_decision(record: Dict[str, Any], path: str = DEFAULT_JOURNAL_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


def write_journal(records: List[Dict[str, Any]], path: str = DEFAULT_JOURNAL_PATH) -> None:
    """Rewrite the whole journal file from ``records`` (one JSON object per line).

    Used by the learnings reconciliation step to persist filled ``outcome``
    fields back onto existing entries. Writes atomically via a temp file + rename
    so a crash mid-write can't truncate the journal.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(p)


def load_journal(path: str = DEFAULT_JOURNAL_PATH) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def summarize_journal(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate realized performance. Unsettled records (outcome is None) count
    toward `total` but are excluded from win-rate / P&L."""
    total = len(records)
    settled = [r for r in records if r.get("outcome")]
    wins = sum(1 for r in settled if r["outcome"].get("won"))
    pnl = sum(float(r["outcome"].get("pnl", 0.0)) for r in settled)

    by_category: Dict[str, Dict[str, Any]] = {}
    for r in records:
        cat = r.get("category") or "unknown"
        c = by_category.setdefault(cat, {"total": 0, "settled": 0, "wins": 0, "pnl": 0.0})
        c["total"] += 1
        if r.get("outcome"):
            c["settled"] += 1
            c["wins"] += 1 if r["outcome"].get("won") else 0
            c["pnl"] += float(r["outcome"].get("pnl", 0.0))

    return {
        "total": total,
        "settled": len(settled),
        "wins": wins,
        "pnl": pnl,
        "win_rate": (wins / len(settled)) if settled else None,
        "by_category": by_category,
    }
