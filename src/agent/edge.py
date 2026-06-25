"""Edge-measurement harness — rigorously prove (or disprove) edge vs. the book.

This is the repo's headline differentiator. The whole Kalshi/prediction-market
AI niche claims profitability and *none of it* proves edge against the sharp
order book out-of-sample. This module does, on settled reality:

  1. ``brier_score`` — mean squared error of my probabilities vs. binary
     outcomes. Lower = better calibrated. The proper scoring rule.
  2. ``log_loss``    — log scoring rule; punishes confident wrongness harder.
  3. ``edge_vs_book``— THE key metric. The price I paid for my side *is* the
     book's implied probability for that side. ``edge_vs_book`` =
     realized win-rate − mean implied probability: did my fills beat the price
     the (sharp) book charged me? Positive = real edge; ~0 = the book was right.
  4. ``forward_only``— out-of-sample honesty guard. Splits trades into
     forward-settled (the event resolved strictly AFTER I traded — the only
     legitimate edge) vs. suspect (resolution at/before the trade → leakage)
     vs. unknown (can't confirm). Conservative: never silently counts a trade
     as forward.
  5. ``edge_report`` — assembles all of the above into one honest verdict that
     REFUSES to claim edge on thin (n<10) or non-forward data.

Design contract (mirrors ``learnings``/``settle``): every function here is
PURE — no ``datetime.now``, no file IO. Any "today" date is passed IN by the
CLI so output stays deterministic and unit-testable. IO + the live fetch live
only in the CLI layer (``cli.py edge``).

It deliberately reuses, not duplicates, the existing measurement plumbing:
``learnings.calibration_table`` for the calibration curve, and the
``realized_edge = win_rate − mean_price`` formula already sketched in
``learnings.edge_breakdown`` is formalized here as ``edge_vs_book``.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agent.learnings import calibration_table

# Below this many forward-settled trades, we refuse to assert edge — the sample
# is too small to distinguish skill from variance. Matches the niche's central
# failure: claiming edge from a handful of lucky settlements.
_MIN_EDGE_N = 10


# ---------------------------------------------------------------------------
# Settled-record helpers
# ---------------------------------------------------------------------------

def _settled(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Records whose ``outcome`` is filled (the market resolved)."""
    return [r for r in records if r.get("outcome")]


def _outcome_o(rec: Dict[str, Any]) -> float:
    """Binary outcome for MY side: 1.0 if my side won, else 0.0."""
    return 1.0 if rec["outcome"].get("won") else 0.0


# ---------------------------------------------------------------------------
# 1. Brier score — mean squared error of probability vs. outcome
# ---------------------------------------------------------------------------

def brier_score(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Mean (p − o)^2 over settled records carrying an ``est_prob``.

    ``p`` = my estimated probability MY side wins; ``o`` = 1.0 if it won else
    0.0. Lower is better (0 = perfect, 0.25 = a coin-flip 50% guess, 1 = always
    maximally wrong). Returns ``{value, n}`` or ``None`` when no scorable record
    exists. Pure / deterministic.
    """
    scorable = [r for r in _settled(records) if r.get("est_prob") is not None]
    n = len(scorable)
    if n == 0:
        return None
    total = 0.0
    for r in scorable:
        p = float(r["est_prob"])
        o = _outcome_o(r)
        total += (p - o) ** 2
    return {"value": total / n, "n": n}


# ---------------------------------------------------------------------------
# 2. Log loss — log scoring rule
# ---------------------------------------------------------------------------

def log_loss(records: List[Dict[str, Any]], eps: float = 1e-15) -> Optional[Dict[str, Any]]:
    """Mean −[o·ln(p) + (1−o)·ln(1−p)] over settled records with ``est_prob``.

    ``p`` is clipped to ``[eps, 1−eps]`` so a confidently-wrong 0/1 prediction
    yields a large-but-finite penalty instead of infinity. Lower is better.
    Returns ``{value, n}`` or ``None`` when no scorable record exists.
    Pure / deterministic.
    """
    scorable = [r for r in _settled(records) if r.get("est_prob") is not None]
    n = len(scorable)
    if n == 0:
        return None
    total = 0.0
    for r in scorable:
        p = min(max(float(r["est_prob"]), eps), 1.0 - eps)
        o = _outcome_o(r)
        total += -(o * math.log(p) + (1.0 - o) * math.log(1.0 - p))
    return {"value": total / n, "n": n}


# ---------------------------------------------------------------------------
# 3. Edge vs. book — did my fills beat the price the book charged?
# ---------------------------------------------------------------------------

def _edge_stats(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Core edge-vs-book stats over settled records carrying a ``price``.

    The price I paid for my side is the book's implied probability for that
    side. So:
        realized_winrate = mean(o)
        mean_implied      = mean(price)
        edge_vs_book      = realized_winrate − mean_implied
        pnl_per_contract  = mean(o − price)   # $1 payout on win, price is cost

    Returns ``{n, realized_winrate, mean_implied, edge_vs_book,
    pnl_per_contract}`` or ``None`` if nothing is scorable. Pure.
    """
    scorable = [r for r in _settled(records) if r.get("price") is not None]
    n = len(scorable)
    if n == 0:
        return None
    o_sum = 0.0
    price_sum = 0.0
    for r in scorable:
        o = _outcome_o(r)
        price = float(r["price"])
        o_sum += o
        price_sum += price
    realized_winrate = o_sum / n
    mean_implied = price_sum / n
    return {
        "n": n,
        "realized_winrate": round(realized_winrate, 4),
        "mean_implied": round(mean_implied, 4),
        "edge_vs_book": round(realized_winrate - mean_implied, 4),
        "pnl_per_contract": round((o_sum - price_sum) / n, 4),
    }


def edge_vs_book(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """THE key metric: did my fills beat the book's implied probabilities?

    Returns::

        {
          "overall":     {n, realized_winrate, mean_implied, edge_vs_book,
                          pnl_per_contract} | None,
          "by_category": {label: {...}, ...},
          "by_side":     {label: {...}, ...},
        }

    ``overall`` is ``None`` when no settled record carries a price. Per-group
    tables include only groups with at least one scorable record. Each settled
    trade's ``price`` (dollars, 0–1) is treated as the book's implied
    probability for the side I bought — so a positive ``edge_vs_book`` means my
    side won MORE often than the price the (sharp) book charged me. Pure /
    deterministic.
    """
    settled = _settled(records)
    by_category: Dict[str, Dict[str, Any]] = {}
    by_side: Dict[str, Dict[str, Any]] = {}

    cat_groups: Dict[str, List[Dict[str, Any]]] = {}
    side_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in settled:
        cat_groups.setdefault(r.get("category") or "unknown", []).append(r)
        side_groups.setdefault(r.get("side") or "unknown", []).append(r)

    for label, group in cat_groups.items():
        stats = _edge_stats(group)
        if stats is not None:
            by_category[label] = stats
    for label, group in side_groups.items():
        stats = _edge_stats(group)
        if stats is not None:
            by_side[label] = stats

    return {
        "overall": _edge_stats(settled),
        "by_category": by_category,
        "by_side": by_side,
    }


# ---------------------------------------------------------------------------
# 4. Forward-only split — the out-of-sample honesty guard
# ---------------------------------------------------------------------------

def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into an aware datetime, or None.

    Tolerates the two shapes the journal/settlements actually emit: a trailing
    ``Z`` (UTC) and an explicit ``+00:00`` offset. Naive timestamps (no offset)
    are assumed UTC so forward/suspect comparisons stay total-orderable.
    Returns ``None`` for missing or unparseable values — the caller treats that
    as 'unknown', never as forward.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _settled_at(rec: Dict[str, Any]) -> Optional[datetime]:
    """When the market resolved, from the settlement timestamp if attached.

    Reconciliation copies the settlement straight into ``outcome``; depending on
    upstream shape the resolution time may surface as ``settled_time`` on the
    outcome or directly on the record. We check both, preferring the outcome.
    """
    outcome = rec.get("outcome") or {}
    return _parse_ts(outcome.get("settled_time")) or _parse_ts(rec.get("settled_time"))


def forward_only(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split SETTLED records into forward / suspect / unknown by resolution time.

    Only legitimate, claimable edge comes from trades whose market resolved
    *strictly after* I placed them — anything else risks acting on an already-
    determined (or about-to-resolve) outcome, the exact leakage that inflates
    every competitor's backtested 'edge'.

      - ``forward``  : settlement time  >  trade ``ts``  (out-of-sample, legit).
      - ``suspect``  : settlement time  <= trade ``ts`` (resolved at/before the
                       trade → cannot be a forward prediction).
      - ``unknown``  : trade ``ts`` or settlement time missing/unparseable —
                       we CANNOT confirm it was forward, so we refuse to count
                       it as such.

    Unsettled records (no ``outcome``) are excluded entirely — there is nothing
    to classify yet. Conservative by construction: a record only lands in
    ``forward`` when both timestamps parse AND the strict ordering holds. Pure /
    deterministic. Returns ``{forward, suspect, unknown}``.
    """
    forward: List[Dict[str, Any]] = []
    suspect: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []
    for rec in _settled(records):
        traded_at = _parse_ts(rec.get("ts"))
        resolved_at = _settled_at(rec)
        if traded_at is None or resolved_at is None:
            unknown.append(rec)
        elif resolved_at > traded_at:
            forward.append(rec)
        else:
            suspect.append(rec)
    return {"forward": forward, "suspect": suspect, "unknown": unknown}


# ---------------------------------------------------------------------------
# 5. Edge report — the honest, gated verdict
# ---------------------------------------------------------------------------

def _verdict(forward_records: List[Dict[str, Any]]) -> str:
    """One honest line. Claims edge ONLY on >=10 forward trades that beat book.

    Computed strictly over the FORWARD subset (the only legitimate evidence):
      - n < 10                         -> "INSUFFICIENT DATA"
      - n >= 10 and edge_vs_book > 0   -> "BEATING THE BOOK by X.X pts ..."
      - n >= 10 and edge_vs_book <= 0  -> "NO MEASURED EDGE ..."
    """
    n = len(forward_records)
    if n < _MIN_EDGE_N:
        return (
            f"INSUFFICIENT DATA (n={n} forward-settled trades; "
            f"need >={_MIN_EDGE_N} to assess edge honestly)"
        )
    stats = _edge_stats(forward_records)
    if stats is None or stats["n"] < _MIN_EDGE_N:
        # Forward trades exist but lack prices — cannot measure edge-vs-book.
        return (
            f"INSUFFICIENT DATA (n={n} forward trades but too few carry a "
            f"fill price to measure edge vs. book)"
        )
    edge = stats["edge_vs_book"]
    pts = edge * 100.0
    if edge > 0:
        return (
            f"BEATING THE BOOK by {pts:.1f} pts over {stats['n']} forward trades "
            f"(won {stats['realized_winrate']:.0%} vs. {stats['mean_implied']:.0%} "
            f"implied; ${stats['pnl_per_contract']:+.3f}/contract)"
        )
    return (
        f"NO MEASURED EDGE (n={stats['n']} forward trades; won "
        f"{stats['realized_winrate']:.0%} vs. {stats['mean_implied']:.0%} implied, "
        f"edge {pts:+.1f} pts)"
    )


def edge_report(records: List[Dict[str, Any]], date: str) -> Dict[str, Any]:
    """Assemble the full edge harness into one honest, deterministic report.

    ``date`` MUST be passed in (the CLI passes today's ISO date) — this function
    never calls ``datetime.now`` so its output stays reproducible.

    Brier, log-loss and the headline ``edge_vs_book`` / verdict are all computed
    over the FORWARD-settled subset only: that is the sole legitimate, out-of-
    sample evidence of edge. Counts for suspect/unknown are surfaced so the
    leakage exposure is visible, never hidden. The calibration table is reused
    verbatim from ``learnings.calibration_table``.

    Returns::

        {
          date, n_settled, n_forward, n_suspect, n_unknown,
          brier, log_loss,                 # over forward subset, {value, n}|None
          calibration_table: [...],        # over forward subset
          edge_vs_book: {overall, by_category, by_side},  # over forward subset
          verdict: "<one honest line>",
        }
    """
    split = forward_only(records)
    forward = split["forward"]
    return {
        "date": date,
        "n_settled": len(_settled(records)),
        "n_forward": len(forward),
        "n_suspect": len(split["suspect"]),
        "n_unknown": len(split["unknown"]),
        "brier": brier_score(forward),
        "log_loss": log_loss(forward),
        "calibration_table": calibration_table(forward),
        "edge_vs_book": edge_vs_book(forward),
        "verdict": _verdict(forward),
    }
