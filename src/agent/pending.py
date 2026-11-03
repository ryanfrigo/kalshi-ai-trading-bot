"""Resolve journal records whose order POST outcome was unknown (status "pending").

READ-ONLY against Kalshi: only GET order lookups, never a place/cancel/amend.
Each pending record is looked up by client_order_id (the exit path's paged lookup:
ticker orders, then archived orders). The lookup either proves what happened or the
record stays pending; nothing is guessed.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from src.agent.journal import is_pending
from src.jobs.execute import _find_by_client_order_id

_TERMINAL_UNFILLED = ("canceled", "expired")
_TERMINAL = ("executed",) + _TERMINAL_UNFILLED


def _avg_fill_price(order: Dict[str, Any], filled: float) -> float:
    """(taker+maker fill cost)/filled in dollars, or 0.0 to fall back to the limit price.

    taker_fill_cost_dollars / maker_fill_cost_dollars are required fixed-point dollar
    strings in Kalshi's GetOrder spec
    (https://docs.kalshi.com/api-reference/orders/get-order). If either fails to
    parse as a finite number we fall back to the limit price.
    """
    try:
        cost = float(order["taker_fill_cost_dollars"]) + float(order["maker_fill_cost_dollars"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    avg = cost / filled
    return avg if math.isfinite(avg) and avg > 0 else 0.0


def _fill(order: Dict[str, Any]) -> float:
    """fill_count_fp as a finite float >= 0; raises when absent or unreadable."""
    v = float(order["fill_count_fp"])
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"bad fill_count_fp {v}")
    return v


def _resolve_one(rec: Dict[str, Any], found: Any) -> Tuple[Dict[str, Any], str]:
    """(record, change); change "" means leave pending. Pure."""
    if found is None:
        return dict(rec, voided=True, voided_reason="order never reached the book"), "voided (absent)"
    try:
        filled = _fill(found)
        oid = found.get("order_id")
        price = _avg_fill_price(found, filled) or float(
            found.get(f"{rec.get('side')}_price_dollars") or rec["price"])
    except (KeyError, TypeError, ValueError):
        return rec, ""
    if filled > 0:
        # A non-terminal order short of the intended count may fill more: keep it
        # pending so a later pass captures the rest.
        if found.get("status") not in _TERMINAL and filled < int(rec.get("count", 0)):
            return rec, ""
        if not oid or not math.isfinite(price):
            return rec, ""
        out = {k: v for k, v in rec.items() if k not in ("status", "order_outcome")}
        # fill_count_fp may be fractional ("2.50"): int only when integral.
        count = int(filled) if filled == int(filled) else filled
        out.update(order_id=oid, count=count, price=price)
        if count != rec.get("count", 0):
            out["original_count"] = rec.get("count", 0)
        return out, f"filled {count} (order {oid})"
    if found.get("status") in _TERMINAL_UNFILLED:
        return dict(rec, voided=True, voided_reason=f"order {found['status']} with zero fills"), \
            f"voided ({found['status']})"
    return rec, ""  # unfilled but not terminal (resting?): not settled yet


async def resolve_pending(records: List[Dict[str, Any]], kalshi_client: Any
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve pending records; returns ``(updated_records, changes)`` like reconcile_fills.

    Records are not mutated in place. A failed read or a page cap leaves the record
    pending. Persist with ``write_journal`` only when ``changes`` is non-empty.
    """
    out: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []
    for rec in records:
        coid = rec.get("client_order_id")
        if not is_pending(rec) or rec.get("voided") or not coid:
            out.append(rec)
            continue
        try:
            found = await _find_by_client_order_id(coid, rec["ticker"], kalshi_client)
        except Exception:
            out.append(rec)
            continue
        updated, change = _resolve_one(rec, found)
        out.append(updated)
        if change:
            changes.append({"ticker": rec.get("ticker"), "client_order_id": coid, "change": change})
    return out, changes
