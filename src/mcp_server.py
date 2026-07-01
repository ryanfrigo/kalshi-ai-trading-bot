"""Kalshi MCP server — a THIN FastMCP wrapper over the existing in-process toolkit.

No official Kalshi MCP exists. This is the distribution layer: it lets any MCP
client (Claude Desktop, Claude Code, etc.) drive the SAME governor-gated tools the
CLI already exposes — ``brief``, ``settle``, ``learnings``, ``edge``, ``hunt`` and
the two mutating hands ``trade`` / ``close``.

TRUST MODEL (read this before connecting):
  - Runs LOCALLY on the user's own machine, against the user's own Kalshi key.
  - The Kalshi API key NEVER leaves the machine — it is read from the environment
    (``KALSHI_API_KEY`` + the private-key file) by ``KalshiClient``, exactly as the
    CLI does. This server opens no outbound channel of its own.
  - It adds NO new authority. Every tool calls an EXISTING function; the mutating
    tools route through ``place_guarded_order`` / ``close_position``, which enforce
    the risk governor (daily-loss + drawdown kill switch + manual halt) and the 10%
    per-position cap. There is no code path here that bypasses them.

SAFETY:
  - The two mutating tools (``trade``, ``close``) default to ``confirm=False`` →
    they return a DRY-RUN preview (the guard checks, the capped size) and place NO
    order. Only ``confirm=True`` places a live order — and even then it goes through
    the full guard stack. The governor halt / kill switch / 10% cap remain
    authoritative.

Thin wrappers only: ALL logic lives in ``src.agent.*`` / ``src.risk.*`` — this file
just instantiates ``KalshiClient`` and forwards. Do not reimplement logic here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("kalshi-trading")

# Annotation presets (MCP hints to the client about a tool's nature).
_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
_MUTATING = ToolAnnotations(destructiveHint=True, openWorldHint=True)


async def _with_client(fn):
    """Instantiate a KalshiClient (reads the local key), run ``fn(client)``, close.

    Importing KalshiClient lazily keeps module import cheap and offline-safe (so the
    server module can be imported / introspected without credentials present).
    """
    from src.clients.kalshi_client import KalshiClient

    client = KalshiClient()
    try:
        return await fn(client)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# READ-ONLY tools — surface the same JSON the CLI already produces.
# ---------------------------------------------------------------------------

@mcp.tool(annotations=_READ_ONLY)
async def brief() -> Dict[str, Any]:
    """Structured situational awareness: governor verdict, equity, positions,
    resting orders. Read-only; places no order. Mirrors ``cli.py brief``."""
    from src.agent.toolbelt import account_brief

    return await _with_client(lambda c: account_brief(c))


@mcp.tool(annotations=_READ_ONLY)
async def settle() -> Dict[str, Any]:
    """Pull Kalshi's authoritative settlements, append new ones to the local log,
    and summarize realized win-rate / P&L by side. Read-only vs. the account
    (writes only the local settlements log). Mirrors ``cli.py settle``."""
    from src.agent.settle import (
        fetch_settlements, settlement_pnl, record_settlements,
        summarize_settlements, load_settlements,
    )

    async def _run(c):
        raw = await fetch_settlements(c, limit=300)
        mine = [s for s in (settlement_pnl(r) for r in raw) if s]
        new = record_settlements(mine)
        summary = summarize_settlements(load_settlements())
        return {"new_this_run": len(new), "new": new, "realized_edge": summary}

    return await _with_client(_run)


@mcp.tool(annotations=_READ_ONLY)
async def learnings(dry: bool = True) -> Dict[str, Any]:
    """Reconcile settled outcomes into the journal and surface candidate learnings
    (calibration + per-category/side edge + deterministic flag rules). Defaults to
    ``dry=True`` (no journal/learnings writes) for a read-only preview; pass
    ``dry=False`` to persist reconciled outcomes + new learnings. Mirrors
    ``cli.py learnings --json``."""
    from datetime import date as _date

    from src.agent.settle import fetch_settlements, settlement_pnl
    from src.agent.journal import load_journal, write_journal, DEFAULT_JOURNAL_PATH
    from src.agent.learnings import (
        reconcile_outcomes, calibration_table, edge_breakdown,
        flag_rules, append_learnings, load_learnings, DEFAULT_LEARNINGS_PATH,
    )

    today = _date.today().isoformat()
    raw = await _with_client(lambda c: fetch_settlements(c, limit=300))
    settlements = [s for s in (settlement_pnl(r) for r in raw) if s]
    journal = load_journal()
    reconciled, newly = reconcile_outcomes(journal, settlements)
    if not dry and newly:
        write_journal(reconciled, DEFAULT_JOURNAL_PATH)

    calibration = calibration_table(reconciled)
    edges = edge_breakdown(reconciled)
    candidates = flag_rules(calibration, edges, today)
    if dry:
        existing = {(c.get("kind"), c.get("claim")) for c in load_learnings(DEFAULT_LEARNINGS_PATH)}
        new_learnings = [c for c in candidates if (c.get("kind"), c.get("claim")) not in existing]
    else:
        new_learnings = append_learnings(candidates, DEFAULT_LEARNINGS_PATH)

    return {
        "date": today,
        "dry": dry,
        "reconciled": newly,
        "settled_total": sum(1 for r in reconciled if r.get("outcome")),
        "calibration": calibration,
        "edges": edges,
        "candidates": candidates,
        "new_learnings": new_learnings,
    }


@mcp.tool(annotations=_READ_ONLY)
async def edge() -> Dict[str, Any]:
    """THE headline: prove (or disprove) edge vs. the sharp Kalshi book. Brier /
    log-loss, realized edge-vs-book, calibration, forward-only honesty split, and a
    gated VERDICT that refuses to claim edge on thin (n<10) or non-forward data.
    Read-only — measures, never trades or writes. Mirrors ``cli.py edge --json``."""
    from datetime import date as _date

    from src.agent.settle import fetch_settlements, settlement_pnl
    from src.agent.journal import load_journal
    from src.agent.learnings import reconcile_outcomes
    from src.agent.edge import edge_report

    today = _date.today().isoformat()
    raw = await _with_client(lambda c: fetch_settlements(c, limit=300))
    settlements = [s for s in (settlement_pnl(r) for r in raw) if s]
    journal = load_journal()
    reconciled, _ = reconcile_outcomes(journal, settlements)  # in-memory only
    return edge_report(reconciled, today)


@mcp.tool(annotations=_READ_ONLY)
async def policy() -> Dict[str, Any]:
    """The data-driven Edge Policy your settled record earns — the pre-trade gate
    that closes the self-improvement loop. Returns category/method BLOCKS (groups
    with >=5 settled trades and negative realized P&L), side WARNINGS (advisory —
    a losing side never disables the strategy), and est_prob HAIRCUTS
    (overconfident bands). Honest gating: a group with n<5 earns no rule.
    Read-only — derives fresh in memory and writes nothing (persisting the active
    gate is ``cli improve``). Mirrors ``cli.py policy --json``."""
    from datetime import date as _date

    from src.agent.settle import fetch_settlements, settlement_pnl
    from src.agent.journal import load_journal
    from src.agent.learnings import edge_breakdown, calibration_table
    from src.agent.policy import build_settled_records, derive_policy

    today = _date.today().isoformat()
    raw = await _with_client(lambda c: fetch_settlements(c, limit=300))
    settlements = [s for s in (settlement_pnl(r) for r in raw) if s]
    journal = load_journal()
    records = build_settled_records(journal, settlements)
    return derive_policy(edge_breakdown(records), calibration_table(records), date=today)


@mcp.tool(annotations=_READ_ONLY)
async def status() -> Dict[str, Any]:
    """Portfolio balance, position value, and active event positions. Read-only.
    Mirrors ``cli.py status`` as structured JSON."""

    async def _run(c):
        bal = await c.get_balance()
        cash = (bal.get("balance", 0) or 0) / 100.0
        pv = (bal.get("portfolio_value", 0) or 0) / 100.0
        positions = await c.get_positions()
        active = [
            {
                "event": p.get("event_ticker"),
                "exposure": float(p.get("event_exposure_dollars", "0") or 0),
                "cost": float(p.get("total_cost_dollars", "0") or 0),
                "realized_pnl": float(p.get("realized_pnl_dollars", "0") or 0),
                "fees": float(p.get("fees_paid_dollars", "0") or 0),
            }
            for p in positions.get("event_positions", [])
            if float(p.get("event_exposure_dollars", "0") or 0) > 0
        ]
        return {
            "cash": round(cash, 2),
            "portfolio": round(pv, 2),
            "equity": round(cash + pv, 2),
            "active_positions": active,
        }

    return await _with_client(_run)


@mcp.tool(annotations=_READ_ONLY)
async def scores() -> List[Dict[str, Any]]:
    """Current category scores, win rates, ROI, and allocation limits from the
    scoring system (score < 30 = BLOCKED). Read-only. Mirrors ``cli.py scores``."""
    from src.strategies.category_scorer import CategoryScorer

    scorer = CategoryScorer()
    await scorer.initialize()
    return await scorer.get_all_scores()


@mcp.tool(annotations=_READ_ONLY)
async def history(limit: int = 50) -> Dict[str, Any]:
    """Closed-trade history: overall win-rate/P&L plus recent trades from the local
    trading database. Read-only. Mirrors ``cli.py history``."""
    from pathlib import Path

    import aiosqlite

    db_path = Path(__file__).resolve().parent.parent / "trading_system.db"
    if not db_path.exists():
        return {"error": "no trading database found", "trades": []}

    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT COUNT(*) total, SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) wins, "
            "SUM(pnl) total_pnl FROM trade_logs"
        )
        ov = await cur.fetchone()
        cur = await db.execute(
            "SELECT market_id, side, entry_price, exit_price, quantity, pnl, "
            "entry_timestamp, strategy FROM trade_logs ORDER BY entry_timestamp DESC LIMIT ?",
            (int(limit),),
        )
        trades = [dict(r) for r in await cur.fetchall()]

    total = (ov["total"] if ov else 0) or 0
    wins = (ov["wins"] if ov else 0) or 0
    pnl = (ov["total_pnl"] if ov else 0.0) or 0.0
    return {
        "total_trades": total,
        "win_rate": round(wins / total, 4) if total else None,
        "total_pnl": round(pnl, 2),
        "recent": trades,
    }


@mcp.tool(annotations=_READ_ONLY)
async def hunt(top: int = 25, books: bool = True) -> Dict[str, Any]:
    """Broad live-book edge-candidate scan across the open Kalshi universe. Returns
    longshot-NO and directional shortlists, enriched with LIVE orderbook prices when
    ``books=True``. RAW RESEARCH MATERIAL, not a buy list. Read-only. Mirrors
    ``scripts/hunt_candidates.py``."""
    from datetime import datetime, timezone

    from scripts.hunt_candidates import fetch_events, scan, enrich

    async def _run(c):
        now = datetime.now(timezone.utc)
        events = await fetch_events(c)
        longshot_no, directional = scan(events, now)
        ls_top, dir_top = longshot_no[:top], directional[:top]
        if books:
            await enrich(c, ls_top)
            await enrich(c, dir_top)
        return {
            "scanned_events": len(events),
            "longshot_no": ls_top,
            "directional": dir_top,
        }

    return await _with_client(_run)


# ---------------------------------------------------------------------------
# MUTATING tools — dry-run by default; route through the guarded functions.
# ---------------------------------------------------------------------------

@mcp.tool(annotations=_MUTATING)
async def trade(
    ticker: str,
    side: str,
    count: int,
    price: Optional[float] = None,
    est_prob: Optional[float] = None,
    rationale: str = "",
    category: str = "",
    max_position_pct: float = 0.10,
    confirm: bool = False,
    override_policy: bool = False,
) -> Dict[str, Any]:
    """Place ONE guarded, journaled order — the agent's hands.

    SAFETY: defaults to a DRY-RUN preview (``confirm=False``) — it runs the full
    guard stack (risk-governor halt check, Edge Policy gate, market-tradeable
    check, 1..99c price sanity, the 10% per-position + cash size cap) and returns
    what WOULD happen, placing NO order. Only ``confirm=True`` places a live
    order, and it still goes through the same guards via ``place_guarded_order``.
    The governor / kill switch / 10% cap are authoritative; this tool adds no path
    around them. ``override_policy=True`` overrides an Edge Policy BLOCK (recorded
    in the journal) — the same full-authority escape hatch as ``cli trade
    --override-policy``.
    """
    from src.agent.toolbelt import place_guarded_order

    return await _with_client(lambda c: place_guarded_order(
        c, ticker=ticker, side=side, count=count, price=price,
        rationale=rationale, est_prob=est_prob, category=category,
        max_position_pct=max_position_pct, dry=not confirm,
        override_policy=override_policy,
    ))


@mcp.tool(annotations=_MUTATING)
async def close(
    ticker: str,
    count: Optional[int] = None,
    price: Optional[float] = None,
    rationale: str = "close position",
    confirm: bool = False,
) -> Dict[str, Any]:
    """Close (sell) an existing position with a marketable limit at the bid.

    SAFETY: defaults to a DRY-RUN preview (``confirm=False``) — reads the live
    holding and returns the sell it WOULD place, placing NO order. Only
    ``confirm=True`` places the live sell via ``close_position``. Selling reduces
    risk, so it is permitted even when the governor is halted (mirroring the CLI),
    but it still passes through ``close_position`` — no bypass.
    """
    from src.agent.toolbelt import close_position

    return await _with_client(lambda c: close_position(
        c, ticker=ticker, count=count, price=price,
        rationale=rationale, dry=not confirm,
    ))


def main() -> None:
    """Run the MCP server over stdio (the transport MCP clients launch)."""
    mcp.run()


if __name__ == "__main__":
    main()
