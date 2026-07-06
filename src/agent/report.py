"""Public track-record renderer — the honest, auto-published face of the loop.

Turns the corpora the agent loop already persists (settlements, decision
journal, Edge Policy) plus an optional live account snapshot into one markdown
page, ``docs/TRACK_RECORD.md``. Losses included, always: the page exists to
*measure* edge in public, not to claim it.

Pure / deterministic: ``render_track_record`` never touches the network, the
filesystem, or the clock — every input (including ``date``) is passed in, so
the output is reproducible and testable. All IO lives in the CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["render_track_record"]


def _money(v: Any) -> str:
    """Format a float dollar amount with sign, e.g. ``+$3.10`` / ``-$310.08``."""
    f = float(v)
    sign = "-" if f < 0 else "+"
    return f"{sign}${abs(f):,.2f}"


def _pct(v: Any) -> str:
    return f"{float(v) * 100:.0f}%"


def _account_section(equity: Optional[Dict[str, Any]]) -> List[str]:
    if not equity:
        return [
            "## Account",
            "",
            "_Live account snapshot unavailable this run (rendered offline)._",
            "",
        ]
    gov = equity.get("governor") or {}
    lines = [
        "## Account (blended — operator + strategy)",
        "",
        "These are **whole-account** figures. This live account is *also* traded "
        "manually by its operator, so equity, drawdown, and day P&L blend the "
        "operator's discretionary positions with the autonomous strategy — a big "
        "manual sports bet can swing them far more than any strategy trade. "
        "**The strategy's own edge is measured below**, from its decision journal "
        "against settled reality, not from this blended equity.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Equity | ${float(equity.get('equity', 0)):,.2f} |",
        f"| Cash | ${float(equity.get('cash', 0)):,.2f} |",
        f"| Open positions | {len(equity.get('positions') or [])} |",
    ]
    if gov:
        lines += [
            f"| Drawdown from peak | {gov.get('drawdown_pct', 0)}% |",
            f"| Day P&L | {_money(gov.get('daily_pnl_cents', 0) / 100)} |",
            f"| Governor halted | {'YES' if gov.get('halted') else 'no'} |",
        ]
    lines.append("")
    return lines


def _verdict_section(edge: Dict[str, Any]) -> List[str]:
    def _metric(m: Optional[Dict[str, Any]]) -> str:
        return f"{m['value']:.4f} (n={m['n']})" if m else "n/a"

    return [
        "## The honest edge verdict (forward-only)",
        "",
        f"> **{edge.get('verdict', 'n/a')}**",
        "",
        "Only *forward-settled* trades — where the market resolved strictly "
        "after the trade — count as out-of-sample evidence. Everything else is "
        "quarantined, never silently counted.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Settled trades (journal) | {edge.get('n_settled', 0)} |",
        f"| Forward-settled (scorable) | {edge.get('n_forward', 0)} |",
        f"| Quarantined (suspect / unknown) | {edge.get('n_suspect', 0)} / {edge.get('n_unknown', 0)} |",
        f"| Brier score | {_metric(edge.get('brier'))} |",
        f"| Log loss | {_metric(edge.get('log_loss'))} |",
        "",
    ]


def _settlements_section(summary: Dict[str, Any]) -> List[str]:
    n = summary.get("n", 0)
    wr = summary.get("win_rate")
    lines = [
        "## Full account settlement history",
        "",
        "Every settlement Kalshi reports for this account — including the "
        "legacy mechanical-bot era whose losses motivated this toolkit. "
        "Published because a track record that hides its losses is marketing, "
        "not measurement.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Settled markets | {n} |",
        f"| Win rate | {_pct(wr) if wr is not None else 'n/a'} |",
        f"| Net realized P&L | {_money(summary.get('pnl', 0))} |",
        "",
    ]
    by_side = summary.get("by_side") or {}
    if by_side:
        lines += ["| Side held | n | Win rate | P&L |", "|---|---|---|---|"]
        for side, d in sorted(by_side.items()):
            side_n = d.get("n", 0)
            side_wr = (d.get("wins", 0) / side_n) if side_n else 0
            lines.append(
                f"| {side.upper()} | {side_n} | {_pct(side_wr)} | {_money(d.get('pnl', 0))} |"
            )
        lines.append("")
    return lines


def _calibration_section(edge: Dict[str, Any]) -> List[str]:
    table = edge.get("calibration_table") or []
    if not table:
        return [
            "## Calibration",
            "",
            "_No forward-settled journaled trades with predictions yet — the "
            "calibration curve appears once enough agent trades settle._",
            "",
        ]
    lines = [
        "## Calibration (predicted vs. actual)",
        "",
        "| est_prob band | n | Predicted | Actual win rate |",
        "|---|---|---|---|",
    ]
    for b in table:
        lines.append(
            f"| {b.get('bucket')} | {b.get('n')} | "
            f"{_pct(b.get('predicted'))} | {_pct(b.get('actual'))} |"
        )
    lines.append("")
    return lines


def _policy_section(policy: Optional[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Edge Policy — what the record currently blocks",
        "",
        "The pre-trade gate the settled record has *earned*: categories the "
        "account provably loses money in are auto-blocked before the next "
        "trade. The gate only ever tightens from evidence.",
        "",
    ]
    if not policy:
        lines += ["_No derived policy yet (run `python cli.py improve`)._", ""]
        return lines
    meta = policy.get("meta") or {}
    lines.append(
        f"Derived from **{meta.get('settled_n', 0)} settled trades** "
        f"on {meta.get('generated_date', '?')}."
    )
    lines.append("")
    blocks = policy.get("blocks") or []
    if blocks:
        lines += ["| Blocked | n | P&L |", "|---|---|---|"]
        for b in blocks:
            lines.append(
                f"| {b.get('dimension')}={b.get('label')} | {b.get('n')} | "
                f"{_money(b.get('pnl', 0))} |"
            )
    else:
        lines.append("_No blocks — no n≥5 group is net-negative._")
    lines.append("")
    haircuts = policy.get("haircuts") or []
    if haircuts:
        lines += ["Overconfidence haircuts:", ""]
        for h in haircuts:
            lines.append(
                f"- band {h.get('band')}: shrink to {_pct(h.get('shrink_to'))} "
                f"(gap {_pct(h.get('gap'))}, n={h.get('n')})"
            )
        lines.append("")
    return lines


def render_track_record(
    *,
    date: str,
    equity: Optional[Dict[str, Any]],
    settle_summary: Dict[str, Any],
    edge: Dict[str, Any],
    policy: Optional[Dict[str, Any]],
) -> str:
    """Render the public track-record page as a markdown string. PURE.

    ``date`` MUST be passed in (the CLI passes today's ISO date) — this
    function never calls ``datetime.now`` so its output stays reproducible.
    ``equity`` is the (optional) ``account_brief`` snapshot; ``settle_summary``
    is ``settle.summarize_settlements`` output; ``edge`` is
    ``edge.edge_report`` output; ``policy`` is the persisted Edge Policy or
    ``None``.
    """
    lines: List[str] = [
        "# Live Track Record",
        "",
        f"_Auto-generated by `python cli.py report` — last updated {date}._",
        "",
        "This is a real Kalshi account, traded autonomously, scored against "
        "Kalshi's authoritative settlements — **losses included**. The page is "
        "the repo's thesis in practice: prove your edge, don't claim it.",
        "",
    ]
    lines += _account_section(equity)
    lines += _verdict_section(edge)
    lines += _settlements_section(settle_summary)
    lines += _calibration_section(edge)
    lines += _policy_section(policy)
    lines += [
        "---",
        "",
        "**How this page is made:** every loop tick runs settle → learn → "
        "improve, persists the settled corpus, re-derives the Edge Policy, and "
        "re-renders this page from those files. See the README's "
        "[Prove Your Edge](../README.md#prove-your-edge-the-headline) and "
        "[Self-Improvement Loop](../README.md#the-self-improvement-loop) "
        "sections for the method, and `src/agent/edge.py` for the scoring "
        "code (pure, deterministic, forward-only gated).",
        "",
    ]
    return "\n".join(lines)
