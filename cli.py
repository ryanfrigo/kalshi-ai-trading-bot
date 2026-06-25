#!/usr/bin/env python3
"""
Kalshi AI Trading Bot -- Unified CLI

Provides a single entry point for all bot operations:
    python cli.py run          Start the trading bot
    python cli.py dashboard    Launch the Streamlit monitoring dashboard
    python cli.py status       Show portfolio balance, positions, and P&L
    python cli.py backtest     Run backtests (placeholder)
    python cli.py health       Verify API connections, database, and configuration
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Start the trading bot (disciplined mode by default)."""
    from src.utils.logging_setup import setup_logging

    log_level = getattr(args, "log_level", "INFO")
    setup_logging(log_level=log_level)

    live = getattr(args, "live", False)
    paper = getattr(args, "paper", False)
    beast = getattr(args, "beast", False)
    disciplined = getattr(args, "disciplined", False)
    safe_compounder = getattr(args, "safe_compounder", False)

    if live and paper:
        print("Error: --live and --paper are mutually exclusive.")
        sys.exit(1)

    live_mode = live and not paper

    if live_mode:
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED")
        print("   This will use real money and place actual trades.")

    # --safe-compounder mode: edge-based NO-side only
    if safe_compounder:
        _run_safe_compounder(
            live_mode=live_mode,
            loop=getattr(args, "loop", False),
            interval=getattr(args, "interval", 300),
        )
        return

    # --beast mode: original aggressive settings (NOT default)
    if beast:
        print("⚠️  BEAST MODE: Aggressive settings enabled.")
        print("   WARNING: Aggressive settings with no guardrails. Use at your own risk.")
        from beast_mode_bot import BeastModeBot
        bot = BeastModeBot(live_mode=live_mode)
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\nTrading bot stopped by user.")
        return

    # DEFAULT: AI directional strategy with disciplined settings active.
    # Despite earlier README copy, this is a single-model OpenRouter call
    # per decision (with a fallback chain), not a parallel ensemble.
    print("🤖  AI DIRECTIONAL MODE (default)")
    print("   Single-model OpenRouter call per decision (fallback chain on error).")
    print("   Category scoring + portfolio guardrails active.")
    print("   Use --safe-compounder for conservative math-only mode.")
    print("   Use --beast to run without guardrails (not recommended).")

    from beast_mode_bot import BeastModeBot
    from src.strategies.category_scorer import CategoryScorer
    from src.strategies.portfolio_enforcer import PortfolioEnforcer

    # Apply disciplined settings overrides
    from src.config import settings as cfg
    cfg.settings.trading.min_confidence_to_trade = 0.45  # LOOSENED from 0.65 (approved 2026-03-29)
    cfg.settings.trading.max_position_size_pct = 3.0
    cfg.settings.trading.kelly_fraction = 0.25
    cfg.max_drawdown = 0.15
    cfg.max_sector_exposure = 0.30

    bot = BeastModeBot(live_mode=live_mode)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user.")


def _run_safe_compounder(
    live_mode: bool = False,
    loop: bool = False,
    interval: int = 300,
) -> None:
    """Run the Safe Compounder strategy.

    When ``loop`` is True, run the strategy repeatedly with ``interval``
    seconds between cycles until the user sends Ctrl-C.
    """
    from src.clients.kalshi_client import KalshiClient
    from src.strategies.safe_compounder import SafeCompounder

    print("🔒 SAFE COMPOUNDER MODE")
    print("   NO-side only | Edge-based | Near-certain outcomes")
    if not live_mode:
        print("   DRY RUN — no real orders will be placed")
    if loop:
        print(f"   Continuous mode — re-running every {interval}s. Ctrl-C to stop.")

    async def _run_once():
        client = KalshiClient()
        try:
            compounder = SafeCompounder(
                client=client,
                dry_run=not live_mode,
            )
            return await compounder.run()
        finally:
            await client.close()

    async def _run_forever():
        cycle = 0
        while True:
            cycle += 1
            print(f"\n──── Cycle {cycle} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ────")
            try:
                await _run_once()
            except Exception as exc:
                # One bad cycle shouldn't kill the loop. Log and keep going.
                print(f"Cycle {cycle} failed: {exc}. Continuing after {interval}s.")
            print(f"\n⏳ Sleeping {interval}s before next cycle...")
            await asyncio.sleep(interval)

    try:
        if loop:
            asyncio.run(_run_forever())
        else:
            asyncio.run(_run_once())
    except KeyboardInterrupt:
        print("\nSafe Compounder stopped by user.")


def cmd_daily(args: argparse.Namespace) -> None:
    """Bounded once-through daily run: kill switch -> governed trade -> snapshot.

    Unlike ``run`` (an infinite loop), ``daily`` executes a single cycle and
    exits — the right shape for a cron/launchd schedule. The risk governor runs
    FIRST: if the account is down past the daily-loss or drawdown limit (or a
    manual halt file exists), new buys are skipped (exits still allowed).
    """
    from src.utils.logging_setup import setup_logging

    setup_logging(log_level=getattr(args, "log_level", "INFO"))
    live = getattr(args, "live", False)

    async def _daily() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.strategies.safe_compounder import SafeCompounder
        from src.risk.risk_governor import RiskGovernor
        from src.data.collector import snapshot_account

        client = KalshiClient()
        try:
            gov = RiskGovernor(kalshi_client=client)
            decision = await gov.check()

            print("=" * 60)
            print(f"  DAILY RUN — {'LIVE' if live else 'DRY-RUN'} — {datetime.now():%Y-%m-%d %H:%M}")
            print("=" * 60)
            print(
                f"  Governor: {'🛑 HALTED' if decision.halted else '✅ OK'} | "
                f"equity ${decision.current_equity_cents/100:.2f} | "
                f"day P&L ${decision.daily_pnl_cents/100:+.2f} | "
                f"dd {decision.drawdown_pct:.1f}%"
            )
            for r in decision.reasons:
                print(f"   ! {r}")

            await snapshot_account(
                client, tag="daily_open",
                meta={"governor": decision.to_dict(), "live": live},
            )

            trade_live = live and not decision.halted
            if decision.halted and live:
                print("  Governor HALTED — skipping new buys (exits still allowed).")

            compounder = SafeCompounder(client=client, dry_run=not trade_live)
            results = await compounder.run()

            snap = await snapshot_account(
                client, tag="daily_close",
                meta={"results": results, "live": live, "halted": decision.halted},
            )

            print("=" * 60)
            print(
                f"  RESULT: orders={results.get('placed', 0)} "
                f"filled={results.get('filled', 0)} "
                f"deployed=${results.get('total_deployed', 0)/100:.2f} "
                f"errors={results.get('errors', 0)} | "
                f"equity ${snap['equity_cents']/100:.2f}"
            )
            print("=" * 60)
        finally:
            await client.close()

    try:
        asyncio.run(_daily())
    except KeyboardInterrupt:
        print("\nDaily run interrupted by user.")


def cmd_brief(args: argparse.Namespace) -> None:
    """Print a structured situational-awareness snapshot for the agent (JSON)."""
    import json
    from src.utils.logging_setup import setup_logging

    setup_logging(log_level="WARNING")

    async def _b() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.agent.toolbelt import account_brief

        client = KalshiClient()
        try:
            print(json.dumps(await account_brief(client), indent=2))
        finally:
            await client.close()

    asyncio.run(_b())


def cmd_trade(args: argparse.Namespace) -> None:
    """Place ONE guarded, journaled order — the agent's hands. Dry by default."""
    import json
    from src.utils.logging_setup import setup_logging

    setup_logging(log_level="WARNING")

    async def _t() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.agent.toolbelt import place_guarded_order

        client = KalshiClient()
        try:
            res = await place_guarded_order(
                client, ticker=args.ticker, side=args.side, count=args.count,
                price=args.price, type_=args.type, rationale=args.rationale or "",
                est_prob=args.est_prob, category=args.category or "",
                max_position_pct=args.max_pct, dry=not args.live,
            )
            print(json.dumps(res, indent=2))
        finally:
            await client.close()

    asyncio.run(_t())


def cmd_close(args: argparse.Namespace) -> None:
    """Close (sell) a single position with a marketable limit. Dry by default."""
    import json
    from src.utils.logging_setup import setup_logging

    setup_logging(log_level="WARNING")

    async def _c() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.agent.toolbelt import close_position

        client = KalshiClient()
        try:
            res = await close_position(
                client, ticker=args.ticker, count=args.count, price=args.price,
                rationale=args.rationale or "close position", dry=not args.live,
            )
            print(json.dumps(res, indent=2))
        finally:
            await client.close()

    asyncio.run(_c())


def cmd_settle(args: argparse.Namespace) -> None:
    """Record settled outcomes and report realized edge (win-rate, P&L)."""
    import json
    from src.utils.logging_setup import setup_logging

    setup_logging(log_level="WARNING")

    async def _s() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.agent.settle import (
            fetch_settlements, settlement_pnl, record_settlements,
            summarize_settlements, load_settlements,
        )

        client = KalshiClient()
        try:
            raw = await fetch_settlements(client, limit=300)
            mine = [s for s in (settlement_pnl(r) for r in raw) if s]
            new = record_settlements(mine)
            summary = summarize_settlements(load_settlements())
            print(json.dumps({"new_this_run": len(new), "new": new, "realized_edge": summary}, indent=2))
        finally:
            await client.close()

    asyncio.run(_s())


def cmd_learnings(args: argparse.Namespace) -> None:
    """Reconcile settled outcomes into the journal, then surface candidate learnings.

    The integrated LEARN step: pull live settlements, join them back into
    decision_journal.jsonl (filling each trade's outcome), print the calibration
    and per-category/side edge tables, run the deterministic flag-rules over those
    tables, and append any genuinely NEW candidate learnings to learnings.jsonl.

    Defaults to writing back reconciled outcomes + new learnings. Pass --dry for a
    read-only preview (no journal or learnings writes). Pass --json for machine
    output instead of the human tables.
    """
    import json
    from datetime import date as _date
    from src.utils.logging_setup import setup_logging

    setup_logging(log_level="WARNING")

    dry = getattr(args, "dry", False)
    as_json = getattr(args, "json", False)
    today = _date.today().isoformat()

    async def _l() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.agent.settle import fetch_settlements, settlement_pnl
        from src.agent.journal import load_journal, write_journal, DEFAULT_JOURNAL_PATH
        from src.agent.learnings import (
            reconcile_outcomes, calibration_table, edge_breakdown,
            flag_rules, append_learnings, load_learnings, DEFAULT_LEARNINGS_PATH,
        )

        client = KalshiClient()
        try:
            raw = await fetch_settlements(client, limit=300)
        finally:
            await client.close()

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

        if as_json:
            print(json.dumps({
                "date": today,
                "dry": dry,
                "reconciled": newly,
                "settled_total": sum(1 for r in reconciled if r.get("outcome")),
                "calibration": calibration,
                "edges": edges,
                "candidates": candidates,
                "new_learnings": new_learnings,
            }, indent=2))
            return

        print("=" * 64)
        print(f"  LEARNINGS — {today}{'  (DRY: no writes)' if dry else ''}")
        print("=" * 64)
        print(f"  Reconciled this run: {newly} new outcome(s)")
        settled_n = sum(1 for r in reconciled if r.get("outcome"))
        print(f"  Settled journal trades: {settled_n} / {len(reconciled)}")
        print()

        print("  CALIBRATION (predicted vs. actual win-rate)")
        if calibration:
            print(f"  {'bucket':<14} {'n':>4} {'pred':>7} {'actual':>7}")
            print(f"  {'-'*14} {'-'*4} {'-'*7} {'-'*7}")
            for b in calibration:
                print(f"  {b['bucket']:<14} {b['n']:>4} {b['predicted']:>6.0%} {b['actual']:>6.0%}")
        else:
            print("  (no settled trades with est_prob yet)")
        print()

        print("  EDGE BREAKDOWN")
        for dim in ("by_category", "by_side", "by_method"):
            table = edges.get(dim, {})
            if not table:
                continue
            print(f"  {dim.replace('by_', 'by ')}:")
            print(f"    {'group':<18} {'n':>4} {'WR':>6} {'P&L':>9} {'entryEdge':>9} {'realEdge':>9}")
            for label, s in sorted(table.items()):
                wr = f"{s['win_rate']:.0%}" if s['win_rate'] is not None else "n/a"
                ee = f"{s['avg_entry_edge']:+.3f}" if s['avg_entry_edge'] is not None else "n/a"
                re_ = f"{s['realized_edge']:+.3f}" if s['realized_edge'] is not None else "n/a"
                print(f"    {label[:18]:<18} {s['n']:>4} {wr:>6} ${s['pnl']:>7.2f} {ee:>9} {re_:>9}")
        print()

        print("  CANDIDATE LEARNINGS")
        if not candidates:
            print("  (none flagged — no n>=5 group is losing and no band is overconfident)")
        for c in candidates:
            is_new = c in new_learnings
            tag = "NEW" if is_new else "seen"
            print(f"  [{tag}] ({c['confidence']}/{c['kind']}) {c['claim']}")
        if not dry:
            print()
            print(f"  Appended {len(new_learnings)} new learning(s) to learnings.jsonl")
        print("=" * 64)

    asyncio.run(_l())


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit monitoring dashboard."""
    import subprocess

    # Prefer the dedicated dashboard launch script if it exists.
    dashboard_script = Path(__file__).parent / "scripts" / "launch_dashboard.py"
    beast_dashboard = Path(__file__).parent / "scripts" / "beast_mode_dashboard.py"

    if dashboard_script.exists():
        subprocess.run([sys.executable, str(dashboard_script)], check=False)
    elif beast_dashboard.exists():
        # Fall back to running the dashboard module directly.
        from src.utils.logging_setup import setup_logging
        from beast_mode_bot import BeastModeBot

        setup_logging(log_level="INFO")
        bot = BeastModeBot(live_mode=False, dashboard_mode=True)
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\nDashboard stopped by user.")
    else:
        print("Error: No dashboard script found.")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Show current portfolio status: balance, positions, and P&L."""

    async def _status() -> None:
        from src.clients.kalshi_client import KalshiClient

        client = KalshiClient()
        try:
            # Fetch balance
            balance_resp = await client.get_balance()
            balance_cents = balance_resp.get("balance", 0)
            balance_usd = balance_cents / 100.0

            # Fetch positions — Kalshi API v2 returns event_positions and market_positions
            portfolio_value_cents = balance_resp.get("portfolio_value", 0)
            portfolio_value_usd = portfolio_value_cents / 100.0

            positions_resp = await client.get_positions()
            event_positions = positions_resp.get("event_positions", [])
            active_positions = [
                p for p in event_positions
                if float(p.get("event_exposure_dollars", "0")) > 0
            ]

            # Display
            print("=" * 56)
            print("  PORTFOLIO STATUS")
            print("=" * 56)
            print(f"  Available Cash:     ${balance_usd:>10,.2f}")
            print(f"  Position Value:     ${portfolio_value_usd:>10,.2f}")
            print(f"  Total Portfolio:    ${balance_usd + portfolio_value_usd:>10,.2f}")
            print(f"  Active Positions:   {len(active_positions):>10}")

            total_exposure = 0.0
            total_realized_pnl = 0.0
            total_fees = 0.0

            if active_positions:
                print()
                print(f"  {'Event':<30} {'Exposure':>10} {'Cost':>10} {'P&L':>10} {'Fees':>8}")
                print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

                for pos in active_positions:
                    ticker = pos.get("event_ticker", "???")
                    exposure = float(pos.get("event_exposure_dollars", "0"))
                    cost = float(pos.get("total_cost_dollars", "0"))
                    pnl = float(pos.get("realized_pnl_dollars", "0"))
                    fees = float(pos.get("fees_paid_dollars", "0"))
                    total_exposure += exposure
                    total_realized_pnl += pnl
                    total_fees += fees
                    print(
                        f"  {ticker:<30} ${exposure:>8.2f} ${cost:>8.2f} "
                        f"${pnl:>8.2f} ${fees:>6.2f}"
                    )

                print()
                print(f"  Total Exposure:     ${total_exposure:>10,.2f}")
                print(f"  Total Realized P&L: ${total_realized_pnl:>10,.2f}")
                print(f"  Total Fees Paid:    ${total_fees:>10,.2f}")

            print("=" * 56)
        finally:
            await client.close()

    try:
        asyncio.run(_status())
    except Exception as exc:
        print(f"Error fetching status: {exc}")
        sys.exit(1)


def cmd_scores(args: argparse.Namespace) -> None:
    """Show current category scores from the scoring system."""

    async def _scores():
        from src.strategies.category_scorer import CategoryScorer
        scorer = CategoryScorer()
        await scorer.initialize()
        scores = await scorer.get_all_scores()
        print(scorer.format_scores_table(scores))
        print()
        print("  Key: Score < 30 = BLOCKED | Alloc = max portfolio % allowed")
        print()

    try:
        asyncio.run(_scores())
    except Exception as exc:
        print(f"Error fetching scores: {exc}")
        sys.exit(1)


def cmd_history(args: argparse.Namespace) -> None:
    """Show trade history with category breakdown."""
    limit = getattr(args, "limit", 50)

    async def _history():
        import aiosqlite

        db_path = Path(__file__).parent / "trading_system.db"
        if not db_path.exists():
            print("No trading database found.")
            return

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row

            # Overall stats
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl
                FROM trade_logs
            """)
            overview = await cursor.fetchone()

            print("=" * 70)
            print("  TRADE HISTORY")
            print("=" * 70)
            if overview and overview["total"]:
                total = overview["total"]
                wins = overview["wins"] or 0
                pnl = overview["total_pnl"] or 0.0
                print(f"  Total Trades:  {total}")
                print(f"  Win Rate:      {wins/total*100:.1f}%")
                print(f"  Total P&L:     ${pnl:.2f}")
                print(f"  Avg per trade: ${(pnl/total):.2f}")
            print()

            # Category breakdown
            cursor = await db.execute("""
                SELECT
                    strategy as category,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl) as total_pnl
                FROM trade_logs
                GROUP BY strategy
                ORDER BY total_pnl DESC
            """)
            cats = await cursor.fetchall()

            if cats:
                print(f"  {'Category':<22} {'Trades':>7} {'WR':>6} {'P&L':>10}")
                print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*10}")
                for row in cats:
                    cat = row["category"] or "unknown"
                    t = row["trades"]
                    w = row["wins"] or 0
                    p = row["total_pnl"] or 0.0
                    wr = f"{w/t*100:.0f}%" if t > 0 else "n/a"
                    print(f"  {cat:<22} {t:>7} {wr:>6} ${p:>9.2f}")
                print()

            # Recent trades
            cursor = await db.execute(f"""
                SELECT market_id, side, entry_price, exit_price, quantity, pnl,
                       entry_timestamp, strategy
                FROM trade_logs
                ORDER BY entry_timestamp DESC
                LIMIT {limit}
            """)
            trades = await cursor.fetchall()

            if trades:
                print(f"  Recent {limit} trades:")
                print(f"  {'Market':<28} {'Side':>4} {'Entry':>6} {'Exit':>6} {'Qty':>4} {'P&L':>8} {'Category'}")
                print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*6} {'-'*4} {'-'*8} {'-'*12}")
                for t in trades:
                    ts = (t["entry_timestamp"] or "")[:10]
                    cat = t["strategy"] or ""
                    print(
                        f"  {t['market_id'][:28]:<28} {t['side']:>4} "
                        f"{t['entry_price']:>6.2f} {t['exit_price']:>6.2f} "
                        f"{t['quantity']:>4} ${t['pnl']:>7.2f}  {cat}"
                    )

            # Blocked trades summary
            cursor2 = await db.execute("""
                SELECT COUNT(*) FROM blocked_trades
            """)
            r2 = await cursor2.fetchone()
            if r2 and r2[0]:
                print(f"\n  ⛔ {r2[0]} trades blocked by portfolio enforcer (use 'python cli.py health' for details)")

            print("=" * 70)

    try:
        asyncio.run(_history())
    except Exception as exc:
        print(f"Error fetching history: {exc}")
        sys.exit(1)


def cmd_close_all(args: argparse.Namespace) -> None:
    """Place sell orders to close every open position on Kalshi.

    Use this AFTER stopping the bot (Ctrl-C). It queries Kalshi directly
    rather than the local DB, so it works even if local state is stale.
    Each position gets a limit sell at the current best bid for its side
    — marketable, but no guaranteed fill if the book is thin.
    """
    import uuid

    auto_yes = getattr(args, "yes", False)
    live_mode = getattr(args, "live", False)

    print("=" * 56)
    print("  CLOSE ALL POSITIONS")
    print("=" * 56)
    if not live_mode:
        print("  DRY RUN — no orders will be sent. Pass --live to actually sell.")
    print()
    print("  WARNING: this places sell orders at the current best bid.")
    print("  You may realize a loss on positions trading below entry.")
    print("  Stop the bot first (Ctrl-C) before running this command.")
    print()

    if live_mode and not auto_yes:
        confirm = input("  Type 'CLOSE ALL' to proceed: ").strip()
        if confirm != "CLOSE ALL":
            print("  Aborted.")
            return

    async def _close() -> None:
        from src.clients.kalshi_client import KalshiClient
        client = KalshiClient()
        try:
            positions_resp = await client.get_positions()
            market_positions = [
                p for p in positions_resp.get("market_positions", [])
                if p.get("position", 0) != 0
            ]

            if not market_positions:
                print("  No open positions on Kalshi.")
                return

            print(f"  Found {len(market_positions)} open position(s).")
            print()

            placed = 0
            failed = 0
            for pos in market_positions:
                ticker = pos["ticker"]
                contracts = pos["position"]            # signed: + YES, - NO
                side = "yes" if contracts > 0 else "no"
                quantity = abs(contracts)

                try:
                    book_resp = await client.get_orderbook(ticker, depth=1)
                    book = book_resp.get("orderbook", {})
                    side_bids = book.get(side, [])
                    if not side_bids:
                        print(f"  ⚠️  {ticker}: no {side.upper()} bids in book — skipping")
                        failed += 1
                        continue
                    # Kalshi orderbook bid entries are [price_cents, count].
                    # Best bid is the highest price.
                    best_bid_cents = max(int(level[0]) for level in side_bids)
                except Exception as exc:
                    print(f"  ❌ {ticker}: orderbook fetch failed — {exc}")
                    failed += 1
                    continue

                if not live_mode:
                    print(
                        f"  [DRY] would sell {quantity} {side.upper()} of {ticker} "
                        f"at {best_bid_cents}¢ (~${best_bid_cents * quantity / 100:.2f})"
                    )
                    placed += 1
                    continue

                order_params = {
                    "ticker": ticker,
                    "client_order_id": str(uuid.uuid4()),
                    "side": side,
                    "action": "sell",
                    "count": quantity,
                    "type_": "limit",
                }
                if side == "yes":
                    order_params["yes_price"] = best_bid_cents
                else:
                    order_params["no_price"] = best_bid_cents

                try:
                    resp = await client.place_order(**order_params)
                    if resp and "order" in resp:
                        print(
                            f"  ✅ {ticker}: sell {quantity} {side.upper()} "
                            f"@ {best_bid_cents}¢ — order_id={resp['order'].get('order_id', '?')}"
                        )
                        placed += 1
                    else:
                        print(f"  ❌ {ticker}: unexpected response {resp}")
                        failed += 1
                except Exception as exc:
                    print(f"  ❌ {ticker}: order failed — {exc}")
                    failed += 1

            print()
            print(f"  Placed: {placed} | Failed: {failed}")
            print()
            print("  Sell orders are limit-priced — they may rest unfilled if the book")
            print("  moves. Check Kalshi or `python cli.py status` after a minute.")
        finally:
            await client.close()

    try:
        asyncio.run(_close())
    except Exception as exc:
        print(f"  Error: {exc}")
        sys.exit(1)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run backtests (placeholder)."""
    print("=" * 56)
    print("  BACKTESTING")
    print("=" * 56)
    print()
    print("  Backtesting engine coming soon.")
    print()
    print("  Planned features:")
    print("    - Historical market replay")
    print("    - Strategy parameter optimization")
    print("    - Walk-forward analysis")
    print("    - Monte Carlo simulation")
    print()
    print("=" * 56)


def cmd_health(args: argparse.Namespace) -> None:
    """Run health checks on configuration, API, and database."""

    checks_passed = 0
    checks_failed = 0

    def ok(label: str, detail: str = "") -> None:
        nonlocal checks_passed
        checks_passed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [PASS] {label}{suffix}")

    def fail(label: str, detail: str = "") -> None:
        nonlocal checks_failed
        checks_failed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [FAIL] {label}{suffix}")

    print("=" * 56)
    print("  HEALTH CHECK")
    print("=" * 56)
    print()

    # 1. .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        ok(".env file exists")
    else:
        fail(".env file missing", "copy env.template to .env and fill in keys")

    # 2. Required environment variables
    from dotenv import load_dotenv
    load_dotenv()

    for var, placeholder in (
        ("KALSHI_API_KEY", "your_kalshi_api_key_here"),
        ("OPENROUTER_API_KEY", "your_openrouter_api_key_here"),
    ):
        val = os.getenv(var, "")
        if val and val not in ("", placeholder):
            ok(f"{var} is set")
        else:
            fail(f"{var} is missing or placeholder")

    # 3. Kalshi API connection
    async def _check_api() -> None:
        from src.clients.kalshi_client import KalshiClient
        client = KalshiClient()
        try:
            balance_resp = await client.get_balance()
            balance_usd = balance_resp.get("balance", 0) / 100.0
            ok("Kalshi API connection", f"balance=${balance_usd:,.2f}")
        except Exception as exc:
            msg = str(exc)
            fail("Kalshi API connection", msg)
            if "401" in msg or "authentication" in msg.lower():
                print(
                    "         A 401 from Kalshi almost always means one of:\n"
                    "           - KALSHI_API_KEY in .env doesn't match the API key ID on Kalshi\n"
                    "           - The private key file (default: kalshi_private_key.pem) is the\n"
                    "             wrong key for that API key, or its path is wrong\n"
                    "           - The API key was created on the Kalshi demo env but you're\n"
                    "             pointing at production (or vice versa)\n"
                    "         Re-download the key pair from Kalshi and verify both KALSHI_API_KEY\n"
                    "         and KALSHI_PRIVATE_KEY_PATH (if set) point to the matching pair."
                )
        finally:
            await client.close()

    try:
        asyncio.run(_check_api())
    except Exception as exc:
        fail("Kalshi API connection", str(exc))

    # 4. Database
    db_path = Path(__file__).parent / "trading_system.db"
    try:
        import aiosqlite

        async def _check_db() -> None:
            from src.utils.database import DatabaseManager
            db_manager = DatabaseManager()
            await db_manager.initialize()
            ok("Database initialization", str(db_path))

        asyncio.run(_check_db())
    except Exception as exc:
        fail("Database initialization", str(exc))

    # 5. Python version
    if sys.version_info >= (3, 12):
        ok("Python version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    else:
        fail("Python version", f"requires >=3.12, found {sys.version}")

    # Summary
    print()
    total = checks_passed + checks_failed
    print(f"  {checks_passed}/{total} checks passed")
    if checks_failed:
        print(f"  {checks_failed} issue(s) need attention")
    else:
        print("  All systems operational.")
    print("=" * 56)

    if checks_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kalshi-bot",
        description="Kalshi AI Trading Bot -- Multi-model AI trading for prediction markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py run                      Start AI Ensemble mode (default, paper)\n"
            "  python cli.py run --live               AI Ensemble with real capital\n"
            "  python cli.py run --safe-compounder    Safe Compounder: conservative, math-only\n"
            "  python cli.py run --safe-compounder --live  Safe Compounder live\n"
            "  python cli.py run --beast              Beast mode (aggressive, not recommended)\n"
            "  python cli.py scores                   Show category scores\n"
            "  python cli.py history                  Show trade history + category breakdown\n"
            "  python cli.py status                   Check portfolio balance and positions\n"
            "  python cli.py health                   Verify all connections and config\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = subparsers.add_parser(
        "run",
        help="Start the trading bot (disciplined mode by default)",
        description=(
            "Launch one of the example trading strategies. Default is the AI "
            "directional strategy: a single LLM call per market via OpenRouter "
            "(fallback chain on error), with category scoring and portfolio "
            "guardrails layered on top. Use --safe-compounder for the "
            "conservative math-only NO-side strategy. Use --beast to run "
            "without guardrails (not recommended). All three are starting "
            "points — fork them, tune them, replace them."
        ),
    )
    live_group = p_run.add_mutually_exclusive_group()
    live_group.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading with real capital (default: paper trading)",
    )
    live_group.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper-trading mode (no real orders)",
    )
    strategy_group = p_run.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--disciplined",
        action="store_true",
        default=True,
        help="Disciplined mode: category scoring + portfolio enforcement (DEFAULT)",
    )
    strategy_group.add_argument(
        "--beast",
        action="store_true",
        help="Beast mode: aggressive settings, no guardrails (not recommended)",
    )
    strategy_group.add_argument(
        "--safe-compounder",
        action="store_true",
        dest="safe_compounder",
        help="Safe Compounder: NO-side only, edge-based, near-certain outcomes",
    )
    p_run.add_argument(
        "--loop",
        action="store_true",
        help="Re-run the strategy continuously (only honored by --safe-compounder today)",
    )
    p_run.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between cycles when --loop is set (default: 300)",
    )
    p_run.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO)",
    )
    p_run.set_defaults(func=cmd_run)

    # --- daily ---
    p_daily = subparsers.add_parser(
        "daily",
        help="Bounded once-through daily run (governor + governed trade + snapshot)",
        description=(
            "Run a single governed trading cycle and exit — the right shape for a "
            "cron/launchd schedule. The risk governor (daily-loss + drawdown kill "
            "switch) runs first; new buys are skipped if halted. Defaults to dry-run; "
            "pass --live to place real orders."
        ),
    )
    dgrp = p_daily.add_mutually_exclusive_group()
    dgrp.add_argument("--live", action="store_true",
                      help="Place real orders (default: dry-run)")
    dgrp.add_argument("--paper", action="store_true",
                      help="Dry-run (no real orders)")
    p_daily.add_argument("--log-level", type=str, default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity (default: INFO)")
    p_daily.set_defaults(func=cmd_daily)

    # --- brief (agent situational awareness) ---
    p_brief = subparsers.add_parser(
        "brief",
        help="Structured situational awareness: governor, equity, positions, resting orders (JSON)",
        description="One-shot agent snapshot for /loop ticks. Read-only.",
    )
    p_brief.set_defaults(func=cmd_brief)

    # --- trade (agent's guarded order tool) ---
    p_trade = subparsers.add_parser(
        "trade",
        help="Place ONE guarded, journaled order (the agent's hands). Dry by default.",
        description=(
            "Place a single order through the full guard stack (risk governor, "
            "price sanity, position+cash caps) and journal the prediction "
            "(est-prob, edge, rationale) for calibration. Defaults to a dry-run "
            "preview; pass --live to actually place."
        ),
    )
    p_trade.add_argument("--ticker", required=True, help="Market ticker")
    p_trade.add_argument("--side", required=True, choices=["yes", "no"])
    p_trade.add_argument("--count", type=int, required=True, help="Intended contracts (auto-capped)")
    p_trade.add_argument("--price", type=float, default=None,
                         help="Limit price in dollars for the chosen side; default = current ask")
    p_trade.add_argument("--type", dest="type", default="limit", choices=["limit", "market"])
    p_trade.add_argument("--est-prob", dest="est_prob", type=float, default=None,
                         help="Your estimated probability the bought side wins (drives edge + calibration)")
    p_trade.add_argument("--rationale", default="", help="Why you're making this trade (journaled)")
    p_trade.add_argument("--category", default="", help="Market category (journaled)")
    p_trade.add_argument("--max-pct", dest="max_pct", type=float, default=0.10,
                         help="Max fraction of equity per position (default 0.10)")
    p_trade.add_argument("--live", action="store_true",
                         help="Actually place the order (default: dry-run preview)")
    p_trade.set_defaults(func=cmd_trade)

    # --- close (agent's guarded sell/exit tool) ---
    p_close = subparsers.add_parser(
        "close",
        help="Close (sell) ONE position with a marketable limit. Dry by default.",
        description=(
            "Sell the side actually held for a ticker (capped to held count) at "
            "the current bid, and journal it. Allowed even when the governor is "
            "halted (selling reduces risk). Defaults to dry-run; pass --live."
        ),
    )
    p_close.add_argument("--ticker", required=True, help="Market ticker to close")
    p_close.add_argument("--count", type=int, default=None,
                         help="Contracts to sell (default: all held)")
    p_close.add_argument("--price", type=float, default=None,
                         help="Limit sell price in dollars (default: current bid)")
    p_close.add_argument("--rationale", default="", help="Why you're closing (journaled)")
    p_close.add_argument("--live", action="store_true",
                         help="Actually place the sell (default: dry-run preview)")
    p_close.set_defaults(func=cmd_close)

    # --- settle (record realized outcomes, measure edge) ---
    p_settle = subparsers.add_parser(
        "settle",
        help="Record settled outcomes and report realized edge (win-rate, P&L)",
        description=(
            "Pull Kalshi's authoritative settlement records for positions held, "
            "append new ones to the local settlements log, and summarize realized "
            "win-rate and P&L by side. This is the learning-loop measurement."
        ),
    )
    p_settle.set_defaults(func=cmd_settle)

    # --- learnings (reconcile outcomes -> calibration/edge -> candidate learnings) ---
    p_learn = subparsers.add_parser(
        "learnings",
        help="Reconcile settled outcomes into the journal and surface candidate learnings",
        description=(
            "The integrated LEARN step. Pull live settlements, join them back into "
            "the decision journal (filling each trade's outcome), print the "
            "calibration table and per-category/side realized edge, run the "
            "deterministic flag-rules over those tables, and append any genuinely "
            "new candidate learnings to learnings.jsonl. Defaults to writing back "
            "reconciled outcomes + new learnings; pass --dry for a read-only "
            "preview. Pass --json for machine output."
        ),
    )
    p_learn.add_argument("--dry", action="store_true",
                         help="Read-only: do not write reconciled outcomes or new learnings")
    p_learn.add_argument("--json", action="store_true",
                         help="Emit machine-readable JSON instead of the human tables")
    p_learn.set_defaults(func=cmd_learnings)

    # --- scores ---
    p_scores = subparsers.add_parser(
        "scores",
        help="Show current category scores",
        description="Display all trading category scores, win rates, ROI, and allocation limits.",
    )
    p_scores.set_defaults(func=cmd_scores)

    # --- history ---
    p_history = subparsers.add_parser(
        "history",
        help="Show trade history with category breakdown",
        description="Display closed trade history grouped by category, win rate, and P&L.",
    )
    p_history.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of recent trades to show (default: 50)",
    )
    p_history.set_defaults(func=cmd_history)

    # --- dashboard ---
    p_dash = subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit monitoring dashboard",
        description="Open a real-time web dashboard showing portfolio performance, positions, risk metrics, and AI decision logs.",
    )
    p_dash.set_defaults(func=cmd_dashboard)

    # --- status ---
    p_status = subparsers.add_parser(
        "status",
        help="Show portfolio balance, positions, and P&L",
        description="Connect to the Kalshi API and display current account balance, open positions, and estimated portfolio value.",
    )
    p_status.set_defaults(func=cmd_status)

    # --- backtest ---
    p_bt = subparsers.add_parser(
        "backtest",
        help="Run backtests (coming soon)",
        description="Backtest trading strategies against historical market data. This feature is under development.",
    )
    p_bt.set_defaults(func=cmd_backtest)

    # --- health ---
    p_health = subparsers.add_parser(
        "health",
        help="Verify API connections, database, and configuration",
        description="Run a series of diagnostic checks: .env presence, API key configuration, Kalshi API connectivity, database initialization, and Python version.",
    )
    p_health.set_defaults(func=cmd_health)

    # --- close-all ---
    p_close = subparsers.add_parser(
        "close-all",
        help="Place limit sell orders to close every open position on Kalshi",
        description=(
            "Best-effort liquidation: query Kalshi for open positions and place "
            "a limit sell at the current best bid for each. Run this AFTER stopping "
            "the bot. Defaults to dry-run; pass --live to actually send orders."
        ),
    )
    p_close.add_argument(
        "--live",
        action="store_true",
        help="Actually place sell orders (default is a dry-run preview)",
    )
    p_close.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive 'CLOSE ALL' confirmation (dangerous)",
    )
    p_close.set_defaults(func=cmd_close_all)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
