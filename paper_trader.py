#!/usr/bin/env python3
"""
Paper Trader — Signal-only mode for the Kalshi AI Trading Bot.

Uses the same market scanning and AI analysis as the live bot, but instead of
placing real orders it logs every signal to SQLite.  A companion HTML dashboard
shows cumulative P&L, win rate, and individual signals.

Usage:
    python paper_trader.py                # Scan once, log signals, generate dashboard
    python paper_trader.py --settle       # Check settled markets and update outcomes
    python paper_trader.py --dashboard    # Regenerate the HTML dashboard only
    python paper_trader.py --loop         # Continuous scanning (Ctrl-C to stop)
    python paper_trader.py --stats        # Print stats to terminal
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timezone

from src.paper.tracker import (
    log_signal,
    settle_signal,
    get_pending_signals,
    get_all_signals,
    get_stats,
)
from src.paper.dashboard import generate_html
from src.config.settings import settings
from src.utils.logging_setup import setup_logging, get_trading_logger

logger = get_trading_logger("paper_trader")

DASHBOARD_OUT = os.path.join(os.path.dirname(__file__), "docs", "paper_dashboard.html")


# ---------------------------------------------------------------------------
# Scanning: reuse the existing ingestion + decision pipeline
# ---------------------------------------------------------------------------

async def scan_and_log():
    """
    Scan markets via the existing ingest pipeline, run ensemble decisions,
    and log any actionable signals to the paper-trading database.
    """
    from src.clients.kalshi_client import KalshiClient
    from src.clients.xai_client import XAIClient
    from src.utils.database import DatabaseManager
    from src.jobs.ingest import run_ingestion
    from src.jobs.decide import make_decision_for_market

    logger.info("📡 Scanning markets for paper trading signals…")

    kalshi = KalshiClient()
    db = DatabaseManager()
    await db.initialize()  # Ensure all tables exist before any DB operations
    xai = XAIClient(db_manager=db)

    # 1. Ingest fresh market data
    try:
        market_queue: asyncio.Queue = asyncio.Queue()
        await run_ingestion(db, market_queue)

        # Drain the queue to collect ingested markets
        markets = []
        while not market_queue.empty():
            markets.append(market_queue.get_nowait())

        if not markets:
            logger.info("No markets returned from ingestion.")
            return 0
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 0

    signals_logged = 0

    # 2. Run decision on each market
    for market in markets:
        try:
            market_id = market.get("ticker") or market.get("market_id", "")
            title = market.get("title", market_id)

            decision = await make_decision_for_market(
                market_data=market,
                kalshi_client=kalshi,
                xai_client=xai,
                db_manager=db,
            )

            if decision is None:
                continue

            action = decision.get("action", "skip")
            if action in ("skip", "hold", None):
                continue

            side = decision.get("side", "NO")
            confidence = decision.get("confidence", 0)
            limit_price = decision.get("limit_price", market.get("no_ask", 0))
            reasoning = decision.get("reasoning", "")

            # Only log signals with meaningful confidence edge
            if confidence < 0.55:
                continue

            signal_id = log_signal(
                market_id=market_id,
                market_title=title,
                side=side,
                entry_price=limit_price,
                confidence=confidence,
                reasoning=reasoning,
                strategy=decision.get("strategy", "directional"),
            )
            signals_logged += 1
            logger.info(
                f"📝 Signal #{signal_id}: {side} {title} @ {limit_price:.0%} "
                f"(conf={confidence:.0%}) — {reasoning[:60]}"
            )

        except Exception as e:
            logger.warning(f"Decision failed for market: {e}")
            continue

    logger.info(f"✅ Logged {signals_logged} paper signals")
    return signals_logged


# ---------------------------------------------------------------------------
# Settlement: check outcomes for pending signals
# ---------------------------------------------------------------------------

async def check_settlements():
    """Check Kalshi for settled markets and update signal outcomes."""
    from src.clients.kalshi_client import KalshiClient

    pending = get_pending_signals()
    if not pending:
        logger.info("No pending signals to settle.")
        return 0

    kalshi = KalshiClient()
    settled_count = 0

    for sig in pending:
        try:
            market = await kalshi.get_market(sig["market_id"])
            if not market:
                continue

            status = str(market.get("status") or "").strip().lower()
            result = str(market.get("result") or "").strip().lower()

            # Only determined/finalized are final (closed just means trading
            # stopped; disputed can still flip), same rule as src/jobs/track.py.
            if status not in ("determined", "finalized"):
                continue

            if result == "void":
                settle_signal(sig["id"], None, void=True)
                logger.info(f"🏁 Signal #{sig['id']} voided (refund): {sig['market_title']}")
                settled_count += 1
                continue
            if result not in ("yes", "no"):
                continue  # missing/empty/scalar result: never booked

            settlement_price = 1.0 if result == "yes" else 0.0
            settle_signal(sig["id"], settlement_price)
            logger.info(f"🏁 Signal #{sig['id']} settled ({result}): {sig['market_title']}")
            settled_count += 1

        except Exception as e:
            logger.warning(f"Settlement check failed for {sig['market_id']}: {e}")

    logger.info(f"✅ Settled {settled_count}/{len(pending)} pending signals")
    return settled_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_stats():
    stats = get_stats()
    print("\n📊 Paper Trading Stats")
    print("=" * 40)
    print(f"  Total signals:  {stats['total_signals']}")
    print(f"  Settled:        {stats['settled']}")
    print(f"  Pending:        {stats['pending']}")
    print(f"  Wins:           {stats['wins']}")
    print(f"  Losses:         {stats['losses']}")
    print(f"  Win rate (before fees):   {stats['win_rate']}%")
    print(f"  Total P&L (before fees):  ${stats['total_pnl']:.2f}")
    print(f"  Avg return (before fees): ${stats['avg_return']:.4f}")
    print(f"  Best trade:     ${stats['best_trade']:.2f}")
    print(f"  Worst trade:    ${stats['worst_trade']:.2f}")
    print("  Note: paper P&L excludes Kalshi trading fees.")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Paper Trader — Kalshi AI signal logger")
    parser.add_argument("--settle", action="store_true", help="Check settled markets")
    parser.add_argument("--dashboard", action="store_true", help="Regenerate HTML dashboard only")
    parser.add_argument("--stats", action="store_true", help="Print stats to terminal")
    parser.add_argument("--loop", action="store_true", help="Continuous scanning")
    parser.add_argument("--interval", type=int, default=900, help="Loop interval in seconds (default 15min)")
    args = parser.parse_args()

    setup_logging()

    if args.stats:
        print_stats()
        return

    if args.dashboard:
        generate_html(DASHBOARD_OUT)
        print(f"✅ Dashboard generated: {DASHBOARD_OUT}")
        return

    if args.settle:
        await check_settlements()
        generate_html(DASHBOARD_OUT)
        print(f"✅ Dashboard updated: {DASHBOARD_OUT}")
        return

    # Default: scan once (or loop)
    while True:
        await scan_and_log()
        await check_settlements()
        generate_html(DASHBOARD_OUT)
        logger.info(f"📊 Dashboard updated: {DASHBOARD_OUT}")

        if not args.loop:
            break

        logger.info(f"💤 Sleeping {args.interval}s until next scan…")
        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
