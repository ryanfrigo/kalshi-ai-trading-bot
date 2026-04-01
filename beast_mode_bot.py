#!/usr/bin/env python3
"""
Kalshi AI Trading Bot — Multi-Model AI Ensemble 🚀

Main entry point for the Unified Advanced Trading System.
Five frontier LLMs debate every trade — positions only open when they agree.
All models route through OpenRouter (single API key).

Ensemble roster (configured in src/config/settings.py):
- Claude Sonnet 4.5 (OpenRouter) — Lead Analyst     30%
- Gemini 3.1 Pro (OpenRouter)    — Forecaster        30%
- GPT-5.4 (OpenRouter)           — Risk Manager      20%
- DeepSeek V3.2 (OpenRouter)     — Bull Researcher   10%
- Grok 4.1 Fast (OpenRouter)     — Bear Researcher   10%

Usage:
    python beast_mode_bot.py              # Paper trading mode (default)
    python beast_mode_bot.py --live       # Live trading mode
    python beast_mode_bot.py --dashboard  # Live dashboard mode

Recommended: use cli.py for the unified CLI interface.
    python cli.py run                     # AI ensemble (default)
    python cli.py run --safe-compounder   # Conservative math-only mode
"""

import asyncio
import argparse
import time
import signal
from datetime import datetime, timedelta
from typing import Optional

from src.jobs.trade import run_trading_job
from src.jobs.ingest import run_ingestion
from src.jobs.track import run_tracking
from src.jobs.evaluate import run_evaluation
from src.utils.logging_setup import setup_logging, get_trading_logger
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.model_router import ModelRouter
from src.config.settings import settings

# Import Beast Mode components
from src.strategies.unified_trading_system import run_unified_trading_system, TradingSystemConfig
from beast_mode_dashboard import BeastModeDashboard


class BeastModeBot:
    """
    Beast Mode Trading Bot - Advanced Multi-Strategy Trading System 🚀
    
    This bot orchestrates all advanced strategies:
    1. Market Making (spread profits)
    2. Directional Trading (AI predictions with portfolio optimization)
    3. Arbitrage Detection (future feature)
    
    Features:
    - Unlimited market deadlines with dynamic exits
    - Cost controls and budget management
    - Real-time performance monitoring
    - Risk management and rebalancing
    """
    
    def __init__(self, live_mode: bool = False, dashboard_mode: bool = False):
        self.live_mode = live_mode
        self.dashboard_mode = dashboard_mode
        self.logger = get_trading_logger("beast_mode_bot")
        self.shutdown_event = asyncio.Event()
        
        # Set live trading in settings
        settings.trading.live_trading_enabled = live_mode
        settings.trading.paper_trading_mode = not live_mode
        
        # Add detailed logging for debugging
        self.logger.info(
            f"🚀 Beast Mode Bot initialized - "
            f"Mode: {'LIVE TRADING' if live_mode else 'PAPER TRADING'}"
        )
        self.logger.info(f"📊 CLI live_mode parameter: {live_mode}")
        self.logger.info(f"⚙️ settings.trading.live_trading_enabled: {settings.trading.live_trading_enabled}")
        self.logger.info(f"⚙️ settings.trading.paper_trading_mode: {settings.trading.paper_trading_mode}")
        
        if live_mode:
            self.logger.warning("⚠️ LIVE TRADING MODE ENABLED - REAL MONEY WILL BE USED")
            self.logger.warning("⚠️ All orders will be placed on the Kalshi exchange")
        else:
            self.logger.info("📝 Paper trading mode - orders will be simulated")

    async def run_dashboard_mode(self):
        """Run in live dashboard mode with real-time updates."""
        try:
            self.logger.info("🚀 Starting Beast Mode Dashboard Mode")
            dashboard = BeastModeDashboard()
            await dashboard.show_live_dashboard()
        except KeyboardInterrupt:
            self.logger.info("👋 Dashboard mode stopped")
        except Exception as e:
            self.logger.error(f"Error in dashboard mode: {e}")

    async def run_trading_mode(self):
        """Run the Beast Mode trading system with all strategies."""
        try:
            self.logger.info("🚀 BEAST MODE TRADING BOT STARTED")
            self.logger.info(f"📊 Trading Mode: {'LIVE' if self.live_mode else 'PAPER'}")
            self.logger.info(f"💰 Daily AI Budget: ${settings.trading.daily_ai_budget}")
            self.logger.info(f"⚡ Features: Market Making + Portfolio Optimization + Dynamic Exits")
            
            # 🚨 CRITICAL FIX: Initialize database FIRST and wait for completion
            self.logger.info("🔧 Initializing database...")
            db_manager = DatabaseManager()
            await self._ensure_database_ready(db_manager)
            self.logger.info("✅ Database initialization complete!")
            
            # Initialize other components
            kalshi_client = KalshiClient()

            # Initialize multi-model router — all models via OpenRouter
            self.model_router = ModelRouter(db_manager=db_manager)
            self.logger.info(
                "ModelRouter initialized for multi-model ensemble (all via OpenRouter)",
                ensemble_enabled=settings.ensemble.enabled,
            )
            
            # Small delay to ensure everything is ready
            await asyncio.sleep(1)
            
            # Start market ingestion first
            self.logger.info("🔄 Starting market ingestion...")
            # Run initial ingestion synchronously so trading cycles have data
            self.logger.info("📥 Running initial market ingestion (this may take 1-2 minutes)...")
            market_queue = asyncio.Queue()
            await run_ingestion(db_manager, market_queue)
            self.logger.info("✅ Initial market ingestion complete. Starting trading cycles.")
            
            # Then start background refresh loop
            ingestion_task = asyncio.create_task(self._run_market_ingestion(db_manager, kalshi_client))
            
            # Run remaining background tasks
            self.logger.info("🚀 Starting trading and monitoring tasks...")
            tasks = [
                ingestion_task,  # Already started
                asyncio.create_task(self._run_trading_cycles(db_manager, kalshi_client)),
                asyncio.create_task(self._run_position_tracking(db_manager, kalshi_client)),
                asyncio.create_task(self._run_performance_evaluation(db_manager))
            ]
            
            # Setup shutdown handler
            def signal_handler():
                self.logger.info("🛑 Shutdown signal received")
                self.shutdown_event.set()
                for task in tasks:
                    task.cancel()
            
            # Handle Ctrl+C gracefully
            for sig in [signal.SIGINT, signal.SIGTERM]:
                signal.signal(sig, lambda s, f: signal_handler())
            
            # Wait for shutdown or completion
            await asyncio.gather(*tasks, return_exceptions=True)
            
            await self.model_router.close()
            await kalshi_client.close()
            
            self.logger.info("🏁 Beast Mode Bot shut down gracefully")
            
        except Exception as e:
            self.logger.error(f"Error in Beast Mode Bot: {e}")
            raise

    async def _ensure_database_ready(self, db_manager: DatabaseManager):
        """Ensure database is fully initialized before starting any tasks."""
        try:
            # Initialize the database first to create all tables
            await db_manager.initialize()
            
            # Verify tables exist by checking one of them
            import aiosqlite
            async with aiosqlite.connect(db_manager.db_path) as db:
                await db.execute("SELECT COUNT(*) FROM positions LIMIT 1")
                await db.execute("SELECT COUNT(*) FROM markets LIMIT 1") 
                await db.execute("SELECT COUNT(*) FROM trade_logs LIMIT 1")
            
            self.logger.info("🎯 Database tables verified and ready")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    async def _run_market_ingestion(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        """Background task for market data ingestion."""
        while not self.shutdown_event.is_set():
            try:
                # Create a queue for market ingestion (though we're not using it in Beast Mode)
                market_queue = asyncio.Queue()
                # ✅ FIXED: Pass the shared database manager
                await run_ingestion(db_manager, market_queue)
                await asyncio.sleep(300)  # Run every 5 minutes (much slower to prevent 429s)
            except Exception as e:
                self.logger.error(f"Error in market ingestion: {e}")
                await asyncio.sleep(60)

    async def _run_trading_cycles(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        """Main Beast Mode trading cycles."""
        cycle_count = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Check daily AI cost limits before starting cycle
                if not await self._check_daily_ai_limits():
                    # Sleep until next day if limits reached
                    await self._sleep_until_next_day()
                    continue
                
                cycle_count += 1
                self.logger.info(f"🔄 Starting Beast Mode Trading Cycle #{cycle_count}")
                
                # Run the Beast Mode unified trading system
                results = await run_trading_job()
                
                if results and results.total_positions > 0:
                    self.logger.info(
                        f"✅ Cycle #{cycle_count} Complete - "
                        f"Positions: {results.total_positions}, "
                        f"Capital Used: ${results.total_capital_used:.0f} ({results.capital_efficiency:.1%}), "
                        f"Expected Return: {results.expected_annual_return:.1%}"
                    )
                else:
                    self.logger.info(f"📊 Cycle #{cycle_count} Complete - No new positions created")
                
                # Wait for next cycle (60 seconds)
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in trading cycle #{cycle_count}: {e}")
                await asyncio.sleep(60)

    async def _check_daily_ai_limits(self) -> bool:
        """
        Check if we should continue trading based on daily AI cost limits.
        Returns True if we can continue, False if we should pause.
        """
        if not settings.trading.enable_daily_cost_limiting:
            return True

        # Delegate to model_router which owns the daily tracker
        if hasattr(self, "model_router") and self.model_router is not None:
            can_proceed = await self.model_router.check_daily_limits()
            if not can_proceed:
                tracker = self.model_router.daily_tracker
                self.logger.warning(
                    "🚫 Daily AI cost limit reached - trading paused",
                    daily_cost=tracker.total_cost,
                    daily_limit=tracker.daily_limit,
                    requests_today=tracker.request_count,
                )
            return can_proceed

        return True

    async def _sleep_until_next_day(self):
        """Sleep until the next day (midnight) when daily limits reset."""
        if not settings.trading.sleep_when_limit_reached:
            # Just sleep for a normal cycle if sleep is disabled
            await asyncio.sleep(60)
            return
        
        # Calculate time until next day
        now = datetime.now()
        next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_next_day = (next_day - now).total_seconds()
        
        # Ensure we don't sleep for more than 24 hours (safety check)
        max_sleep = 24 * 60 * 60  # 24 hours
        sleep_time = min(seconds_until_next_day, max_sleep)
        
        if sleep_time > 0:
            hours_to_sleep = sleep_time / 3600
            self.logger.info(
                f"💤 Sleeping until next day to reset AI limits - {hours_to_sleep:.1f} hours"
            )
            
            # Sleep in chunks to allow for graceful shutdown
            chunk_size = 300  # 5 minutes per chunk
            while sleep_time > 0 and not self.shutdown_event.is_set():
                current_chunk = min(chunk_size, sleep_time)
                await asyncio.sleep(current_chunk)
                sleep_time -= current_chunk
            
            self.logger.info("🌅 Daily AI limits reset - resuming trading")
        else:
            # Safety fallback
            await asyncio.sleep(60)

    async def _run_position_tracking(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        """Background task for position tracking and exit strategies."""
        while not self.shutdown_event.is_set():
            try:
                # ✅ FIXED: Pass the shared database manager
                await run_tracking(db_manager)
                await asyncio.sleep(120)  # Check positions every 2 minutes (slower to reduce API load)
            except Exception as e:
                self.logger.error(f"Error in position tracking: {e}")
                await asyncio.sleep(30)

    async def _run_performance_evaluation(self, db_manager: DatabaseManager):
        """Background task for performance evaluation."""
        while not self.shutdown_event.is_set():
            try:
                await run_evaluation()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in performance evaluation: {e}")
                await asyncio.sleep(300)

    async def run(self):
        """Main entry point for Beast Mode Bot."""
        if self.dashboard_mode:
            await self.run_dashboard_mode()
        else:
            await self.run_trading_mode()


async def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Beast Mode Trading Bot 🚀 - Advanced Multi-Strategy Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python beast_mode_bot.py              # Paper trading mode
  python beast_mode_bot.py --live       # Live trading mode  
  python beast_mode_bot.py --dashboard  # Live dashboard mode
  python beast_mode_bot.py --live --log-level DEBUG  # Live mode with debug logs

Beast Mode Features:
  • Market Making (40% allocation) - Profit from spreads
  • Directional Trading (50% allocation) - AI predictions with portfolio optimization
  • Arbitrage Detection (10% allocation) - Cross-market opportunities
  • No time restrictions - Trade any deadline with dynamic exits
  • Kelly Criterion portfolio optimization
  • Real-time risk management and rebalancing
  • Cost controls and budget management
        """
    )
    
    parser.add_argument(
        "--live", 
        action="store_true", 
        help="Run in LIVE trading mode (default: paper trading)"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run in live dashboard mode for monitoring"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--reset-limits",
        action="store_true",
        help="Reset daily AI cost limits and exit (clears exhausted state)"
    )
    
    args = parser.parse_args()
    
    # Handle --reset-limits
    if args.reset_limits:
        import glob
        pkl_files = glob.glob("logs/daily_*usage*.pkl")
        if pkl_files:
            for f in pkl_files:
                os.remove(f)
                print(f"✅ Deleted {f}")
            print("🔄 Daily AI limits reset. Restart the bot to continue trading.")
        else:
            print("ℹ️  No daily limit files found — limits are already clean.")
        return
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Warn about live mode
    if args.live and not args.dashboard:
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED")
        print("💰 This will use real money and place actual trades!")
        print("🚀 LIVE TRADING MODE CONFIRMED")
    
    # Create and run Beast Mode Bot
    bot = BeastModeBot(live_mode=args.live, dashboard_mode=args.dashboard)
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Beast Mode Bot stopped by user")
    except Exception as e:
        print(f"❌ Beast Mode Bot error: {e}")
        raise 