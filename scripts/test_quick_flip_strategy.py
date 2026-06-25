#!/usr/bin/env python3
"""
Quick Flip Strategy Demo

Test the new quick flip scalping strategy that:
1. Buys low-priced contracts (1¢, 5¢, etc.)
2. Immediately places sell orders for quick profits (2¢, 10¢, etc.)
3. Uses AI to identify likely price movements
4. Manages multiple concurrent positions
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager
from src.utils.logging_setup import setup_logging
from experimental.strategies.quick_flip_scalping import (
    run_quick_flip_strategy, 
    QuickFlipConfig,
    QuickFlipScalpingStrategy
)


async def test_quick_flip_opportunities():
    """Test identifying quick flip opportunities without executing trades."""
    
    setup_logging()
    logger = logging.getLogger("quick_flip_test")
    
    logger.info("🎯 Testing Quick Flip Strategy - Opportunity Identification")
    
    # Initialize clients
    kalshi_client = KalshiClient()
    db_manager = DatabaseManager()
    await db_manager.initialize()
    xai_client = XAIClient(db_manager=db_manager)  # Pass db_manager for LLM logging
    
    try:
        # Create strategy instance
        config = QuickFlipConfig(
            min_entry_price=1,      # Look for 1¢ opportunities  
            max_entry_price=10,     # Up to 10¢ entries
            min_profit_margin=1.0,  # 100% minimum profit (1¢ → 2¢)
            max_position_size=50,   # Small test positions
            max_concurrent_positions=10,  # Limited for testing
            capital_per_trade=25.0, # $25 per trade for testing
            confidence_threshold=0.5,  # Lower threshold for testing
            max_hold_minutes=15     # Quick exit
        )
        
        strategy = QuickFlipScalpingStrategy(
            db_manager, kalshi_client, xai_client, config
        )
        
        # Get some markets to analyze
        markets = await db_manager.get_eligible_markets(volume_min=100, max_days_to_expiry=365)
        
        if not markets:
            logger.error("No markets available for testing")
            return
        
        logger.info(f"📊 Analyzing {len(markets)} markets for quick flip opportunities")
        
        # Identify opportunities (but don't execute)
        opportunities = await strategy.identify_quick_flip_opportunities(
            markets, available_capital=500.0  # $500 test capital
        )
        
        logger.info(f"🎯 Found {len(opportunities)} quick flip opportunities:")
        
        for i, opp in enumerate(opportunities[:5]):  # Show top 5
            logger.info(
                f"  {i+1}. {opp.market_title[:50]}..."
                f"\n     Entry: {opp.side} at {opp.entry_price}¢ → Exit at {opp.exit_price}¢"
                f"\n     Quantity: {opp.quantity} contracts"
                f"\n     Expected Profit: ${opp.expected_profit:.2f}"
                f"\n     Confidence: {opp.confidence_score:.1%}"
                f"\n     Reason: {opp.movement_indicator}"
                f"\n"
            )
        
        if opportunities:
            logger.info("✅ Quick flip opportunity identification successful!")
            logger.info("💡 This strategy would place immediate sell orders after buying")
            logger.info("⚡ Targets rapid 100%+ returns on low-priced contracts")
        else:
            logger.info("ℹ️ No opportunities found with current market conditions")
        
    except Exception as e:
        logger.error(f"Error in quick flip test: {e}")
        import traceback
        logger.error(f"Full error: {traceback.format_exc()}")
    
    finally:
        await db_manager.close()


async def test_quick_flip_full_strategy():
    """Test the full quick flip strategy execution (paper trading mode)."""
    
    setup_logging()
    logger = logging.getLogger("quick_flip_full_test")
    
    logger.info("🚀 Testing Full Quick Flip Strategy - PAPER MODE")
    
    # Initialize clients
    kalshi_client = KalshiClient()
    db_manager = DatabaseManager()
    await db_manager.initialize()
    xai_client = XAIClient(db_manager=db_manager)  # Pass db_manager for LLM logging
    
    try:
        # Configure for conservative testing
        config = QuickFlipConfig(
            min_entry_price=1,
            max_entry_price=8,      # Very low-priced entries only
            min_profit_margin=1.5,  # 150% minimum profit (1¢ → 2.5¢)
            max_position_size=20,   # Small positions
            max_concurrent_positions=5,  # Very limited
            capital_per_trade=20.0, # Small capital per trade
            confidence_threshold=0.7,  # High confidence required
            max_hold_minutes=10     # Quick exits
        )
        
        logger.info("📋 Quick Flip Configuration:")
        logger.info(f"  • Entry Range: {config.min_entry_price}¢ - {config.max_entry_price}¢")
        logger.info(f"  • Min Profit Margin: {config.min_profit_margin*100:.0f}%")
        logger.info(f"  • Max Position Size: {config.max_position_size} contracts")
        logger.info(f"  • Capital per Trade: ${config.capital_per_trade}")
        logger.info(f"  • Confidence Threshold: {config.confidence_threshold:.1%}")
        
        # Run the strategy
        results = await run_quick_flip_strategy(
            db_manager=db_manager,
            kalshi_client=kalshi_client,
            xai_client=xai_client,
            available_capital=100.0,  # $100 test capital
            config=config
        )
        
        logger.info("📊 Quick Flip Strategy Results:")
        logger.info(f"  • Opportunities Analyzed: {results.get('opportunities_analyzed', 0)}")
        logger.info(f"  • Positions Created: {results.get('positions_created', 0)}")
        logger.info(f"  • Sell Orders Placed: {results.get('sell_orders_placed', 0)}")
        logger.info(f"  • Capital Used: ${results.get('total_capital_used', 0):.2f}")
        logger.info(f"  • Expected Profit: ${results.get('expected_profit', 0):.2f}")
        logger.info(f"  • Failed Executions: {results.get('failed_executions', 0)}")
        
        if results.get('positions_created', 0) > 0:
            logger.info("✅ Quick flip strategy executed successfully!")
            logger.info("💡 In live mode, this would place real orders and immediate sell limits")
        else:
            logger.info("ℹ️ No trades executed - either no opportunities or all filtered out")
        
    except Exception as e:
        logger.error(f"Error in full strategy test: {e}")
        import traceback
        logger.error(f"Full error: {traceback.format_exc()}")
    
    finally:
        await db_manager.close()


async def demo_strategy_concept():
    """Demonstrate the strategy concept with examples."""
    
    logger = logging.getLogger("quick_flip_demo")
    
    logger.info("🎯 QUICK FLIP SCALPING STRATEGY CONCEPT")
    logger.info("=" * 60)
    
    logger.info("📈 STRATEGY OVERVIEW:")
    logger.info("  1. Find contracts priced at 1¢-20¢ (low cost, high % return potential)")
    logger.info("  2. Use AI to identify likely short-term price movement catalysts")
    logger.info("  3. Buy contracts at low prices")
    logger.info("  4. IMMEDIATELY place sell limit orders at 2x+ the price")
    logger.info("  5. Hold for maximum 30 minutes, then cut losses if needed")
    logger.info("")
    
    logger.info("💡 EXAMPLE SCENARIOS:")
    logger.info("  Scenario A: Buy YES at 1¢ → Sell at 2¢ = 100% return")
    logger.info("  Scenario B: Buy NO at 5¢ → Sell at 10¢ = 100% return") 
    logger.info("  Scenario C: Buy YES at 3¢ → Sell at 7¢ = 133% return")
    logger.info("")
    
    logger.info("⚡ ADVANTAGES:")
    logger.info("  • Low capital requirement per trade")
    logger.info("  • High percentage returns on small price movements")
    logger.info("  • Quick turnover - capital not tied up long-term")
    logger.info("  • Can run many concurrent positions")
    logger.info("  • Limited downside risk per position")
    logger.info("")
    
    logger.info("🛡️ RISK MANAGEMENT:")
    logger.info("  • AI confidence filtering (only high-confidence trades)")
    logger.info("  • Position size limits")
    logger.info("  • Time-based stop losses (30 min max hold)")
    logger.info("  • Capital allocation limits")
    logger.info("")
    
    logger.info("🚀 INTEGRATION:")
    logger.info("  • Added to unified trading system (20% capital allocation)")
    logger.info("  • Runs parallel with market making and directional strategies")
    logger.info("  • Uses existing order placement and tracking infrastructure")


if __name__ == "__main__":
    print("🎯 Quick Flip Scalping Strategy Test Suite")
    print("=" * 50)
    
    async def main():
        # Demo the concept
        await demo_strategy_concept()
        print()
        
        # Test opportunity identification
        await test_quick_flip_opportunities()
        print()
        
        # Test full strategy (paper mode)
        await test_quick_flip_full_strategy()
    
    asyncio.run(main()) 