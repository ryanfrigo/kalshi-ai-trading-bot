#!/usr/bin/env python3

import asyncio
import sys
import json
sys.path.append('.')

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager
from src.utils.market_prices import signed_position
import aiosqlite
from datetime import datetime, timedelta

async def run_quick_analysis():
    """Quick comprehensive analysis using Grok4."""
    
    print("🚀 QUICK TRADING PERFORMANCE ANALYSIS WITH GROK4")
    print("=" * 60)
    
    # Initialize clients
    kalshi_client = KalshiClient()
    xai_client = XAIClient()
    db = DatabaseManager()
    await db.initialize()
    
    try:
        # 1. Gather current portfolio data
        print("\n📊 Gathering portfolio data...")
        
        # Kalshi positions
        positions_response = await kalshi_client.get_positions()
        kalshi_positions = positions_response.get('market_positions', [])
        active_positions = [p for p in kalshi_positions if signed_position(p) != 0]
        
        # Balance
        balance_response = await kalshi_client.get_balance()
        available_cash = balance_response.get('balance', 0) / 100
        
        # 2. Get historical performance
        async with aiosqlite.connect(db.db_path) as database:
            # Trade stats
            cursor = await database.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    MIN(pnl) as worst_loss,
                    MAX(pnl) as best_win
                FROM trade_logs 
                WHERE pnl IS NOT NULL
            """)
            trade_stats = await cursor.fetchone()
            
            # Position distribution
            cursor = await database.execute("""
                SELECT status, COUNT(*) 
                FROM positions 
                GROUP BY status
            """)
            position_stats = dict(await cursor.fetchall())
            
            # Market patterns
            cursor = await database.execute("""
                SELECT 
                    SUBSTR(market_id, 1, 10) as market_prefix,
                    COUNT(*) as trade_count,
                    AVG(confidence) as avg_confidence
                FROM positions 
                GROUP BY market_prefix
                ORDER BY trade_count DESC
                LIMIT 5
            """)
            market_patterns = await cursor.fetchall()
        
        # 3. Prepare analysis prompt for Grok4
        analysis_prompt = f"""
You are an expert quantitative trading analyst reviewing a Kalshi prediction market trading system.

CURRENT PORTFOLIO STATE:
- Active Positions: {len(active_positions)} markets
- Total Contracts: {sum(abs(signed_position(p)) for p in active_positions)}
- Available Cash: ${available_cash:.2f}
- Capital Utilization: {((sum(abs(signed_position(p)) for p in active_positions) * 0.50) / (available_cash + sum(abs(signed_position(p)) for p in active_positions) * 0.50)) * 100:.1f}%

HISTORICAL PERFORMANCE:
- Total Completed Trades: {trade_stats[0] if trade_stats else 0}
- Winning Trades: {trade_stats[1] if trade_stats else 0}
- Win Rate: {(trade_stats[1] / trade_stats[0] * 100) if trade_stats and trade_stats[0] > 0 else 0:.1f}%
- Total P&L: ${trade_stats[3] if trade_stats else 0:.2f}
- Average P&L per Trade: ${trade_stats[2] if trade_stats else 0:.2f}
- Worst Loss: ${trade_stats[4] if trade_stats else 0:.2f}
- Best Win: ${trade_stats[5] if trade_stats else 0:.2f}

POSITION DISTRIBUTION:
{json.dumps(position_stats, indent=2)}

TOP MARKET PATTERNS (by trade count):
{json.dumps(market_patterns, indent=2)}

KEY CONTEXT:
- Recent positions are mostly unrealized P&L (haven't closed yet)
- Earlier positions include profitable manual trades to learn from
- System has 16 active positions with high capital utilization (91.4%)
- Mix of automated and manual trading decisions

ANALYSIS REQUEST:
Provide a comprehensive performance analysis with specific recommendations:

1. **Performance Diagnosis**: What are the main strengths and issues?
2. **Capital Management**: Is the high capital utilization (91.4%) optimal?
3. **Position Strategy**: Are we trading appropriate market types?
4. **Risk Assessment**: What are the key risks with current approach?
5. **Improvement Recommendations**: 3-5 specific actionable improvements

For each recommendation, provide:
- Specific implementation steps
- Expected impact on performance
- Priority level (Critical/High/Medium/Low)

Focus on actionable insights that can immediately improve the trading system.
"""

        print("\n🤖 Running Grok4 analysis...")
        
        # Call Grok4 through XAI client
        try:
            response = await xai_client.get_completion(
                prompt=analysis_prompt,
                max_tokens=3000,
                temperature=0.3
            )
            
            if response:
                if isinstance(response, dict):
                    analysis_text = response.get('content', str(response))
                else:
                    analysis_text = str(response)
                
                print("\n" + "="*60)
                print("🎯 GROK4 TRADING SYSTEM ANALYSIS")
                print("="*60)
                print(analysis_text)
                print("="*60)
                
                # Save analysis
                with open('grok4_trading_analysis.txt', 'w') as f:
                    f.write(f"GROK4 Trading Analysis - {datetime.now().isoformat()}\n")
                    f.write("="*60 + "\n\n")
                    f.write(analysis_text)
                
                print(f"\n💾 Analysis saved to: grok4_trading_analysis.txt")
                
            else:
                print("❌ No analysis generated by Grok4")
                
        except Exception as ai_error:
            print(f"⚠️ Grok4 analysis failed: {ai_error}")
            print("\n📋 FALLBACK ANALYSIS:")
            print("-" * 40)
            
            # Provide basic analysis
            win_rate = (trade_stats[1] / trade_stats[0] * 100) if trade_stats and trade_stats[0] > 0 else 0
            
            if win_rate == 0 and trade_stats and trade_stats[0] > 0:
                print("🔴 CRITICAL: 0% win rate on completed trades - strategy needs immediate revision")
            
            if available_cash < 50:
                print("🔴 CRITICAL: Very low available cash - may block new opportunities")
            
            if len(active_positions) > 10:
                print("⚠️ HIGH: Many active positions - ensure proper risk management")
            
            print("\nRECOMMENDATIONS:")
            print("1. [CRITICAL] Review confidence thresholds - current results suggest overconfidence")
            print("2. [HIGH] Implement capital preservation mode to rebuild cash reserves")
            print("3. [MEDIUM] Analyze successful manual trades to improve automated strategy")
            print("4. [MEDIUM] Consider reducing position sizes until win rate improves")
            print("5. [LOW] Enable live trading only after demonstrating consistent profitability")
        
    finally:
        await kalshi_client.close()
        await xai_client.close()

if __name__ == "__main__":
    asyncio.run(run_quick_analysis()) 