#!/usr/bin/env python3

import asyncio
import sys
sys.path.append('.')
from src.clients.kalshi_client import KalshiClient
from src.strategies.safe_compounder import _fetch_all_pages
from src.utils.database import DatabaseManager
from src.clients.xai_client import XAIClient
from src.utils.market_prices import held_side_mid, signed_position
import aiosqlite
from datetime import datetime, timedelta

async def comprehensive_analysis():
    """Comprehensive analysis of trading performance and system state."""
    
    print("🎯 KALSHI TRADING SYSTEM PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    kalshi_client = KalshiClient()
    db = DatabaseManager()
    await db.initialize()
    
    try:
        # 1. CURRENT POSITIONS ANALYSIS
        print("\n📊 CURRENT POSITIONS ANALYSIS")
        print("-" * 40)
        
        # Kalshi actual positions
        kalshi_positions = await _fetch_all_pages(kalshi_client.get_positions, 'market_positions')
        
        non_zero_positions = [p for p in kalshi_positions if signed_position(p) != 0]
        total_contracts = sum(abs(signed_position(p)) for p in non_zero_positions)
        
        print(f"✅ Active Kalshi Positions: {len(non_zero_positions)} markets")
        print(f"📈 Total Contracts Held: {total_contracts}")
        
        if non_zero_positions:
            print("\nTop Positions:")
            sorted_positions = sorted(non_zero_positions, key=lambda x: abs(signed_position(x)), reverse=True)
            for i, pos in enumerate(sorted_positions[:5]):
                ticker = pos.get('ticker') or '(unnamed position)'
                position = signed_position(pos)
                direction = "LONG" if position > 0 else "SHORT"
                print(f"  {i+1}. {ticker[:25]}: {direction} {abs(position)} contracts")
        
        # 2. DATABASE ANALYSIS
        print(f"\n💾 DATABASE ANALYSIS")
        print("-" * 40)
        
        # Get all positions from database
        async with aiosqlite.connect(db.db_path) as database:
            # Count all positions by status
            cursor = await database.execute("SELECT status, COUNT(*) FROM positions GROUP BY status")
            status_counts = await cursor.fetchall()
            
            print("Position Status Distribution:")
            for status, count in status_counts:
                print(f"  {status}: {count}")
            
            # Get position details
            cursor = await database.execute("SELECT market_id, side, quantity, entry_price, live FROM positions WHERE status = 'open' ORDER BY entry_price * quantity DESC")
            open_positions = await cursor.fetchall()
            
            total_exposure = sum(p[2] * p[3] for p in open_positions)
            print(f"\nTotal Database Exposure: ${total_exposure:.2f}")
            
            # Live vs non-live
            live_positions = [p for p in open_positions if p[4] == 1]
            non_live_positions = [p for p in open_positions if p[4] == 0]
            
            print(f"Live positions: {len(live_positions)}")
            print(f"Non-live positions: {len(non_live_positions)}")
        
        # 3. BALANCE AND CAPITAL ANALYSIS
        print(f"\n💰 CAPITAL ANALYSIS")
        print("-" * 40)
        
        balance_response = await kalshi_client.get_balance()
        available_cash = balance_response.get('balance', 0) / 100
        
        print(f"Available Cash: ${available_cash:.2f}")
        
        # Calculate total position value from Kalshi
        total_position_value = 0
        unvalued = []
        for pos in kalshi_positions:
            quantity = signed_position(pos)
            if quantity != 0:
                ticker = pos.get('ticker')
                if not ticker:
                    unvalued.append("(unnamed position)")
                    continue
                try:
                    market_data = await kalshi_client.get_market(ticker)
                    market_info = market_data['market']
                    current_price = held_side_mid(market_info, quantity)
                    position_value = abs(quantity) * current_price
                    total_position_value += position_value
                except Exception:
                    unvalued.append(ticker)
        
        total_portfolio_value = available_cash + total_position_value
        print(f"Position Value: ${total_position_value:.2f}")
        print(f"Total Portfolio: ${total_portfolio_value:.2f}")
        if total_portfolio_value:
            print(f"Capital Utilization: {(total_position_value/total_portfolio_value)*100:.1f}%")
        else:
            print("Capital Utilization: n/a (no cash and no valued positions)")
        if unvalued:
            print(f"⚠️ PARTIAL: could not value {', '.join(unvalued)}. Totals read low.")
        
        # 4. RECENT TRADING ACTIVITY
        print(f"\n📈 RECENT TRADING ACTIVITY")
        print("-" * 40)
        
        # Look at recent logs for trading decisions
        try:
            async with aiosqlite.connect(db.db_path) as database:
                # Check for any trade logs
                cursor = await database.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_logs'")
                trade_logs_table = await cursor.fetchone()
                
                if trade_logs_table:
                    cursor = await database.execute("SELECT COUNT(*) FROM trade_logs")
                    trade_count = await cursor.fetchone()
                    print(f"Total completed trades: {trade_count[0] if trade_count else 0}")
                else:
                    print("No trade_logs table found - no completed trades yet")
                    
                # Check recent position activity
                cursor = await database.execute("SELECT COUNT(*) FROM positions WHERE date(timestamp) >= date('now', '-7 days')")
                recent_positions = await cursor.fetchone()
                print(f"Positions created in last 7 days: {recent_positions[0] if recent_positions else 0}")
                
        except Exception as e:
            print(f"Error analyzing recent activity: {e}")
        
        # 5. SYSTEM HEALTH CHECK
        print(f"\n🏥 SYSTEM HEALTH CHECK")
        print("-" * 40)
        
        # Check for any stuck or problematic positions
        discrepancies = 0
        for db_pos in open_positions:
            market_id, side, quantity, entry_price, live = db_pos
            # Find corresponding Kalshi position
            kalshi_pos = next((p for p in kalshi_positions if p.get('ticker') == market_id), None)
            
            expected_quantity = quantity if side == "YES" else -quantity

            if kalshi_pos:
                kalshi_quantity = signed_position(kalshi_pos)
                
                if kalshi_quantity != expected_quantity:
                    discrepancies += 1
                    print(f"⚠️  {market_id}: DB={expected_quantity}, Kalshi={kalshi_quantity}")
            elif live == 1:
                discrepancies += 1
                print(f"⚠️  {market_id}: DB={expected_quantity}, Kalshi=no position")
        
        if discrepancies == 0:
            print("✅ All database positions match Kalshi positions")
        else:
            print(f"❌ Found {discrepancies} position discrepancies")
        
        # 6. TRADE HISTORY
        print(f"\n🎯 TRADE HISTORY")
        print("-" * 40)
        
        # Calculate win rate if we have trade logs
        try:
            async with aiosqlite.connect(db.db_path) as database:
                cursor = await database.execute("SELECT AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), COUNT(*) FROM trade_logs WHERE pnl IS NOT NULL")
                win_rate, settled = await cursor.fetchone()
                
                if win_rate is not None:
                    print(f"📈 Historical Win Rate: {win_rate:.1%} (settled trades: {settled})")
                else:
                    print("No settled trades yet")
                        
        except Exception as e:
            print("📊 Win rate data not available yet")
            
    finally:
        await kalshi_client.close()

if __name__ == "__main__":
    asyncio.run(comprehensive_analysis())