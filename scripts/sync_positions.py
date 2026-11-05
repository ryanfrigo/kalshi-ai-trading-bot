#!/usr/bin/env python3
"""
Position Sync Tool

This script syncs your current Kalshi positions into the database
so the dashboard shows accurate data.

It reads every position and its market first, then replaces every local position
row in one transaction. A failed read changes nothing. Entry prices become today's
mid, and stops, strategy and exit-order ids on the old rows are lost.
"""

import asyncio
import sys
import os
from dataclasses import asdict
from datetime import datetime
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.kalshi_client import KalshiClient
from src.strategies.safe_compounder import _fetch_all_pages
from src.utils.database import DatabaseManager, Position, TradeLog
from src.utils.logging_setup import get_trading_logger
from src.utils.market_prices import held_side_mid, signed_position

logger = get_trading_logger(__name__)


async def _read_all_positions(kalshi_client) -> List[Dict]:
    """Every market_positions row, or raise. A null list can't be told from a bad reply,
    and this list decides which local rows survive."""
    rows: List[Dict] = []

    async def page(**kw):
        resp = await kalshi_client.get_positions(**kw)
        if not isinstance(resp, dict) or not isinstance(resp.get('market_positions'), list):
            raise RuntimeError("positions reply has no market_positions list")
        rows.extend(resp['market_positions'])
        return resp

    await _fetch_all_pages(page, 'market_positions')
    return rows


async def sync_positions_to_database():
    """Sync current Kalshi positions to database."""
    
    kalshi_client = KalshiClient()
    db_manager = DatabaseManager()
    
    try:
        await db_manager.initialize()
        
        print("🔄 Syncing positions from Kalshi to database...")
        
        # Get current Kalshi positions
        market_positions = await _read_all_positions(kalshi_client)

        print(f"📊 Found {len(market_positions)} positions on Kalshi")

        # Get current database positions
        db_positions = await db_manager.get_open_positions()
        print(f"💾 Found {len(db_positions)} open positions in database")

        # Build every new row before touching the table: any failure here deletes nothing
        new_positions = []
        total_value = 0

        for kalshi_pos in market_positions:
            ticker = kalshi_pos.get('ticker')
            position_count = signed_position(kalshi_pos)
            if position_count == 0:
                continue
            if not ticker:
                raise RuntimeError(f"position row has no ticker: {kalshi_pos}")

            # Get market data for pricing
            market_data = await kalshi_client.get_market(ticker)
            if not market_data or 'market' not in market_data:
                raise RuntimeError(f"no market data for {ticker}")
            market_info = market_data['market']

            # Determine side and current price
            side = 'YES' if position_count > 0 else 'NO'
            current_price = held_side_mid(market_info, position_count)

            new_positions.append(Position(
                market_id=ticker,
                side=side,
                entry_price=current_price,  # Use current price as entry
                quantity=abs(position_count),
                timestamp=datetime.now(),
                rationale="Synced from Kalshi",
                confidence=0.5,  # Default confidence
                live=True,
                status='open',
                strategy='manual_sync'  # Mark as synced
            ))

            position_value = abs(position_count) * current_price
            total_value += position_value

            print(f"   ➕ {ticker}: {side} {abs(position_count)} @ ${current_price:.2f} = ${position_value:.2f}")

        # One transaction: a failed insert rolls the delete back
        print("🔄 Replacing all database positions...")
        import aiosqlite
        async with aiosqlite.connect(db_manager.db_path) as db:
            await db.execute("DELETE FROM positions")
            for position in new_positions:
                row = asdict(position)
                row['timestamp'] = position.timestamp.isoformat()
                await db.execute("""
                    INSERT OR REPLACE INTO positions (market_id, side, entry_price, quantity, timestamp, rationale, confidence, live, status, strategy, stop_loss_price, take_profit_price, max_hold_hours, target_confidence_change)
                    VALUES (:market_id, :side, :entry_price, :quantity, :timestamp, :rationale, :confidence, :live, :status, :strategy, :stop_loss_price, :take_profit_price, :max_hold_hours, :target_confidence_change)
                """, row)
                await db.execute("UPDATE markets SET has_position = 1 WHERE market_id = ?", (position.market_id,))
            await db.commit()

        active_positions = len(new_positions)

        print(f"\n✅ Sync complete!")
        print(f"📊 Active positions: {active_positions}")
        print(f"💰 Total position value: ${total_value:.2f}")
        print(f"🚀 Dashboard should now show accurate data!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error syncing positions: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await kalshi_client.close()
        await db_manager.close()


async def close_database_position(db_manager: DatabaseManager, position: Position):
    """Close a database position and create a trade log."""
    
    # Create a trade log for the closed position
    trade_log = TradeLog(
        market_id=position.market_id,
        side=position.side,
        entry_price=position.entry_price,
        exit_price=position.entry_price,  # Use same price for zero P&L
        quantity=position.quantity,
        pnl=0.0,  # Zero P&L since we're just cleaning up
        entry_timestamp=position.timestamp,
        exit_timestamp=datetime.now(),
        rationale="Position sync cleanup",
        strategy=position.strategy or 'unknown'
    )
    
    # Add trade log and close position
    await db_manager.add_trade_log(trade_log)
    
    # Update position status to closed
    import aiosqlite
    async with aiosqlite.connect(db_manager.db_path) as db:
        await db.execute(
            "UPDATE positions SET status = 'closed' WHERE market_id = ? AND side = ?",
            (position.market_id, position.side)
        )
        await db.commit()


async def verify_sync():
    """Verify the sync worked correctly."""
    
    print("\n🔍 Verifying sync results...")
    
    kalshi_client = KalshiClient()
    db_manager = DatabaseManager()
    
    try:
        await db_manager.initialize()
        
        # Get Kalshi positions
        kalshi_positions = await _read_all_positions(kalshi_client)
        kalshi_active = len([p for p in kalshi_positions if signed_position(p) != 0])
        
        # Get database positions
        db_positions = await db_manager.get_open_positions()
        
        print(f"📊 Kalshi active positions: {kalshi_active}")
        print(f"💾 Database open positions: {len(db_positions)}")
        
        if kalshi_active == len(db_positions):
            print("✅ Sync successful - counts match!")
        else:
            print("⚠️ Counts don't match - may need to run sync again")
            
    except Exception as e:
        print(f"❌ Error verifying sync: {e}")
        
    finally:
        await kalshi_client.close()
        await db_manager.close()


async def main():
    """Main function."""
    
    print("🔄 Position Database Sync Tool")
    print("=" * 50)
    print("This will sync your current Kalshi positions into the database")
    print("so the dashboard shows accurate trade counts and P&L data.")
    print()
    
    # Run sync
    success = await sync_positions_to_database()
    
    if success:
        await verify_sync()
        print("\n🎉 Position sync complete!")
        print("📊 Refresh your dashboard to see updated data")
        print("🚀 Run: python launch_dashboard.py")
    else:
        print("\n❌ Sync failed, no database rows changed - check errors above")

    return success


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
