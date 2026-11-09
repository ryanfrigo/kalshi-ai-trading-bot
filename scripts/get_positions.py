#!/usr/bin/env python3

import asyncio
import sys
sys.path.append('.')
from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager
from src.utils.market_prices import signed_position

async def get_positions():
    # Get Kalshi positions
    kalshi_client = KalshiClient()
    try:
        positions_response = await kalshi_client.get_positions()
        print('=== KALSHI POSITIONS ===')
        market_positions = positions_response.get('market_positions', [])
        print(f'Total positions: {len(market_positions)}')
        for pos in market_positions:
            print(f'{pos.get("ticker")}: {signed_position(pos)} contracts')
        
        # Get database positions  
        db = DatabaseManager()
        await db.initialize()
        db_positions = await db.get_open_positions()
        print(f'\n=== DATABASE POSITIONS ===')
        print(f'Total positions: {len(db_positions)}')
        for pos in db_positions:
            print(f'{pos.market_id}: {pos.side} {pos.quantity} @ ${pos.entry_price:.2f} (live: {pos.live})')
        
        # Get closed positions too for performance analysis
        import aiosqlite
        async with aiosqlite.connect(db.db_path) as database:
            cursor = await database.execute("SELECT market_id, side, quantity, entry_price, status FROM positions WHERE status = 'closed' ORDER BY timestamp DESC LIMIT 10")
            closed_positions = await cursor.fetchall()
        
        print(f'\n=== RECENT CLOSED POSITIONS ===')
        for pos in closed_positions:
            print(f'{pos[0]}: {pos[1]} {pos[2]} @ ${pos[3]:.2f} (status: {pos[4]})')
            
    finally:
        await kalshi_client.close()

if __name__ == "__main__":
    asyncio.run(get_positions())