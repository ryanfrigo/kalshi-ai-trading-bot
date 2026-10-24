#!/usr/bin/env python3

import asyncio
import sys
sys.path.append('.')
from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager

async def get_positions():
    # Get Kalshi positions
    kalshi_client = KalshiClient()
    try:
        positions_response = await kalshi_client.get_positions()
        print('=== KALSHI POSITIONS ===')
        market_positions = positions_response.get('market_positions', [])

        # v2 returns position_fp (a signed string: + YES, - NO); the v1 'position'
        # key is gone, so this used to print every position as "None contracts".
        def _f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return 0.0

        def _size(p):
            return _f(p.get('position_fp') or p.get('position') or 0)

        market_positions = [p for p in market_positions if _size(p) != 0]
        print(f'Total positions: {len(market_positions)}')
        for pos in market_positions:
            size = _size(pos)
            side = 'YES' if size > 0 else 'NO'
            print(f'{pos.get("ticker")}: {side} {abs(size):g}')
        
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