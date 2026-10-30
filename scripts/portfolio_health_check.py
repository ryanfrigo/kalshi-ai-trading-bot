#!/usr/bin/env python3
"""
Portfolio Health Check Utility

This script provides a clear view of your portfolio finances:
- Available Cash (for new trades)
- Position Value (current market value of holdings)
- Total Portfolio Value (cash + positions)
- Key utilization metrics
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager
from src.utils.market_prices import held_side_mid, signed_position


async def get_portfolio_health() -> Dict:
    """Get comprehensive portfolio health metrics."""
    
    kalshi_client = KalshiClient()
    
    try:
        # Get available cash
        balance_response = await kalshi_client.get_balance()
        available_cash = balance_response.get('balance', 0) / 100
        
        # Get current positions
        positions_response = await kalshi_client.get_positions()
        market_positions = positions_response.get('market_positions', [])
        
        total_position_value = 0
        position_details = []
        
        # Calculate value of each position
        for position in market_positions:
            ticker = position.get('ticker')
            position_count = signed_position(position)

            if ticker and position_count != 0:
                try:
                    market_data = await kalshi_client.get_market(ticker)
                    if market_data and 'market' in market_data:
                        market_info = market_data['market']

                        current_price = held_side_mid(market_info, position_count)
                        side = 'YES' if position_count > 0 else 'NO'
                        
                        position_value = abs(position_count) * current_price
                        total_position_value += position_value
                        
                        position_details.append({
                            'ticker': ticker,
                            'side': side,
                            'contracts': abs(position_count),
                            'price': current_price,
                            'value': position_value,
                            'title': market_info.get('title', ticker)
                        })
                        
                except Exception as e:
                    print(f"Warning: Could not value position {ticker}: {e}")
        
        # Calculate totals and metrics
        total_portfolio_value = available_cash + total_position_value
        utilization_pct = (total_position_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        cash_pct = (available_cash / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        
        return {
            'available_cash': available_cash,
            'position_value': total_position_value,
            'total_portfolio_value': total_portfolio_value,
            'positions_count': len([p for p in position_details if p['contracts'] > 0]),
            'utilization_pct': utilization_pct,
            'cash_pct': cash_pct,
            'position_details': position_details
        }
        
    finally:
        await kalshi_client.close()


def print_portfolio_summary(health: Dict):
    """Print a formatted portfolio summary."""
    
    print("💰 PORTFOLIO HEALTH CHECK")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Financial overview
    print("💵 FINANCIAL OVERVIEW")
    print("-" * 30)
    print(f"💰 Total Portfolio Value:  ${health['total_portfolio_value']:.2f}")
    print(f"💵 Available Cash:         ${health['available_cash']:.2f}")
    print(f"📊 Position Value:         ${health['position_value']:.2f}")
    print()
    
    # Utilization metrics
    print("📈 UTILIZATION METRICS")
    print("-" * 30)
    print(f"📊 Portfolio Utilization:  {health['utilization_pct']:.1f}%")
    print(f"💸 Cash Reserve:           {health['cash_pct']:.1f}%")
    print(f"🎯 Active Positions:       {health['positions_count']}")
    print()
    
    # Health indicators
    print("🩺 HEALTH INDICATORS")
    print("-" * 30)
    
    if health['cash_pct'] < 10:
        print("🔴 LOW CASH: Consider closing some positions to free up trading capital")
    elif health['cash_pct'] < 20:
        print("🟡 MODERATE CASH: Watch cash levels for new opportunities")
    else:
        print("🟢 HEALTHY CASH: Good liquidity for new trades")
    
    if health['utilization_pct'] > 90:
        print("🔴 HIGH UTILIZATION: Portfolio heavily invested, limited flexibility")
    elif health['utilization_pct'] > 70:
        print("🟡 MODERATE UTILIZATION: Well-invested but still some flexibility")
    else:
        print("🟢 CONSERVATIVE UTILIZATION: Room for additional positions")
    
    print()


def print_position_details(health: Dict):
    """Print detailed position information."""
    
    if not health['position_details']:
        print("📊 No active positions")
        return
    
    print("📊 POSITION DETAILS")
    print("-" * 80)
    print(f"{'Ticker':<25} {'Side':<4} {'Contracts':<10} {'Price':<8} {'Value':<10} {'Market':<20}")
    print("-" * 80)
    
    # Sort by value (highest first)
    positions = sorted(health['position_details'], key=lambda x: x['value'], reverse=True)
    
    for pos in positions:
        title_short = pos['title'][:18] + "..." if len(pos['title']) > 20 else pos['title']
        print(f"{pos['ticker']:<25} {pos['side']:<4} {pos['contracts']:<10} ${pos['price']:<7.2f} ${pos['value']:<9.2f} {title_short:<20}")
    
    print("-" * 80)
    print(f"{'TOTAL':<50} ${health['position_value']:<9.2f}")
    print()


async def main():
    """Main function."""
    try:
        print("🔍 Analyzing portfolio...")
        health = await get_portfolio_health()
        
        print_portfolio_summary(health)
        print_position_details(health)
        
        # Trading recommendations
        print("💡 TRADING RECOMMENDATIONS")
        print("-" * 40)
        
        if health['cash_pct'] < 15:
            print("• Consider closing 1-2 low-confidence positions to free up cash")
            print(f"• Target: Increase cash to ${health['total_portfolio_value'] * 0.2:.2f} (20%)")
        
        if health['utilization_pct'] > 85:
            print("• Portfolio is highly concentrated - diversify or reduce exposure")
            
        if health['available_cash'] < 20:
            print("• Low cash balance may limit ability to take advantage of new opportunities")
            
        max_position = max(health['position_details'], key=lambda x: x['value']) if health['position_details'] else None
        if max_position and max_position['value'] > health['total_portfolio_value'] * 0.15:
            print(f"• Large position risk: {max_position['ticker']} is {(max_position['value']/health['total_portfolio_value']*100):.1f}% of portfolio")
        
        print(f"\n🚀 Ready for dashboard! Run: python launch_dashboard.py")
        
    except Exception as e:
        print(f"❌ Error analyzing portfolio: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 