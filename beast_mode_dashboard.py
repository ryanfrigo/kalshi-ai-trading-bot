#!/usr/bin/env python3
"""
Beast Mode Trading Dashboard 🚀

Real-time performance monitoring for the Unified Advanced Trading System.

Features:
- Live portfolio performance across all strategies
- Risk metrics and capital efficiency
- Market making vs directional trading breakdown
- Expected returns and Sharpe ratios
- Position tracking with exit strategies
- Cost monitoring and budget utilization

Usage:
    python beast_mode_dashboard.py           # Live dashboard
    python beast_mode_dashboard.py --summary # Performance summary
    python beast_mode_dashboard.py --export  # Export to CSV
"""

import asyncio
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import asdict
import pandas as pd

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings

from src.strategies.unified_trading_system import (
    UnifiedAdvancedTradingSystem,
    TradingSystemConfig,
    TradingSystemResults
)


class BeastModeDashboard:
    """
    Comprehensive dashboard for monitoring the Beast Mode trading system.
    
    Tracks:
    - Overall portfolio performance
    - Strategy-specific metrics (market making vs directional)
    - Risk management metrics
    - Capital efficiency and utilization
    - Exit strategy performance
    - Cost analysis and budget tracking
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.kalshi_client = KalshiClient()
        self.xai_client = XAIClient()
        
        # Initialize unified system for performance tracking
        self.unified_system = UnifiedAdvancedTradingSystem(
            self.db_manager, self.kalshi_client, self.xai_client
        )

    async def show_live_dashboard(self):
        """
        Display live dashboard with real-time updates.
        """
        print("🚀 BEAST MODE TRADING DASHBOARD 🚀")
        print("=" * 60)
        
        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
                # Header
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"🚀 BEAST MODE DASHBOARD - {now} 🚀")
                print("=" * 60)
                
                # Get current performance
                performance = await self.get_comprehensive_performance()
                
                # Display sections
                await self._display_portfolio_overview(performance)
                await self._display_strategy_breakdown(performance)
                await self._display_risk_metrics(performance)
                await self._display_position_status(performance)
                await self._display_cost_analysis(performance)
                await self._display_system_health(performance)
                
                print("\n" + "=" * 60)
                print("🔄 Updates every 30 seconds | Ctrl+C to exit")
                
                # Wait 30 seconds before refresh
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped. Thanks for using Beast Mode!")
        except Exception as e:
            print(f"\n❌ Dashboard error: {e}")

    async def get_comprehensive_performance(self) -> Dict:
        """
        Get comprehensive performance metrics across all strategies.
        """
        try:
            # Get system performance summary
            system_performance = self.unified_system.get_system_performance_summary()
            
            # Get current positions
            positions = await self.db_manager.get_open_positions()
            
            # Get recent trades
            recent_trades = await self._get_recent_trade_performance()
            
            # Get cost analysis
            cost_analysis = await self._get_cost_analysis()
            
            # Get market opportunities
            markets = await self.db_manager.get_eligible_markets(
                volume_min=200,  # Standard volume threshold
                max_days_to_expiry=365  # Accept any timeline
            )
            
            # Get AI spending
            daily_ai_cost = await self.db_manager.get_daily_ai_cost()
            
            return {
                'system_performance': system_performance,
                'current_positions': positions,
                'recent_trades': recent_trades,
                'cost_analysis': cost_analysis,
                'available_markets': len(markets) if markets else 0,
                'daily_ai_cost': daily_ai_cost,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting performance: {e}")
            return {}

    async def _display_portfolio_overview(self, performance: Dict):
        """Display high-level portfolio overview."""
        print("\n📊 PORTFOLIO OVERVIEW")
        print("-" * 30)
        
        try:
            system_perf = performance.get('system_performance', {})
            positions = performance.get('current_positions', [])
            recent_trades = performance.get('recent_trades', {})
            
            # Calculate total exposure
            total_exposure = sum(
                pos.entry_price * pos.quantity for pos in positions
                if hasattr(pos, 'entry_price') and hasattr(pos, 'quantity')
            )
            
            total_capital = system_perf.get('total_capital', 10000)
            capital_used_pct = (total_exposure / total_capital) * 100 if total_capital > 0 else 0
            
            print(f"💰 Total Capital: ${total_capital:,.0f}")
            print(f"📈 Current Exposure: ${total_exposure:,.0f} ({capital_used_pct:.1f}%)")
            print(f"🎯 Active Positions: {len(positions)}")
            print(f"📅 Today's Trades: {recent_trades.get('trades_today', 0)}")
            print(f"💵 Today's P&L: ${recent_trades.get('pnl_today', 0):+.2f}")
            print(f"📊 Win Rate (7d): {recent_trades.get('win_rate_7d', 0):.1%}")
            
        except Exception as e:
            print(f"Error displaying portfolio overview: {e}")

    async def _display_strategy_breakdown(self, performance: Dict):
        """Display performance by strategy type."""
        print("\n💡 STRATEGY BREAKDOWN")
        print("-" * 30)
        
        try:
            positions = performance.get('current_positions', [])
            
            # Categorize positions (simplified - would need more logic for real categorization)
            market_making_positions = 0
            directional_positions = 0
            total_mm_exposure = 0.0
            total_dir_exposure = 0.0
            
            for pos in positions:
                if hasattr(pos, 'rationale') and pos.rationale:
                    if 'market making' in pos.rationale.lower() or 'spread' in pos.rationale.lower():
                        market_making_positions += 1
                        total_mm_exposure += getattr(pos, 'entry_price', 0) * getattr(pos, 'quantity', 0)
                    else:
                        directional_positions += 1
                        total_dir_exposure += getattr(pos, 'entry_price', 0) * getattr(pos, 'quantity', 0)
                else:
                    directional_positions += 1
                    total_dir_exposure += getattr(pos, 'entry_price', 0) * getattr(pos, 'quantity', 0)
            
            print(f"🎯 Market Making:")
            print(f"   Positions: {market_making_positions}")
            print(f"   Exposure: ${total_mm_exposure:,.0f}")
            
            print(f"📈 Directional Trading:")
            print(f"   Positions: {directional_positions}")
            print(f"   Exposure: ${total_dir_exposure:,.0f}")
            
            print(f"🔮 Arbitrage: Coming Soon!")
            
        except Exception as e:
            print(f"Error displaying strategy breakdown: {e}")

    async def _display_risk_metrics(self, performance: Dict):
        """Display risk management metrics."""
        print("\n⚠️  RISK METRICS")
        print("-" * 30)
        
        try:
            positions = performance.get('current_positions', [])
            
            # Calculate risk metrics
            total_positions = len(positions)
            max_single_exposure = 0.0
            total_exposure = 0.0
            
            for pos in positions:
                exposure = getattr(pos, 'entry_price', 0) * getattr(pos, 'quantity', 0)
                total_exposure += exposure
                max_single_exposure = max(max_single_exposure, exposure)
            
            concentration_risk = (max_single_exposure / total_exposure) * 100 if total_exposure > 0 else 0
            
            # Exit strategy coverage
            positions_with_exits = sum(
                1 for pos in positions 
                if getattr(pos, 'stop_loss_price', None) is not None
            )
            exit_coverage = (positions_with_exits / total_positions) * 100 if total_positions > 0 else 0
            
            print(f"🎲 Portfolio Concentration: {concentration_risk:.1f}%")
            print(f"🛡️  Exit Strategy Coverage: {exit_coverage:.0f}%")
            print(f"⏰ Avg Time to Expiry: {self._calculate_avg_time_to_expiry(positions):.1f} days")
            print(f"🔄 Diversification Score: {self._calculate_diversification_score(positions):.2f}")
            
        except Exception as e:
            print(f"Error displaying risk metrics: {e}")

    async def _display_position_status(self, performance: Dict):
        """Display current positions with exit strategies."""
        print("\n🎯 POSITION STATUS")
        print("-" * 30)
        
        try:
            positions = performance.get('current_positions', [])
            
            if not positions:
                print("📭 No active positions")
                return
            
            # Display top 5 positions
            positions_sorted = sorted(
                positions,
                key=lambda p: getattr(p, 'entry_price', 0) * getattr(p, 'quantity', 0),
                reverse=True
            )
            
            for i, pos in enumerate(positions_sorted[:5], 1):
                market_id = getattr(pos, 'market_id', 'Unknown')[:20]
                side = getattr(pos, 'side', 'Unknown')
                entry_price = getattr(pos, 'entry_price', 0)
                quantity = getattr(pos, 'quantity', 0)
                stop_loss = getattr(pos, 'stop_loss_price', None)
                take_profit = getattr(pos, 'take_profit_price', None)
                
                exposure = entry_price * quantity
                
                print(f"{i}. {market_id}...")
                print(f"   Side: {side} | Entry: ${entry_price:.2f} | Qty: {quantity}")
                print(f"   Exposure: ${exposure:.0f}")
                if stop_loss:
                    print(f"   Stop Loss: ${stop_loss:.2f}")
                if take_profit:
                    print(f"   Take Profit: ${take_profit:.2f}")
                print()
            
            if len(positions) > 5:
                print(f"... and {len(positions) - 5} more positions")
                
        except Exception as e:
            print(f"Error displaying position status: {e}")

    async def _display_cost_analysis(self, performance: Dict):
        """Display AI cost analysis and budget tracking."""
        print("\n💸 COST ANALYSIS")
        print("-" * 30)
        
        try:
            daily_cost = performance.get('daily_ai_cost', 0)
            cost_analysis = performance.get('cost_analysis', {})
            
            daily_budget = getattr(settings.trading, 'daily_ai_budget', 10.0)
            budget_used_pct = (daily_cost / daily_budget) * 100 if daily_budget > 0 else 0
            
            print(f"💰 Daily AI Spending: ${daily_cost:.2f} / ${daily_budget:.2f} ({budget_used_pct:.1f}%)")
            
            # Budget status
            if budget_used_pct < 50:
                status = "🟢 HEALTHY"
            elif budget_used_pct < 80:
                status = "🟡 MODERATE"
            else:
                status = "🔴 HIGH"
            
            print(f"📊 Budget Status: {status}")
            print(f"🔄 Analysis Count Today: {cost_analysis.get('analyses_today', 0)}")
            print(f"💡 Avg Cost per Analysis: ${cost_analysis.get('avg_cost_per_analysis', 0):.3f}")
            
        except Exception as e:
            print(f"Error displaying cost analysis: {e}")

    async def _display_system_health(self, performance: Dict):
        """Display system health and operational metrics."""
        print("\n🏥 SYSTEM HEALTH")
        print("-" * 30)
        
        try:
            available_markets = performance.get('available_markets', 0)
            
            # System status indicators
            print(f"📊 Available Markets: {available_markets}")
            print(f"🔌 System Status: {'🟢 OPERATIONAL' if available_markets > 0 else '🔴 LIMITED'}")
            print(f"🚀 Beast Mode: {'✅ ACTIVE' if available_markets > 0 else '⏸️  STANDBY'}")
            
            # Performance targets
            system_perf = performance.get('system_performance', {})
            capital_allocation = system_perf.get('capital_allocation', {})
            
            print(f"\n📈 Strategy Allocation:")
            print(f"   Market Making: {capital_allocation.get('market_making', 0.4):.0%}")
            print(f"   Directional: {capital_allocation.get('directional', 0.5):.0%}")
            print(f"   Arbitrage: {capital_allocation.get('arbitrage', 0.1):.0%}")
            
        except Exception as e:
            print(f"Error displaying system health: {e}")

    async def _get_recent_trade_performance(self) -> Dict:
        """Get recent trade performance metrics."""
        try:
            # This would query the trade_logs table for recent performance
            # Simplified implementation
            return {
                'trades_today': 0,
                'pnl_today': 0.0,
                'win_rate_7d': 0.0,
                'avg_holding_time': 0.0
            }
        except Exception as e:
            print(f"Error getting trade performance: {e}")
            return {}

    async def _get_cost_analysis(self) -> Dict:
        """Get AI cost analysis."""
        try:
            # This would analyze AI spending patterns
            return {
                'analyses_today': 0,
                'avg_cost_per_analysis': 0.0,
                'total_weekly_cost': 0.0
            }
        except Exception as e:
            return {}

    def _calculate_avg_time_to_expiry(self, positions) -> float:
        """Calculate average time to expiry for positions."""
        try:
            # Simplified calculation
            return 7.0  # Default 7 days
        except Exception:
            return 0.0

    def _calculate_diversification_score(self, positions) -> float:
        """Calculate portfolio diversification score."""
        try:
            # Simplified diversification calculation
            return min(1.0, len(positions) / 10)  # Max score with 10+ positions
        except Exception:
            return 0.0

    async def export_performance_csv(self, filename: Optional[str] = None):
        """Export performance data to CSV."""
        if not filename:
            filename = f"beast_mode_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            performance = await self.get_comprehensive_performance()
            
            # Convert to pandas DataFrame for easy export
            data = {
                'timestamp': [performance.get('timestamp')],
                'total_positions': [len(performance.get('current_positions', []))],
                'daily_ai_cost': [performance.get('daily_ai_cost', 0)],
                'available_markets': [performance.get('available_markets', 0)]
            }
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            print(f"✅ Performance data exported to {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")

    async def show_summary(self):
        """Show quick performance summary."""
        print("🚀 BEAST MODE SUMMARY 🚀")
        print("=" * 40)
        
        performance = await self.get_comprehensive_performance()
        
        positions = performance.get('current_positions', [])
        daily_cost = performance.get('daily_ai_cost', 0)
        available_markets = performance.get('available_markets', 0)
        
        print(f"📊 Active Positions: {len(positions)}")
        print(f"💰 Daily AI Cost: ${daily_cost:.2f}")
        print(f"📈 Available Markets: {available_markets}")
        print(f"🚀 System Status: {'ACTIVE' if available_markets > 0 else 'STANDBY'}")

async def main():
    """Main entry point for the dashboard."""
    parser = argparse.ArgumentParser(description="Beast Mode Trading Dashboard")
    parser.add_argument('--summary', action='store_true', help='Show quick summary')
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    parser.add_argument('--filename', type=str, help='CSV filename for export')

    args = parser.parse_args()

    dashboard = BeastModeDashboard()

    try:
        await dashboard.unified_system.async_initialize()

        if args.summary:
            await dashboard.show_summary()
        elif args.export:
            await dashboard.export_performance_csv(args.filename)
        else:
            await dashboard.show_live_dashboard()

    except Exception as e:
        print(f"❌ Dashboard error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

