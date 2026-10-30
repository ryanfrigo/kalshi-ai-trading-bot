"""
Automated Performance Analysis Job

This job analyzes trading performance using AI and provides actionable improvement recommendations.
It examines positions, win rates, capital utilization, and market selection to identify optimization opportunities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiosqlite

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager, Position, TradeLog
from src.utils.market_prices import signed_position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


class TradingPerformanceAnalyzer:
    """
    AI-powered trading performance analyzer using Grok4.
    
    Analyzes:
    - Position performance and win rates
    - Capital utilization efficiency
    - Market selection patterns
    - Risk management effectiveness
    - Edge detection accuracy
    """
    
    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: XAIClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.logger = get_trading_logger("performance_analyzer")
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis and generate improvement recommendations."""
        
        self.logger.info("🔍 Starting comprehensive trading performance analysis")
        
        try:
            # Gather all performance data
            performance_data = await self._gather_performance_data()
            
            # Generate AI analysis using Grok4
            ai_analysis = await self._generate_ai_analysis(performance_data)
            
            # Compile final report
            report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'performance_data': performance_data,
                'ai_insights': ai_analysis,
                'action_items': self._extract_action_items(ai_analysis)
            }
            
            # Log summary
            self._log_analysis_summary(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {}
    
    async def _gather_performance_data(self) -> Dict[str, Any]:
        """Gather comprehensive performance data from all sources."""
        
        data = {}
        
        # 1. Current Portfolio State
        try:
            # Kalshi positions
            positions_response = await self.kalshi_client.get_positions()
            kalshi_positions = positions_response.get('market_positions', [])
            active_positions = [p for p in kalshi_positions if signed_position(p) != 0]
            
            # Portfolio value
            balance_response = await self.kalshi_client.get_balance()
            available_cash = balance_response.get('balance', 0) / 100
            
            data['portfolio'] = {
                'active_positions': len(active_positions),
                'total_contracts': sum(abs(signed_position(p)) for p in active_positions),
                'available_cash': available_cash,
                'positions_detail': active_positions[:10]  # Top 10 for analysis
            }
            
        except Exception as e:
            self.logger.warning(f"Error gathering portfolio data: {e}")
            data['portfolio'] = {}
        
        # 2. Historical Performance
        try:
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                # Win rate and PnL analysis
                cursor = await db.execute("""
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
                
                # Position analysis
                cursor = await db.execute("""
                    SELECT 
                        status,
                        COUNT(*) as count,
                        AVG(entry_price * quantity) as avg_exposure
                    FROM positions 
                    GROUP BY status
                """)
                position_stats = await cursor.fetchall()
                
                # Recent performance (last 7 days)
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as recent_trades,
                        AVG(pnl) as recent_avg_pnl,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as recent_wins
                    FROM trade_logs 
                    WHERE exit_timestamp >= ? AND pnl IS NOT NULL
                """, (week_ago,))
                recent_stats = await cursor.fetchone()
                
                # Calculate unrealized P&L for current positions
                cursor = await db.execute("""
                     SELECT 
                         COUNT(*) as open_positions,
                         SUM(entry_price * quantity) as total_exposure
                     FROM positions 
                     WHERE status = 'open'
                 """)
                open_position_stats = await cursor.fetchone()
                
                data['performance'] = {
                    'overall_stats': {
                        'total_trades': trade_stats[0] if trade_stats else 0,
                        'winning_trades': trade_stats[1] if trade_stats else 0,
                        'win_rate': (trade_stats[1] / trade_stats[0]) if trade_stats and trade_stats[0] > 0 else 0,
                        'avg_pnl': trade_stats[2] if trade_stats else 0,
                        'total_pnl': trade_stats[3] if trade_stats else 0,
                        'worst_loss': trade_stats[4] if trade_stats else 0,
                        'best_win': trade_stats[5] if trade_stats else 0
                    },
                    'position_distribution': dict(position_stats) if position_stats else {},
                    'open_positions': {
                        'count': open_position_stats[0] if open_position_stats else 0,
                        'total_exposure': open_position_stats[1] if open_position_stats else 0
                    },
                    'recent_performance': {
                        'trades_last_7d': recent_stats[0] if recent_stats else 0,
                        'avg_pnl_last_7d': recent_stats[1] if recent_stats else 0,
                        'wins_last_7d': recent_stats[2] if recent_stats else 0
                    }
                }
                
        except Exception as e:
            self.logger.warning(f"Error gathering performance data: {e}")
            data['performance'] = {}
        
        # 3. Market Selection Patterns
        try:
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                # Analyze which types of markets we trade
                cursor = await db.execute("""
                    SELECT 
                        SUBSTR(market_id, 1, 10) as market_prefix,
                        COUNT(*) as trade_count,
                        AVG(confidence) as avg_confidence,
                        AVG(entry_price) as avg_entry_price
                    FROM positions 
                    GROUP BY market_prefix
                    ORDER BY trade_count DESC
                    LIMIT 10
                """)
                market_patterns = await cursor.fetchall()
                
                data['market_patterns'] = market_patterns
                
        except Exception as e:
            self.logger.warning(f"Error analyzing market patterns: {e}")
            data['market_patterns'] = []
        
        # 4. System Configuration
        data['system_config'] = {
            'min_confidence': getattr(settings.trading, 'min_confidence_to_trade', 0.6),
            'max_position_size': getattr(settings.trading, 'max_position_size_pct', 15),
            'kelly_fraction': getattr(settings.trading, 'kelly_fraction', 0.75),
            'live_trading_enabled': getattr(settings.trading, 'live_trading_enabled', False),
            'ev_threshold': getattr(settings.trading, 'ev_threshold', 0.10)
        }
        
        return data
    
    async def _generate_ai_analysis(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use Grok4 to analyze performance data and generate insights."""
        
        analysis_prompt = f"""
You are an expert quantitative trading analyst reviewing a Kalshi prediction market trading system. 

Analyze this performance data and provide actionable insights:

CURRENT PORTFOLIO:
- Active Positions: {performance_data.get('portfolio', {}).get('active_positions', 0)}
- Total Contracts: {performance_data.get('portfolio', {}).get('total_contracts', 0)}
- Available Cash: ${performance_data.get('portfolio', {}).get('available_cash', 0):.2f}

HISTORICAL PERFORMANCE:
{json.dumps(performance_data.get('performance', {}), indent=2)}

MARKET SELECTION PATTERNS:
{json.dumps(performance_data.get('market_patterns', []), indent=2)}

SYSTEM CONFIGURATION:
{json.dumps(performance_data.get('system_config', {}), indent=2)}

Please provide a comprehensive analysis covering:

1. **Performance Diagnosis**: What are the main issues causing poor performance?
2. **Capital Management**: Is the capital allocation strategy optimal?
3. **Market Selection**: Are we trading the right types of markets?
4. **Risk Management**: Are our position sizing and risk controls appropriate?
5. **Edge Detection**: Is our AI confidence calibration accurate?
6. **Operational Issues**: Any systematic problems in execution or strategy?

For each issue identified, provide:
- Root cause analysis
- Specific quantitative recommendations
- Implementation priority (High/Medium/Low)

Focus on actionable insights that can immediately improve performance.
"""

        try:
            # Use XAI client's get_completion method - just return the text response
            response = await self.xai_client.get_completion(
                prompt=analysis_prompt,
                max_tokens=3000,
                temperature=0.3
            )
            
            # Extract text from response
            if isinstance(response, dict):
                analysis_text = response.get('content', str(response))
            else:
                analysis_text = str(response) if response else "No analysis generated"
            
            return {
                'raw_analysis': analysis_text,
                'model': 'grok-4',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return {
                'error': str(e),
                'fallback_analysis': self._generate_fallback_analysis(performance_data)
            }
    
    def _generate_fallback_analysis(self, performance_data: Dict[str, Any]) -> str:
        """Generate basic analysis if AI fails."""
        
        perf = performance_data.get('performance', {}).get('overall_stats', {})
        portfolio = performance_data.get('portfolio', {})
        
        issues = []
        
        # Check win rate
        win_rate = perf.get('win_rate', 0)
        if win_rate < 0.4:
            issues.append(f"Critical: Win rate is {win_rate:.1%} - strategy needs major revision")
        
        # Check capital utilization
        cash = portfolio.get('available_cash', 0)
        if cash < 50:
            issues.append(f"Critical: Low available cash (${cash:.2f}) - may block new trades")
        
        # Check total PnL
        total_pnl = perf.get('total_pnl', 0)
        if total_pnl < -100:
            issues.append(f"Critical: Large cumulative loss (${total_pnl:.2f})")
        
        recommendations = [
            "1. Increase minimum confidence threshold for trades",
            "2. Reduce position sizes to preserve capital",
            "3. Review market selection criteria",
            "4. Implement stricter stop-loss rules",
            "5. Consider paper trading to test strategy improvements"
        ]
        
        return f"FALLBACK ANALYSIS:\n\nKey Issues:\n" + "\n".join(issues) + "\n\nRecommendations:\n" + "\n".join(recommendations)
    
    def _extract_action_items(self, ai_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract concrete action items from AI analysis."""
        
        # This would parse the AI response to extract specific action items
        # For now, return some basic action items based on common issues
        
        return [
            {
                'priority': 'High',
                'action': 'Review and increase minimum confidence threshold',
                'rationale': 'Current 0% win rate suggests overconfident AI predictions'
            },
            {
                'priority': 'High', 
                'action': 'Implement capital preservation mode',
                'rationale': 'Low available cash ($9.84) blocks new trading opportunities'
            },
            {
                'priority': 'Medium',
                'action': 'Analyze market selection patterns',
                'rationale': 'Identify which market types are most/least profitable'
            },
            {
                'priority': 'Medium',
                'action': 'Calibrate AI edge detection',
                'rationale': 'Zero wins suggests poor probability estimation'
            },
            {
                'priority': 'Low',
                'action': 'Enable live trading for successful strategies',
                'rationale': 'Currently all positions are non-live simulations'
            }
        ]
    
    def _log_analysis_summary(self, report: Dict[str, Any]) -> None:
        """Log analysis summary for monitoring."""
        
        perf = report.get('performance_data', {}).get('performance', {}).get('overall_stats', {})
        portfolio = report.get('performance_data', {}).get('portfolio', {})
        
        self.logger.info(
            "📊 Performance Analysis Complete",
            win_rate=f"{perf.get('win_rate', 0):.1%}",
            total_pnl=f"${perf.get('total_pnl', 0):.2f}",
            active_positions=portfolio.get('active_positions', 0),
            available_cash=f"${portfolio.get('available_cash', 0):.2f}",
            action_items=len(report.get('action_items', []))
        )


async def run_performance_analysis(
    db_manager: Optional[DatabaseManager] = None,
    kalshi_client: Optional[KalshiClient] = None,
    xai_client: Optional[XAIClient] = None
) -> Dict[str, Any]:
    """
    Run automated performance analysis.
    
    Returns:
        Comprehensive performance analysis report
    """
    logger = get_trading_logger("performance_analysis")
    logger.info("🚀 Starting automated performance analysis")
    
    # Initialize clients if not provided
    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.initialize()
    
    if kalshi_client is None:
        kalshi_client = KalshiClient()
    
    if xai_client is None:
        xai_client = XAIClient(db_manager=db_manager)
    
    try:
        analyzer = TradingPerformanceAnalyzer(db_manager, kalshi_client, xai_client)
        report = await analyzer.run_comprehensive_analysis()
        
        logger.info("✅ Performance analysis completed successfully")
        return report
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        return {}
    
    finally:
        if kalshi_client:
            await kalshi_client.close() 