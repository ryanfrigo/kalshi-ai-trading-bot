"""
Automated Performance Analysis System

A production-ready system that automatically analyzes trading performance using Grok4,
implements risk checks, tracks manual vs automated performance, and generates actionable
recommendations based on the Kalshi trading system state.

Key Features:
- Regular automated analysis using Grok4
- Risk management checks (cash reserves, position limits, diversification)
- Manual vs automated trade performance tracking
- Actionable recommendations with priority levels
- Integration with existing trading system
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import aiosqlite

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager, Position, TradeLog
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


class Priority(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ActionItem:
    """Represents an actionable recommendation."""
    priority: Priority
    category: str
    action: str
    rationale: str
    impact: str
    implementation_steps: List[str]
    target_date: datetime
    implemented: bool = False


@dataclass
class RiskCheck:
    """Represents a risk management check result."""
    check_name: str
    status: str  # "PASS", "WARNING", "CRITICAL"
    current_value: float
    threshold_value: float
    recommendation: str
    impact: str


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    total_trades: int
    manual_trades: int
    automated_trades: int
    manual_win_rate: float
    automated_win_rate: float
    overall_win_rate: float
    total_pnl: float
    manual_pnl: float
    automated_pnl: float
    unrealized_pnl: float
    capital_utilization: float
    available_cash: float
    active_positions: int
    avg_position_size: float
    largest_position_pct: float


class AutomatedPerformanceAnalyzer:
    """
    Automated trading performance analyzer with Grok4 intelligence.
    
    Implements systematic risk checks, performance tracking, and generates
    actionable recommendations based on current portfolio state.
    """
    
    def __init__(self):
        self.logger = get_trading_logger("automated_performance_analyzer")
        self.kalshi_client = None
        self.xai_client = None
        self.db = None
        
    async def initialize(self):
        """Initialize clients and database connections."""
        self.kalshi_client = KalshiClient()
        self.xai_client = XAIClient()
        self.db = DatabaseManager()
        await self.db.initialize()
        self.logger.info("Automated Performance Analyzer initialized")
    
    async def close(self):
        """Clean up connections."""
        if self.kalshi_client:
            await self.kalshi_client.close()
        if self.xai_client:
            await self.xai_client.close()
        self.logger.info("Automated Performance Analyzer closed")
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive automated performance analysis.
        
        Returns:
            Complete analysis report with metrics, risk checks, and action items
        """
        self.logger.info("🚀 Starting automated performance analysis")
        
        try:
            # Gather current state
            portfolio_data = await self._gather_portfolio_data()
            performance_metrics = await self._calculate_performance_metrics()
            
            # Run risk checks
            risk_checks = await self._run_risk_checks(portfolio_data, performance_metrics)
            
            # Generate AI analysis using Grok4
            grok_analysis = await self._generate_grok_analysis(portfolio_data, performance_metrics, risk_checks)
            
            # Create action items
            action_items = await self._generate_action_items(risk_checks, performance_metrics)
            
            # Compile full report
            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_data': portfolio_data,
                'performance_metrics': performance_metrics.__dict__,
                'risk_checks': [check.__dict__ for check in risk_checks],
                'grok_analysis': grok_analysis,
                'action_items': [item.__dict__ for item in action_items],
                'summary': {
                    'total_critical_issues': len([c for c in risk_checks if c.status == "CRITICAL"]),
                    'total_warnings': len([c for c in risk_checks if c.status == "WARNING"]),
                    'critical_actions': len([a for a in action_items if a.priority == Priority.CRITICAL]),
                    'overall_health_score': self._calculate_health_score(risk_checks, performance_metrics)
                }
            }
            
            # Save report
            await self._save_analysis_report(report)
            
            self.logger.info(
                "✅ Automated analysis completed",
                critical_issues=report['summary']['total_critical_issues'],
                warnings=report['summary']['total_warnings'],
                health_score=report['summary']['overall_health_score']
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            raise
    
    async def _gather_portfolio_data(self) -> Dict[str, Any]:
        """Gather current portfolio state from Kalshi API."""
        self.logger.debug("Gathering portfolio data from Kalshi")
        
        # Get positions and balance
        positions_response = await self.kalshi_client.get_positions()
        balance_response = await self.kalshi_client.get_balance()
        
        kalshi_positions = positions_response.get('market_positions', [])
        active_positions = [p for p in kalshi_positions if p.get('position', 0) != 0]
        
        return {
            'active_positions': len(active_positions),
            'total_contracts': sum(abs(p.get('position', 0)) for p in active_positions),
            'available_cash': balance_response.get('balance', 0) / 100,
            'positions_detail': active_positions,
            'total_portfolio_value': balance_response.get('balance', 0) / 100 + sum(
                abs(p.get('position', 0)) * 0.50 for p in active_positions  # Rough position valuation
            )
        }
    
    async def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        self.logger.debug("Calculating performance metrics")
        
        async with aiosqlite.connect(self.db.db_path) as database:
            # Get trade statistics with date context
            cursor = await database.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    MIN(pnl) as worst_loss,
                    MAX(pnl) as best_win,
                    MIN(DATE(entry_timestamp)) as earliest_trade,
                    MAX(DATE(entry_timestamp)) as latest_trade
                FROM trade_logs 
                WHERE pnl IS NOT NULL
            """)
            trade_stats = await cursor.fetchone()
            
            # Get manual vs automated breakdown
            cursor = await database.execute("""
                SELECT 
                    COUNT(*) as manual_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as manual_wins,
                    SUM(pnl) as manual_pnl
                FROM trade_logs 
                WHERE pnl IS NOT NULL AND rationale LIKE '%manual%'
            """)
            manual_stats = await cursor.fetchone()
            
            cursor = await database.execute("""
                SELECT 
                    COUNT(*) as auto_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as auto_wins,
                    SUM(pnl) as auto_pnl
                FROM trade_logs 
                WHERE pnl IS NOT NULL AND (rationale NOT LIKE '%manual%' OR rationale IS NULL)
            """)
            auto_stats = await cursor.fetchone()
            
            # Get current positions info
            cursor = await database.execute("""
                SELECT 
                    COUNT(*) as open_positions,
                    AVG(entry_price * quantity) as avg_position_size,
                    MAX(entry_price * quantity) as largest_position
                FROM positions 
                WHERE status = 'open'
            """)
            position_stats = await cursor.fetchone()
            
            # Get portfolio data for calculations
            portfolio_data = await self._gather_portfolio_data()
            
            total_trades = trade_stats[0] if trade_stats else 0
            manual_trades = manual_stats[0] if manual_stats else 0
            auto_trades = auto_stats[0] if auto_stats else 0
            
            # Store trade date range for context
            self.trade_date_range = {
                'earliest': trade_stats[6] if trade_stats and trade_stats[6] else 'Unknown',
                'latest': trade_stats[7] if trade_stats and trade_stats[7] else 'Unknown'
            }
            
            return PerformanceMetrics(
                total_trades=total_trades,
                manual_trades=manual_trades,
                automated_trades=auto_trades,
                manual_win_rate=(manual_stats[1] / manual_trades) if manual_trades > 0 else 0,
                automated_win_rate=(auto_stats[1] / auto_trades) if auto_trades > 0 else 0,
                overall_win_rate=(trade_stats[1] / total_trades) if total_trades > 0 else 0,
                total_pnl=trade_stats[3] if trade_stats else 0,
                manual_pnl=manual_stats[2] if manual_stats else 0,
                automated_pnl=auto_stats[2] if auto_stats else 0,
                unrealized_pnl=0,  # Will be calculated from current positions
                capital_utilization=((portfolio_data['total_portfolio_value'] - portfolio_data['available_cash']) / portfolio_data['total_portfolio_value']) * 100 if portfolio_data['total_portfolio_value'] > 0 else 0,
                available_cash=portfolio_data['available_cash'],
                active_positions=portfolio_data['active_positions'],
                avg_position_size=position_stats[1] if position_stats else 0,
                largest_position_pct=(position_stats[2] / portfolio_data['total_portfolio_value']) * 100 if portfolio_data['total_portfolio_value'] > 0 and position_stats[2] else 0
            )
    
    async def _run_risk_checks(self, portfolio_data: Dict[str, Any], metrics: PerformanceMetrics) -> List[RiskCheck]:
        """Run automated risk management checks based on Grok4 recommendations."""
        self.logger.debug("Running automated risk checks")
        
        checks = []
        
        # 1. Cash Reserve Check (Grok4: maintain 15-20%)
        cash_pct = (metrics.available_cash / portfolio_data['total_portfolio_value']) * 100 if portfolio_data['total_portfolio_value'] > 0 else 0
        if cash_pct < 15:
            status = "CRITICAL" if cash_pct < 10 else "WARNING"
            checks.append(RiskCheck(
                check_name="Cash Reserve Threshold",
                status=status,
                current_value=cash_pct,
                threshold_value=15.0,
                recommendation="Close some positions to build cash reserves",
                impact=f"Risk of liquidity crisis. Need ${(portfolio_data['total_portfolio_value'] * 0.15) - metrics.available_cash:.2f} more cash"
            ))
        else:
            checks.append(RiskCheck(
                check_name="Cash Reserve Threshold",
                status="PASS",
                current_value=cash_pct,
                threshold_value=15.0,
                recommendation="Cash reserves adequate",
                impact="Good liquidity buffer maintained"
            ))
        
        # 2. Position Limit Check (Grok4: max 10-12 positions)
        if metrics.active_positions > 12:
            checks.append(RiskCheck(
                check_name="Position Concentration",
                status="WARNING",
                current_value=metrics.active_positions,
                threshold_value=12.0,
                recommendation=f"Close {metrics.active_positions - 12} positions to improve concentration",
                impact="Position dilution reducing capital efficiency"
            ))
        else:
            checks.append(RiskCheck(
                check_name="Position Concentration",
                status="PASS",
                current_value=metrics.active_positions,
                threshold_value=12.0,
                recommendation="Position count within optimal range",
                impact="Good capital concentration maintained"
            ))
        
        # 3. Capital Utilization Check (Grok4: max 80%)
        if metrics.capital_utilization > 80:
            status = "CRITICAL" if metrics.capital_utilization > 90 else "WARNING"
            checks.append(RiskCheck(
                check_name="Capital Utilization",
                status=status,
                current_value=metrics.capital_utilization,
                threshold_value=80.0,
                recommendation="Reduce position sizes or close positions",
                impact="High utilization increases volatility risk"
            ))
        else:
            checks.append(RiskCheck(
                check_name="Capital Utilization",
                status="PASS",
                current_value=metrics.capital_utilization,
                threshold_value=80.0,
                recommendation="Capital utilization well managed",
                impact="Good risk buffer maintained"
            ))
        
        # 4. Position Size Check (Grok4: max 5-10% per position)
        if metrics.largest_position_pct > 10:
            checks.append(RiskCheck(
                check_name="Single Position Risk",
                status="WARNING",
                current_value=metrics.largest_position_pct,
                threshold_value=10.0,
                recommendation="Reduce size of largest position",
                impact="Concentration risk in single position"
            ))
        else:
            checks.append(RiskCheck(
                check_name="Single Position Risk",
                status="PASS",
                current_value=metrics.largest_position_pct,
                threshold_value=10.0,
                recommendation="Position sizes well diversified",
                impact="Good risk distribution across positions"
            ))
        
        # 5. Performance Divergence Check (Manual vs Automated)
        if metrics.manual_trades > 0 and metrics.automated_trades > 0:
            performance_gap = metrics.manual_win_rate - metrics.automated_win_rate
            if performance_gap > 0.2:  # 20% difference
                checks.append(RiskCheck(
                    check_name="Manual vs Automated Performance",
                    status="WARNING",
                    current_value=performance_gap * 100,
                    threshold_value=20.0,
                    recommendation="Analyze manual trading patterns to improve automation",
                    impact="Automation underperforming manual trading significantly"
                ))
            else:
                checks.append(RiskCheck(
                    check_name="Manual vs Automated Performance",
                    status="PASS",
                    current_value=performance_gap * 100,
                    threshold_value=20.0,
                    recommendation="Performance gap acceptable",
                    impact="Automation performing reasonably vs manual"
                ))
        
        return checks
    
    async def _generate_grok_analysis(self, portfolio_data: Dict[str, Any], metrics: PerformanceMetrics, risk_checks: List[RiskCheck]) -> Dict[str, Any]:
        """Generate AI analysis using Grok4."""
        self.logger.debug("Generating Grok4 analysis")
        
        # Prepare analysis prompt with current state
        analysis_prompt = f"""
You are an expert quantitative trading analyst. Analyze this Kalshi prediction market trading system:

        **IMPORTANT DATA LIMITATION**: The trade_logs database only contains recent data from {getattr(self, 'trade_date_range', {}).get('earliest', 'today')} to {getattr(self, 'trade_date_range', {}).get('latest', 'today')}. Historical profitable manual trades are NOT included in this analysis.

**CURRENT PORTFOLIO STATE:**
- Active Positions: {portfolio_data['active_positions']}
- Available Cash: ${metrics.available_cash:.2f}
- Capital Utilization: {metrics.capital_utilization:.1f}%
- Total Portfolio Value: ${portfolio_data['total_portfolio_value']:.2f}

**RECENT PERFORMANCE METRICS (LIMITED DATA):**
- Total Recorded Trades: {metrics.total_trades} (recent data only)
- Manual Trades: {metrics.manual_trades} (Win Rate: {metrics.manual_win_rate:.1%})
- Automated Trades: {metrics.automated_trades} (Win Rate: {metrics.automated_win_rate:.1%})
- Overall Win Rate: {metrics.overall_win_rate:.1%} (WARNING: Based on incomplete data)
- Total P&L: ${metrics.total_pnl:.2f} (recent trades only)

**RISK CHECK RESULTS:**
{self._format_risk_checks_for_prompt(risk_checks)}

**ANALYSIS REQUEST:**
Provide a focused analysis on:
1. Most critical issues requiring immediate action
2. Performance trends (manual vs automated)
3. Specific optimization recommendations
4. Risk mitigation priorities

Be concise and actionable. Focus on the top 3 priorities.
"""

        try:
            # Use get_completion which delegates to OpenRouter
            response_content = await self.xai_client.get_completion(
                prompt=analysis_prompt,
                max_tokens=3000,
                temperature=0.3,
                strategy="performance_analysis",
                query_type="analysis",
            )
            cost = getattr(self.xai_client, 'total_cost', 0.0)

            return {
                'analysis_text': response_content,
                'model': settings.trading.primary_model,
                'timestamp': datetime.now().isoformat(),
                'cost': cost,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Grok4 analysis failed: {e}")
            return {
                'analysis_text': self._generate_fallback_analysis(metrics, risk_checks),
                'model': 'fallback',
                'timestamp': datetime.now().isoformat(),
                'cost': 0,
                'status': 'fallback',
                'error': str(e)
            }
    
    def _format_risk_checks_for_prompt(self, risk_checks: List[RiskCheck]) -> str:
        """Format risk checks for AI prompt."""
        formatted = ""
        for check in risk_checks:
            formatted += f"- {check.check_name}: {check.status} ({check.current_value:.1f}{'%' if 'pct' in check.check_name.lower() or 'utilization' in check.check_name.lower() else ''} vs {check.threshold_value:.1f}{'%' if 'pct' in check.check_name.lower() or 'utilization' in check.check_name.lower() else ''} threshold)\n"
        return formatted
    
    def _generate_fallback_analysis(self, metrics: PerformanceMetrics, risk_checks: List[RiskCheck]) -> str:
        """Generate fallback analysis when Grok4 is unavailable."""
        critical_issues = [c for c in risk_checks if c.status == "CRITICAL"]
        warnings = [c for c in risk_checks if c.status == "WARNING"]
        
        analysis = "FALLBACK ANALYSIS:\n\n"
        
        if critical_issues:
            analysis += "CRITICAL ISSUES:\n"
            for issue in critical_issues:
                analysis += f"- {issue.check_name}: {issue.recommendation}\n"
            analysis += "\n"
        
        if warnings:
            analysis += "WARNINGS:\n"
            for warning in warnings:
                analysis += f"- {warning.check_name}: {warning.recommendation}\n"
            analysis += "\n"
        
        # Performance summary with data limitation warning
        analysis += f"PERFORMANCE SUMMARY (WARNING - INCOMPLETE DATA):\n"
        analysis += f"- Overall Win Rate: {metrics.overall_win_rate:.1%} (RECENT DATA ONLY - missing historical profitable manual trades)\n"
        analysis += f"- Manual vs Automated: {metrics.manual_win_rate:.1%} vs {metrics.automated_win_rate:.1%}\n"
        analysis += f"- Capital Utilization: {metrics.capital_utilization:.1f}%\n"
        analysis += f"- Available Cash: ${metrics.available_cash:.2f}\n"
        analysis += f"- DATA LIMITATION: Analysis missing historical profitable manual trades\n"
        
        return analysis
    
    async def _generate_action_items(self, risk_checks: List[RiskCheck], metrics: PerformanceMetrics) -> List[ActionItem]:
        """Generate specific action items based on risk checks and performance."""
        action_items = []
        
        # Generate action items from critical risk checks
        for check in risk_checks:
            if check.status == "CRITICAL":
                if "Cash Reserve" in check.check_name:
                    action_items.append(ActionItem(
                        priority=Priority.CRITICAL,
                        category="Risk Management",
                        action="Build cash reserves immediately",
                        rationale=f"Cash reserves at {check.current_value:.1f}% - critical liquidity risk",
                        impact="Prevents forced liquidations and enables opportunistic trading",
                        implementation_steps=[
                            "Close 2-3 lowest conviction positions",
                            f"Target ${(metrics.available_cash * 0.15 / (check.current_value / 100)) - metrics.available_cash:.2f} additional cash",
                            "Set automated cash threshold alerts",
                            "Review position sizing rules"
                        ],
                        target_date=datetime.now() + timedelta(days=1)
                    ))
                
                elif "Capital Utilization" in check.check_name:
                    action_items.append(ActionItem(
                        priority=Priority.CRITICAL,
                        category="Risk Management",
                        action="Reduce capital utilization",
                        rationale=f"Utilization at {check.current_value:.1f}% - excessive risk exposure",
                        impact="Reduces portfolio volatility and margin call risk",
                        implementation_steps=[
                            "Close positions to reach 80% utilization target",
                            "Implement position sizing limits",
                            "Set up automated utilization monitoring",
                            "Review Kelly Criterion implementation"
                        ],
                        target_date=datetime.now() + timedelta(days=2)
                    ))
        
        # Generate action items from warnings
        for check in risk_checks:
            if check.status == "WARNING":
                if "Position Concentration" in check.check_name:
                    action_items.append(ActionItem(
                        priority=Priority.HIGH,
                        category="Portfolio Management",
                        action="Reduce position count for better concentration",
                        rationale=f"{check.current_value} positions causing capital dilution",
                        impact="Improves capital efficiency and reduces management overhead",
                        implementation_steps=[
                            f"Close {int(check.current_value - check.threshold_value)} lowest edge positions",
                            "Implement maximum position limit in automation",
                            "Review position selection criteria",
                            "Set up position count alerts"
                        ],
                        target_date=datetime.now() + timedelta(days=3)
                    ))
                
                elif "Manual vs Automated" in check.check_name:
                    action_items.append(ActionItem(
                        priority=Priority.MEDIUM,
                        category="System Optimization",
                        action="Analyze and bridge manual-automated performance gap",
                        rationale=f"Manual outperforming automated by {check.current_value:.1f}%",
                        impact="Could improve automated trading win rate significantly",
                        implementation_steps=[
                            "Extract patterns from profitable manual trades",
                            "Implement hybrid decision framework",
                            "Add manual approval for large positions",
                            "Backtest manual trading rules on automation"
                        ],
                        target_date=datetime.now() + timedelta(days=7)
                    ))
        
        # Add system improvement action items
        if metrics.automated_trades > 0 and metrics.automated_win_rate < 0.6:
            action_items.append(ActionItem(
                priority=Priority.MEDIUM,
                category="System Optimization",
                action="Improve automated trading performance",
                rationale=f"Automated win rate at {metrics.automated_win_rate:.1%} - below target",
                impact="Could significantly improve overall returns",
                implementation_steps=[
                    "Implement edge-based filtering (>10% probability difference)",
                    "Add volatility-adjusted position sizing",
                    "Set time-to-resolution limits (<60 days)",
                    "Add automated stop-loss and take-profit rules"
                ],
                target_date=datetime.now() + timedelta(days=14)
            ))
        
        return action_items
    
    def _calculate_health_score(self, risk_checks: List[RiskCheck], metrics: PerformanceMetrics) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # Deduct points for risk check failures
        for check in risk_checks:
            if check.status == "CRITICAL":
                score -= 25
            elif check.status == "WARNING":
                score -= 10
        
        # Adjust for performance
        if metrics.overall_win_rate > 0.7:
            score += 10
        elif metrics.overall_win_rate < 0.4:
            score -= 15
        
        # Adjust for manual vs automated performance
        if metrics.manual_trades > 0 and metrics.automated_trades > 0:
            gap = abs(metrics.manual_win_rate - metrics.automated_win_rate)
            if gap > 0.3:
                score -= 10
        
        return max(0, min(100, score))
    
    async def _save_analysis_report(self, report: Dict[str, Any]) -> None:
        """Save analysis report to database and file."""
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save summary to database
        async with aiosqlite.connect(self.db.db_path) as database:
            await database.execute("""
                INSERT OR REPLACE INTO analysis_reports 
                (timestamp, health_score, critical_issues, warnings, action_items, report_file)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                report['timestamp'],
                report['summary']['overall_health_score'],
                report['summary']['total_critical_issues'],
                report['summary']['total_warnings'],
                len(report['action_items']),
                filename
            ))
            await database.commit()
        
        self.logger.info(f"Analysis report saved: {filename}")


async def run_performance_analysis() -> Dict[str, Any]:
    """
    Main entry point for automated performance analysis.
    
    Returns:
        Complete analysis report
    """
    analyzer = AutomatedPerformanceAnalyzer()
    
    try:
        await analyzer.initialize()
        report = await analyzer.run_full_analysis()
        return report
    finally:
        await analyzer.close()


# CLI interface for manual runs
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run automated performance analysis")
    parser.add_argument("--save-report", action="store_true", help="Save detailed report to file")
    args = parser.parse_args()
    
    async def main():
        report = await run_performance_analysis()
        
        print("🎯 AUTOMATED PERFORMANCE ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Health Score: {report['summary']['overall_health_score']:.1f}/100")
        print(f"Critical Issues: {report['summary']['total_critical_issues']}")
        print(f"Warnings: {report['summary']['total_warnings']}")
        print(f"Action Items: {len(report['action_items'])}")
        
        # Show critical action items
        critical_actions = [a for a in report['action_items'] if a['priority'] == 'CRITICAL']
        if critical_actions:
            print("\n🔴 CRITICAL ACTIONS REQUIRED:")
            for action in critical_actions:
                print(f"- {action['action']}")
                print(f"  Rationale: {action['rationale']}")
                print(f"  Target: {action['target_date']}")
        
        if 'grok_analysis' in report and report['grok_analysis']['status'] == 'success':
            print("\n🤖 GROK4 ANALYSIS:")
            print("-" * 40)
            print(report['grok_analysis']['analysis_text'])
        
        if args.save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_analysis_report_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved: {filename}")
    
    asyncio.run(main()) 