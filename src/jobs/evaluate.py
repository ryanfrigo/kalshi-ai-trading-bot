"""
Enhanced evaluation system with cost monitoring and trading performance analysis.
"""

import asyncio
import aiosqlite
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.utils.database import DatabaseManager
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


def _read_xai_tracker_from_pickle() -> Optional[dict]:
    """
    Read the in-memory DailyUsageTracker that xai_client persists to disk.

    Returns a plain dict with keys: date, total_cost, request_count,
    daily_limit, is_exhausted — or None if the file is missing/unreadable.
    """
    usage_file = "logs/daily_ai_usage.pkl"
    try:
        if os.path.exists(usage_file):
            with open(usage_file, "rb") as f:
                tracker = pickle.load(f)
            today = datetime.now().strftime("%Y-%m-%d")
            if getattr(tracker, "date", None) == today:
                return {
                    "date": tracker.date,
                    "total_cost": getattr(tracker, "total_cost", 0.0),
                    "request_count": getattr(tracker, "request_count", 0),
                    "daily_limit": getattr(tracker, "daily_limit", 50.0),
                    "is_exhausted": getattr(tracker, "is_exhausted", False),
                }
    except Exception:
        pass
    return None


async def analyze_ai_costs(db_manager: DatabaseManager) -> Dict:
    """Analyze AI spending patterns and provide cost optimization recommendations."""
    logger = get_trading_logger("cost_analysis")
    
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    async with aiosqlite.connect(db_manager.db_path) as db:
        # Daily cost summary
        daily_costs = {}
        for date in [today, yesterday]:
            cursor = await db.execute("""
                SELECT total_ai_cost, analysis_count, decision_count 
                FROM daily_cost_tracking WHERE date = ?
            """, (date,))
            row = await cursor.fetchone()
            daily_costs[date] = {
                'cost': row[0] if row else 0.0,
                'analyses': row[1] if row else 0,
                'decisions': row[2] if row else 0
            }
        
        # Weekly cost trend
        cursor = await db.execute("""
            SELECT SUM(total_ai_cost), SUM(analysis_count), SUM(decision_count)
            FROM daily_cost_tracking WHERE date >= ?
        """, (week_ago,))
        weekly_stats = await cursor.fetchone()
        
        # Most expensive markets by analysis cost
        cursor = await db.execute("""
            SELECT market_id, COUNT(*) as analysis_count, SUM(cost_usd) as total_cost,
                   AVG(cost_usd) as avg_cost, analysis_type
            FROM market_analyses 
            WHERE DATE(analysis_timestamp) >= ?
            GROUP BY market_id
            ORDER BY total_cost DESC
            LIMIT 10
        """, (week_ago,))
        expensive_markets = await cursor.fetchall()
        
        # Analysis type breakdown
        cursor = await db.execute("""
            SELECT analysis_type, COUNT(*) as count, SUM(cost_usd) as total_cost
            FROM market_analyses 
            WHERE DATE(analysis_timestamp) >= ?
            GROUP BY analysis_type
        """, (week_ago,))
        analysis_breakdown = await cursor.fetchall()
    
    # ---------------------------------------------------------------------------
    # Merge DB costs with in-memory xAI tracker (belt-and-suspenders).
    # xAI clients now persist costs to DB via upsert_daily_cost(), but if the
    # bot was last run on an older version the pickle may still hold more data.
    # ---------------------------------------------------------------------------
    xai_tracker = _read_xai_tracker_from_pickle()
    xai_cost_today = xai_tracker["total_cost"] if xai_tracker else 0.0
    xai_requests_today = xai_tracker["request_count"] if xai_tracker else 0
    trading_paused = (xai_tracker["is_exhausted"] if xai_tracker else False)

    # Use the higher of DB cost vs. pickle cost (DB is authoritative going
    # forward; pickle is the fallback for backwards-compat).
    db_today_cost = daily_costs[today]['cost']
    today_cost = max(db_today_cost, xai_cost_today)

    # Backfill the daily_costs dict so callers see the merged value
    daily_costs[today]['cost'] = today_cost
    daily_costs[today]['xai_requests'] = xai_requests_today

    today_decisions = daily_costs[today]['decisions']
    cost_per_decision = today_cost / max(1, today_decisions)

    weekly_cost = weekly_stats[0] if weekly_stats and weekly_stats[0] else 0.0
    weekly_analyses = weekly_stats[1] if weekly_stats and weekly_stats[1] else 0
    weekly_decisions = weekly_stats[2] if weekly_stats and weekly_stats[2] else 0

    # Use the *actual* enforced limit (daily_ai_cost_limit) for all threshold
    # calculations — not the softer daily_ai_budget display value.
    actual_limit = getattr(settings.trading, 'daily_ai_cost_limit', 50.0)
    soft_budget  = getattr(settings.trading, 'daily_ai_budget', 10.0)

    # Generate recommendations
    recommendations = []

    if trading_paused:
        recommendations.append(
            f"🚫 Trading PAUSED — daily xAI limit reached: "
            f"${today_cost:.3f} / ${actual_limit:.2f} "
            f"({xai_requests_today} requests)"
        )

    if today_cost > soft_budget * 0.8:
        recommendations.append(
            f"⚠️  Near soft budget threshold: ${today_cost:.3f} / ${soft_budget:.2f}"
        )

    if today_cost > actual_limit * 0.8:
        recommendations.append(
            f"🔴 Near hard xAI limit: ${today_cost:.3f} / ${actual_limit:.2f}"
        )

    if cost_per_decision > settings.trading.max_ai_cost_per_decision:
        recommendations.append(f"💰 High cost per decision: ${cost_per_decision:.3f}")

    if weekly_cost > soft_budget * 5:  # More than 5 days of soft budget in a week
        recommendations.append("📈 Weekly spending trending high - consider tighter controls")

    if weekly_analyses > weekly_decisions * 3:  # Too many analyses relative to decisions
        recommendations.append("🔄 High analysis-to-decision ratio - improve filtering")

    # Log comprehensive cost report
    logger.info(
        "AI Cost Analysis Report",
        today_cost=today_cost,
        yesterday_cost=daily_costs[yesterday]['cost'],
        weekly_cost=weekly_cost,
        cost_per_decision=cost_per_decision,
        weekly_analyses=weekly_analyses,
        weekly_decisions=weekly_decisions,
        budget_utilization=today_cost / soft_budget if soft_budget else 0,
        hard_limit_utilization=today_cost / actual_limit if actual_limit else 0,
        trading_paused=trading_paused,
        xai_requests_today=xai_requests_today,
        recommendations=recommendations,
    )

    return {
        'daily_costs': daily_costs,
        'weekly_cost': weekly_cost,
        'cost_per_decision': cost_per_decision,
        'expensive_markets': expensive_markets,
        'analysis_breakdown': analysis_breakdown,
        'recommendations': recommendations,
        'trading_paused': trading_paused,
        'xai_requests_today': xai_requests_today,
        'actual_daily_limit': actual_limit,
    }

async def analyze_trading_performance(db_manager: DatabaseManager) -> Dict:
    """Analyze trading performance and position management effectiveness."""
    logger = get_trading_logger("trading_performance")
    
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    async with aiosqlite.connect(db_manager.db_path) as db:
        # Overall P&L
        cursor = await db.execute("""
            SELECT COUNT(*) as total_trades, SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades
            FROM trade_logs WHERE DATE(exit_timestamp) >= ?
        """, (week_ago,))
        perf_stats = await cursor.fetchone()
        
        # Exit reason analysis
        cursor = await db.execute("""
            SELECT 
                CASE 
                    WHEN rationale LIKE '%market_resolution%' THEN 'market_resolution'
                    WHEN rationale LIKE '%stop_loss%' THEN 'stop_loss'
                    WHEN rationale LIKE '%take_profit%' THEN 'take_profit'
                    WHEN rationale LIKE '%time_based%' THEN 'time_based'
                    ELSE 'other'
                END as exit_reason,
                COUNT(*) as count,
                AVG(pnl) as avg_pnl
            FROM trade_logs 
            WHERE DATE(exit_timestamp) >= ?
            GROUP BY exit_reason
        """, (week_ago,))
        exit_reasons = await cursor.fetchall()
        
        # Current open positions analysis
        cursor = await db.execute("""
            SELECT COUNT(*) as open_positions,
                   AVG((julianday('now') - julianday(timestamp)) * 24) as avg_hours_held
            FROM positions WHERE status = 'open'
        """)
        position_stats = await cursor.fetchone()
    
    total_trades = perf_stats[0] if perf_stats and perf_stats[0] is not None else 0
    total_pnl = perf_stats[1] if perf_stats and perf_stats[1] is not None else 0.0
    winning_trades = perf_stats[3] if perf_stats and perf_stats[3] is not None else 0
    win_rate = (winning_trades / max(1, total_trades)) if total_trades > 0 else 0.0
    
    avg_pnl = perf_stats[2] if perf_stats and perf_stats[2] is not None else 0.0
    open_positions = position_stats[0] if position_stats and position_stats[0] is not None else 0
    avg_hours_held = position_stats[1] if position_stats and position_stats[1] is not None else 0.0
    
    logger.info(
        "Trading Performance Report",
        total_trades=total_trades,
        total_pnl=total_pnl,
        win_rate=win_rate,
        avg_pnl=avg_pnl,
        open_positions=open_positions,
        avg_hours_held=avg_hours_held
    )
    
    return {
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'exit_reasons': exit_reasons,
        'position_stats': position_stats
    }

async def run_evaluation():
    """
    Enhanced evaluation job that analyzes both costs and trading performance.
    """
    logger = get_trading_logger("evaluation")
    logger.info("Starting enhanced evaluation job.")
    
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    try:
        # Analyze AI costs and efficiency
        cost_analysis = await analyze_ai_costs(db_manager)
        
        # Analyze trading performance
        performance_analysis = await analyze_trading_performance(db_manager)
        
        # Generate overall system health summary
        daily_cost = cost_analysis['daily_costs'][datetime.now().strftime('%Y-%m-%d')]['cost']
        actual_limit = cost_analysis.get('actual_daily_limit', settings.trading.daily_ai_budget)
        budget_utilization = daily_cost / actual_limit if actual_limit else 0
        trading_paused = cost_analysis.get('trading_paused', False)
        
        health_status = "🟢 HEALTHY"
        if trading_paused:
            health_status = "🔴 PAUSED — DAILY LIMIT REACHED"
        elif budget_utilization > 0.9:
            health_status = "🔴 OVER BUDGET"
        elif budget_utilization > 0.7:
            health_status = "🟡 HIGH USAGE"
        
        logger.info(
            "System Health Summary",
            status=health_status,
            daily_budget_used=f"{budget_utilization:.1%}",
            total_recommendations=len(cost_analysis['recommendations']),
            open_positions=performance_analysis.get('position_stats', [0])[0] if performance_analysis.get('position_stats') else 0
        )
        
        # If costs are high, suggest immediate actions
        if budget_utilization > 0.8:
            logger.warning(
                "HIGH COST ALERT: Consider immediate actions",
                suggestions=[
                    "Increase analysis_cooldown_hours",
                    "Raise min_volume_for_ai_analysis threshold", 
                    "Enable skip_news_for_low_volume",
                    "Reduce max_analyses_per_market_per_day"
                ]
            )
    
    except Exception as e:
        logger.error("Error in evaluation job", error=str(e), exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
