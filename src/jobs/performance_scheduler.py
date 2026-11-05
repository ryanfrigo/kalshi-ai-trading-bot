"""
Performance Analysis Scheduler

Automated scheduler that runs performance analysis at regular intervals,
monitors system health, and alerts on critical issues. Implements the 
Grok4 recommendations for systematic monitoring.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import threading
from dataclasses import dataclass

from src.jobs.automated_performance_analyzer import run_performance_analysis
from src.utils.logging_setup import get_trading_logger


def _schedule():
    """Import the optional `schedule` package only when the scheduler runs."""
    try:
        import schedule
    except ImportError as e:
        raise ImportError(
            "PerformanceScheduler needs the 'schedule' package: pip install schedule"
        ) from e
    return schedule


@dataclass
class ScheduleConfig:
    """Configuration for analysis schedule."""
    daily_analysis_time: str = "09:00"  # Daily analysis at 9 AM
    weekly_deep_analysis_day: str = "monday"  # Weekly deep analysis
    critical_check_interval_minutes: int = 60  # Check for critical issues every hour
    health_score_threshold: float = 50.0  # Alert if health score drops below this
    enable_email_alerts: bool = False  # Enable email alerts (requires config)
    enable_file_alerts: bool = True  # Write alerts to file


class PerformanceScheduler:
    """
    Automated scheduler for performance analysis.
    
    Features:
    - Daily automated analysis
    - Weekly deep analysis with full reports
    - Continuous monitoring for critical issues
    - Alert system for health score drops
    - Integration with trading system lifecycle
    """
    
    def __init__(self, config: Optional[ScheduleConfig] = None):
        self.config = config or ScheduleConfig()
        self.logger = get_trading_logger("performance_scheduler")
        self.running = False
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.last_health_score = 100.0
        
    def start(self) -> None:
        """Start the automated scheduler."""
        schedule = _schedule()
        self.logger.info("🚀 Starting Performance Analysis Scheduler")
        
        # Schedule daily analysis
        schedule.every().day.at(self.config.daily_analysis_time).do(self._run_daily_analysis)
        
        # Schedule weekly deep analysis
        getattr(schedule.every(), self.config.weekly_deep_analysis_day).at("08:00").do(self._run_weekly_analysis)
        
        # Schedule critical issue monitoring
        schedule.every(self.config.critical_check_interval_minutes).minutes.do(self._check_critical_issues)
        
        self.running = True
        self.logger.info(
            "📅 Scheduler configured",
            daily_time=self.config.daily_analysis_time,
            weekly_day=self.config.weekly_deep_analysis_day,
            critical_check_minutes=self.config.critical_check_interval_minutes
        )
        
        # Start the scheduler in a separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        # Run initial analysis
        asyncio.create_task(self._run_initial_analysis())
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self.logger.info("🛑 Stopping Performance Analysis Scheduler")
        self.running = False
        schedule = _schedule()
        schedule.clear()
    
    def _run_scheduler(self) -> None:
        """Run the schedule checker in a loop."""
        schedule = _schedule()
        while self.running:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
    
    async def _run_initial_analysis(self) -> None:
        """Run initial analysis on startup."""
        self.logger.info("🔍 Running initial performance analysis")
        try:
            report = await run_performance_analysis()
            self.last_analysis = report
            self.last_health_score = report['summary']['overall_health_score']
            
            # Check if immediate action needed
            if report['summary']['total_critical_issues'] > 0:
                await self._handle_critical_alert(report, "Initial Analysis")
            
            self.logger.info(
                "✅ Initial analysis completed",
                health_score=self.last_health_score,
                critical_issues=report['summary']['total_critical_issues']
            )
            
        except Exception as e:
            self.logger.error(f"❌ Initial analysis failed: {e}")
    
    def _run_daily_analysis(self) -> None:
        """Run daily performance analysis."""
        self.logger.info("📊 Running scheduled daily analysis")
        asyncio.create_task(self._execute_daily_analysis())
    
    async def _execute_daily_analysis(self) -> None:
        """Execute the daily analysis."""
        try:
            report = await run_performance_analysis()
            self.last_analysis = report
            
            # Check for health score degradation
            current_health = report['summary']['overall_health_score']
            health_change = current_health - self.last_health_score
            
            self.logger.info(
                "📈 Daily analysis completed",
                health_score=current_health,
                health_change=health_change,
                critical_issues=report['summary']['total_critical_issues'],
                warnings=report['summary']['total_warnings']
            )
            
            # Alert on significant health score drop
            if health_change < -10:
                await self._handle_health_degradation_alert(report, health_change)
            
            # Alert on critical issues
            if report['summary']['total_critical_issues'] > 0:
                await self._handle_critical_alert(report, "Daily Analysis")
            
            self.last_health_score = current_health
            
            # Save daily summary
            await self._save_daily_summary(report)
            
        except Exception as e:
            self.logger.error(f"❌ Daily analysis failed: {e}")
            await self._handle_analysis_failure("Daily Analysis", e)
    
    def _run_weekly_analysis(self) -> None:
        """Run weekly deep analysis."""
        self.logger.info("📋 Running scheduled weekly deep analysis")
        asyncio.create_task(self._execute_weekly_analysis())
    
    async def _execute_weekly_analysis(self) -> None:
        """Execute the weekly deep analysis."""
        try:
            report = await run_performance_analysis()
            self.last_analysis = report
            
            # Generate comprehensive weekly report
            weekly_report = await self._generate_weekly_report(report)
            
            self.logger.info(
                "📊 Weekly analysis completed",
                health_score=report['summary']['overall_health_score'],
                total_action_items=len(report['action_items'])
            )
            
            # Save weekly report
            await self._save_weekly_report(weekly_report)
            
            # Always send weekly summary (even if no critical issues)
            await self._send_weekly_summary(weekly_report)
            
        except Exception as e:
            self.logger.error(f"❌ Weekly analysis failed: {e}")
            await self._handle_analysis_failure("Weekly Analysis", e)
    
    def _check_critical_issues(self) -> None:
        """Check for critical issues that need immediate attention."""
        if self.last_analysis is None:
            return
        
        # This is a lighter check - just review last analysis for urgent issues
        critical_count = self.last_analysis['summary']['total_critical_issues']
        health_score = self.last_analysis['summary']['overall_health_score']
        
        # Alert if health score drops below threshold
        if health_score < self.config.health_score_threshold:
            asyncio.create_task(self._handle_low_health_alert(health_score))
        
        # Check if we have new critical issues (would require new analysis)
        if critical_count > 0:
            self.logger.warning(
                f"⚠️ Ongoing critical issues detected: {critical_count}",
                health_score=health_score
            )
    
    async def _handle_critical_alert(self, report: Dict[str, Any], analysis_type: str) -> None:
        """Handle critical issue alerts."""
        critical_issues = report['summary']['total_critical_issues']
        health_score = report['summary']['overall_health_score']
        
        alert_message = f"""
🚨 CRITICAL TRADING SYSTEM ALERT

Analysis Type: {analysis_type}
Timestamp: {datetime.now().isoformat()}
Health Score: {health_score:.1f}/100
Critical Issues: {critical_issues}

CRITICAL ACTION ITEMS:
"""
        
        # Add critical action items
        for action in report['action_items']:
            if action['priority'] == 'CRITICAL':
                alert_message += f"""
• {action['action']}
  Rationale: {action['rationale']}
  Target: {action['target_date']}
  Impact: {action['impact']}
"""
        
        alert_message += f"""

IMMEDIATE STEPS REQUIRED:
1. Review critical action items above
2. Check current positions and cash reserves
3. Consider manual intervention if needed
4. Monitor system closely until issues resolved

Full report saved to: {report.get('report_file', 'database')}
"""
        
        # Send alert
        await self._send_alert(alert_message, "CRITICAL", report)
        
        self.logger.error(
            "🚨 Critical alert sent",
            analysis_type=analysis_type,
            critical_issues=critical_issues,
            health_score=health_score
        )
    
    async def _handle_health_degradation_alert(self, report: Dict[str, Any], health_change: float) -> None:
        """Handle health score degradation alerts."""
        alert_message = f"""
⚠️ TRADING SYSTEM HEALTH DEGRADATION

Timestamp: {datetime.now().isoformat()}
Health Score Change: {health_change:+.1f} points
Current Health Score: {report['summary']['overall_health_score']:.1f}/100

This indicates worsening system performance or increased risk exposure.
Review the latest analysis report for specific recommendations.
"""
        
        await self._send_alert(alert_message, "WARNING", report)
        
        self.logger.warning(
            "⚠️ Health degradation alert sent",
            health_change=health_change,
            current_score=report['summary']['overall_health_score']
        )
    
    async def _handle_low_health_alert(self, health_score: float) -> None:
        """Handle low health score alerts."""
        alert_message = f"""
🔻 LOW TRADING SYSTEM HEALTH

Timestamp: {datetime.now().isoformat()}
Health Score: {health_score:.1f}/100 (Below threshold: {self.config.health_score_threshold})

System health is critically low. Immediate review recommended.
Consider reducing exposure and addressing outstanding issues.
"""
        
        await self._send_alert(alert_message, "CRITICAL")
        
        self.logger.error(
            "🔻 Low health alert sent",
            health_score=health_score,
            threshold=self.config.health_score_threshold
        )
    
    async def _handle_analysis_failure(self, analysis_type: str, error: Exception) -> None:
        """Handle analysis failures."""
        alert_message = f"""
❌ ANALYSIS SYSTEM FAILURE

Analysis Type: {analysis_type}
Timestamp: {datetime.now().isoformat()}
Error: {str(error)}

The automated performance analysis system encountered an error.
Manual monitoring may be required until the issue is resolved.
"""
        
        await self._send_alert(alert_message, "SYSTEM_ERROR")
        
        self.logger.error(
            "❌ Analysis failure alert sent",
            analysis_type=analysis_type,
            error=str(error)
        )
    
    async def _generate_weekly_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive weekly report."""
        weekly_report = {
            'timestamp': datetime.now().isoformat(),
            'type': 'weekly_summary',
            'period': 'all_recorded_trades',  # the analyzer has no date filter
            'current_analysis': report,
            'summary': {
                'health_score': report['summary']['overall_health_score'],
                'health_trend': 'unknown',  # not calculated
                'total_action_items': len(report['action_items']),
                'critical_actions': len([a for a in report['action_items'] if a['priority'] == 'CRITICAL']),
                'system_recommendations': []
            }
        }
        
        # Add system recommendations based on analysis
        if report['summary']['total_critical_issues'] > 0:
            weekly_report['summary']['system_recommendations'].append(
                "Address critical issues immediately to prevent system degradation"
            )
        
        if report['performance_metrics']['manual_win_rate'] > report['performance_metrics']['automated_win_rate'] + 0.2:
            weekly_report['summary']['system_recommendations'].append(
                "Significant manual vs automated performance gap - review automation logic"
            )
        
        if report['performance_metrics']['capital_utilization'] > 80:
            weekly_report['summary']['system_recommendations'].append(
                "High capital utilization - consider reducing position sizes"
            )
        
        return weekly_report
    
    async def _save_daily_summary(self, report: Dict[str, Any]) -> None:
        """Save daily analysis summary."""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"daily_summary_{timestamp}.json"
        
        summary = {
            'date': timestamp,
            'health_score': report['summary']['overall_health_score'],
            'critical_issues': report['summary']['total_critical_issues'],
            'warnings': report['summary']['total_warnings'],
            'action_items': len(report['action_items']),
            'performance_metrics': report['performance_metrics']
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.debug(f"Daily summary saved: {filename}")
    
    async def _save_weekly_report(self, weekly_report: Dict[str, Any]) -> None:
        """Save weekly comprehensive report."""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"weekly_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(weekly_report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Weekly report saved: {filename}")
    
    async def _send_weekly_summary(self, weekly_report: Dict[str, Any]) -> None:
        """Send weekly summary notification."""
        summary_message = f"""
📊 WEEKLY TRADING SYSTEM SUMMARY

Week Ending: {datetime.now().strftime('%Y-%m-%d')}
Health Score: {weekly_report['summary']['health_score']:.1f}/100
Action Items: {weekly_report['summary']['total_action_items']}
Critical Actions: {weekly_report['summary']['critical_actions']}

SYSTEM RECOMMENDATIONS:
"""
        
        for rec in weekly_report['summary']['system_recommendations']:
            summary_message += f"• {rec}\n"
        
        if not weekly_report['summary']['system_recommendations']:
            summary_message += "• System operating within normal parameters\n"
        
        summary_message += f"""
Full weekly report available in: weekly_report_{datetime.now().strftime('%Y%m%d')}.json
"""
        
        await self._send_alert(summary_message, "WEEKLY_SUMMARY", weekly_report)
    
    async def _send_alert(self, message: str, alert_type: str, report_data: Optional[Dict] = None) -> None:
        """Send alert through configured channels."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Always log the alert
        self.logger.info(f"📧 Sending {alert_type} alert")
        
        # Save to file if enabled
        if self.config.enable_file_alerts:
            alert_filename = f"alert_{alert_type.lower()}_{timestamp}.txt"
            with open(alert_filename, 'w') as f:
                f.write(message)
            self.logger.debug(f"Alert saved to file: {alert_filename}")
        
        # TODO: Add email alerts if configured
        if self.config.enable_email_alerts:
            # await self._send_email_alert(message, alert_type)
            pass
        
        # Print to console for immediate visibility
        print(f"\n{'='*60}")
        print(f"ALERT: {alert_type}")
        print(f"{'='*60}")
        print(message)
        print(f"{'='*60}\n")


# Singleton instance
_scheduler_instance: Optional[PerformanceScheduler] = None


def start_performance_scheduler(config: Optional[ScheduleConfig] = None) -> PerformanceScheduler:
    """Start the global performance scheduler."""
    global _scheduler_instance
    
    if _scheduler_instance is None or not _scheduler_instance.running:
        _scheduler_instance = PerformanceScheduler(config)
        _scheduler_instance.start()
    
    return _scheduler_instance


def stop_performance_scheduler() -> None:
    """Stop the global performance scheduler."""
    global _scheduler_instance
    
    if _scheduler_instance and _scheduler_instance.running:
        _scheduler_instance.stop()
        _scheduler_instance = None


def get_scheduler_status() -> Dict[str, Any]:
    """Get current scheduler status."""
    global _scheduler_instance
    
    if _scheduler_instance is None:
        return {'running': False, 'last_analysis': None}
    
    return {
        'running': _scheduler_instance.running,
        'last_analysis': _scheduler_instance.last_analysis['timestamp'] if _scheduler_instance.last_analysis else None,
        'last_health_score': _scheduler_instance.last_health_score,
        'config': _scheduler_instance.config.__dict__
    }


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Analysis Scheduler")
    parser.add_argument("--start", action="store_true", help="Start the scheduler")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--daily-time", default="09:00", help="Daily analysis time (HH:MM)")
    parser.add_argument("--weekly-day", default="monday", help="Weekly analysis day")
    args = parser.parse_args()
    
    if args.start:
        config = ScheduleConfig(
            daily_analysis_time=args.daily_time,
            weekly_deep_analysis_day=args.weekly_day
        )
        
        scheduler = start_performance_scheduler(config)
        print("🚀 Performance scheduler started")
        print(f"Daily analysis: {args.daily_time}")
        print(f"Weekly analysis: {args.weekly_day}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping scheduler...")
            stop_performance_scheduler()
            print("✅ Scheduler stopped")
    
    elif args.status:
        status = get_scheduler_status()
        print("📊 SCHEDULER STATUS")
        print(f"Running: {status['running']}")
        print(f"Last Analysis: {status['last_analysis']}")
        print(f"Last Health Score: {status.get('last_health_score', 'N/A')}")
    
    else:
        parser.print_help() 