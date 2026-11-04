"""Tests for PerformanceScheduler alerting, reporting and scheduling helpers."""
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import asyncio
import pytest

sys.modules.setdefault("schedule", types.ModuleType("schedule"))

from src.jobs import performance_scheduler as ps  # noqa: E402


def _action(priority="HIGH"):
    return {
        "priority": priority,
        "action": "do a thing",
        "rationale": "because",
        "target_date": "2026-01-01",
        "impact": "some",
    }


def _report(health=100.0, critical=0, warnings=0, manual=0.6, automated=0.6,
            capital=50.0, actions=None):
    return {
        "summary": {
            "overall_health_score": health,
            "total_critical_issues": critical,
            "total_warnings": warnings,
        },
        "action_items": [] if actions is None else actions,
        "performance_metrics": {
            "manual_win_rate": manual,
            "automated_win_rate": automated,
            "capital_utilization": capital,
        },
        "timestamp": "2026-01-01T00:00:00",
    }


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _daily(monkeypatch, scheduler, report=None, side_effect=None):
    """Patch the analyzer plus the alert/save collaborators for a daily run."""
    analyzer = AsyncMock(return_value=report, side_effect=side_effect)
    monkeypatch.setattr(ps, "run_performance_analysis", analyzer)
    monkeypatch.setattr(scheduler, "_handle_health_degradation_alert", AsyncMock())
    monkeypatch.setattr(scheduler, "_handle_critical_alert", AsyncMock())
    monkeypatch.setattr(scheduler, "_handle_analysis_failure", AsyncMock())
    monkeypatch.setattr(scheduler, "_save_daily_summary", AsyncMock())
    return scheduler


def test_schedule_config_defaults():
    config = ps.ScheduleConfig()
    assert config.daily_analysis_time == "09:00"
    assert config.weekly_deep_analysis_day == "monday"
    assert config.critical_check_interval_minutes == 60
    assert config.health_score_threshold == 50.0
    assert config.enable_email_alerts is False
    assert config.enable_file_alerts is True


def test_new_scheduler_initial_state():
    scheduler = ps.PerformanceScheduler()
    assert scheduler.running is False
    assert scheduler.last_analysis is None
    assert scheduler.last_health_score == 100.0


def test_daily_analysis_alerts_on_health_drop(monkeypatch):
    scheduler = _daily(monkeypatch, ps.PerformanceScheduler(), _report(health=85.0))
    asyncio.run(scheduler._execute_daily_analysis())

    scheduler._handle_health_degradation_alert.assert_awaited_once_with(
        scheduler.last_analysis, -15.0
    )
    assert scheduler.last_health_score == 85.0


def test_daily_analysis_no_alert_on_small_drop(monkeypatch):
    scheduler = _daily(monkeypatch, ps.PerformanceScheduler(), _report(health=95.0))
    asyncio.run(scheduler._execute_daily_analysis())

    scheduler._handle_health_degradation_alert.assert_not_awaited()


def test_daily_analysis_critical_alert_only_when_critical_issues(monkeypatch):
    scheduler = _daily(monkeypatch, ps.PerformanceScheduler(), _report(critical=2))
    asyncio.run(scheduler._execute_daily_analysis())
    scheduler._handle_critical_alert.assert_awaited_once_with(
        scheduler.last_analysis, "Daily Analysis"
    )

    clean = _daily(monkeypatch, ps.PerformanceScheduler(), _report(critical=0))
    asyncio.run(clean._execute_daily_analysis())
    clean._handle_critical_alert.assert_not_awaited()


def test_daily_analysis_failure_alerts_and_keeps_health(monkeypatch):
    scheduler = ps.PerformanceScheduler()
    _daily(monkeypatch, scheduler, side_effect=RuntimeError("boom"))
    asyncio.run(scheduler._execute_daily_analysis())

    scheduler._handle_analysis_failure.assert_awaited_once()
    analysis_type, error = scheduler._handle_analysis_failure.await_args.args
    assert analysis_type == "Daily Analysis"
    assert isinstance(error, RuntimeError)
    assert scheduler.last_health_score == 100.0


def test_daily_analysis_writes_summary_file(monkeypatch, tmp_path):
    report = _report(health=90.0, critical=1, warnings=3, actions=[_action(), _action()])
    monkeypatch.setattr(ps, "run_performance_analysis", AsyncMock(return_value=report))
    scheduler = ps.PerformanceScheduler()
    monkeypatch.setattr(scheduler, "_handle_health_degradation_alert", AsyncMock())
    monkeypatch.setattr(scheduler, "_handle_critical_alert", AsyncMock())

    asyncio.run(scheduler._execute_daily_analysis())

    files = list(tmp_path.glob("daily_summary_*.json"))
    assert len(files) == 1
    import json
    summary = json.loads(files[0].read_text())
    assert summary["health_score"] == 90.0
    assert summary["critical_issues"] == 1
    assert summary["warnings"] == 3
    assert summary["action_items"] == len(report["action_items"])


def test_check_critical_issues_without_analysis(monkeypatch):
    scheduler = ps.PerformanceScheduler()
    monkeypatch.setattr(ps.asyncio, "create_task", MagicMock())

    assert scheduler._check_critical_issues() is None
    ps.asyncio.create_task.assert_not_called()


def test_check_critical_issues_below_threshold(monkeypatch):
    scheduler = ps.PerformanceScheduler()
    monkeypatch.setattr(ps.asyncio, "create_task", MagicMock())

    scheduler.last_analysis = _report(health=40.0)
    scheduler._check_critical_issues()
    ps.asyncio.create_task.assert_called_once()
    ps.asyncio.create_task.call_args[0][0].close()

    ps.asyncio.create_task.reset_mock()
    scheduler.last_analysis = _report(health=60.0)
    scheduler._check_critical_issues()
    ps.asyncio.create_task.assert_not_called()


def test_weekly_report_counts_critical_actions():
    report = _report(actions=[_action("CRITICAL"), _action("CRITICAL"), _action("HIGH"), _action("MEDIUM")])
    weekly = asyncio.run(ps.PerformanceScheduler()._generate_weekly_report(report))

    assert weekly["summary"]["critical_actions"] == 2
    assert weekly["summary"]["total_action_items"] == len(report["action_items"])


def test_weekly_report_recommendations():
    scheduler = ps.PerformanceScheduler()
    recs = lambda report: asyncio.run(scheduler._generate_weekly_report(report))["summary"]["system_recommendations"]

    assert any("Address critical issues" in r for r in recs(_report(critical=1)))
    assert not any("Address critical issues" in r for r in recs(_report(critical=0)))

    assert any("manual vs automated" in r for r in recs(_report(manual=0.8, automated=0.5)))
    assert not any("manual vs automated" in r for r in recs(_report(manual=0.7, automated=0.5)))

    assert any("High capital utilization" in r for r in recs(_report(capital=81)))
    assert not any("High capital utilization" in r for r in recs(_report(capital=80)))


def test_send_alert_file_toggle(capsys, tmp_path):
    scheduler = ps.PerformanceScheduler()
    asyncio.run(scheduler._send_alert("hello alert", "WEEKLY_SUMMARY"))
    capsys.readouterr()

    files = list(tmp_path.glob("alert_weekly_summary_*.txt"))
    assert len(files) == 1
    assert files[0].read_text() == "hello alert"

    scheduler.config.enable_file_alerts = False
    asyncio.run(scheduler._send_alert("second alert", "WEEKLY_SUMMARY"))
    capsys.readouterr()
    assert len(list(tmp_path.glob("alert_weekly_summary_*.txt"))) == 1


def test_weekly_summary_default_message(monkeypatch):
    scheduler = ps.PerformanceScheduler()
    monkeypatch.setattr(scheduler, "_send_alert", AsyncMock())
    weekly_report = {
        "timestamp": "2026-01-01T00:00:00",
        "summary": {
            "health_score": 90.0,
            "total_action_items": 0,
            "critical_actions": 0,
            "system_recommendations": [],
        },
    }

    asyncio.run(scheduler._send_weekly_summary(weekly_report))

    message = scheduler._send_alert.call_args.args[0]
    assert "System operating within normal parameters" in message


def test_get_scheduler_status_without_instance(monkeypatch):
    monkeypatch.setattr(ps, "_scheduler_instance", None)
    assert ps.get_scheduler_status() == {"running": False, "last_analysis": None}
