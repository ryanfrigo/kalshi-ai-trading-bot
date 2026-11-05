"""Scheduler and dashboard flows on top of the analyzer report. Fakes only; no client is built."""
import fnmatch
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import asyncio
import pytest

from src.jobs import automated_performance_analyzer as apa
from src.jobs import performance_dashboard_integration as pdi
from src.jobs import performance_scheduler as ps
from src.utils.database import DatabaseManager

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _action(priority="CRITICAL", steps=("first step", "second", "third")):
    return {
        "priority": priority, "category": "Risk", "action": f"{priority} action",
        "rationale": "why", "impact": "what", "implementation_steps": list(steps),
        "target_date": "2026-01-02",
    }


def _report(health=70.0, critical=1, warnings=0, actions=None):
    return {
        "timestamp": "2026-01-01T00:00:00",
        "summary": {"overall_health_score": health, "total_critical_issues": critical,
                    "total_warnings": warnings},
        "action_items": [_action()] if actions is None else actions,
        "performance_metrics": {"manual_win_rate": 0.5, "automated_win_rate": 0.5,
                                "capital_utilization": 50.0, "available_cash": 12.5,
                                "active_positions": 2, "overall_win_rate": 0.6, "total_pnl": 3.0},
    }


# --- report files stay out of git -----------------------------------------------------

@pytest.mark.parametrize("name", [
    "performance_analysis_20260101_000000.json", "daily_summary_20260101.json",
    "weekly_report_20260101.json", "alert_critical_20260101_000000.txt",
    "alert_weekly_summary_20260101_000000.txt",
])
def test_report_files_with_balances_are_gitignored(name):
    patterns = [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]
    assert any(fnmatch.fnmatch(name, p) for p in patterns if p and not p.startswith("#"))


# --- scheduler: lifecycle -------------------------------------------------------------

async def test_start_registers_three_jobs_and_runs_the_initial_analysis(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(ps, "_schedule", lambda: fake)
    analysis = AsyncMock(return_value=_report(health=88.0, critical=0))
    monkeypatch.setattr(ps, "run_performance_analysis", analysis)
    scheduler = ps.PerformanceScheduler(ps.ScheduleConfig(daily_analysis_time="07:30",
                                                          weekly_deep_analysis_day="friday",
                                                          critical_check_interval_minutes=15))
    # The worker loop exits after one pass.
    monkeypatch.setattr(ps, "time", SimpleNamespace(sleep=lambda _s: setattr(scheduler, "running", False)))

    scheduler.start()
    scheduler.scheduler_thread.join(timeout=5)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    fake.every.return_value.day.at.assert_called_once_with("07:30")
    fake.every.return_value.day.at.return_value.do.assert_called_once_with(scheduler._run_daily_analysis)
    fake.every.return_value.friday.at.return_value.do.assert_called_once_with(scheduler._run_weekly_analysis)
    fake.every.assert_any_call(15)
    fake.run_pending.assert_called()
    analysis.assert_awaited_once()
    assert scheduler.last_health_score == 88.0


def test_stop_clears_the_schedule(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(ps, "_schedule", lambda: fake)
    scheduler = ps.PerformanceScheduler()
    scheduler.running = True
    scheduler.stop()
    assert scheduler.running is False
    fake.clear.assert_called_once()


def _raised_in_worker_thread(call):
    errors = []

    def worker():
        try:
            call()
        except Exception as e:  # noqa: BLE001 - the error is what the test reads
            errors.append(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return errors


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.xfail(strict=True, reason=(
    "start() runs jobs in a thread with no event loop, so asyncio.create_task raises and "
    "kills the thread on the first due job. Fixing this starts scheduled paid AI calls: "
    "owner's call."
))
@pytest.mark.parametrize("job", ["_run_daily_analysis", "_run_weekly_analysis"])
def test_scheduled_jobs_can_run_in_the_scheduler_thread(monkeypatch, job):
    monkeypatch.setattr(ps, "run_performance_analysis", AsyncMock(return_value=_report()))
    scheduler = ps.PerformanceScheduler()
    assert _raised_in_worker_thread(getattr(scheduler, job)) == []


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.xfail(strict=True, reason="Same fault: the low-health alert uses create_task in the thread.")
def test_low_health_check_can_run_in_the_scheduler_thread():
    scheduler = ps.PerformanceScheduler()
    scheduler.last_analysis = _report(health=10.0)
    assert _raised_in_worker_thread(scheduler._check_critical_issues) == []


def test_start_helper_reuses_a_running_instance_and_stop_clears_it(monkeypatch):
    monkeypatch.setattr(ps, "_scheduler_instance", None)
    monkeypatch.setattr(ps.PerformanceScheduler, "start", lambda self: setattr(self, "running", True))
    monkeypatch.setattr(ps, "_schedule", lambda: MagicMock())
    first = ps.start_performance_scheduler()
    assert ps.start_performance_scheduler() is first
    ps.stop_performance_scheduler()
    assert ps._scheduler_instance is None
    ps.stop_performance_scheduler()


def test_start_helper_replaces_a_stopped_instance(monkeypatch):
    stale = ps.PerformanceScheduler()
    monkeypatch.setattr(ps, "_scheduler_instance", stale)
    monkeypatch.setattr(ps.PerformanceScheduler, "start", lambda self: setattr(self, "running", True))
    assert ps.start_performance_scheduler() is not stale


def test_status_reports_the_running_instance(monkeypatch):
    scheduler = ps.PerformanceScheduler()
    scheduler.running = True
    scheduler.last_analysis = _report()
    scheduler.last_health_score = 70.0
    monkeypatch.setattr(ps, "_scheduler_instance", scheduler)
    status = ps.get_scheduler_status()
    assert status["running"] is True
    assert status["last_analysis"] == "2026-01-01T00:00:00"
    assert status["last_health_score"] == 70.0
    assert status["config"]["daily_analysis_time"] == "09:00"


# --- scheduler: analyses and alerts ---------------------------------------------------

async def test_initial_analysis_alerts_on_critical_issues(monkeypatch):
    monkeypatch.setattr(ps, "run_performance_analysis", AsyncMock(return_value=_report(critical=2)))
    scheduler = ps.PerformanceScheduler()
    scheduler._handle_critical_alert = AsyncMock()
    await scheduler._run_initial_analysis()
    scheduler._handle_critical_alert.assert_awaited_once()
    assert scheduler.last_analysis["summary"]["total_critical_issues"] == 2


async def test_initial_analysis_stays_quiet_without_critical_issues(monkeypatch):
    monkeypatch.setattr(ps, "run_performance_analysis", AsyncMock(return_value=_report(critical=0)))
    scheduler = ps.PerformanceScheduler()
    scheduler._handle_critical_alert = AsyncMock()
    await scheduler._run_initial_analysis()
    scheduler._handle_critical_alert.assert_not_awaited()


async def test_initial_analysis_failure_keeps_the_old_state(monkeypatch):
    monkeypatch.setattr(ps, "run_performance_analysis", AsyncMock(side_effect=RuntimeError("no")))
    scheduler = ps.PerformanceScheduler()
    await scheduler._run_initial_analysis()
    assert scheduler.last_analysis is None
    assert scheduler.last_health_score == 100.0


async def test_critical_alert_lists_only_critical_actions():
    scheduler = ps.PerformanceScheduler()
    scheduler._send_alert = AsyncMock()
    report = _report(actions=[_action("CRITICAL"), _action("HIGH")])
    await scheduler._handle_critical_alert(report, "Daily Analysis")
    message, kind, data = scheduler._send_alert.await_args.args
    assert kind == "CRITICAL"
    assert data is report
    assert "Analysis Type: Daily Analysis" in message
    assert "CRITICAL action" in message
    assert "HIGH action" not in message


async def test_critical_alert_from_a_real_analyzer_report_names_the_action(tmp_path):
    a = apa.AutomatedPerformanceAnalyzer()
    a.db = DatabaseManager(str(tmp_path / "t.db"))
    await a.db.initialize()
    a.kalshi_client = SimpleNamespace(
        get_positions=AsyncMock(return_value={"market_positions": [{"position_fp": "10.00"}]}),
        get_balance=AsyncMock(return_value={"balance": 0}),
    )
    a.xai_client = SimpleNamespace(get_completion=AsyncMock(return_value="text"), total_cost=0.0)
    report = await a.run_full_analysis()

    scheduler = ps.PerformanceScheduler()
    scheduler._send_alert = AsyncMock()
    await scheduler._handle_critical_alert(report, "Daily Analysis")
    assert "Build cash reserves immediately" in scheduler._send_alert.await_args.args[0]
    weekly = await scheduler._generate_weekly_report(report)
    assert weekly["summary"]["critical_actions"] >= 1


async def test_degradation_alert_shows_the_signed_change():
    scheduler = ps.PerformanceScheduler()
    scheduler._send_alert = AsyncMock()
    await scheduler._handle_health_degradation_alert(_report(health=60.0), -15.0)
    message, kind, _ = scheduler._send_alert.await_args.args
    assert kind == "WARNING"
    assert "-15.0 points" in message
    assert "60.0/100" in message


async def test_low_health_alert_shows_the_threshold():
    scheduler = ps.PerformanceScheduler(ps.ScheduleConfig(health_score_threshold=40.0))
    scheduler._send_alert = AsyncMock()
    await scheduler._handle_low_health_alert(12.0)
    message, kind = scheduler._send_alert.await_args.args
    assert kind == "CRITICAL"
    assert "12.0/100" in message
    assert "threshold: 40.0" in message


async def test_failure_alert_carries_the_error():
    scheduler = ps.PerformanceScheduler()
    scheduler._send_alert = AsyncMock()
    await scheduler._handle_analysis_failure("Weekly Analysis", RuntimeError("db locked"))
    message, kind = scheduler._send_alert.await_args.args
    assert kind == "SYSTEM_ERROR"
    assert "Weekly Analysis" in message
    assert "db locked" in message


async def test_weekly_run_saves_the_report_and_sends_the_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "run_performance_analysis", AsyncMock(return_value=_report()))
    scheduler = ps.PerformanceScheduler()
    scheduler._send_alert = AsyncMock()
    await scheduler._execute_weekly_analysis()
    files = list(tmp_path.glob("weekly_report_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["summary"]["critical_actions"] == 1
    assert scheduler._send_alert.await_args.args[1] == "WEEKLY_SUMMARY"
    assert scheduler.last_analysis["timestamp"] == "2026-01-01T00:00:00"


async def test_weekly_report_does_not_claim_a_window_or_trend_it_never_computed():
    weekly = await ps.PerformanceScheduler()._generate_weekly_report(_report())
    assert weekly["period"] == "all_recorded_trades"
    assert weekly["summary"]["health_trend"] == "unknown"


async def test_weekly_failure_sends_a_failure_alert(monkeypatch):
    monkeypatch.setattr(ps, "run_performance_analysis", AsyncMock(side_effect=RuntimeError("no")))
    scheduler = ps.PerformanceScheduler()
    scheduler._handle_analysis_failure = AsyncMock()
    await scheduler._execute_weekly_analysis()
    scheduler._handle_analysis_failure.assert_awaited_once()


async def test_weekly_summary_lists_recommendations():
    scheduler = ps.PerformanceScheduler()
    scheduler._send_alert = AsyncMock()
    weekly = {"summary": {"health_score": 70.0, "total_action_items": 1, "critical_actions": 1,
                          "system_recommendations": ["reduce it"]}}
    await scheduler._send_weekly_summary(weekly)
    message = scheduler._send_alert.await_args.args[0]
    assert "• reduce it" in message
    assert "normal parameters" not in message


async def test_critical_check_only_logs_when_health_is_fine():
    scheduler = ps.PerformanceScheduler()
    scheduler.last_analysis = _report(health=90.0, critical=1)
    scheduler._handle_low_health_alert = AsyncMock()
    scheduler._check_critical_issues()
    scheduler._handle_low_health_alert.assert_not_called()


# --- dashboard ------------------------------------------------------------------------

@pytest.fixture
async def dash(tmp_path):
    d = pdi.PerformanceDashboardIntegration()
    d.db = DatabaseManager(str(tmp_path / "d.db"))
    await d.db.initialize()
    return d


async def _save(d, report_file, ts="2026-01-01T00:00:00", health=70.0, critical=1, warnings=2):
    async with aiosqlite.connect(d.db.db_path) as db:
        await db.execute(
            "INSERT INTO analysis_reports (timestamp, health_score, critical_issues, warnings,"
            " action_items, report_file) VALUES (?,?,?,?,?,?)",
            (ts, health, critical, warnings, 1, report_file),
        )
        await db.commit()


async def test_metrics_with_no_analysis_are_placeholders_with_no_timestamp(dash):
    m = await dash.get_current_metrics()
    assert m.health_score == 50.0
    assert m.last_analysis is None
    assert m.available_cash == 0.0


async def test_metrics_read_the_newest_report_file(dash, tmp_path):
    (tmp_path / "old.json").write_text(json.dumps(_report(health=10.0)))
    (tmp_path / "new.json").write_text(json.dumps(_report(health=70.0)))
    await _save(dash, "old.json", ts="2026-01-01T00:00:00")
    await _save(dash, "new.json", ts="2026-01-02T00:00:00")
    m = await dash.get_current_metrics()
    assert m.health_score == 70.0
    assert m.available_cash == 12.5


async def test_metrics_fall_back_to_the_row_when_the_file_is_gone(dash):
    await _save(dash, "gone.json", health=33.0, critical=4, warnings=5)
    m = await dash.get_current_metrics()
    assert (m.health_score, m.critical_issues, m.warnings) == (33.0, 4, 5)
    assert m.last_analysis == "2026-01-01T00:00:00"
    assert m.available_cash == 0.0


async def test_metrics_with_a_corrupt_file_read_as_a_critical_failure(dash, tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    await _save(dash, "bad.json")
    m = await dash.get_current_metrics()
    assert (m.health_score, m.critical_issues, m.trending) == (0.0, 1, "down")


@pytest.mark.parametrize("before,after,trend", [
    (70.0, 76.0, "up"), (70.0, 75.0, "stable"), (70.0, 65.0, "stable"), (70.0, 64.0, "down"),
])
async def test_trend_needs_more_than_five_points(dash, before, after, trend):
    await _save(dash, "gone.json", ts="2026-01-01T00:00:00", health=before)
    await dash.get_current_metrics()
    await _save(dash, "gone.json", ts="2026-01-02T00:00:00", health=after)
    assert (await dash.get_current_metrics()).trending == trend


async def test_alerts_keep_critical_and_high_only(dash, tmp_path):
    actions = [_action("CRITICAL"), _action("HIGH", steps=()), _action("MEDIUM"), _action("LOW")]
    (tmp_path / "r.json").write_text(json.dumps(_report(actions=actions)))
    await _save(dash, "r.json")
    alerts = await dash.get_critical_alerts()
    assert [a.priority for a in alerts] == ["CRITICAL", "HIGH"]
    assert alerts[0].action_required == "first step"
    assert alerts[1].action_required == "Review required"
    assert alerts[0].category == "Risk"


async def test_alerts_are_empty_when_there_is_nothing_to_read(dash, tmp_path):
    assert await dash.get_critical_alerts() == []
    await _save(dash, None)
    assert await dash.get_critical_alerts() == []
    await _save(dash, "gone.json", ts="2026-01-02T00:00:00")
    assert await dash.get_critical_alerts() == []
    (tmp_path / "bad.json").write_text("{not json")
    await _save(dash, "bad.json", ts="2026-01-03T00:00:00")
    assert await dash.get_critical_alerts() == []


@pytest.mark.parametrize("health,critical,warnings,status,color", [
    (90.0, 1, 0, "CRITICAL", "red"),
    (90.0, 0, 1, "WARNING", "yellow"),
    (80.0, 0, 0, "HEALTHY", "green"),
    (79.9, 0, 0, "DEGRADED", "orange"),
])
async def test_summary_status(dash, health, critical, warnings, status, color):
    await _save(dash, "gone.json", health=health, critical=critical, warnings=warnings)
    summary = await dash.get_dashboard_summary()
    assert (summary["system_status"], summary["status_color"]) == (status, color)
    assert summary["scheduler"]["running"] is False
    assert summary["metrics"]["health_score"] == health
    assert summary["quick_actions"][0]["action"] == "run_analysis"


async def test_emergency_lists_critical_actions_and_steps(dash, monkeypatch):
    report = _report(actions=[_action("CRITICAL"), _action("HIGH")])
    monkeypatch.setattr(pdi, "run_performance_analysis", AsyncMock(return_value=report))
    result = await dash.trigger_emergency_analysis()
    assert [a["action"] for a in result["urgent_actions"]] == ["CRITICAL action"]
    assert result["urgent_actions"][0]["steps"] == ["first step", "second"]
    assert "Manual oversight required" in result["immediate_steps"]


async def test_emergency_without_critical_issues_has_no_steps(dash, monkeypatch):
    monkeypatch.setattr(pdi, "run_performance_analysis", AsyncMock(return_value=_report(critical=0, actions=[])))
    result = await dash.trigger_emergency_analysis()
    assert result["immediate_steps"] == []
    assert result["urgent_actions"] == []


async def test_emergency_failure_reports_it(dash, monkeypatch):
    monkeypatch.setattr(pdi, "run_performance_analysis", AsyncMock(side_effect=RuntimeError("no")))
    result = await dash.trigger_emergency_analysis()
    assert result["error"] == "no"
    assert (result["health_score"], result["critical_issues"]) == (0.0, 1)


async def test_api_functions_share_one_instance(monkeypatch, tmp_path):
    made = []
    real = pdi.DatabaseManager
    monkeypatch.setattr(pdi, "_dashboard_integration", None)
    monkeypatch.setattr(pdi, "DatabaseManager", lambda: made.append(1) or real(str(tmp_path / "api.db")))
    monkeypatch.setattr(pdi, "run_performance_analysis", AsyncMock(side_effect=RuntimeError("no")))
    metrics = await pdi.dashboard_get_metrics()
    alerts = await pdi.dashboard_get_alerts()
    summary = await pdi.dashboard_get_summary()
    emergency = await pdi.dashboard_trigger_analysis()
    assert metrics["last_analysis"] is None
    assert alerts == []
    assert summary["system_status"] == "DEGRADED"
    assert emergency["error"] == "no"
    assert len(made) == 1


# --- analyzer health cutoffs QA flagged as untested -----------------------------------

@pytest.mark.parametrize("win,statuses,expected", [
    (0.7, ["WARNING"], 90),   # 0.7 itself earns no bonus
    (0.4, [], 100),           # 0.4 itself takes no penalty
])
def test_health_score_win_rate_cutoffs_are_exclusive(win, statuses, expected):
    a = apa.AutomatedPerformanceAnalyzer()
    checks = [apa.RiskCheck("X", s, 0, 0, "", "") for s in statuses]
    metrics = apa.PerformanceMetrics(
        total_trades=1, manual_trades=0, automated_trades=1, manual_win_rate=0, automated_win_rate=win,
        overall_win_rate=win, total_pnl=0, manual_pnl=0, automated_pnl=0, unrealized_pnl=0,
        capital_utilization=0, available_cash=0, active_positions=0, avg_position_size=0,
        largest_position_pct=0,
    )
    assert a._calculate_health_score(checks, metrics) == expected
