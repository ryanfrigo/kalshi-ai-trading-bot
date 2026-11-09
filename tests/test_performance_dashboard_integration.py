"""Unit tests for the pure helpers in performance_dashboard_integration.

Only pure helpers are exercised: no DB, no analyzer, no network.
"""

import dataclasses
import datetime as _dt

from src.jobs import performance_dashboard_integration as pdi


def _m(**kw):
    defaults = dict(
        health_score=90.0,
        critical_issues=0,
        warnings=0,
        available_cash=500.0,
        capital_utilization=10.0,
        active_positions=1,
        win_rate=0.5,
        total_pnl=0.0,
        last_analysis=None,
        trending="stable",
    )
    defaults.update(kw)
    return pdi.DashboardMetrics(**defaults)


def _a(priority="HIGH", title="T"):
    return pdi.CriticalAlert(priority, title, "msg", "act", "2026-01-01", "cat")


def _actions(metrics):
    return [action["action"] for action in pdi.PerformanceDashboardIntegration()._generate_quick_actions(metrics, [])]


def test_extract_dashboard_metrics_maps_report_fields():
    report = {
        "summary": {
            "overall_health_score": 78.5,
            "total_critical_issues": 2,
            "total_warnings": 3,
        },
        "performance_metrics": {
            "available_cash": 123.4,
            "capital_utilization": 42.0,
            "active_positions": 7,
            "overall_win_rate": 0.62,
            "total_pnl": -15.5,
        },
        "timestamp": "2026-01-01T09:00:00",
    }

    m = pdi.PerformanceDashboardIntegration()._extract_dashboard_metrics(report)

    assert m.health_score == 78.5
    assert m.critical_issues == 2
    assert m.warnings == 3
    assert m.available_cash == 123.4
    assert m.capital_utilization == 42.0
    assert m.active_positions == 7
    assert m.win_rate == 0.62
    assert m.total_pnl == -15.5
    assert m.last_analysis == "2026-01-01T09:00:00"
    assert m.trending == "stable"


def test_extract_dashboard_metrics_empty_report_uses_defaults():
    m = pdi.PerformanceDashboardIntegration()._extract_dashboard_metrics({})

    assert m.health_score == 0.0
    assert m.critical_issues == 0
    assert m.warnings == 0
    assert m.available_cash == 0.0
    assert m.last_analysis is None


def test_generate_quick_actions_healthy_has_only_run_analysis():
    assert _actions(_m()) == ["run_analysis"]


def test_generate_quick_actions_cash_boundary():
    assert "build_cash" in _actions(_m(available_cash=49.99))
    assert "build_cash" not in _actions(_m(available_cash=50))


def test_generate_quick_actions_positions_boundary():
    assert "reduce_positions" in _actions(_m(active_positions=13))
    assert "reduce_positions" not in _actions(_m(active_positions=12))


def test_generate_quick_actions_utilization_boundary():
    assert "reduce_utilization" in _actions(_m(capital_utilization=80.1))
    assert "reduce_utilization" not in _actions(_m(capital_utilization=80))


def test_generate_quick_actions_all_triggers():
    actions = _actions(_m(available_cash=10.0, active_positions=20, capital_utilization=95.0))

    assert len(actions) == 4
    assert actions[0] == "run_analysis"
    assert set(actions) == {"run_analysis", "build_cash", "reduce_positions", "reduce_utilization"}


def test_get_health_trend_without_data():
    obj = pdi.PerformanceDashboardIntegration()

    assert obj._get_health_trend() == "No trend data available"


def test_get_health_trend_boundaries():
    obj = pdi.PerformanceDashboardIntegration()

    cases = [
        (80, "excellent"),
        (79.9, "good"),
        (60, "good"),
        (59.9, "needs attention"),
        (40, "needs attention"),
        (39.9, "critical"),
    ]
    for score, expected in cases:
        obj._last_metrics = _m(health_score=score)
        assert expected in obj._get_health_trend(), (score, obj._get_health_trend())


def test_get_key_improvements_empty():
    assert pdi.PerformanceDashboardIntegration()._get_key_improvements([]) == [
        "✅ No critical issues detected"
    ]


def test_get_key_improvements_priority_prefixes():
    obj = pdi.PerformanceDashboardIntegration()

    assert obj._get_key_improvements([_a("CRITICAL", "A")]) == ["🚨 A"]
    assert obj._get_key_improvements([_a("HIGH", "B")]) == ["⚠️ B"]


def test_get_key_improvements_returns_first_three_only():
    alerts = [_a("HIGH", t) for t in "ABCDE"]

    assert pdi.PerformanceDashboardIntegration()._get_key_improvements(alerts) == [
        "⚠️ A",
        "⚠️ B",
        "⚠️ C",
    ]


def test_calculate_risk_level():
    obj = pdi.PerformanceDashboardIntegration()

    assert obj._calculate_risk_level(_m(critical_issues=1, capital_utilization=0)) == "HIGH"
    assert obj._calculate_risk_level(_m(warnings=1)) == "MEDIUM"
    assert obj._calculate_risk_level(_m(capital_utilization=81)) == "MEDIUM"
    assert obj._calculate_risk_level(_m(capital_utilization=61)) == "LOW"
    assert obj._calculate_risk_level(_m(capital_utilization=60)) == "MINIMAL"
    assert obj._calculate_risk_level(_m(capital_utilization=80)) == "LOW"


def test_calculate_next_analysis_time(monkeypatch):
    class _FrozenDatetime(_dt.datetime):
        frozen = _dt.datetime(2026, 1, 1, 8, 0)

        @classmethod
        def now(cls, tz=None):
            return cls.frozen

    monkeypatch.setattr(pdi, "datetime", _FrozenDatetime)
    obj = pdi.PerformanceDashboardIntegration()

    _FrozenDatetime.frozen = _dt.datetime(2026, 1, 1, 8, 0)
    assert obj._calculate_next_analysis_time() == "2026-01-01T09:00:00"

    _FrozenDatetime.frozen = _dt.datetime(2026, 1, 1, 10, 0)
    assert obj._calculate_next_analysis_time() == "2026-01-02T09:00:00"

    _FrozenDatetime.frozen = _dt.datetime(2026, 1, 1, 9, 0)
    assert obj._calculate_next_analysis_time() == "2026-01-02T09:00:00"


def test_dataclasses_fields():
    assert dataclasses.is_dataclass(pdi.DashboardMetrics)
    assert dataclasses.asdict(_a()).keys() == {
        "priority",
        "title",
        "message",
        "action_required",
        "target_date",
        "category",
    }
