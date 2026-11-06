"""Tests for the automated performance analyzer: metrics, risk checks, action items, report.

Every client is a fake. FakeKalshi has only the two read methods the analyzer may
use, so any attempt to place or cancel an order raises AttributeError.
"""
import json
from datetime import datetime, timedelta

import aiosqlite
import pytest

from src.jobs import automated_performance_analyzer as apa
from src.jobs.automated_performance_analyzer import (
    AutomatedPerformanceAnalyzer,
    PerformanceMetrics,
    Priority,
    RiskCheck,
)
from src.utils.database import DatabaseManager


class FakeKalshi:
    def __init__(self, balance_cents=100_000, positions=()):
        self.balance_cents = balance_cents
        self.positions = list(positions)
        self.closed = False

    async def get_positions(self, **_):
        return {"market_positions": self.positions}

    async def get_balance(self):
        return {"balance": self.balance_cents}

    async def close(self):
        self.closed = True


class FakeAI:
    def __init__(self, reply="three priorities", error=None, total_cost=0.25):
        self.reply = reply
        self.error = error
        self.total_cost = total_cost
        self.prompts = []
        self.closed = False

    async def get_completion(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        if self.error:
            raise self.error
        return self.reply

    async def close(self):
        self.closed = True


def _metrics(**over):
    base = dict(
        total_trades=10, manual_trades=0, automated_trades=10, manual_win_rate=0.0,
        automated_win_rate=0.65, overall_win_rate=0.65, total_pnl=5.0, manual_pnl=0.0,
        automated_pnl=5.0, unrealized_pnl=0, capital_utilization=50.0, available_cash=500.0,
        active_positions=3, avg_position_size=10.0, largest_position_pct=5.0,
    )
    base.update(over)
    return PerformanceMetrics(**base)


def _portfolio(total=1000.0):
    return {"active_positions": 3, "total_contracts": 30, "available_cash": 500.0,
            "positions_detail": [], "total_portfolio_value": total}


def _check(name, status, current=50.0, threshold=15.0):
    return RiskCheck(name, status, current, threshold, "do it", "impact")


def _by_name(checks):
    return {c.check_name: c for c in checks}


@pytest.fixture
async def analyzer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = AutomatedPerformanceAnalyzer()
    a.db = DatabaseManager(str(tmp_path / "t.db"))
    await a.db.initialize()
    a.kalshi_client = FakeKalshi()
    a.xai_client = FakeAI()
    return a


async def _add_trade(a, pnl, rationale="auto", ts="2026-01-05T00:00:00", market="M"):
    async with aiosqlite.connect(a.db.db_path) as db:
        await db.execute(
            "INSERT INTO trade_logs (market_id, side, entry_price, exit_price, quantity, pnl,"
            " entry_timestamp, exit_timestamp, rationale) VALUES (?,?,?,?,?,?,?,?,?)",
            (market, "YES", 0.5, 0.6, 1, pnl, ts, ts, rationale),
        )
        await db.commit()


async def _add_position(a, market, price, qty, status="open"):
    async with aiosqlite.connect(a.db.db_path) as db:
        await db.execute(
            "INSERT INTO positions (market_id, side, entry_price, quantity, timestamp, status)"
            " VALUES (?,?,?,?,?,?)",
            (market, "YES", price, qty, "2026-01-05T00:00:00", status),
        )
        await db.commit()


# --- portfolio data -------------------------------------------------------------------

async def test_portfolio_counts_only_nonzero_positions_and_values_contracts_at_50c(analyzer):
    analyzer.kalshi_client = FakeKalshi(
        balance_cents=12_345,
        positions=[{"position_fp": "10.00"}, {"position_fp": "-4.00"}, {"position_fp": "0.00"}, {}],
    )
    data = await analyzer._gather_portfolio_data()
    assert data["active_positions"] == 2
    assert data["total_contracts"] == 14
    assert data["available_cash"] == pytest.approx(123.45)
    assert data["total_portfolio_value"] == pytest.approx(123.45 + 7.0)
    assert len(data["positions_detail"]) == 2


async def test_portfolio_with_an_empty_balance_reply_raises(analyzer):
    class Empty(FakeKalshi):
        async def get_balance(self):
            return {}

    analyzer.kalshi_client = Empty()
    with pytest.raises(ValueError, match="no balance field"):
        await analyzer._gather_portfolio_data()


async def test_portfolio_with_a_real_zero_balance_reads_zero(analyzer):
    analyzer.kalshi_client = FakeKalshi(balance_cents=0)
    data = await analyzer._gather_portfolio_data()
    assert data["available_cash"] == 0
    assert data["total_portfolio_value"] == 0


async def test_portfolio_with_an_unreadable_positions_reply_raises(analyzer):
    class Empty(FakeKalshi):
        async def get_positions(self, **_):
            return {}

    analyzer.kalshi_client = Empty()
    with pytest.raises(RuntimeError, match="malformed market_positions page"):
        await analyzer._gather_portfolio_data()


# --- metrics --------------------------------------------------------------------------

async def test_metrics_split_manual_and_automated(analyzer):
    await _add_trade(analyzer, 2.0, "manual entry")
    await _add_trade(analyzer, -1.0, "MANUAL exit")
    await _add_trade(analyzer, 3.0, "edge 5%")
    await _add_trade(analyzer, 4.0, None)
    await _add_trade(analyzer, -2.0, "edge 3%")
    m = await analyzer._calculate_performance_metrics()
    assert (m.total_trades, m.manual_trades, m.automated_trades) == (5, 2, 3)
    assert m.manual_win_rate == pytest.approx(0.5)
    assert m.automated_win_rate == pytest.approx(2 / 3)
    assert m.overall_win_rate == pytest.approx(0.6)
    assert m.total_pnl == pytest.approx(6.0)
    assert m.manual_pnl == pytest.approx(1.0)
    assert m.automated_pnl == pytest.approx(5.0)


async def test_metrics_record_the_trade_date_range(analyzer):
    await _add_trade(analyzer, 1.0, ts="2026-01-05T10:00:00")
    await _add_trade(analyzer, 1.0, ts="2026-03-09T10:00:00")
    await analyzer._calculate_performance_metrics()
    assert analyzer.trade_date_range == {"earliest": "2026-01-05", "latest": "2026-03-09"}


async def test_metrics_on_empty_ledger_are_zero_not_none(analyzer):
    m = await analyzer._calculate_performance_metrics()
    assert m.total_trades == 0
    assert m.overall_win_rate == 0
    assert m.total_pnl == 0
    assert m.manual_pnl == 0
    assert m.automated_pnl == 0
    assert m.avg_position_size == 0
    assert m.largest_position_pct == 0
    assert analyzer.trade_date_range == {"earliest": "Unknown", "latest": "Unknown"}


async def test_metrics_utilization_and_largest_position_use_portfolio_value(analyzer):
    # $100 cash + 100 contracts valued at 50c = $150 total.
    analyzer.kalshi_client = FakeKalshi(balance_cents=10_000, positions=[{"position_fp": "100.00"}])
    await _add_position(analyzer, "A", 0.30, 50)
    await _add_position(analyzer, "B", 0.10, 50)
    await _add_position(analyzer, "C", 0.90, 1000, status="closed")
    m = await analyzer._calculate_performance_metrics()
    assert m.capital_utilization == pytest.approx(50 / 150 * 100)
    assert m.available_cash == pytest.approx(100.0)
    assert m.active_positions == 1
    assert m.avg_position_size == pytest.approx(10.0)
    assert m.largest_position_pct == pytest.approx(15 / 150 * 100)


async def test_metrics_on_zero_portfolio_do_not_divide_by_zero(analyzer):
    analyzer.kalshi_client = FakeKalshi(balance_cents=0)
    await _add_position(analyzer, "A", 0.30, 50)
    m = await analyzer._calculate_performance_metrics()
    assert m.capital_utilization == 0
    assert m.largest_position_pct == 0


# --- risk checks ----------------------------------------------------------------------

@pytest.mark.parametrize("cash,status", [
    (99.0, "CRITICAL"), (100.0, "WARNING"), (149.0, "WARNING"), (150.0, "PASS"), (500.0, "PASS"),
])
async def test_cash_reserve_thresholds(analyzer, cash, status):
    checks = await analyzer._run_risk_checks(_portfolio(), _metrics(available_cash=cash))
    check = _by_name(checks)["Cash Reserve Threshold"]
    assert check.status == status
    assert check.current_value == pytest.approx(cash / 10)
    assert check.threshold_value == 15.0


async def test_cash_warning_names_the_shortfall(analyzer):
    checks = await analyzer._run_risk_checks(_portfolio(), _metrics(available_cash=120.0))
    assert "$30.00" in _by_name(checks)["Cash Reserve Threshold"].impact


async def test_zero_value_with_positions_held_stays_critical(analyzer):
    checks = await analyzer._run_risk_checks(_portfolio(total=0), _metrics(available_cash=0))
    check = _by_name(checks)["Cash Reserve Threshold"]
    assert (check.status, check.current_value) == ("CRITICAL", 0)
    assert "Close some positions" in check.recommendation


async def test_empty_account_is_a_warning_with_nothing_to_close(analyzer):
    checks = await analyzer._run_risk_checks(
        _portfolio(total=0), _metrics(available_cash=0, active_positions=0))
    check = _by_name(checks)["Cash Reserve Threshold"]
    assert (check.status, check.current_value, check.threshold_value) == ("WARNING", 0, 15.0)
    assert check.recommendation == "No capital and no open positions, so there is nothing to close"
    assert check.impact == "The bot cannot trade until the account holds cash"


async def test_no_positions_with_cash_still_passes(analyzer):
    checks = await analyzer._run_risk_checks(
        _portfolio(total=1000.0), _metrics(available_cash=1000.0, active_positions=0))
    assert _by_name(checks)["Cash Reserve Threshold"].status == "PASS"


async def test_no_positions_with_low_cash_share_stays_critical(analyzer):
    checks = await analyzer._run_risk_checks(
        _portfolio(total=1000.0), _metrics(available_cash=50.0, active_positions=0))
    assert _by_name(checks)["Cash Reserve Threshold"].status == "CRITICAL"


@pytest.mark.parametrize("count,status", [(12, "PASS"), (13, "WARNING")])
async def test_position_count_threshold(analyzer, count, status):
    checks = await analyzer._run_risk_checks(_portfolio(), _metrics(active_positions=count))
    assert _by_name(checks)["Position Concentration"].status == status


async def test_position_count_warning_says_how_many_to_close(analyzer):
    checks = await analyzer._run_risk_checks(_portfolio(), _metrics(active_positions=15))
    assert "Close 3 positions" in _by_name(checks)["Position Concentration"].recommendation


@pytest.mark.parametrize("util,status", [
    (80.0, "PASS"), (80.1, "WARNING"), (90.0, "WARNING"), (90.1, "CRITICAL"),
])
async def test_capital_utilization_thresholds(analyzer, util, status):
    checks = await analyzer._run_risk_checks(_portfolio(), _metrics(capital_utilization=util))
    assert _by_name(checks)["Capital Utilization"].status == status


@pytest.mark.parametrize("pct,status", [(10.0, "PASS"), (10.1, "WARNING")])
async def test_single_position_threshold(analyzer, pct, status):
    checks = await analyzer._run_risk_checks(_portfolio(), _metrics(largest_position_pct=pct))
    assert _by_name(checks)["Single Position Risk"].status == status


async def test_manual_gap_check_needs_both_kinds_of_trade(analyzer):
    for over in ({"manual_trades": 0}, {"automated_trades": 0}):
        checks = await analyzer._run_risk_checks(_portfolio(), _metrics(**over))
        assert "Manual vs Automated Performance" not in _by_name(checks)


@pytest.mark.parametrize("manual,auto,status", [(0.75, 0.6, "PASS"), (0.81, 0.6, "WARNING"), (0.1, 0.9, "PASS")])
async def test_manual_gap_threshold(analyzer, manual, auto, status):
    metrics = _metrics(manual_trades=4, automated_trades=4, manual_win_rate=manual, automated_win_rate=auto)
    checks = await analyzer._run_risk_checks(_portfolio(), metrics)
    assert _by_name(checks)["Manual vs Automated Performance"].status == status


# --- AI analysis ----------------------------------------------------------------------

async def test_ai_analysis_success_carries_text_cost_and_prompt_facts(analyzer):
    checks = [_check("Capital Utilization", "WARNING", 85.0, 80.0)]
    result = await analyzer._generate_grok_analysis(_portfolio(), _metrics(), checks)
    assert result["status"] == "success"
    assert result["analysis_text"] == "three priorities"
    assert result["cost"] == 0.25
    prompt, kwargs = analyzer.xai_client.prompts[0]
    assert kwargs["strategy"] == "performance_analysis"
    assert "Available Cash: $500.00" in prompt
    assert "Capital Utilization: WARNING (85.0% vs 80.0% threshold)" in prompt


async def test_ai_error_falls_back_with_no_cost(analyzer):
    analyzer.xai_client = FakeAI(error=RuntimeError("boom"))
    checks = [_check("Cash Reserve Threshold", "CRITICAL"), _check("Position Concentration", "WARNING")]
    result = await analyzer._generate_grok_analysis(_portfolio(), _metrics(), checks)
    assert result["status"] == "fallback"
    assert result["model"] == "fallback"
    assert result["cost"] == 0
    assert result["error"] == "boom"
    assert "CRITICAL ISSUES:\n- Cash Reserve Threshold" in result["analysis_text"]
    assert "WARNINGS:\n- Position Concentration" in result["analysis_text"]


@pytest.mark.parametrize("reply", [None, ""])
async def test_ai_with_no_reply_is_a_fallback_not_a_success(analyzer, reply):
    # XAIClient.get_completion returns None when the daily cap is closed or a call fails.
    analyzer.xai_client = FakeAI(reply=reply)
    result = await analyzer._generate_grok_analysis(_portfolio(), _metrics(), [])
    assert result["status"] == "fallback"
    assert result["cost"] == 0
    assert result["analysis_text"].startswith("FALLBACK ANALYSIS")


async def test_report_text_claims_no_profitable_manual_history(analyzer):
    analyzer.xai_client = FakeAI(error=RuntimeError("boom"))
    result = await analyzer._generate_grok_analysis(_portfolio(), _metrics(), [])
    prompt, _ = analyzer.xai_client.prompts[0]
    assert "profitable" not in prompt.lower()
    assert "profitable" not in result["analysis_text"].lower()


def test_fallback_without_issues_has_only_the_summary():
    a = AutomatedPerformanceAnalyzer()
    text = a._generate_fallback_analysis(_metrics(), [_check("X", "PASS")])
    assert "CRITICAL ISSUES" not in text
    assert "WARNINGS" not in text
    assert "Capital Utilization: 50.0%" in text


def test_prompt_format_marks_only_utilization_as_percent():
    a = AutomatedPerformanceAnalyzer()
    text = a._format_risk_checks_for_prompt([
        _check("Position Concentration", "PASS", 3.0, 12.0),
        _check("Capital Utilization", "PASS", 50.0, 80.0),
    ])
    assert "Position Concentration: PASS (3.0 vs 12.0 threshold)" in text
    assert "Capital Utilization: PASS (50.0% vs 80.0% threshold)" in text


# --- action items ---------------------------------------------------------------------

async def test_critical_cash_action_targets_the_shortfall(analyzer):
    checks = [_check("Cash Reserve Threshold", "CRITICAL", 5.0)]
    items = await analyzer._generate_action_items(checks, _metrics(available_cash=50.0))
    assert len(items) == 1
    assert items[0].priority is Priority.CRITICAL
    assert "Target $100.00 additional cash" in items[0].implementation_steps
    assert items[0].target_date < datetime.now() + timedelta(days=1, seconds=5)


@pytest.mark.parametrize("cash_pct", [0, -5.0])
async def test_critical_cash_at_or_below_zero_percent_still_makes_an_action(analyzer, cash_pct):
    checks = [_check("Cash Reserve Threshold", "CRITICAL", cash_pct)]
    items = await analyzer._generate_action_items(checks, _metrics(available_cash=0.0))
    assert items[0].action == "Build cash reserves immediately"
    assert "Target cash at 15% of portfolio value" in items[0].implementation_steps


async def test_critical_utilization_action(analyzer):
    checks = [_check("Capital Utilization", "CRITICAL", 95.0, 80.0)]
    items = await analyzer._generate_action_items(checks, _metrics())
    assert [i.action for i in items] == ["Reduce capital utilization"]
    assert items[0].priority is Priority.CRITICAL


async def test_warning_actions_and_ordering(analyzer):
    checks = [
        _check("Position Concentration", "WARNING", 15.0, 12.0),
        _check("Manual vs Automated Performance", "WARNING", 30.0, 20.0),
        _check("Capital Utilization", "WARNING", 85.0, 80.0),
    ]
    items = await analyzer._generate_action_items(checks, _metrics())
    assert [i.priority for i in items] == [Priority.HIGH, Priority.MEDIUM]
    assert "Close 3 lowest edge positions" in items[0].implementation_steps


@pytest.mark.parametrize("auto_trades,rate,expected", [(10, 0.59, 1), (10, 0.6, 0), (0, 0.0, 0)])
async def test_low_automated_win_rate_adds_one_item(analyzer, auto_trades, rate, expected):
    items = await analyzer._generate_action_items(
        [], _metrics(automated_trades=auto_trades, automated_win_rate=rate))
    assert len(items) == expected


async def test_passing_checks_make_no_actions(analyzer):
    items = await analyzer._generate_action_items([_check("Cash Reserve Threshold", "PASS")], _metrics())
    assert items == []


# --- health score ---------------------------------------------------------------------

@pytest.mark.parametrize("statuses,win,expected", [
    ([], 0.5, 100),
    (["CRITICAL"], 0.5, 75),
    (["WARNING"], 0.5, 90),
    (["PASS"], 0.5, 100),
    (["CRITICAL", "WARNING"], 0.5, 65),
    ([], 0.71, 100),
    (["WARNING"], 0.71, 100),
    ([], 0.39, 85),
    (["CRITICAL"] * 5, 0.39, 0),
])
def test_health_score(statuses, win, expected):
    a = AutomatedPerformanceAnalyzer()
    checks = [_check("X", s) for s in statuses]
    assert a._calculate_health_score(checks, _metrics(overall_win_rate=win)) == expected


@pytest.mark.parametrize("manual,auto,expected", [(0.9, 0.5, 90), (0.5, 0.9, 90), (0.75, 0.5, 100)])
def test_health_score_manual_gap_counts_either_way(manual, auto, expected):
    a = AutomatedPerformanceAnalyzer()
    metrics = _metrics(overall_win_rate=0.5, manual_trades=2, automated_trades=2,
                       manual_win_rate=manual, automated_win_rate=auto)
    assert a._calculate_health_score([], metrics) == expected


# --- full run -------------------------------------------------------------------------

async def test_full_run_on_a_clean_install_completes(analyzer):
    analyzer.kalshi_client = FakeKalshi(balance_cents=100_000)
    report = await analyzer.run_full_analysis()
    assert report["performance_metrics"]["total_trades"] == 0
    assert report["grok_analysis"]["status"] == "success"
    assert report["summary"]["overall_health_score"] <= 100


async def test_full_run_with_no_cash_completes_and_reports_it(analyzer):
    analyzer.kalshi_client = FakeKalshi(balance_cents=0, positions=[{"position_fp": "10.00"}])
    report = await analyzer.run_full_analysis()
    assert report["summary"]["total_critical_issues"] >= 1
    assert report["summary"]["critical_actions"] >= 1


async def test_full_run_on_an_empty_account_has_no_close_advice(analyzer):
    analyzer.kalshi_client = FakeKalshi(balance_cents=0)
    report = await analyzer.run_full_analysis()
    assert report["summary"]["total_critical_issues"] == 0
    assert report["summary"]["critical_actions"] == 0
    assert report["summary"]["total_warnings"] >= 1
    assert "Build cash reserves immediately" not in [a["action"] for a in report["action_items"]]


async def test_report_priorities_are_the_strings_every_reader_compares(analyzer):
    # performance_scheduler and performance_dashboard_integration compare
    # action['priority'] with 'CRITICAL', and read the saved JSON too.
    analyzer.kalshi_client = FakeKalshi(balance_cents=0, positions=[{"position_fp": "10.00"}])
    report = await analyzer.run_full_analysis()
    priorities = [a["priority"] for a in report["action_items"]]
    assert "CRITICAL" in priorities
    assert all(isinstance(p, str) for p in priorities)

    async with aiosqlite.connect(analyzer.db.db_path) as db:
        row = await (await db.execute("SELECT report_file FROM analysis_reports")).fetchone()
    saved = json.load(open(row[0]))
    assert "CRITICAL" in [a["priority"] for a in saved["action_items"]]


async def test_report_is_saved_to_file_and_table(analyzer, tmp_path):
    analyzer.kalshi_client = FakeKalshi(balance_cents=100_000)
    await _add_trade(analyzer, 1.0)
    report = await analyzer.run_full_analysis()
    async with aiosqlite.connect(analyzer.db.db_path) as db:
        row = await (await db.execute(
            "SELECT timestamp, health_score, critical_issues, warnings, action_items, report_file"
            " FROM analysis_reports")).fetchone()
    assert row[0] == report["timestamp"]
    assert row[1] == report["summary"]["overall_health_score"]
    assert row[2] == report["summary"]["total_critical_issues"]
    assert row[3] == report["summary"]["total_warnings"]
    assert row[4] == len(report["action_items"])
    assert (tmp_path / row[5]).exists()


async def test_full_run_only_uses_the_two_read_calls(analyzer):
    # FakeKalshi has no order methods, so a place or cancel would raise and fail this run.
    analyzer.kalshi_client = FakeKalshi(balance_cents=0, positions=[{"position_fp": "10.00"}])
    assert not hasattr(analyzer.kalshi_client, "place_order")
    await analyzer.run_full_analysis()


async def test_full_run_reraises_a_failed_read(analyzer):
    class Down(FakeKalshi):
        async def get_balance(self):
            raise ConnectionError("down")

    analyzer.kalshi_client = Down()
    with pytest.raises(ConnectionError):
        await analyzer.run_full_analysis()


async def test_full_run_with_an_empty_balance_reply_makes_no_ai_call_and_saves_no_report(analyzer):
    class Empty(FakeKalshi):
        async def get_balance(self):
            return {}

    analyzer.kalshi_client = Empty()
    with pytest.raises(ValueError, match="no balance field"):
        await analyzer.run_full_analysis()
    assert analyzer.xai_client.prompts == []
    async with aiosqlite.connect(analyzer.db.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_reports'")
        if await cursor.fetchone():
            row = await (await db.execute("SELECT COUNT(*) FROM analysis_reports")).fetchone()
            assert row[0] == 0


# --- lifecycle ------------------------------------------------------------------------

async def test_close_without_initialize_is_safe():
    await AutomatedPerformanceAnalyzer().close()


async def test_close_closes_both_clients(analyzer):
    kalshi, ai = analyzer.kalshi_client, analyzer.xai_client
    await analyzer.close()
    assert kalshi.closed and ai.closed


async def test_entry_point_closes_on_failure(monkeypatch):
    seen = {}

    async def fake_init(self):
        pass

    async def fake_run(self):
        raise RuntimeError("no")

    async def fake_close(self):
        seen["closed"] = True

    monkeypatch.setattr(AutomatedPerformanceAnalyzer, "initialize", fake_init)
    monkeypatch.setattr(AutomatedPerformanceAnalyzer, "run_full_analysis", fake_run)
    monkeypatch.setattr(AutomatedPerformanceAnalyzer, "close", fake_close)
    with pytest.raises(RuntimeError):
        await apa.run_performance_analysis()
    assert seen["closed"]


async def test_entry_point_returns_the_report(monkeypatch):
    async def fake_init(self):
        pass

    async def fake_run(self):
        return {"ok": True}

    async def fake_close(self):
        pass

    monkeypatch.setattr(AutomatedPerformanceAnalyzer, "initialize", fake_init)
    monkeypatch.setattr(AutomatedPerformanceAnalyzer, "run_full_analysis", fake_run)
    monkeypatch.setattr(AutomatedPerformanceAnalyzer, "close", fake_close)
    assert await apa.run_performance_analysis() == {"ok": True}
