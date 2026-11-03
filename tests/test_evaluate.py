"""Pin the behaviour of src/jobs/evaluate.py (AI-cost and P&L evaluation).

Scope note: evaluate.py is monitoring only. It has no edge, EV or fee math and
never touches the Kalshi client, so there is nothing to place, cancel or modify.
The "math" under test is the cost/threshold/win-rate arithmetic, checked against
hand-computed values. No network, no credentials. Each test runs with cwd in
tmp_path so the relative `logs/daily_ai_usage.pkl` and `trading_system.db` paths
can never hit the repo's real files.
"""
import os
import pickle
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import aiosqlite
import pytest

from src.config.settings import settings
from src.jobs import evaluate as ev
from src.utils.database import DatabaseManager

TODAY = datetime.now().strftime("%Y-%m-%d")


def day(n):
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


class RecLogger:
    def __init__(self):
        self.records = []

    def __getattr__(self, name):
        if name in ("debug", "info", "warning", "error"):
            return lambda msg, *a, **k: self.records.append((name, msg, k))
        raise AttributeError(name)

    def find(self, level, msg):
        return [k for lv, m, k in self.records if lv == level and m == msg]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings.trading, "daily_ai_budget", 10.0)
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 10.0)
    monkeypatch.setattr(settings.trading, "max_ai_cost_per_decision", 0.08)
    monkeypatch.setattr(ev, "_read_xai_tracker_from_pickle", lambda: None)
    logger = RecLogger()
    monkeypatch.setattr(ev, "get_trading_logger", lambda name: logger)
    return logger


@pytest.fixture
def log(_isolated):
    return _isolated


@pytest.fixture
async def db(tmp_path):
    m = DatabaseManager(str(tmp_path / "e.db"))
    await m.initialize()
    return m


async def sql(db, q, args=()):
    async with aiosqlite.connect(db.db_path) as c:
        await c.execute(q, args)
        await c.commit()


async def cost_row(db, date, cost, analyses, decisions):
    await sql(db, "INSERT INTO daily_cost_tracking VALUES (?,?,?,?)", (date, cost, analyses, decisions))


async def trade(db, pnl, exit_dt, rationale=None):
    await sql(db, "INSERT INTO trade_logs (market_id, side, entry_price, exit_price, quantity, pnl, "
                  "entry_timestamp, exit_timestamp, rationale) VALUES ('M','YES',.5,.6,10,?,?,?,?)",
              (pnl, (exit_dt - timedelta(hours=1)).isoformat(), exit_dt.isoformat(), rationale))


def tracker(**kw):
    base = dict(date=TODAY, total_cost=0.0, request_count=0, daily_limit=50.0, is_exhausted=False)
    base.update(kw)
    return base


# --------------------------------------------------------------------------- xai pickle reader


def write_pickle(**kw):
    os.makedirs("logs", exist_ok=True)
    t = SimpleNamespace(**{**dict(date=TODAY, total_cost=1.5, request_count=7, daily_limit=20.0,
                                  is_exhausted=False), **kw})
    with open("logs/daily_ai_usage.pkl", "wb") as f:
        pickle.dump(t, f)


# capture the real function before the autouse fixture replaces it
_real_reader_fn = ev._read_xai_tracker_from_pickle


def test_pickle_today_is_read():
    write_pickle()
    assert _real_reader_fn() == {"date": TODAY, "total_cost": 1.5, "request_count": 7,
                                 "daily_limit": 20.0, "is_exhausted": False}


def test_pickle_stale_date_missing_file_and_garbage_return_none():
    assert _real_reader_fn() is None  # no file
    write_pickle(date=day(1))
    assert _real_reader_fn() is None  # yesterday's tracker is ignored
    open("logs/daily_ai_usage.pkl", "wb").write(b"not a pickle")
    assert _real_reader_fn() is None  # corrupt file swallowed


def test_pickle_missing_attrs_use_defaults():
    os.makedirs("logs", exist_ok=True)
    with open("logs/daily_ai_usage.pkl", "wb") as f:
        pickle.dump(SimpleNamespace(date=TODAY), f)
    assert _real_reader_fn() == {"date": TODAY, "total_cost": 0.0, "request_count": 0,
                                 "daily_limit": 50.0, "is_exhausted": False}


# --------------------------------------------------------------------------- analyze_ai_costs


async def test_costs_empty_db_all_zero(db, log):
    r = await ev.analyze_ai_costs(db)
    assert r["daily_costs"][TODAY] == {"cost": 0.0, "analyses": 0, "decisions": 0, "xai_requests": 0}
    assert r["daily_costs"][day(1)] == {"cost": 0.0, "analyses": 0, "decisions": 0}
    assert r["weekly_cost"] == 0.0 and r["cost_per_decision"] == 0.0
    assert r["recommendations"] == [] and r["trading_paused"] is False
    assert r["expensive_markets"] == [] and r["analysis_breakdown"] == []
    assert r["actual_daily_limit"] == 10.0


async def test_costs_hand_computed_cost_per_decision_and_weekly(db):
    await cost_row(db, TODAY, 2.40, 8, 3)
    await cost_row(db, day(1), 1.00, 5, 2)
    await cost_row(db, day(8), 100.0, 100, 0)  # outside the 7-day window
    r = await ev.analyze_ai_costs(db)
    assert r["daily_costs"][TODAY]["cost"] == pytest.approx(2.40)
    assert r["daily_costs"][day(1)] == {"cost": 1.0, "analyses": 5, "decisions": 2}
    assert r["cost_per_decision"] == pytest.approx(0.80)       # 2.40 / 3
    assert r["weekly_cost"] == pytest.approx(3.40)             # 2.40 + 1.00, old row excluded
    assert r["recommendations"] == ["💰 High cost per decision: $0.800"]


async def test_costs_zero_decisions_divides_by_one(db):
    await cost_row(db, TODAY, 0.05, 4, 0)
    r = await ev.analyze_ai_costs(db)
    assert r["cost_per_decision"] == pytest.approx(0.05)
    assert not any("High cost per decision" in x for x in r["recommendations"])


@pytest.mark.parametrize("cost,high", [(0.24, False), (0.2400001, True)])
async def test_cost_per_decision_boundary_strict(db, cost, high):
    await cost_row(db, TODAY, cost, 3, 3)  # 0.08 per decision at 0.24
    r = await ev.analyze_ai_costs(db)
    assert any("High cost per decision" in x for x in r["recommendations"]) is high


@pytest.mark.parametrize("cost,soft,hard", [(8.0, False, False), (8.01, True, True),
                                           (7.99, False, False)])
async def test_budget_recommendations_boundary_default_limits(db, cost, soft, hard):
    await cost_row(db, TODAY, cost, 1, 100)  # many decisions: no per-decision rec
    r = await ev.analyze_ai_costs(db)
    recs = " ".join(r["recommendations"])
    assert ("Near soft budget" in recs) is soft
    assert ("Near hard xAI limit" in recs) is hard


async def test_soft_and_hard_thresholds_are_independent(db, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_budget", 5.0)
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 20.0)
    await cost_row(db, TODAY, 4.01, 1, 100)       # > 0.8*5, < 0.8*20
    recs = " ".join((await ev.analyze_ai_costs(db))["recommendations"])
    assert "Near soft budget threshold: $4.010 / $5.00" in recs and "Near hard" not in recs
    await sql(db, "UPDATE daily_cost_tracking SET total_ai_cost = 16.01")
    recs = " ".join((await ev.analyze_ai_costs(db))["recommendations"])
    assert "Near soft budget" in recs and "Near hard xAI limit: $16.010 / $20.00" in recs


@pytest.mark.parametrize("weekly,flag", [(50.0, False), (50.01, True)])
async def test_weekly_spend_threshold_is_5x_soft_budget(db, weekly, flag):
    await cost_row(db, day(2), weekly, 1, 100)
    r = await ev.analyze_ai_costs(db)
    assert any("Weekly spending trending high" in x for x in r["recommendations"]) is flag


@pytest.mark.parametrize("analyses,decisions,flag", [(15, 5, False), (16, 5, True), (1, 0, True), (0, 0, False)])
async def test_analysis_to_decision_ratio_boundary(db, analyses, decisions, flag):
    await cost_row(db, day(1), 0.0, analyses, decisions)
    r = await ev.analyze_ai_costs(db)
    assert any("analysis-to-decision ratio" in x for x in r["recommendations"]) is flag


async def test_expensive_markets_and_breakdown_window_and_order(db):
    old = (datetime.now() - timedelta(days=8)).isoformat()
    rows_ = [("A", datetime.now().isoformat(), 0.30, "deep"), ("A", datetime.now().isoformat(), 0.10, "deep"),
             ("B", datetime.now().isoformat(), 0.50, "quick"), ("C", old, 9.99, "deep"),
             ("D", (datetime.now() - timedelta(days=7)).isoformat(), 0.05, "quick")]  # exactly 7d: kept
    for m, ts, c, t in rows_:
        await sql(db, "INSERT INTO market_analyses (market_id, analysis_timestamp, decision_action, cost_usd, "
                      "analysis_type) VALUES (?,?,'SKIP',?,?)", (m, ts, c, t))
    r = await ev.analyze_ai_costs(db)
    top = [(x[0], x[1], pytest.approx(x[2])) for x in r["expensive_markets"]]
    assert top == [("B", 1, pytest.approx(0.50)), ("A", 2, pytest.approx(0.40)), ("D", 1, pytest.approx(0.05))]
    assert sorted((t, n, round(c, 2)) for t, n, c in r["analysis_breakdown"]) == [
        ("deep", 2, 0.40), ("quick", 2, 0.55)]


async def test_xai_pickle_higher_than_db_wins(db, monkeypatch):
    monkeypatch.setattr(ev, "_read_xai_tracker_from_pickle", lambda: tracker(total_cost=3.0, request_count=12))
    await cost_row(db, TODAY, 1.0, 1, 1)
    r = await ev.analyze_ai_costs(db)
    assert r["daily_costs"][TODAY]["cost"] == 3.0 and r["daily_costs"][TODAY]["xai_requests"] == 12
    assert r["xai_requests_today"] == 12
    assert r["cost_per_decision"] == pytest.approx(3.0)


async def test_db_higher_than_pickle_wins(db, monkeypatch):
    monkeypatch.setattr(ev, "_read_xai_tracker_from_pickle", lambda: tracker(total_cost=0.5, request_count=2))
    await cost_row(db, TODAY, 1.0, 1, 1)
    assert (await ev.analyze_ai_costs(db))["daily_costs"][TODAY]["cost"] == 1.0


@pytest.mark.parametrize("cost,reqs,paused", [(0.0, 0, False),   # stale/fresh-day false positive cleared
                                              (2.0, 0, True), (0.0, 3, True), (5.0, 9, True)])
async def test_exhausted_flag_only_kept_when_usage_confirms(db, monkeypatch, cost, reqs, paused):
    monkeypatch.setattr(ev, "_read_xai_tracker_from_pickle",
                        lambda: tracker(total_cost=cost, request_count=reqs, is_exhausted=True))
    r = await ev.analyze_ai_costs(db)
    assert r["trading_paused"] is paused
    assert any("Trading PAUSED" in x for x in r["recommendations"]) is paused


async def test_nan_tracker_cost_does_not_poison_result(db, monkeypatch):
    monkeypatch.setattr(ev, "_read_xai_tracker_from_pickle",
                        lambda: tracker(total_cost=float("nan"), request_count=1))
    await cost_row(db, TODAY, 1.0, 1, 1)
    r = await ev.analyze_ai_costs(db)
    assert r["daily_costs"][TODAY]["cost"] == 1.0 and r["cost_per_decision"] == 1.0


@pytest.mark.xfail(strict=True, reason="src/jobs/evaluate.py:114 max(db_cost, tracker cost) raises TypeError "
                   "when the pickled tracker has total_cost=None (getattr default only covers a missing attr)")
async def test_none_tracker_cost_does_not_crash(db, monkeypatch):
    monkeypatch.setattr(ev, "_read_xai_tracker_from_pickle", lambda: tracker(total_cost=None, request_count=1))
    await cost_row(db, TODAY, 1.0, 1, 1)
    r = await ev.analyze_ai_costs(db)
    assert r["daily_costs"][TODAY]["cost"] == 1.0


async def test_nan_tracker_cost_understates_when_db_is_zero(db, monkeypatch):
    """Pins the fail-open side of the NaN case: nothing in the DB + NaN tracker => reported $0."""
    monkeypatch.setattr(ev, "_read_xai_tracker_from_pickle",
                        lambda: tracker(total_cost=float("nan"), request_count=4))
    r = await ev.analyze_ai_costs(db)
    assert r["daily_costs"][TODAY]["cost"] == 0.0


# --------------------------------------------------------------------------- analyze_trading_performance


async def test_perf_empty_db(db):
    r = await ev.analyze_trading_performance(db)
    assert (r["total_trades"], r["total_pnl"], r["win_rate"]) == (0, 0.0, 0.0)
    assert r["exit_reasons"] == [] and r["position_stats"][0] == 0


async def test_perf_hand_computed_pnl_and_win_rate(db):
    now = datetime.now()
    for p in (10.0, -4.0, 0.0, 6.0):
        await trade(db, p, now)
    await trade(db, 1000.0, now - timedelta(days=8))  # outside window
    r = await ev.analyze_trading_performance(db)
    assert r["total_trades"] == 4
    assert r["total_pnl"] == pytest.approx(12.0)
    assert r["win_rate"] == pytest.approx(0.5)  # pnl == 0 is not a win: 2 of 4


async def test_perf_window_boundary_7_days_inclusive(db):
    await trade(db, 1.0, datetime.now() - timedelta(days=7))
    await trade(db, 1.0, datetime.now() - timedelta(days=8))
    assert (await ev.analyze_trading_performance(db))["total_trades"] == 1


async def test_perf_all_losses_zero_win_rate(db):
    await trade(db, -2.0, datetime.now())
    await trade(db, -3.0, datetime.now())
    r = await ev.analyze_trading_performance(db)
    assert r["win_rate"] == 0.0 and r["total_pnl"] == pytest.approx(-5.0)


async def test_exit_reason_classification_and_precedence(db):
    now = datetime.now()
    await trade(db, 5.0, now, "closed via market_resolution")
    await trade(db, -3.0, now, "hit stop_loss")
    await trade(db, 4.0, now, "take_profit reached")
    await trade(db, 1.0, now, "time_based exit")
    await trade(db, 2.0, now, "stop_loss then take_profit")  # first CASE arm wins -> stop_loss
    await trade(db, 9.0, now, None)                          # NULL rationale -> other
    r = await ev.analyze_trading_performance(db)
    got = {name: (n, round(avg, 6)) for name, n, avg in r["exit_reasons"]}
    assert got == {"market_resolution": (1, 5.0), "stop_loss": (2, -0.5), "take_profit": (1, 4.0),
                   "time_based": (1, 1.0), "other": (1, 9.0)}


async def test_open_positions_counts_only_open(db):
    ts = (datetime.now() - timedelta(hours=1)).isoformat()
    for i, status in enumerate(("open", "open", "pending", "closed")):
        await sql(db, "INSERT INTO positions (market_id, side, entry_price, quantity, timestamp, status) "
                      "VALUES (?,'YES',.5,1,?,?)", (f"M{i}", ts, status))
    assert (await ev.analyze_trading_performance(db))["position_stats"][0] == 2


@pytest.mark.xfail(strict=True, reason="src/jobs/evaluate.py:225 julianday('now') is UTC but positions."
                   "timestamp is naive local isoformat (database.py), so avg_hours_held is off by the "
                   "UTC offset (negative for a fresh position west of UTC)")
async def test_avg_hours_held_is_timezone_correct(db, monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        ts = (datetime.now() - timedelta(hours=2)).isoformat()
        await sql(db, "INSERT INTO positions (market_id, side, entry_price, quantity, timestamp, status) "
                      "VALUES ('M','YES',.5,1,?,'open')", (ts,))
        hours = (await ev.analyze_trading_performance(db))["position_stats"][1]
        assert hours == pytest.approx(2.0, abs=0.1)
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


# --------------------------------------------------------------------------- run_evaluation


def fake_analyses(monkeypatch, cost=0.0, limit=10.0, paused=False, reqs=0, recs=(), positions=(3, 1.5)):
    async def costs(_db):
        return {"daily_costs": {TODAY: {"cost": cost}}, "recommendations": list(recs),
                "trading_paused": paused, "xai_requests_today": reqs, "actual_daily_limit": limit}

    async def perf(_db):
        return {"position_stats": positions}
    monkeypatch.setattr(ev, "analyze_ai_costs", costs)
    monkeypatch.setattr(ev, "analyze_trading_performance", perf)


async def status_for(monkeypatch, log, **kw):
    log.records.clear()
    fake_analyses(monkeypatch, **kw)
    await ev.run_evaluation()
    (summary,) = log.find("info", "System Health Summary")
    return summary


@pytest.mark.parametrize("cost,status", [
    (0.0, "🟢 HEALTHY"), (7.0, "🟢 HEALTHY"),              # 0.70 exactly: strict >
    (7.1, "🟡 HIGH USAGE"), (9.0, "🟡 HIGH USAGE"),         # 0.90 exactly: strict >
    (9.1, "🔴 OVER BUDGET"), (25.0, "🔴 OVER BUDGET"),
])
async def test_health_status_boundaries(monkeypatch, log, cost, status):
    s = await status_for(monkeypatch, log, cost=cost)
    assert s["status"] == status
    assert s["daily_budget_used"] == f"{cost / 10.0:.1%}"


async def test_paused_overrides_budget_status_only_with_usage(monkeypatch, log):
    assert (await status_for(monkeypatch, log, cost=0.0, paused=True, reqs=0))["status"] == "🟢 HEALTHY"
    assert (await status_for(monkeypatch, log, cost=0.0, paused=True, reqs=1))["status"].startswith("🔴 PAUSED")
    assert (await status_for(monkeypatch, log, cost=0.5, paused=True, reqs=0))["status"].startswith("🔴 PAUSED")
    assert (await status_for(monkeypatch, log, cost=9.5, paused=False))["status"] == "🔴 OVER BUDGET"


@pytest.mark.parametrize("cost,alert", [(8.0, False), (8.01, True)])
async def test_high_cost_alert_threshold(monkeypatch, log, cost, alert):
    await status_for(monkeypatch, log, cost=cost)
    assert bool(log.find("warning", "HIGH COST ALERT: Consider immediate actions")) is alert


async def test_summary_reports_recommendation_count_and_open_positions(monkeypatch, log):
    s = await status_for(monkeypatch, log, recs=["a", "b"], positions=(4, 2.0))
    assert s["total_recommendations"] == 2 and s["open_positions"] == 4


async def test_run_evaluation_swallows_sub_analysis_error(monkeypatch, log):
    async def boom(_db):
        raise RuntimeError("db locked")
    monkeypatch.setattr(ev, "analyze_ai_costs", boom)
    await ev.run_evaluation()  # does not raise
    (err,) = log.find("error", "Error in evaluation job")
    assert err["error"] == "db locked"


async def test_run_evaluation_end_to_end_on_temp_db(log):
    """Real analyses against a fresh DatabaseManager() created in tmp cwd; healthy and quiet."""
    await ev.run_evaluation()
    assert os.path.exists("trading_system.db")
    (s,) = log.find("info", "System Health Summary")
    assert s["status"] == "🟢 HEALTHY" and s["open_positions"] == 0
    assert not log.find("error", "Error in evaluation job")


@pytest.mark.xfail(strict=True, reason="src/jobs/evaluate.py:277 actual_limit == 0 yields utilization 0 -> "
                   "'HEALTHY' even with spend over a zero limit (analyze_ai_costs flags it, summary does not)")
async def test_zero_limit_with_spend_is_not_healthy(monkeypatch, log):
    s = await status_for(monkeypatch, log, cost=5.0, limit=0)
    assert s["status"] != "🟢 HEALTHY"
