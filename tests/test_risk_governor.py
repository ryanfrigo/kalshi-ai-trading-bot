"""Tests for the risk governor — the realized-P&L kill switch.

The governor is the one component that stands between the autonomous loop and a
drained account. It must HALT new buys when the account is down past the daily
loss limit or the drawdown-from-peak limit, while always allowing exits, and it
must honor a manual kill-switch file the owner can drop at any time.
"""
import asyncio

import pytest

from src.risk.risk_governor import RiskGovernor, evaluate_risk, roll_state


# --- pure evaluate_risk -----------------------------------------------------

def test_no_loss_not_halted():
    d = evaluate_risk(current=100_000, sod=100_000, peak=100_000,
                      max_daily_loss_pct=10.0, max_drawdown_pct=15.0)
    assert d.halted is False
    assert d.reasons == []


def test_daily_loss_at_threshold_halts():
    # Down exactly 10% from start-of-day -> must halt (inclusive boundary).
    d = evaluate_risk(current=90_000, sod=100_000, peak=100_000,
                      max_daily_loss_pct=10.0, max_drawdown_pct=15.0)
    assert d.halted is True
    assert any("DAILY_LOSS" in r for r in d.reasons)


def test_daily_loss_just_under_threshold_ok():
    # Down 9.9% -> still trading.
    d = evaluate_risk(current=90_100, sod=100_000, peak=100_000,
                      max_daily_loss_pct=10.0, max_drawdown_pct=15.0)
    assert d.halted is False


def test_drawdown_halts_even_when_daily_loss_ok():
    # Start-of-day was already below peak; today only down 5% from sod but
    # 15% from the running peak -> drawdown halt fires.
    d = evaluate_risk(current=85_000, sod=89_000, peak=100_000,
                      max_daily_loss_pct=10.0, max_drawdown_pct=15.0)
    assert d.halted is True
    assert any("DRAWDOWN" in r for r in d.reasons)
    assert not any("DAILY_LOSS" in r for r in d.reasons)


def test_both_reasons_present():
    d = evaluate_risk(current=80_000, sod=100_000, peak=100_000,
                      max_daily_loss_pct=10.0, max_drawdown_pct=15.0)
    assert d.halted is True
    assert any("DAILY_LOSS" in r for r in d.reasons)
    assert any("DRAWDOWN" in r for r in d.reasons)


def test_gain_not_halted_and_pnl_positive():
    d = evaluate_risk(current=110_000, sod=100_000, peak=100_000,
                      max_daily_loss_pct=10.0, max_drawdown_pct=15.0)
    assert d.halted is False
    assert d.daily_pnl_cents == 10_000
    assert d.daily_loss_pct < 0  # negative loss == gain


# --- roll_state (start-of-day + peak bookkeeping) ---------------------------

def test_roll_state_fresh():
    s = roll_state({}, current=100_000, today="2026-06-18")
    assert s == {"date": "2026-06-18", "sod_equity_cents": 100_000,
                 "peak_equity_cents": 100_000}


def test_roll_state_same_day_peak_rises():
    prev = {"date": "2026-06-18", "sod_equity_cents": 100_000, "peak_equity_cents": 100_000}
    s = roll_state(prev, current=120_000, today="2026-06-18")
    assert s["sod_equity_cents"] == 100_000  # SOD never moves intraday
    assert s["peak_equity_cents"] == 120_000  # peak ratchets up


def test_roll_state_same_day_drop_keeps_peak_and_sod():
    prev = {"date": "2026-06-18", "sod_equity_cents": 100_000, "peak_equity_cents": 120_000}
    s = roll_state(prev, current=95_000, today="2026-06-18")
    assert s["sod_equity_cents"] == 100_000
    assert s["peak_equity_cents"] == 120_000  # peak does not fall


def test_roll_state_new_day_resets_sod_carries_peak():
    prev = {"date": "2026-06-18", "sod_equity_cents": 100_000, "peak_equity_cents": 120_000}
    s = roll_state(prev, current=110_000, today="2026-06-19")
    assert s["date"] == "2026-06-19"
    assert s["sod_equity_cents"] == 110_000  # new SOD = today's open
    assert s["peak_equity_cents"] == 120_000  # rolling peak carries across days


# --- RiskGovernor (state file + manual kill switch) -------------------------

def _gov(tmp_path, **kw):
    return RiskGovernor(
        state_path=str(tmp_path / "state.json"),
        halt_flag_path=str(tmp_path / "TRADING_HALTED"),
        clock=lambda: __import__("datetime").datetime(2026, 6, 18, 12, 0, 0),
        **kw,
    )


def test_check_with_injected_equity_halts_on_loss(tmp_path):
    gov = _gov(tmp_path)
    # First call establishes SOD baseline at 100_000.
    d1 = asyncio.run(gov.check(current_equity_cents=100_000))
    assert d1.halted is False
    # Same day, now down 12% -> halt.
    d2 = asyncio.run(gov.check(current_equity_cents=88_000))
    assert d2.halted is True
    assert any("DAILY_LOSS" in r for r in d2.reasons)


def test_manual_halt_trips_and_clears(tmp_path):
    gov = _gov(tmp_path)
    asyncio.run(gov.check(current_equity_cents=100_000))
    assert gov.is_manually_halted() is False

    gov.trip_manual_halt("owner pulled the plug")
    assert gov.is_manually_halted() is True
    d = asyncio.run(gov.check(current_equity_cents=100_000))  # healthy equity...
    assert d.halted is True  # ...but manual halt overrides
    assert any("MANUAL_HALT" in r for r in d.reasons)

    gov.clear_manual_halt()
    assert gov.is_manually_halted() is False
    d2 = asyncio.run(gov.check(current_equity_cents=100_000))
    assert d2.halted is False


def test_state_persists_across_governor_instances(tmp_path):
    g1 = _gov(tmp_path)
    asyncio.run(g1.check(current_equity_cents=100_000))  # SOD = 100_000
    # A fresh instance (new process) must read the same baseline, not reset it.
    g2 = _gov(tmp_path)
    d = asyncio.run(g2.check(current_equity_cents=90_000))
    assert d.halted is True  # down 10% from the persisted SOD
