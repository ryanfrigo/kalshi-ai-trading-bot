"""Tests for the risk governor — the realized-P&L kill switch.

The governor is the one component that stands between the autonomous loop and a
drained account. It must HALT new buys when the account is down past the daily
loss limit or the drawdown-from-peak limit, while always allowing exits, and it
must honor a manual kill-switch file the owner can drop at any time.
"""
import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest

from src.risk.risk_governor import RiskGovernor, equity_cents_from_balance, evaluate_risk, roll_state


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


# --- corrupt state fails closed (must not reset the baselines) --------------

def test_missing_state_file_still_works_first_run(tmp_path):
    gov = _gov(tmp_path)
    assert not gov.state_path.exists()
    d = asyncio.run(gov.check(current_equity_cents=100_000))
    assert d.halted is False
    assert d.sod_equity_cents == 100_000
    assert gov.state_path.exists()  # first run persists the baseline


def test_corrupt_json_halts_and_leaves_file_untouched(tmp_path):
    gov = _gov(tmp_path)
    gov.state_path.parent.mkdir(parents=True, exist_ok=True)
    gov.state_path.write_text("{not json")
    d = asyncio.run(gov.check(current_equity_cents=100_000))
    assert d.halted is True
    assert d.reasons == [f"STATE_CORRUPT: {gov.state_path}"]
    assert gov.state_path.read_text() == "{not json"  # operator's evidence, not rewritten


def test_non_dict_json_halts_and_leaves_file_untouched(tmp_path):
    gov = _gov(tmp_path)
    gov.state_path.parent.mkdir(parents=True, exist_ok=True)
    gov.state_path.write_text("[]")
    d = asyncio.run(gov.check(current_equity_cents=100_000))
    assert d.halted is True
    assert d.reasons == [f"STATE_CORRUPT: {gov.state_path}"]
    assert gov.state_path.read_text() == "[]"


# --- atomic save ------------------------------------------------------------

def test_failed_replace_leaves_previous_state_byte_identical(tmp_path, monkeypatch):
    gov = _gov(tmp_path)
    asyncio.run(gov.check(current_equity_cents=100_000))
    before = gov.state_path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        gov.save_state({"date": "2026-06-18", "sod_equity_cents": 1, "peak_equity_cents": 1})
    monkeypatch.undo()

    assert gov.state_path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # tmp cleaned up


def test_unserializable_state_leaves_previous_state_byte_identical(tmp_path):
    gov = _gov(tmp_path)
    asyncio.run(gov.check(current_equity_cents=100_000))
    before = gov.state_path.read_bytes()

    with pytest.raises(TypeError):
        gov.save_state({"date": "2026-06-18", "sod_equity_cents": object(), "peak_equity_cents": 1})

    assert gov.state_path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


# --- schema validation ------------------------------------------------------

def test_valid_written_state_loads(tmp_path):
    gov = _gov(tmp_path)
    asyncio.run(gov.check(current_equity_cents=100_000))
    assert gov.load_state() == {"date": "2026-06-18", "sod_equity_cents": 100_000,
                                "peak_equity_cents": 100_000}


@pytest.mark.parametrize("payload", [
    '{"foo": 1}',
    '{}',
    '{"date": "2026-06-18", "sod_equity_cents": 1}',
    '{"date": 20260618, "sod_equity_cents": 1, "peak_equity_cents": 1}',
    '{"date": "2026-06-18", "sod_equity_cents": "abc", "peak_equity_cents": 1}',
    '{"date": "2026-06-18", "sod_equity_cents": -5, "peak_equity_cents": 1}',
    '{"date": "2026-06-18", "sod_equity_cents": 1, "peak_equity_cents": true}',
])
def test_schema_invalid_state_halts_and_leaves_file_untouched(tmp_path, payload):
    gov = _gov(tmp_path)
    gov.state_path.parent.mkdir(parents=True, exist_ok=True)
    gov.state_path.write_text(payload)
    d = asyncio.run(gov.check(current_equity_cents=100_000))
    assert d.halted is True
    assert d.reasons == [f"STATE_CORRUPT: {gov.state_path}"]
    assert gov.state_path.read_text() == payload


# --- manual halt flag edge cases ---------------------------------------------

def test_an_empty_halt_flag_still_halts_with_a_default_reason(tmp_path):
    gov = _gov(tmp_path)
    (tmp_path / "TRADING_HALTED").write_text("   \n")
    d = asyncio.run(gov.check(current_equity_cents=100_000))
    assert d.halted is True
    assert d.reasons == ["MANUAL_HALT: manual halt flag present"]


def test_an_unreadable_halt_flag_still_halts(tmp_path):
    gov = _gov(tmp_path)
    (tmp_path / "TRADING_HALTED").mkdir()  # exists, but read_text raises OSError
    d = asyncio.run(gov.check(current_equity_cents=100_000))
    assert d.halted is True
    assert d.reasons == ["MANUAL_HALT: manual halt flag present"]


def test_the_manual_halt_reason_comes_first_and_keeps_limit_reasons(tmp_path):
    gov = _gov(tmp_path)
    asyncio.run(gov.check(current_equity_cents=100_000))
    gov.trip_manual_halt("  owner stop  ")
    d = asyncio.run(gov.check(current_equity_cents=88_000))
    assert d.halted is True
    assert d.reasons[0] == "MANUAL_HALT: owner stop"
    assert any(r.startswith("DAILY_LOSS") for r in d.reasons[1:])


def test_clearing_a_halt_that_is_not_set_is_a_no_op(tmp_path):
    gov = _gov(tmp_path)
    gov.clear_manual_halt()
    assert gov.is_manually_halted() is False


def test_trip_manual_halt_creates_missing_parent_dirs(tmp_path):
    flag = tmp_path / "a" / "b" / "TRADING_HALTED"
    gov = RiskGovernor(state_path=str(tmp_path / "s.json"), halt_flag_path=str(flag))
    gov.trip_manual_halt("x")
    assert gov.is_manually_halted() is True
    assert flag.read_text() == "x"


# --- a balance reply missing a figure is a failed read, never a baseline ------

@pytest.mark.parametrize("bal", [
    {},
    {"status_code": 200, "empty_body": True},
    {"balance": 10000},
    {"portfolio_value": 90000},
    {"balance": 10000, "portfolio_value": None},
    {"balance": None, "portfolio_value": 90000},
])
def test_part_of_a_balance_is_no_equity(bal):
    assert equity_cents_from_balance(bal) == 0


def test_a_whole_balance_is_cash_plus_portfolio():
    assert equity_cents_from_balance({"balance": 10000, "portfolio_value": 90000}) == 100000
    assert equity_cents_from_balance({"balance": 0, "portfolio_value": 90000}) == 90000
    assert equity_cents_from_balance({"balance": 10000, "portfolio_value": 0}) == 10000


class _BalanceClient:
    def __init__(self, bal):
        self.bal = bal

    async def get_balance(self):
        return self.bal


def test_a_partial_balance_never_loosens_the_daily_limit(tmp_path):
    today = [datetime(2026, 11, 4, 12, tzinfo=timezone.utc)]
    state = tmp_path / "state.json"
    client = _BalanceClient({"balance": 10000, "portfolio_value": 90000})
    gov = RiskGovernor(
        kalshi_client=client, state_path=str(state), halt_flag_path=str(tmp_path / "halt"),
        max_daily_loss_pct=10.0, max_drawdown_pct=90.0, clock=lambda: today[0],
    )
    assert asyncio.run(gov.check()).halted is False

    today[0] += timedelta(days=1)
    before = state.read_text()
    client.bal = {"balance": 10000}
    partial = asyncio.run(gov.check())
    assert partial.halted is True and partial.reasons[0].startswith("EQUITY_UNREADABLE")
    assert state.read_text() == before

    client.bal = {"balance": 10000, "portfolio_value": 70000}
    assert asyncio.run(gov.check()).sod_equity_cents == 80000
    client.bal = {"balance": 10000, "portfolio_value": 60000}
    down = asyncio.run(gov.check())
    assert down.halted is True and down.reasons[0].startswith("DAILY_LOSS")


def test_a_partial_balance_refuses_a_guarded_order(tmp_path):
    from src.agent.toolbelt import place_guarded_order

    class Client(_BalanceClient):
        async def get_market(self, ticker):
            raise AssertionError("the governor must refuse before the market is read")

    client = Client({"balance": 10000})
    gov = RiskGovernor(kalshi_client=client, state_path=str(tmp_path / "s.json"), halt_flag_path=str(tmp_path / "h"))
    res = asyncio.run(place_guarded_order(client, "T", "no", 1, price=0.5, est_prob=0.6, governor=gov, dry=True))
    assert res["ok"] is False and res["reason"] == "governor_halted"
    assert not (tmp_path / "s.json").exists()
