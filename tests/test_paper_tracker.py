"""Pin the behaviour of src/paper/tracker.py (paper-trading signal ledger).

No network, no credentials, no Kalshi: the DB is a tmp SQLite file and the one
settlement path that talks to an exchange (paper_trader.check_settlements) is driven
with a recording fake client that only has ``get_market``. The paper-vs-live
boundary itself is pinned in tests/test_paper_live_boundary.py and not repeated here;
test_tracker_has_no_exchange_surface only checks tracker.py's own imports.

Bugs found are strict xfails carrying file:line, and are NOT fixed here.
"""
import asyncio
import ast
import math
import sqlite3
from pathlib import Path

import pytest

from src.paper import tracker

TRACKER_SRC = Path(tracker.__file__)


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    path = tmp_path / "paper.db"
    monkeypatch.setattr(tracker, "DB_PATH", str(path))
    return path


def sig(side="NO", price=0.85, market="MKT-A", conf=0.7, **kw):
    return tracker.log_signal(market, "Title " + market, side, price, conf, "why", **kw)


def row(i):
    return next(r for r in tracker.get_all_signals() if r["id"] == i)


# ---------------------------------------------------------------- P&L, hand computed

@pytest.mark.parametrize("side,entry,settle,pnl,outcome", [
    # NO bought at 0.85: pays 0.85, gets $1 if NO wins.
    ("NO", 0.85, 0.0, 0.15, "win"),     # 1 - .85
    ("NO", 0.85, 1.0, -0.85, "loss"),   # -.85
    ("NO", 0.10, 0.0, 0.90, "win"),
    ("NO", 0.10, 1.0, -0.10, "loss"),
    # YES bought at 0.30
    ("YES", 0.30, 1.0, 0.70, "win"),
    ("YES", 0.30, 0.0, -0.30, "loss"),
    ("YES", 0.97, 1.0, 0.03, "win"),
    ("YES", 0.97, 0.0, -0.97, "loss"),
])
def test_pnl_per_contract_hand_computed(side, entry, settle, pnl, outcome):
    i = sig(side, entry)
    tracker.settle_signal(i, settle)
    r = row(i)
    assert r["pnl"] == pytest.approx(pnl, abs=1e-9)
    assert r["outcome"] == outcome
    assert r["settlement_price"] == settle
    assert r["settled_at"]


def test_pnl_rounded_to_4dp():
    i = sig("NO", 1 / 3)
    tracker.settle_signal(i, 0.0)
    assert row(i)["pnl"] == 0.6667  # round(2/3, 4)


def test_win_and_loss_pnl_are_complements_of_one():
    """Same entry: winning pnl - losing pnl == $1 payout."""
    a, b = sig("NO", 0.42), sig("NO", 0.42)
    tracker.settle_signal(a, 0.0)
    tracker.settle_signal(b, 1.0)
    assert row(a)["pnl"] - row(b)["pnl"] == pytest.approx(1.0)


@pytest.mark.xfail(strict=True, reason="src/paper/tracker.py:129,133,138,141 pnl ignores Kalshi trading fees: "
                   "NO @0.60 win should net 0.40 - ceil(.07*.6*.4*100)/100 = 0.38, tracker books 0.40")
def test_pnl_is_net_of_fees():
    i = sig("NO", 0.60)
    tracker.settle_signal(i, 0.0)
    assert row(i)["pnl"] == pytest.approx(0.38)


@pytest.mark.xfail(strict=True, reason="src/paper/tracker.py:129,133,138,141 pnl ignores fees: "
                   "a LOSS also pays the entry fee, YES @0.50 loss should be -0.50 - 0.02 = -0.52")
def test_loss_includes_entry_fee():
    i = sig("YES", 0.50)
    tracker.settle_signal(i, 0.0)
    assert row(i)["pnl"] == pytest.approx(-0.52)


# ---------------------------------------------------------------- settlement edges

def test_settle_unknown_id_is_a_noop():
    i = sig()
    tracker.settle_signal(999, 1.0)
    assert row(i)["outcome"] == "pending"
    assert len(tracker.get_all_signals()) == 1


@pytest.mark.xfail(strict=True, reason="src/paper/tracker.py:127,137 a 0.5 settlement (void / pushed / "
                   "disputed-pending) makes BOTH sides win: NO wins on <=0.5 and YES wins on >=0.5")
def test_half_settlement_does_not_pay_both_sides():
    n, y = sig("NO", 0.5, "M1"), sig("YES", 0.5, "M1")
    tracker.settle_signal(n, 0.5)
    tracker.settle_signal(y, 0.5)
    assert not (row(n)["outcome"] == "win" and row(y)["outcome"] == "win")


def test_half_settlement_current_behaviour_both_win_documented():
    """Companion to the xfail above: today's behaviour, so a change is noticed."""
    n, y = sig("NO", 0.5, "M1"), sig("YES", 0.5, "M1")
    tracker.settle_signal(n, 0.5)
    tracker.settle_signal(y, 0.5)
    assert (row(n)["outcome"], row(y)["outcome"]) == ("win", "win")


@pytest.mark.xfail(strict=True, reason="src/paper/tracker.py:116-148 settling an already-settled signal "
                   "silently overwrites outcome/pnl (no outcome='pending' guard on the UPDATE)")
def test_resettle_does_not_overwrite_a_settled_signal():
    i = sig("NO", 0.85)
    tracker.settle_signal(i, 0.0)
    tracker.settle_signal(i, 1.0)
    assert row(i)["outcome"] == "win"


@pytest.mark.xfail(strict=True, reason="src/paper/tracker.py:125-135 side other than 'NO' (e.g. 'no', "
                   "'garbage') falls into the YES branch and is scored as a YES bet")
@pytest.mark.parametrize("bad_side", ["no", "No", "garbage", ""])
def test_unrecognised_side_is_not_scored_as_yes(bad_side):
    i = sig(bad_side, 0.85)
    tracker.settle_signal(i, 0.0)          # YES lost / NO-ish won
    assert row(i)["outcome"] in ("pending", "win")


@pytest.mark.parametrize("bad", [float("nan"), None, "yes"])
def test_settle_with_malformed_price(bad):
    """NaN is scored as a loss for either side (comparisons False); None/str raise."""
    i = sig("NO")
    if bad is None or isinstance(bad, str):
        with pytest.raises(TypeError):
            tracker.settle_signal(i, bad)
        assert row(i)["outcome"] == "pending"
    else:
        tracker.settle_signal(i, bad)
        assert row(i)["outcome"] == "loss"


@pytest.mark.xfail(strict=True, reason="src/paper/tracker.py:127,137 NaN settlement price is booked as a "
                   "'loss' with pnl -entry instead of being rejected")
def test_nan_settlement_is_rejected():
    i = sig("NO")
    tracker.settle_signal(i, float("nan"))
    assert row(i)["outcome"] == "pending"


# ---------------------------------------------------------------- paper_trader.check_settlements

class FakeKalshi:
    """Read-only: only get_market exists, so any order call is an AttributeError."""
    markets = {}
    calls = []

    def __init__(self, *a, **k):
        pass

    async def get_market(self, ticker):
        FakeKalshi.calls.append(ticker)
        m = FakeKalshi.markets.get(ticker)
        if isinstance(m, Exception):
            raise m
        return m


@pytest.fixture
def settle_run(monkeypatch):
    import paper_trader
    import src.clients.kalshi_client as kc
    monkeypatch.setattr(kc, "KalshiClient", FakeKalshi)
    FakeKalshi.markets, FakeKalshi.calls = {}, []

    def run(markets):
        FakeKalshi.markets = markets
        return asyncio.run(paper_trader.check_settlements())
    return run


@pytest.mark.parametrize("status", ["settled", "finalized"])
@pytest.mark.parametrize("result,expected", [("yes", "loss"), ("no", "win"), ("YES", "loss"), ("No", "win")])
def test_final_status_settles_no_signal(settle_run, status, result, expected):
    i = sig("NO", 0.85)
    assert settle_run({"MKT-A": {"status": status, "result": result}}) == 1
    assert row(i)["outcome"] == expected
    assert FakeKalshi.calls == ["MKT-A"]


@pytest.mark.parametrize("status", ["active", "open", "initialized", "determined", "disputed", "amended", ""])
def test_non_final_status_stays_pending(settle_run, status):
    """'determined' and 'disputed' are Kalshi's pre-final states: the result can still change."""
    i = sig("NO")
    assert settle_run({"MKT-A": {"status": status, "result": "no"}}) == 0
    assert row(i)["outcome"] == "pending"


@pytest.mark.xfail(strict=True, reason="paper_trader.py:157 'closed' counts as final but only means trading "
                   "has stopped; result is still empty, and :161 turns '' into a NO win")
def test_closed_status_with_no_result_stays_pending(settle_run):
    i = sig("NO")
    settle_run({"MKT-A": {"status": "closed", "result": ""}})
    assert row(i)["outcome"] == "pending"


@pytest.mark.xfail(strict=True, reason="paper_trader.py:161 any result other than 'yes' (void, scalar, "
                   "'', missing) is settled as 0.0, so a voided market books a NO win")
@pytest.mark.parametrize("market", [{"status": "settled", "result": "void"},
                                    {"status": "settled", "result": ""},
                                    {"status": "settled"}])
def test_void_or_missing_result_is_not_scored(settle_run, market):
    i = sig("NO", 0.85)
    settle_run({"MKT-A": market})
    assert row(i)["outcome"] == "pending"


def test_result_none_is_swallowed_and_stays_pending(settle_run):
    """None.lower() raises inside the try: logged, signal stays pending, count 0."""
    i = sig("NO")
    assert settle_run({"MKT-A": {"status": "settled", "result": None}}) == 0
    assert row(i)["outcome"] == "pending"


def test_missing_market_and_api_error_stay_pending_and_do_not_stop_the_loop(settle_run):
    a, b, c = sig("NO", 0.8, "GONE"), sig("NO", 0.8, "BOOM"), sig("YES", 0.3, "OK")
    n = settle_run({"GONE": None, "BOOM": RuntimeError("500"), "OK": {"status": "finalized", "result": "yes"}})
    assert n == 1
    assert (row(a)["outcome"], row(b)["outcome"], row(c)["outcome"]) == ("pending", "pending", "win")
    assert row(c)["pnl"] == pytest.approx(0.70)


def test_no_pending_signals_never_builds_a_client(settle_run):
    assert settle_run({}) == 0
    assert FakeKalshi.calls == []


def test_settled_signals_are_not_refetched(settle_run):
    i = sig("NO")
    settle_run({"MKT-A": {"status": "settled", "result": "no"}})
    FakeKalshi.calls.clear()
    assert settle_run({"MKT-A": {"status": "settled", "result": "yes"}}) == 0
    assert FakeKalshi.calls == []
    assert row(i)["outcome"] == "win"


# ---------------------------------------------------------------- persistence / round-trip

def test_log_signal_round_trip():
    i = sig("YES", 0.31, "RT-1", conf=0.66, strategy="market_making")
    r = row(i)
    assert (r["market_id"], r["market_title"], r["side"], r["entry_price"], r["confidence"],
            r["reasoning"], r["strategy"]) == ("RT-1", "Title RT-1", "YES", 0.31, 0.66, "why", "market_making")
    assert r["outcome"] == "pending"
    assert r["pnl"] is None and r["settlement_price"] is None and r["settled_at"] is None
    assert r["timestamp"].endswith("+00:00")


def test_ids_autoincrement_and_survive_reconnect():
    a, b = sig(), sig()
    assert b == a + 1
    assert len(tracker.get_all_signals()) == 2


def test_settlement_persists_across_connections(db):
    i = sig("NO", 0.85)
    tracker.settle_signal(i, 0.0)
    raw = sqlite3.connect(db).execute("SELECT outcome, pnl FROM signals WHERE id=?", (i,)).fetchone()
    assert raw == ("win", 0.15)


def test_get_connection_creates_parent_dir_and_is_idempotent(tmp_path, monkeypatch):
    p = tmp_path / "a" / "b" / "p.db"
    monkeypatch.setattr(tracker, "DB_PATH", str(p))
    tracker.get_connection().close()
    tracker.get_connection().close()
    assert p.exists()
    assert tracker.get_all_signals() == []


def test_pending_signals_filter_and_order():
    a, b, c = sig(market="A"), sig(market="B"), sig(market="C")
    tracker.settle_signal(b, 0.0)
    assert [r["id"] for r in tracker.get_pending_signals()] == [a, c]


def test_unicode_and_sql_metacharacters_round_trip_safely():
    nasty = "Will it rain? '); DROP TABLE signals;-- ☔"
    i = tracker.log_signal("M'1", nasty, "NO", 0.5, 0.6, nasty)
    r = row(i)
    assert r["market_title"] == nasty and r["market_id"] == "M'1" and r["reasoning"] == nasty
    assert len(tracker.get_all_signals()) == 1


# ---------------------------------------------------------------- stats

def test_stats_empty_db():
    s = tracker.get_stats()
    assert s["total_signals"] == 0 and s["settled"] == 0 and s["win_rate"] == 0.0


def test_stats_pending_only():
    sig(); sig()
    s = tracker.get_stats()
    assert (s["total_signals"], s["pending"], s["settled"], s["total_pnl"]) == (2, 2, 0, 0.0)


def test_stats_hand_computed():
    a, b, c, _ = sig("NO", 0.80), sig("NO", 0.90), sig("YES", 0.25), sig()
    tracker.settle_signal(a, 0.0)   # +0.20
    tracker.settle_signal(b, 1.0)   # -0.90
    tracker.settle_signal(c, 1.0)   # +0.75
    s = tracker.get_stats()
    assert s["total_signals"] == 4 and s["settled"] == 3 and s["pending"] == 1
    assert (s["wins"], s["losses"]) == (2, 1)
    assert s["win_rate"] == 66.7
    assert s["total_pnl"] == pytest.approx(0.05)         # .20 - .90 + .75
    assert s["avg_return"] == pytest.approx(0.0167, abs=1e-4)
    assert s["best_trade"] == 0.75 and s["worst_trade"] == -0.9


# ---------------------------------------------------------------- malformed input

@pytest.mark.parametrize("kw", [
    dict(market_id=None), dict(market_title=None), dict(entry_price=None),
])
def test_log_signal_not_null_columns_reject_none(kw):
    args = dict(market_id="M", market_title="T", side="NO", entry_price=0.5, confidence=0.5, reasoning="r")
    args.update(kw)
    with pytest.raises(sqlite3.IntegrityError):
        tracker.log_signal(**args)
    assert tracker.get_all_signals() == []


@pytest.mark.xfail(strict=True, reason="src/paper/tracker.py:87-101 log_signal stores entry_price unvalidated; "
                   "values outside (0,1) or non-finite make pnl (1-entry / -entry) meaningless")
@pytest.mark.parametrize("bad", [-0.2, 0.0, 1.0, 1.5, 85.0, float("nan"), float("inf")])
def test_log_signal_rejects_out_of_range_entry_price(bad):
    with pytest.raises(ValueError):
        tracker.log_signal("M", "T", "NO", bad, 0.5, "r")


@pytest.mark.xfail(strict=True, reason="src/paper/tracker.py:87-101 log_signal accepts any side string "
                   "('maybe'); only 'NO' is recognised on settlement")
def test_log_signal_rejects_bad_side():
    with pytest.raises(ValueError):
        tracker.log_signal("M", "T", "maybe", 0.5, 0.5, "r")


def test_nan_entry_price_is_stored_as_null_and_rejected_by_not_null():
    """sqlite turns NaN into NULL, so the NOT NULL constraint is the only guard."""
    with pytest.raises(sqlite3.IntegrityError):
        tracker.log_signal("M", "T", "NO", float("nan"), 0.5, "r")


def test_stats_ignore_null_pnl_rows():
    """A settled row with NULL pnl (hand-edited db) is counted settled but not summed."""
    i = sig("NO", 0.8)
    tracker.settle_signal(i, 0.0)
    conn = tracker.get_connection()
    conn.execute("UPDATE signals SET pnl = NULL WHERE id = ?", (i,))
    conn.commit(); conn.close()
    s = tracker.get_stats()
    assert s["settled"] == 1 and s["total_pnl"] == 0.0 and s["best_trade"] == 0.0


# ---------------------------------------------------------------- scope: no exchange surface

def test_tracker_has_no_exchange_surface():
    tree = ast.parse(TRACKER_SRC.read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= {"sqlite3", "os", "datetime", "typing", "dataclasses"}, mods
