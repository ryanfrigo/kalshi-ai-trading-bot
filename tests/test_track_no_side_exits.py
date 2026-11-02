"""NO-side exit direction in should_exit_position (src/jobs/track.py).

For a NO position, entry_price and the current price passed in are the price of the NO
contract (decide.py stores no_price; run_tracking passes the NO bid), so a NO position
profits when that price RISES and loses when it FALLS. These tests fix that contract.
"""
from datetime import datetime, timedelta

import pytest

from src.jobs.track import should_exit_position
from src.utils.database import Position

pytestmark = pytest.mark.asyncio


def _no(entry=0.40, **kw):
    return Position(market_id="N-1", side="NO", entry_price=entry, quantity=5,
                    timestamp=kw.pop("timestamp", datetime.now()), live=True,
                    status="open", **kw)


async def _exit(pos, no_bid, yes_bid=0.5):
    return await should_exit_position(pos, yes_bid, no_bid, "active")


# ---- branches that behave correctly today (YES/NO symmetric ones) -------------
def _yes(**kw):
    return Position(market_id="Y", side="YES", entry_price=0.40, quantity=5,
                    timestamp=kw.pop("timestamp", datetime.now()), live=True,
                    status="open", **kw)


async def test_time_based_exit_fires_after_max_hold():
    pos = _yes(stop_loss_price=0.20, max_hold_hours=2, timestamp=datetime.now() - timedelta(hours=3))
    assert (await should_exit_position(pos, 0.40, 0.60, "active"))[:2] == (True, "time_based")


async def test_time_based_exit_does_not_fire_early():
    pos = _yes(stop_loss_price=0.20, max_hold_hours=10, timestamp=datetime.now() - timedelta(hours=1))
    assert (await should_exit_position(pos, 0.40, 0.60, "active"))[0] is False


async def test_yes_take_profit_fires_when_price_rises():
    pos = Position(market_id="Y", side="YES", entry_price=0.40, quantity=5,
                   timestamp=datetime.now(), live=True, status="open",
                   stop_loss_price=0.30, take_profit_price=0.50)
    assert (await should_exit_position(pos, 0.55, 0.45, "active"))[:2] == (True, "take_profit")


async def test_legacy_yes_without_stop_gets_emergency_exit_at_10pct_loss():
    pos = Position(market_id="Y", side="YES", entry_price=0.50, quantity=5,
                   timestamp=datetime.now(), live=True, status="open")
    assert (await should_exit_position(pos, 0.44, 0.56, "active"))[:2] == (
        True, "emergency_stop_loss_10pct")
    assert (await should_exit_position(pos, 0.47, 0.53, "active"))[0] is False


# ---- NO direction: prices are the NO contract's own price ---------------------



async def test_no_position_gaining_does_not_trip_stop_loss():
    # entry .40, NO bid rose to .45 (+12%); .428 is a legacy inverted stop above entry
    pos = _no(0.40, stop_loss_price=0.428)
    assert (await _exit(pos, 0.45))[0] is False


async def test_no_position_losing_does_not_take_profit():
    # entry .40, NO bid fell to .30 (-25%): take_profit_price .32 is "below entry"
    pos = _no(0.40, stop_loss_price=0.428, take_profit_price=0.32)
    should_exit, reason, _ = await _exit(pos, 0.30)
    assert not (should_exit and reason == "take_profit")


async def test_no_position_losing_badly_triggers_a_stop():
    # a -50% NO position must exit; today nothing fires and 'emergency' is inverted
    pos = _no(0.40)
    assert (await _exit(pos, 0.20))[0] is True


async def test_no_position_gaining_does_not_trip_emergency_stop():
    pos = _no(0.40)  # legacy: no stop/take-profit set
    assert (await _exit(pos, 0.45))[0] is False


async def test_no_legacy_inverted_levels_are_recomputed_from_entry():
    # Rows written before the fix hold stop .43 (above entry) / tp .32 (below).
    # Trusted under the new rule they would fire on any price; recomputed from
    # entry .40 (confidence .7) they are stop .37 / tp .48.
    pos = _no(0.40, stop_loss_price=0.43, take_profit_price=0.32)
    assert (await _exit(pos, 0.40))[0] is False
    assert (await _exit(pos, 0.36))[:2] == (True, "stop_loss_triggered_pnl_-0.15")
    assert (await _exit(pos, 0.48))[:2] == (True, "take_profit")


async def test_no_take_profit_fires_when_no_price_rises():
    pos = _no(0.40, stop_loss_price=0.37, take_profit_price=0.48)
    assert (await _exit(pos, 0.50))[:2] == (True, "take_profit")


async def test_no_legacy_emergency_stop_fires_at_10pct_loss():
    pos = _no(0.50)
    assert (await _exit(pos, 0.45))[:2] == (True, "emergency_stop_loss_10pct")
    assert (await _exit(pos, 0.47))[0] is False


# ---- YES lock-in: behaviour must not change -----------------------------------
async def test_yes_stop_fires_when_price_falls_and_logs_loss():
    pos = _yes(stop_loss_price=0.37, take_profit_price=0.48)
    assert (await should_exit_position(pos, 0.36, 0.64, "active"))[:2] == (
        True, "stop_loss_triggered_pnl_-0.15")


async def test_yes_gaining_below_target_holds():
    pos = _yes(stop_loss_price=0.37, take_profit_price=0.48)
    assert (await should_exit_position(pos, 0.45, 0.55, "active"))[0] is False


async def test_yes_losing_does_not_take_profit():
    pos = _yes(stop_loss_price=0.20, take_profit_price=0.48)
    should_exit, reason, _ = await should_exit_position(pos, 0.30, 0.70, "active")
    assert not should_exit and reason == ""


async def test_yes_stored_levels_are_trusted_not_recomputed():
    # A YES stop of .39 is tighter than the recomputed .37; it must be used as stored.
    pos = _yes(stop_loss_price=0.39)
    assert (await should_exit_position(pos, 0.385, 0.615, "active"))[0] is True


async def test_calculator_levels_yes_unchanged_and_no_matches_yes():
    from src.utils.stop_loss_calculator import StopLossCalculator as C
    yes = C.calculate_stop_loss_levels(0.40, "YES", confidence=0.7)
    assert (yes["stop_loss_price"], yes["take_profit_price"]) == (0.37, 0.48)
    assert C.calculate_stop_loss_levels(0.40, "NO", confidence=0.7) == yes
    assert C.calculate_simple_stop_loss(0.50, "YES", 0.10) == 0.45
    assert C.calculate_simple_stop_loss(0.50, "NO", 0.10) == 0.45
