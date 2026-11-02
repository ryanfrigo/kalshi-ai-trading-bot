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


# ---- NO direction: these encode the correct behavior --------------------------
_BUG = ("BUG: src/jobs/track.py should_exit_position (+ src/utils/stop_loss_calculator.py) "
        "apply the YES-price convention to NO positions, but run_tracking feeds the NO "
        "contract price. A NO position's stop-loss is set ABOVE entry and fires when the "
        "NO price RISES (a gain); take-profit is set BELOW entry and fires when it FALLS "
        "(a loss). Exits are inverted for every NO position.")


@pytest.mark.xfail(strict=True, reason=_BUG)
async def test_no_position_gaining_does_not_trip_stop_loss():
    # entry .40, NO bid rose to .45 (+12%): stop_loss_price .428 is "above entry"
    pos = _no(0.40, stop_loss_price=0.428)
    assert (await _exit(pos, 0.45))[0] is False


@pytest.mark.xfail(strict=True, reason=_BUG)
async def test_no_position_losing_does_not_take_profit():
    # entry .40, NO bid fell to .30 (-25%): take_profit_price .32 is "below entry"
    pos = _no(0.40, stop_loss_price=0.428, take_profit_price=0.32)
    should_exit, reason, _ = await _exit(pos, 0.30)
    assert not (should_exit and reason == "take_profit")


@pytest.mark.xfail(strict=True, reason=_BUG)
async def test_no_position_losing_badly_triggers_a_stop():
    # a -50% NO position must exit; today nothing fires and 'emergency' is inverted
    pos = _no(0.40)
    assert (await _exit(pos, 0.20))[0] is True


@pytest.mark.xfail(strict=True, reason=_BUG)
async def test_no_position_gaining_does_not_trip_emergency_stop():
    pos = _no(0.40)  # legacy: no stop/take-profit set
    assert (await _exit(pos, 0.45))[0] is False
