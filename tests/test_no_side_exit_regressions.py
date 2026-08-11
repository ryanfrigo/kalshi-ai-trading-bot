"""Regression tests for the NO-side exit-logic churn loop.

Four independent defects combined to make every NO position stop out within
~1 minute of entry: 160 trades, 0 wins, across only 4 markets (two re-entered
71 times each). Each is pinned here because all four were silent — the suite
was green throughout.

1. `unified_trading_system` priced positions from the legacy `no_price` field,
   which Kalshi API v2 does not return, so it defaulted to a phantom 50c.
2. Exit levels were therefore anchored to 0.50 while the real fill was ~0.95.
3. `StopLossCalculator` worked in YES-price space while every caller fed it
   own-side prices, so a NO stop sat ABOVE entry and triggered on any rise.
4. Resolution compared Kalshi's lowercase `result` against an uppercase
   `Position.side`, so a settled winner booked as a total loss.
"""
import os
from datetime import datetime

import pytest

from src.jobs.track import should_exit_position
from src.utils.database import Position
from src.utils.stop_loss_calculator import StopLossCalculator


def _no_position(entry_price=0.955, stop_loss_price=None, take_profit_price=None):
    """A NO position priced like the near-certainties the strategy actually picks."""
    return Position(
        market_id="KXGREENLAND-29-27",
        side="NO",
        entry_price=entry_price,
        quantity=60,
        timestamp=datetime.now(),
        rationale="regression fixture",
        confidence=0.82,
        live=False,
        status="open",
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
    )


class TestOwnSidePriceConvention:
    """Defect 3: stop below entry / target above entry, for BOTH sides."""

    @pytest.mark.parametrize("side", ["YES", "NO"])
    def test_stop_below_entry_and_target_above(self, side):
        levels = StopLossCalculator.calculate_stop_loss_levels(
            entry_price=0.95, side=side, confidence=0.82
        )
        assert levels["stop_loss_price"] < 0.95, (
            f"{side} stop must sit BELOW entry — a stop above entry triggers on a "
            f"price rise, i.e. it exits a winning position"
        )
        assert levels["take_profit_price"] > 0.95, f"{side} target must sit ABOVE entry"

    @pytest.mark.parametrize("side", ["YES", "NO"])
    def test_simple_stop_below_entry(self, side):
        stop = StopLossCalculator.calculate_simple_stop_loss(
            entry_price=0.95, side=side, stop_loss_pct=0.10
        )
        assert stop < 0.95

    def test_no_stop_does_not_trigger_on_a_tiny_adverse_move(self):
        """The exact production case: NO bought at 0.955, book mid 0.9535."""
        assert (
            StopLossCalculator.is_stop_loss_triggered(
                position_side="NO",
                entry_price=0.955,
                current_price=0.9535,
                stop_loss_price=0.888,
            )
            is False
        )

    def test_no_stop_triggers_on_a_real_drop(self):
        assert (
            StopLossCalculator.is_stop_loss_triggered(
                position_side="NO",
                entry_price=0.955,
                current_price=0.85,
                stop_loss_price=0.888,
            )
            is True
        )

    def test_stopped_out_no_position_reports_a_loss(self):
        """Exit reasons logged 'stop_loss_triggered_pnl_12.60' on losing trades."""
        pnl = StopLossCalculator.calculate_pnl_at_stop_loss(
            entry_price=0.955, stop_loss_price=0.888, quantity=60, side="NO"
        )
        assert pnl < 0, "a stop-out is a loss; the NO branch used to report it positive"


class TestNoPositionSurvivesEntry:
    """Defects 2+3 together: the symptom the user actually saw."""

    @pytest.mark.asyncio
    async def test_fresh_no_position_is_not_exited_immediately(self):
        """Entry at the ask, mark at the mid — must NOT exit on the first pass."""
        position = _no_position(entry_price=0.955)
        levels = StopLossCalculator.calculate_stop_loss_levels(
            entry_price=position.entry_price, side="NO", confidence=0.82
        )
        position.stop_loss_price = levels["stop_loss_price"]
        position.take_profit_price = levels["take_profit_price"]

        should_exit, reason, _ = await should_exit_position(
            position=position,
            current_yes_price=0.0465,
            current_no_price=0.9535,
            market_status="active",
            market_result=None,
        )
        assert should_exit is False, f"exited immediately via {reason!r}"

    @pytest.mark.asyncio
    async def test_phantom_50c_levels_are_detected_as_incoherent(self):
        """Defect 2: 0.50-anchored levels on a 0.955 fill.

        Fixing the stop direction alone is not enough — a target of 0.40 sits
        BELOW the 0.9535 mark, so it fires a bogus take-profit on the next pass.
        `run_tracking` detects levels that fail to bracket the entry price and
        re-anchors them; this pins the detection rule those legacy rows hit.
        """
        position = _no_position(
            entry_price=0.955, stop_loss_price=0.535, take_profit_price=0.40
        )

        levels_incoherent = (
            (position.stop_loss_price or 0) >= position.entry_price
            or (position.take_profit_price or 1.0) <= position.entry_price
        )
        assert levels_incoherent is True, "0.535/0.40 against a 0.955 entry must be flagged"

        # Confirm the danger is real if they were left in place: the stale target
        # is below the mark, so an exit WOULD fire without re-anchoring.
        should_exit, reason, _ = await should_exit_position(
            position=position,
            current_yes_price=0.0465,
            current_no_price=0.9535,
            market_status="active",
            market_result=None,
        )
        assert should_exit is True and reason == "take_profit"

        # After re-anchoring from the real entry price, the position survives.
        levels = StopLossCalculator.calculate_stop_loss_levels(
            entry_price=position.entry_price, side="NO", confidence=0.82
        )
        position.stop_loss_price = levels["stop_loss_price"]
        position.take_profit_price = levels["take_profit_price"]

        should_exit_after, reason_after, _ = await should_exit_position(
            position=position,
            current_yes_price=0.0465,
            current_no_price=0.9535,
            market_status="active",
            market_result=None,
        )
        assert should_exit_after is False, f"still exits via {reason_after!r} after re-anchoring"


class TestResolutionCaseNormalisation:
    """Defect 4: latent until positions are held long enough to settle."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "side,result,expected",
        [
            ("NO", "no", 1.0),
            ("NO", "yes", 0.0),
            ("YES", "yes", 1.0),
            ("YES", "no", 0.0),
            ("NO", "NO", 1.0),
            ("YES", " Yes ", 1.0),
        ],
    )
    async def test_settlement_price_matches_case_insensitively(self, side, result, expected):
        position = _no_position()
        position.side = side

        should_exit, reason, exit_price = await should_exit_position(
            position=position,
            current_yes_price=0.5,
            current_no_price=0.5,
            market_status="closed",
            market_result=result,
        )
        assert should_exit is True
        assert reason == "market_resolution"
        assert exit_price == expected, (
            f"side={side!r} result={result!r} settled at {exit_price} not {expected}; "
            f"a raw == comparison books every winner as a total loss"
        )


class TestCostTrackerMirroring:
    """The daily cycle throttle read a tracker that never accrued cost."""

    def test_openrouter_publishes_last_request_cost(self):
        from src.clients.openrouter_client import OpenRouterClient

        assert hasattr(OpenRouterClient, "_update_daily_cost")

        client = OpenRouterClient.__new__(OpenRouterClient)
        client._last_request_cost = 0.0

        class _Tracker:
            total_cost = 0.0
            request_count = 0
            daily_limit = 2.0
            is_exhausted = False
            last_exhausted_time = None

        client.daily_tracker = _Tracker()
        client._save_daily_tracker = lambda: None

        OpenRouterClient._update_daily_cost(client, 0.0042)

        assert client._last_request_cost == pytest.approx(0.0042), (
            "XAIClient mirrors this attribute into the tracker beast_mode_bot "
            "reads; without it the bot-level daily throttle never fires"
        )
        assert client.daily_tracker.total_cost == pytest.approx(0.0042)
