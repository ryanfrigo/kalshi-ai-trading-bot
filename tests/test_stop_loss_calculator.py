"""Tests for src/utils/stop_loss_calculator.py.

Pure math: no network, no files, no credentials. Assertions are the
mathematically correct answers, not a pin of today's output. Known bug
flavors (silent NaN swallowing by clamp/default idioms, missing clamps)
are marked xfail(strict=True) against the correct behavior with the
offending line cited, so a fix turns the suite red until the xfail is
retired deliberately.
"""

import math
import pytest

from src.utils.stop_loss_calculator import (
    StopLossCalculator,
    calculate_stop_loss_levels,
)

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# calculate_stop_loss_levels
# ---------------------------------------------------------------------------

class TestStopLossLevelsConfidenceTiers:
    ENTRY = 0.40  # picked so every price multiplies out away from .xx5 ties

    @pytest.mark.parametrize(
        "confidence, stop_pct, take_profit_pct",
        [
            (0.8, 5.0, 30.0),    # boundary: high tier starts at 0.8
            (0.9, 5.0, 30.0),
            (1.0, 5.0, 30.0),
            (0.79, 7.0, 20.0),   # just below the 0.8 boundary -> medium
            (0.6, 7.0, 20.0),    # boundary: medium tier starts at 0.6
            (0.65, 7.0, 20.0),
            (0.001, 10.0, 15.0),  # near zero -> widest stop (exact 0.0 is line-62 bug, xfail below)
            (-0.5, 10.0, 15.0),
        ],
    )
    def test_tiers(self, confidence, stop_pct, take_profit_pct):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=confidence
        )
        assert result["stop_loss_pct"] == pytest.approx(stop_pct)
        assert result["take_profit_pct"] == pytest.approx(take_profit_pct)

    def test_confidence_below_medium_uses_widest_stop_prices(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.599
        )
        assert result["stop_loss_price"] == pytest.approx(0.36)   # 0.40 * 0.90
        assert result["take_profit_price"] == pytest.approx(0.46)  # 0.40 * 1.15

    def test_high_confidence_prices(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.8
        )
        assert result["stop_loss_price"] == pytest.approx(0.38)   # 0.40 * 0.95
        assert result["take_profit_price"] == pytest.approx(0.52)  # 0.40 * 1.30

    def test_medium_confidence_prices(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.7
        )
        assert result["stop_loss_price"] == pytest.approx(0.37)   # 0.40 * 0.93
        assert result["take_profit_price"] == pytest.approx(0.48)  # 0.40 * 1.20

    def test_default_confidence_is_medium(self):
        result = StopLossCalculator.calculate_stop_loss_levels(self.ENTRY, "YES")
        assert result["stop_loss_pct"] == pytest.approx(7.0)

    def test_none_confidence_uses_medium_default(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=None
        )
        assert result["stop_loss_pct"] == pytest.approx(7.0)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:62 `confidence or 0.7` swallows an "
            "explicit confidence of 0.0 as omitted and returns the 7% medium tier "
            "instead of the 10% zero-confidence tier"
        ),
    )
    def test_explicit_zero_confidence_is_not_omitted(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.0
        )
        assert result["stop_loss_pct"] == pytest.approx(10.0)

    def test_nan_confidence_falls_to_widest_stop(self):
        # NaN compares False against every tier boundary, so the `else` (10%)
        # branch runs — conservative fail-safe, therefore correct.
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=NAN
        )
        assert result["stop_loss_pct"] == pytest.approx(10.0)
        assert result["take_profit_pct"] == pytest.approx(15.0)


class TestStopLossLevelsVolatility:
    ENTRY = 0.40

    def test_baseline_volatility_is_no_adjustment(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", market_volatility=0.2
        )
        assert result["stop_loss_pct"] == pytest.approx(7.0)

    def test_higher_volatility_widens_stop(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.7, market_volatility=0.5
        )
        # adjustment = 1 + (0.5 - 0.2) = 1.3 -> 7% * 1.3 = 9.1%
        assert result["stop_loss_pct"] == pytest.approx(9.1)
        assert result["stop_loss_price"] == pytest.approx(0.36)  # 0.40 * 0.909

    def test_adjustment_is_capped_at_one_point_five(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.7, market_volatility=100.0
        )
        assert result["stop_loss_pct"] == pytest.approx(10.5)  # 7% * 1.5

    def test_adjustment_cap_boundary_exact(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.7, market_volatility=0.7
        )
        assert result["stop_loss_pct"] == pytest.approx(10.5)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:78 volatility adjustment has no "
            "lower clamp, so vol just under 0.2 shrinks the stop below the "
            "documented MIN_STOP_LOSS_PCT of 5% (vol=0.01 -> 4.05%)"
        ),
    )
    def test_low_volatility_never_drops_below_minimum_stop(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.7, market_volatility=0.01
        )
        assert result["stop_loss_pct"] == pytest.approx(5.0)
        assert result["stop_loss_price"] == pytest.approx(0.38)  # 0.40 * 0.95

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:63 an explicit market_volatility "
            "of 0.0 is swallowed by `or 0.2` and priced as 0.2 (7% stop) instead "
            "of calmer-than-baseline, which the 5% minimum floors to 5%"
        ),
    )
    def test_explicit_zero_volatility_is_not_omitted(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            self.ENTRY, "YES", confidence=0.7, market_volatility=0.0
        )
        assert result["stop_loss_pct"] == pytest.approx(5.0)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:78 NaN volatility survives the "
            "default and makes min(1.5, 1+(nan-0.2)) evaluate to 1.5 (max stop) "
            "from garbage data instead of raising"
        ),
    )
    def test_nan_volatility_must_raise(self):
        with pytest.raises(ValueError):
            StopLossCalculator.calculate_stop_loss_levels(
                self.ENTRY, "YES", market_volatility=NAN
            )


class TestStopLossLevelsEntryPriceAndBounds:
    def test_price_stays_clamped_low(self):
        # 10% stop on a 1-cent entry would be 0.9 cents -> clamped to 0.01
        result = StopLossCalculator.calculate_stop_loss_levels(
            0.01, "YES", confidence=0.4, market_volatility=100.0
        )
        assert result["stop_loss_price"] == pytest.approx(0.01)
        assert result["take_profit_price"] == pytest.approx(0.01)  # max(0.01, 0.0115)

    def test_take_profit_clamped_at_ninety_nine_cents(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            0.90, "YES", confidence=0.8
        )
        assert result["take_profit_price"] == pytest.approx(0.99)  # 0.90 * 1.30
        assert result["stop_loss_pct"] == pytest.approx(5.0)

    def test_entry_below_one_cent_clamps_up(self):
        result = StopLossCalculator.calculate_stop_loss_levels(0.001, "YES")
        assert result["stop_loss_price"] == pytest.approx(0.01)
        assert result["take_profit_price"] == pytest.approx(0.01)

    def test_entry_above_ninety_nine_cents_clamps_down(self):
        result = StopLossCalculator.calculate_stop_loss_levels(5.0, "YES")
        # entry treated as 0.99: stop 0.99*0.93 = 0.9207 -> 0.92, tp clamps 0.99
        assert result["stop_loss_price"] == pytest.approx(0.92)
        assert result["take_profit_price"] == pytest.approx(0.99)

    def test_entry_prices_stay_within_one_cent_to_ninety_nine_cents(self):
        for entry in (0.01, 0.05, 0.5, 0.95, 0.99):
            result = StopLossCalculator.calculate_stop_loss_levels(
                entry, "NO", confidence=0.3, market_volatility=2.0
            )
            assert 0.01 <= result["stop_loss_price"] <= 0.99
            assert 0.01 <= result["take_profit_price"] <= 0.99

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:61 the clamp max(0.01, min(0.99, e)) "
            "silently turns NaN entry into a 99¢ position (min ignores NaN) and "
            "+inf into a real entry instead of raising for a non-finite price"
        ),
    )
    def test_non_finite_entry_price_must_raise(self):
        with pytest.raises((ValueError, TypeError)):
            StopLossCalculator.calculate_stop_loss_levels(NAN, "YES")
        with pytest.raises((ValueError, TypeError)):
            StopLossCalculator.calculate_stop_loss_levels(INF, "YES")

    def test_none_entry_price_raises(self):
        # None must not quietly produce prices; any raise is acceptable.
        with pytest.raises((TypeError, ValueError)):
            StopLossCalculator.calculate_stop_loss_levels(None, "YES")

    def test_side_does_not_change_held_contract_math(self):
        yes = StopLossCalculator.calculate_stop_loss_levels(0.40, "YES")
        no = StopLossCalculator.calculate_stop_loss_levels(0.40, "NO")
        assert yes == no


class TestStopLossLevelsHoldTime:
    @pytest.mark.parametrize(
        "days, hours",
        [
            (0.24, 6),   # 2.88 -> floored at 6
            (0.25, 6),   # 3.0 -> floored at 6
            (0.5, 6),    # exactly the floor boundary
            (0.6, 7),    # 7.2 -> int 7
            (1.0, 12),   # 12.0 exactly
            (6.0, 72),   # 6*24*0.5 = 72, exactly the cap boundary
            (6.1, 72),   # 73.2 -> capped
            (100.0, 72),
            (-5.0, 6),   # negative days -> floored at 6
        ],
    )
    def test_hold_time(self, days, hours):
        result = StopLossCalculator.calculate_stop_loss_levels(
            0.40, "YES", time_to_expiry_days=days
        )
        assert result["max_hold_hours"] == hours
        assert isinstance(result["max_hold_hours"], int)

    def test_default_hold_time_is_seventy_two_hours(self):
        result = StopLossCalculator.calculate_stop_loss_levels(0.40, "YES")
        assert result["max_hold_hours"] == 72

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:64 an explicit time_to_expiry_days "
            "of 0.0 is falsy and swallowed by `or 30.0`, so an about-to-expire "
            "position gets a 72h hold instead of the 6h floor"
        ),
    )
    def test_explicit_zero_days_to_expiry_is_not_omitted(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            0.40, "YES", time_to_expiry_days=0.0
        )
        assert result["max_hold_hours"] == 6


class TestStopLossLevelsShape:
    def test_returns_all_documented_keys(self):
        result = StopLossCalculator.calculate_stop_loss_levels(0.40, "YES")
        assert set(result) == {
            "stop_loss_price",
            "take_profit_price",
            "max_hold_hours",
            "stop_loss_pct",
            "take_profit_pct",
            "target_confidence_change",
        }

    def test_prices_are_rounded_to_two_decimals(self):
        result = StopLossCalculator.calculate_stop_loss_levels(
            0.80, "YES", confidence=0.7, market_volatility=0.5
        )
        # raw stop = 0.80 * (1 - 0.091) = 0.7272 -> 0.73
        assert result["stop_loss_price"] == pytest.approx(0.73)
        assert result["stop_loss_price"] == round(result["stop_loss_price"], 2)

    def test_target_confidence_change_is_fixed(self):
        result = StopLossCalculator.calculate_stop_loss_levels(0.40, "YES")
        assert result["target_confidence_change"] == pytest.approx(0.15)

    def test_stop_only_below_and_target_only_above(self):
        for entry in (0.02, 0.35, 0.6, 0.98):
            for conf in (0.2, 0.7, 0.95):
                result = StopLossCalculator.calculate_stop_loss_levels(
                    entry, "YES", confidence=conf
                )
                if result["stop_loss_price"] > 0.01:  # clamped values can exceed
                    assert result["stop_loss_price"] <= entry
                if result["take_profit_price"] < 0.99:
                    assert result["take_profit_price"] >= entry


# ---------------------------------------------------------------------------
# calculate_simple_stop_loss
# ---------------------------------------------------------------------------

class TestSimpleStopLoss:
    @pytest.mark.parametrize(
        "entry, pct, expected",
        [
            (0.50, 0.10, 0.45),
            (0.80, 0.05, 0.76),
            (0.20, 0.07, 0.19),  # default pct
            (0.30, 0.30, 0.21),
            (0.68, 0.50, 0.34),
        ],
    )
    def test_values(self, entry, pct, expected):
        assert StopLossCalculator.calculate_simple_stop_loss(entry, "YES", pct) == pytest.approx(expected)

    def test_default_stop_loss_pct_is_seven(self):
        result = StopLossCalculator.calculate_simple_stop_loss(0.20, "NO")
        assert result == pytest.approx(0.19)

    def test_full_loss_stop_clamps_to_floor(self):
        # a 90% stop on a 2-cent entry is 0.2 cents -> clamped to 1 cent
        assert StopLossCalculator.calculate_simple_stop_loss(0.02, "YES", 0.90) == pytest.approx(0.01)

    def test_side_does_not_change_math(self):
        assert StopLossCalculator.calculate_simple_stop_loss(0.5, "NO", 0.1) == (
            StopLossCalculator.calculate_simple_stop_loss(0.5, "YES", 0.1)
        )

    def test_result_always_in_valid_price_band(self):
        for entry in (0.01, 0.5, 0.99):
            for pct in (0.0, 0.07, 0.5, 1.0):
                result = StopLossCalculator.calculate_simple_stop_loss(entry, "YES", pct)
                assert 0.01 <= result <= 0.99
        # pct=0 keeps the entry price itself
        assert StopLossCalculator.calculate_simple_stop_loss(0.5, "YES", 0.0) == pytest.approx(0.5)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:133 a negative stop_loss_pct sets "
            "the stop ABOVE entry (0.5 - -50% = 0.75), a guaranteed instant stop, "
            "instead of raising for an invalid pct"
        ),
    )
    def test_negative_stop_loss_pct_must_raise(self):
        with pytest.raises(ValueError):
            StopLossCalculator.calculate_simple_stop_loss(0.50, "YES", -0.50)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:134 the clamp turns a NaN entry/pct "
            "into min(0.99, nan) = 0.99 (no stop at all) and inf into a 1-cent "
            "stop, silently, instead of raising for non-finite input"
        ),
    )
    def test_non_finite_inputs_must_raise(self):
        with pytest.raises((ValueError, TypeError)):
            StopLossCalculator.calculate_simple_stop_loss(NAN, "YES")
        with pytest.raises((ValueError, TypeError)):
            StopLossCalculator.calculate_simple_stop_loss(0.50, "YES", NAN)
        with pytest.raises((ValueError, TypeError)):
            StopLossCalculator.calculate_simple_stop_loss(INF, "YES")


# ---------------------------------------------------------------------------
# is_stop_loss_triggered
# ---------------------------------------------------------------------------

class TestIsStopLossTriggered:
    def test_current_below_stop_triggers(self):
        assert StopLossCalculator.is_stop_loss_triggered("YES", 0.50, 0.44, 0.45) is True

    def test_current_equal_to_stop_triggers(self):
        assert StopLossCalculator.is_stop_loss_triggered("NO", 0.50, 0.45, 0.45) is True

    def test_current_above_stop_does_not_trigger(self):
        assert StopLossCalculator.is_stop_loss_triggered("YES", 0.50, 0.46, 0.45) is False

    def test_side_is_irrelevant_to_trigger_math(self):
        for side in ("YES", "NO"):
            assert StopLossCalculator.is_stop_loss_triggered(side, 0.60, 0.55, 0.56)
            assert not StopLossCalculator.is_stop_loss_triggered(side, 0.60, 0.57, 0.56)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:158 a NaN current or stop price makes "
            "current_price <= stop_loss_price evaluate False, so a position with "
            "corrupt price data is never stopped (fail-open) instead of raising"
        ),
    )
    def test_nan_prices_must_raise(self):
        with pytest.raises(ValueError):
            StopLossCalculator.is_stop_loss_triggered("YES", 0.50, NAN, 0.45)
        with pytest.raises(ValueError):
            StopLossCalculator.is_stop_loss_triggered("NO", 0.50, 0.44, NAN)


# ---------------------------------------------------------------------------
# calculate_pnl_at_stop_loss
# ---------------------------------------------------------------------------

class TestPnlAtStopLoss:
    def test_loss_is_negative(self):
        # (0.45 - 0.50) * 100 = -5.00
        assert StopLossCalculator.calculate_pnl_at_stop_loss(0.50, 0.45, 100, "YES") == pytest.approx(-5.0)

    def test_gain_if_stop_above_entry(self):
        assert StopLossCalculator.calculate_pnl_at_stop_loss(0.40, 0.50, 10, "NO") == pytest.approx(1.0)

    def test_side_is_irrelevant(self):
        assert StopLossCalculator.calculate_pnl_at_stop_loss(0.50, 0.45, 100, "YES") == (
            StopLossCalculator.calculate_pnl_at_stop_loss(0.50, 0.45, 100, "NO")
        )

    def test_zero_quantity_is_zero_pnl(self):
        assert StopLossCalculator.calculate_pnl_at_stop_loss(0.50, 0.45, 0, "YES") == 0.0

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src/utils/stop_loss_calculator.py:176 non-finite prices multiply straight "
            "through to a NaN P&L, which would poison any position ledger, instead "
            "of raising"
        ),
    )
    def test_non_finite_prices_must_raise(self):
        with pytest.raises(ValueError):
            StopLossCalculator.calculate_pnl_at_stop_loss(NAN, 0.45, 100, "YES")
        with pytest.raises(ValueError):
            StopLossCalculator.calculate_pnl_at_stop_loss(0.50, INF, 100, "YES")


# ---------------------------------------------------------------------------
# module-level convenience function
# ---------------------------------------------------------------------------

class TestConvenienceFunction:
    def test_matches_classmethod_for_all_inputs(self):
        kwargs = dict(
            confidence=0.8,
            market_volatility=0.5,
            time_to_expiry_days=2.0,
        )
        expected = StopLossCalculator.calculate_stop_loss_levels(0.40, "YES", **kwargs)
        assert calculate_stop_loss_levels(0.40, "YES", **kwargs) == expected

    def test_default_confidence_only_signature(self):
        assert calculate_stop_loss_levels(0.40, "YES", None, market_volatility=0.2) == (
            StopLossCalculator.calculate_stop_loss_levels(
                entry_price=0.40, side="YES", confidence=None, market_volatility=0.2
            )
        )
