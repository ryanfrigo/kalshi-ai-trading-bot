"""Characterization tests for position-sizing math.

These tests pin the EXACT numeric behavior of every Kelly implementation
before and after consolidation into src/utils/position_sizing.py. If a
refactor changes any number here, that refactor changed trading behavior
and must be reviewed — these are not aspirational tests, they are a
contract.

All tests are pure math: no network, no credentials, CI-safe.
"""

import numpy as np
import pytest
from unittest.mock import Mock

from src.strategies.safe_compounder import kelly_fraction
from src.strategies.portfolio_optimization import (
    AdvancedPortfolioOptimizer,
    MarketOpportunity,
    _calculate_simple_kelly,
)
from src.utils import position_sizing
from src.utils.position_sizing import binary_market_payout_odds


def make_opportunity(**overrides) -> MarketOpportunity:
    """A fully-populated MarketOpportunity with neutral defaults."""
    defaults = dict(
        market_id="TEST-MKT",
        market_title="Test market",
        predicted_probability=0.65,
        market_probability=0.50,
        confidence=0.8,
        edge=0.15,
        volatility=0.1,
        expected_return=0.1,
        max_loss=0.5,
        time_to_expiry=30.0,
        correlation_score=0.0,
        kelly_fraction=0.0,
        fractional_kelly=0.0,
        risk_adjusted_fraction=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown_contribution=0.0,
    )
    defaults.update(overrides)
    return MarketOpportunity(**defaults)


# ---------------------------------------------------------------------------
# Kernel: kelly_fraction(prob_win, payout_ratio)  [safe_compounder]
# ---------------------------------------------------------------------------

class TestKellyKernel:
    def test_even_odds_with_edge(self):
        # f* = (0.6*1 - 0.4) / 1 = 0.2
        assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)

    def test_no_edge_is_zero(self):
        assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)

    def test_short_odds(self):
        # f* = (0.7*0.5 - 0.3) / 0.5 = 0.1
        assert kelly_fraction(0.7, 0.5) == pytest.approx(0.1)

    def test_negative_edge_floors_at_zero(self):
        assert kelly_fraction(0.4, 1.0) == 0.0

    def test_invalid_payout_returns_zero(self):
        assert kelly_fraction(0.6, 0.0) == 0.0
        assert kelly_fraction(0.6, -1.0) == 0.0

    def test_invalid_probability_returns_zero(self):
        assert kelly_fraction(0.0, 2.0) == 0.0
        assert kelly_fraction(-0.1, 1.0) == 0.0

    def test_certain_win_bets_everything(self):
        assert kelly_fraction(1.0, 2.0) == pytest.approx(1.0)

    def test_safe_compounder_reexports_canonical_kernel(self):
        # The strategy must use the shared kernel, not a private copy.
        assert kelly_fraction is position_sizing.kelly_fraction


# ---------------------------------------------------------------------------
# binary_market_payout_odds  [shared odds helper]
# ---------------------------------------------------------------------------

class TestPayoutOdds:
    def test_yes_odds(self):
        # YES at $0.40: profit 0.60 on 0.40 staked -> b = 1.5
        assert binary_market_payout_odds(0.40, bet_yes=True) == pytest.approx(1.5)

    def test_no_odds(self):
        # NO at P=0.40: profit 0.40 on 0.60 staked -> b = 2/3
        assert binary_market_payout_odds(0.40, bet_yes=False) == pytest.approx(2 / 3)

    def test_even_market(self):
        assert binary_market_payout_odds(0.50, bet_yes=True) == pytest.approx(1.0)
        assert binary_market_payout_odds(0.50, bet_yes=False) == pytest.approx(1.0)

    def test_degenerate_prices_raise(self):
        with pytest.raises(ZeroDivisionError):
            binary_market_payout_odds(0.0, bet_yes=True)
        with pytest.raises(ZeroDivisionError):
            binary_market_payout_odds(1.0, bet_yes=False)


# ---------------------------------------------------------------------------
# _calculate_simple_kelly  [portfolio_optimization module function]
# ---------------------------------------------------------------------------

class TestSimpleKelly:
    def test_yes_bet_capped_at_20_percent(self):
        # b = 1.0, raw kelly = 0.65 - 0.35 = 0.30 -> capped to 0.2
        opp = make_opportunity(predicted_probability=0.65, market_probability=0.50, edge=0.15)
        assert _calculate_simple_kelly(opp) == pytest.approx(0.2)

    def test_yes_bet_under_cap(self):
        # raw kelly = 0.55 - 0.45 = 0.10
        opp = make_opportunity(predicted_probability=0.55, market_probability=0.50, edge=0.05)
        assert _calculate_simple_kelly(opp) == pytest.approx(0.1)

    def test_yes_bet_exactly_at_cap(self):
        # raw kelly = 0.60 - 0.40 = 0.20 (boundary)
        opp = make_opportunity(predicted_probability=0.60, market_probability=0.50, edge=0.10)
        assert _calculate_simple_kelly(opp) == pytest.approx(0.2)

    def test_no_bet_capped(self):
        # edge < 0 -> NO branch: p = 0.65, b = 0.5/0.5 = 1 -> raw 0.30 -> cap 0.2
        opp = make_opportunity(predicted_probability=0.35, market_probability=0.50, edge=-0.15)
        assert _calculate_simple_kelly(opp) == pytest.approx(0.2)

    def test_no_bet_under_cap(self):
        # p = 0.55, b = 1 -> 0.10
        opp = make_opportunity(predicted_probability=0.45, market_probability=0.50, edge=-0.05)
        assert _calculate_simple_kelly(opp) == pytest.approx(0.1)

    def test_degenerate_price_yes_falls_back_to_5_percent(self):
        # market_probability = 0 with YES branch -> division by zero -> legacy 0.05
        opp = make_opportunity(predicted_probability=0.7, market_probability=0.0, edge=0.7)
        assert _calculate_simple_kelly(opp) == pytest.approx(0.05)

    def test_degenerate_price_no_falls_back_to_5_percent(self):
        # market_probability = 1 with NO branch -> division by zero -> legacy 0.05
        opp = make_opportunity(predicted_probability=0.9, market_probability=1.0, edge=-0.1)
        assert _calculate_simple_kelly(opp) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# PortfolioOptimizer._calculate_kelly_fractions  [KCE pipeline]
# ---------------------------------------------------------------------------

def make_optimizer(market_state: str = "normal") -> AdvancedPortfolioOptimizer:
    opt = AdvancedPortfolioOptimizer(
        db_manager=Mock(),
        kalshi_client=Mock(),
        xai_client=Mock(),
    )
    # Pin the knobs so tests are independent of settings/env drift.
    opt.kelly_fraction_multiplier = 0.25
    opt.max_position_fraction = 0.25
    opt.market_state = market_state
    return opt


class TestKellyFractionsKCE:
    def test_standard_pipeline(self):
        # standard = (1*0.65 - 0.35)/1 = 0.30; regime 1.0; decay 30/30 = 1.0;
        # * confidence 0.8 = 0.24; * multiplier 0.25 = 0.06
        opt = make_optimizer()
        opp = make_opportunity()
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == pytest.approx(0.06)
        assert opp.kelly_fraction == pytest.approx(0.30)       # raw standard Kelly
        assert opp.fractional_kelly == pytest.approx(0.06)
        assert opp.risk_adjusted_fraction == pytest.approx(0.06)

    def test_win_prob_at_most_half_gives_zero(self):
        # edge > 0 but win_prob <= 0.5 -> gate fails -> 0
        opt = make_optimizer()
        opp = make_opportunity(predicted_probability=0.45, market_probability=0.40, edge=0.05)
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == 0.0
        assert opp.kelly_fraction == 0.0

    def test_no_edge_gives_zero(self):
        opt = make_optimizer()
        opp = make_opportunity(predicted_probability=0.65, market_probability=0.70, edge=-0.05)
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == 0.0

    def test_degenerate_market_price_uses_even_odds(self):
        # market_probability = 0 -> odds fallback 1.0
        # standard = (1*0.7 - 0.3)/1 = 0.4; conf 1.0; decay 1.0 (60d capped); * 0.25 = 0.1
        opt = make_optimizer()
        opp = make_opportunity(
            predicted_probability=0.7, market_probability=0.0, edge=0.7,
            confidence=1.0, time_to_expiry=60.0,
        )
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == pytest.approx(0.1)
        assert opp.kelly_fraction == pytest.approx(0.4)

    def test_time_decay_scales_linearly(self):
        # 3 days -> decay 0.1; standard 0.30 * 0.1 = 0.03; conf 1.0; * 0.25 = 0.0075
        opt = make_optimizer()
        opp = make_opportunity(confidence=1.0, time_to_expiry=3.0)
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == pytest.approx(0.0075)

    def test_time_decay_floors_at_10_percent(self):
        # 0.5 days -> 0.5/30 = 0.0167 -> floored to 0.1 (same as 3 days)
        opt = make_optimizer()
        opp = make_opportunity(confidence=1.0, time_to_expiry=0.5)
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == pytest.approx(0.0075)

    def test_volatile_regime_dampens(self):
        # standard 0.30 * regime 0.7 = 0.21; conf 1.0; * 0.25 = 0.0525
        opt = make_optimizer(market_state="volatile")
        opp = make_opportunity(confidence=1.0)
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == pytest.approx(0.0525)

    def test_trending_regime_boosts(self):
        # standard 0.30 * regime 1.2 = 0.36; conf 1.0; * 0.25 = 0.09
        opt = make_optimizer(market_state="trending")
        opp = make_opportunity(confidence=1.0)
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == pytest.approx(0.09)

    def test_final_fraction_clamped_to_max_position(self):
        # Huge edge: standard = (1*0.99 - 0.01)/1 = 0.98; conf 1.0 -> frac 0.245 < 0.25 cap...
        # push over the cap with trending regime: 0.98 * 1.2 = 1.176 * 0.25 = 0.294 -> clamp 0.25
        opt = make_optimizer(market_state="trending")
        opp = make_opportunity(
            predicted_probability=0.99, market_probability=0.50, edge=0.49, confidence=1.0,
        )
        result = opt._calculate_kelly_fractions([opp])
        assert result["TEST-MKT"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# PortfolioOptimizer._calculate_portfolio_metrics  [correlation indexing]
# ---------------------------------------------------------------------------

class TestPortfolioMetricsCorrelationIndexing:
    """The correlation matrix is built over ALL opportunities in input order.

    When only a SUBSET is allocated, the portfolio metrics must pair each
    allocated market with its OWN correlations — not the top-left n x n block
    of the full matrix. Slicing [:n, :n] silently pairs each allocated market
    with whichever market happened to be first in the input, corrupting
    volatility/Sharpe/VaR whenever the allocated set isn't the leading n.
    """

    def test_uses_allocated_markets_own_correlations(self):
        opt = make_optimizer()
        opt.total_capital = 10_000

        # Three opportunities in input order A, B, C.
        a = make_opportunity(market_id="A", volatility=0.10)
        b = make_opportunity(market_id="B", volatility=0.20)
        c = make_opportunity(market_id="C", volatility=0.30)
        opportunities = [a, b, c]

        # Correlation matrix over [A, B, C]. The A-B block (top-left, what the
        # bug uses) is HIGHLY correlated (0.9); the B-C block (correct for an
        # allocation of B and C) is weakly correlated (0.2).
        corr = np.array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )

        # Allocate to B and C only — i.e. NOT the leading n markets.
        allocation = {"B": 0.5, "C": 0.5}

        metrics = opt._calculate_portfolio_metrics(allocation, opportunities, corr)

        w = np.array([0.5, 0.5])
        vols = np.array([0.20, 0.30])
        # Correct: pair B,C with their real 0.2 cross-correlation.
        correct_cov = np.outer(vols, vols) * corr[np.ix_([1, 2], [1, 2])]
        expected_vol = float(np.sqrt(w @ correct_cov @ w))  # ~0.19621
        # Buggy: top-left A-B block (0.9 correlation).
        buggy_cov = np.outer(vols, vols) * corr[:2, :2]
        buggy_vol = float(np.sqrt(w @ buggy_cov @ w))  # ~0.24393

        assert metrics["portfolio_volatility"] == pytest.approx(expected_vol, rel=1e-9)
        assert metrics["portfolio_volatility"] != pytest.approx(buggy_vol, rel=1e-6)
