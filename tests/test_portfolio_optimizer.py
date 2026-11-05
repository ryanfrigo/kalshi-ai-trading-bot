"""AdvancedPortfolioOptimizer and its runner. Fully offline: every client is a mock."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.config.settings import settings
from src.strategies.portfolio import optimizer as optimizer_module
from src.strategies.portfolio import runner as runner_module
from src.strategies.portfolio.models import MarketOpportunity
from src.strategies.portfolio.optimizer import AdvancedPortfolioOptimizer

EMPTY_METRICS = {
    "total_capital_used": 0.0,
    "expected_portfolio_return": 0.0,
    "portfolio_volatility": 0.0,
    "portfolio_sharpe": 0.0,
    "max_portfolio_drawdown": 0.0,
    "diversification_ratio": 1.0,
    "portfolio_var_95": 0.0,
    "portfolio_cvar_95": 0.0,
    "aggregate_kelly_fraction": 0.0,
    "portfolio_growth_rate": 0.0,
}


def opp(market_id="m0", pp=0.7, mp=0.5, conf=0.8, vol=0.3, tte=30.0, **over):
    """An opportunity built the way create_market_opportunities_from_markets builds one."""
    edge = pp - mp
    fields = dict(
        market_id=market_id, market_title="t", predicted_probability=pp, market_probability=mp,
        confidence=conf, edge=edge, volatility=vol, expected_return=abs(edge) * conf, max_loss=0.5,
        time_to_expiry=tte, correlation_score=0.0, kelly_fraction=0.0, fractional_kelly=0.0,
        risk_adjusted_fraction=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        max_drawdown_contribution=vol * 2.0,
    )
    fields.update(over)
    return MarketOpportunity(**fields)


def many(n, **kw):
    return [opp(f"m{i}", **kw) for i in range(n)]


@pytest.fixture
def o():
    opt = AdvancedPortfolioOptimizer(MagicMock(), MagicMock(), MagicMock())
    opt.total_capital = 10000.0
    opt.max_position_fraction = 0.03
    opt.min_position_size = 5
    opt.kelly_fraction_multiplier = 0.25
    opt.max_correlation = 0.70
    opt.logger = MagicMock()
    return opt


def metrics_of(allocation):
    return {k: getattr(allocation, k) for k in EMPTY_METRICS}


# ------------------------------------------------------------------ construction
def test_defaults_come_from_settings_and_the_fallbacks():
    opt = AdvancedPortfolioOptimizer(MagicMock(), MagicMock(), MagicMock())
    assert opt.max_position_fraction == settings.trading.max_single_position
    assert opt.kelly_fraction_multiplier == settings.trading.kelly_fraction
    assert opt.total_capital == 10000
    assert opt.min_position_size == 5
    assert opt.max_correlation == 0.70
    assert opt.market_state == "normal"


# ------------------------------------------------------------------ optimize_portfolio
def test_no_opportunities_is_an_empty_allocation(o):
    o._enhance_opportunities_with_metrics = AsyncMock()
    result = asyncio.run(o.optimize_portfolio([]))
    assert result.allocations == {}
    assert metrics_of(result) == EMPTY_METRICS
    o._enhance_opportunities_with_metrics.assert_not_awaited()


def test_batch_keeps_the_best_by_confidence_times_return(o, monkeypatch):
    monkeypatch.setattr(settings.trading, "max_opportunities_per_batch", 2, raising=False)
    seen = {}

    async def enhance(batch):
        seen["ids"] = [x.market_id for x in batch]
        return []
    o._enhance_opportunities_with_metrics = enhance
    batch = [opp("next", conf=0.5, expected_return=0.5), opp("best", conf=0.9, expected_return=0.3),
             opp("cut", conf=0.2, expected_return=0.9)]
    asyncio.run(o.optimize_portfolio(batch))
    assert seen["ids"] == ["best", "next"]


def test_batch_at_the_limit_is_left_alone(o, monkeypatch):
    monkeypatch.setattr(settings.trading, "max_opportunities_per_batch", 2, raising=False)
    seen = {}

    async def enhance(batch):
        seen["ids"] = [x.market_id for x in batch]
        return []
    o._enhance_opportunities_with_metrics = enhance
    asyncio.run(o.optimize_portfolio([opp("a", conf=0.1), opp("b", conf=0.9)]))
    assert seen["ids"] == ["a", "b"]


def test_batch_limit_defaults_to_fifty(o):
    seen = {}

    async def enhance(batch):
        seen["n"] = len(batch)
        return []
    o._enhance_opportunities_with_metrics = enhance
    asyncio.run(o.optimize_portfolio(many(51)))
    assert seen["n"] == 50


def test_a_failure_in_the_pipeline_allocates_nothing(o):
    o._enhance_opportunities_with_metrics = AsyncMock(side_effect=RuntimeError("boom"))
    result = asyncio.run(o.optimize_portfolio(many(3)))
    assert result.allocations == {}
    assert metrics_of(result) == EMPTY_METRICS
    assert "boom" in o.logger.error.call_args.args[0]


def test_kelly_values_reach_the_risk_step(o):
    seen = {}
    real = o._apply_risk_constraints

    def spy(allocation, enhanced):
        seen["enhanced"] = enhanced
        return real(allocation, enhanced)
    o._apply_risk_constraints = spy
    asyncio.run(o.optimize_portfolio([opp("a"), opp("b")]))
    first = seen["enhanced"][0]
    assert first.kelly_fraction == pytest.approx(0.03)
    assert first.fractional_kelly == pytest.approx(0.015)
    assert first.risk_adjusted_fraction == pytest.approx(0.015)


@pytest.mark.parametrize("batch", [
    many(1), many(2), many(7), many(3, pp=0.3, mp=0.55), many(3, pp=0.56, conf=0.5),
    [opp("a"), opp("b"), opp("c", pp=0.3, mp=0.55)], many(3, tte=0.1),
], ids=["one", "two", "seven", "all-no", "thin", "mixed", "near-expiry"])
def test_no_position_passes_the_cap_and_capital_used_adds_up(o, batch):
    result = asyncio.run(o.optimize_portfolio(batch))
    assert result.allocations
    assert all(0 < f <= o.max_position_fraction + 1e-9 for f in result.allocations.values())
    assert result.total_capital_used == pytest.approx(sum(result.allocations.values()) * 10000.0)


def test_end_to_end_reports_metrics_for_what_was_allocated(o):
    result = asyncio.run(o.optimize_portfolio(many(3)))
    assert set(result.allocations) == {"m0", "m1", "m2"}
    assert result.expected_portfolio_return > 0
    assert result.portfolio_volatility > 0
    assert result.portfolio_sharpe > 0
    assert result.aggregate_kelly_fraction > 0
    assert "complete" in o.logger.info.call_args.args[0]


@pytest.mark.xfail(strict=True, reason="sizing ignores Kelly: thin edges are forced up to the cap "
                                       "by the 5% minimum sum or the rank-weight fallback")
@pytest.mark.parametrize("n", [1, 2, 5])
def test_allocation_does_not_exceed_fractional_kelly(o, n):
    batch = many(n, pp=0.56, conf=0.5)
    result = asyncio.run(o.optimize_portfolio(batch))
    enhanced = asyncio.run(o._enhance_opportunities_with_metrics(batch))
    kelly = o._calculate_kelly_fractions(enhanced)
    assert all(f <= kelly[m] + 1e-9 for m, f in result.allocations.items())


@pytest.mark.xfail(strict=True, reason="_calculate_kelly_fractions gives every NO side 0, then the "
                                       "fallback allocates it the full cap")
def test_no_side_position_is_sized_from_its_kelly_fraction(o):
    batch = many(3, pp=0.48, mp=0.55, conf=0.5)     # a 7-point NO edge at half confidence
    result = asyncio.run(o.optimize_portfolio(batch))
    enhanced = asyncio.run(o._enhance_opportunities_with_metrics(batch))
    kelly = o._calculate_kelly_fractions(enhanced)
    assert all(f <= kelly[m] + 1e-9 for m, f in result.allocations.items())


# ------------------------------------------------------------------ enhancement
def test_enhancement_computes_metrics_and_resets_kelly_state(o):
    source = opp(vol=0.4, expected_return=0.2, correlation_score=0.9, kelly_fraction=0.5,
                 fractional_kelly=0.5, risk_adjusted_fraction=0.5, volume=600.0)
    out = asyncio.run(o._enhance_opportunities_with_metrics([source]))[0]
    assert out is not source
    assert out.sharpe_ratio == pytest.approx(0.5)
    assert out.sortino_ratio == pytest.approx(0.2 / (0.4 * 0.7))
    assert out.max_drawdown_contribution == pytest.approx(0.8)
    assert (out.correlation_score, out.kelly_fraction, out.fractional_kelly,
            out.risk_adjusted_fraction) == (0.0, 0.0, 0.0, 0.0)
    assert out.volume == 600.0
    for name in ("market_id", "market_title", "predicted_probability", "market_probability",
                 "confidence", "edge", "volatility", "expected_return", "max_loss", "time_to_expiry"):
        assert getattr(out, name) == getattr(source, name)


def test_enhancement_skips_a_broken_opportunity_and_keeps_the_rest(o):
    broken = SimpleNamespace(market_id="broken")
    out = asyncio.run(o._enhance_opportunities_with_metrics([broken, opp("good")]))
    assert [x.market_id for x in out] == ["good"]
    assert "broken" in o.logger.error.call_args.args[0]


# ------------------------------------------------------------------ Kelly fractions
def test_kelly_applies_each_factor(o):
    o.max_position_fraction = 1.0
    target = opp(pp=0.7, mp=0.5, conf=0.8, tte=30.0)
    fractions = o._calculate_kelly_fractions([target])
    # full Kelly 0.4 at even odds, times confidence 0.8, times quarter Kelly
    assert fractions["m0"] == pytest.approx(0.08)
    assert target.kelly_fraction == pytest.approx(0.4)
    assert target.fractional_kelly == pytest.approx(0.08)
    assert target.risk_adjusted_fraction == pytest.approx(0.08)


def test_kelly_is_capped_at_the_position_limit_but_records_the_raw_figure(o):
    target = opp(pp=0.9, mp=0.5, conf=1.0)
    assert o._calculate_kelly_fractions([target])["m0"] == 0.03
    assert target.fractional_kelly == pytest.approx(0.2)
    assert target.risk_adjusted_fraction == 0.03


def test_kelly_uses_the_market_price_for_odds(o):
    o.max_position_fraction = 1.0
    target = opp(pp=0.7, mp=0.6, conf=1.0)      # b = 0.4/0.6, f* = (0.7b - 0.3)/b = 0.25
    assert o._calculate_kelly_fractions([target])["m0"] == pytest.approx(0.25 * 0.25)


@pytest.mark.parametrize("price", [0.0, 1.0])
def test_kelly_uses_even_odds_at_a_degenerate_price(o, price):
    o.max_position_fraction = 1.0
    target = opp(pp=0.7, mp=price, conf=1.0, edge=0.2)
    assert target.market_probability == price
    assert o._calculate_kelly_fractions([target])["m0"] == pytest.approx(0.4 * 0.25)


@pytest.mark.parametrize("tte, factor", [(60.0, 1.0), (30.0, 1.0), (15.0, 0.5), (3.0, 0.1), (0.1, 0.1)])
def test_kelly_decays_with_time_to_expiry_down_to_a_floor(o, tte, factor):
    o.max_position_fraction = 1.0
    target = opp(pp=0.7, mp=0.5, conf=1.0, tte=tte)
    assert o._calculate_kelly_fractions([target])["m0"] == pytest.approx(0.4 * factor * 0.25)


@pytest.mark.parametrize("state, factor", [("normal", 1.0), ("volatile", 0.7), ("trending", 1.2),
                                           ("unknown", 1.0)])
def test_kelly_follows_the_market_regime(o, state, factor):
    o.max_position_fraction = 1.0
    o.market_state = state
    assert o._get_regime_multiplier() == factor
    target = opp(pp=0.7, mp=0.5, conf=1.0)
    assert o._calculate_kelly_fractions([target])["m0"] == pytest.approx(0.4 * factor * 0.25)


@pytest.mark.parametrize("kw", [dict(pp=0.3, mp=0.5), dict(pp=0.5, mp=0.4), dict(pp=0.7, mp=0.5, edge=0.0)],
                         ids=["negative-edge", "coin-flip", "no-edge"])
def test_kelly_is_zero_without_a_favourable_edge(o, kw):
    target = opp(**kw)
    assert o._calculate_kelly_fractions([target])["m0"] == 0.0
    assert target.kelly_fraction == 0.0


def test_kelly_counts_a_broken_opportunity_as_zero(o):
    fractions = o._calculate_kelly_fractions([SimpleNamespace(market_id="broken"), opp("good")])
    assert fractions["broken"] == 0.0
    assert fractions["good"] > 0
    assert "broken" in o.logger.error.call_args.args[0]


# ------------------------------------------------------------------ correlation
def test_detect_regime_settles_on_normal(o):
    o.market_state = "volatile"
    asyncio.run(o._detect_market_regime())
    assert o.market_state == "normal"


@pytest.mark.parametrize("n", [0, 1])
def test_correlation_matrix_of_one_or_none_is_a_single_cell(o, n):
    assert asyncio.run(o._estimate_correlation_matrix(many(n))).tolist() == [[1.0]]


def test_correlation_matrix_fills_each_pair_and_records_the_highest(o):
    batch = many(3)
    values = {("m0", "m1"): 0.2, ("m1", "m0"): 0.2, ("m0", "m2"): -0.4, ("m2", "m0"): -0.4,
              ("m1", "m2"): 0.1, ("m2", "m1"): 0.1}

    async def pair(a, b):
        return values[(a.market_id, b.market_id)]
    o._estimate_pairwise_correlation = pair
    o._ensure_positive_semidefinite = lambda m: m
    matrix = asyncio.run(o._estimate_correlation_matrix(batch))
    assert matrix.tolist() == [[1.0, 0.2, -0.4], [0.2, 1.0, 0.1], [-0.4, 0.1, 1.0]]
    assert [x.correlation_score for x in batch] == [0.4, 0.2, 0.4]


def test_correlation_matrix_is_made_positive_semidefinite(o):
    o._ensure_positive_semidefinite = MagicMock(return_value=np.eye(2) * 7)
    assert asyncio.run(o._estimate_correlation_matrix(many(2))).tolist() == [[7.0, 0.0], [0.0, 7.0]]


def test_correlation_matrix_failure_falls_back_to_identity(o):
    o._estimate_pairwise_correlation = AsyncMock(side_effect=RuntimeError("boom"))
    assert asyncio.run(o._estimate_correlation_matrix(many(2))).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert "boom" in o.logger.error.call_args.args[0]


def test_pairwise_correlation_weights_its_parts(o):
    a = opp("a", tte=30.0, vol=0.5)
    b = opp("b", tte=15.0, vol=0.25)
    o._get_category_correlation = AsyncMock(return_value=0.5)
    o._get_content_similarity = AsyncMock(return_value=0.2)
    expected = 0.4 * 0.5 + 0.2 * 0.5 + 0.3 * 0.2 + 0.1 * 0.75
    assert asyncio.run(o._estimate_pairwise_correlation(a, b)) == pytest.approx(expected)
    o._get_category_correlation.assert_awaited_once_with("a", "b")
    o._get_content_similarity.assert_awaited_once_with(a, b)


def test_pairwise_correlation_is_order_independent_and_floors_its_parts(o):
    a = opp("a", tte=100.0, vol=2.0)
    b = opp("b", tte=10.0, vol=0.5)
    expected = 0.4 * 0.1 + 0.3 * 0.1     # time and volatility parts floor at 0, not negative
    assert asyncio.run(o._estimate_pairwise_correlation(a, b)) == pytest.approx(expected)
    assert asyncio.run(o._estimate_pairwise_correlation(b, a)) == pytest.approx(expected)


def test_pairwise_correlation_is_capped(o):
    o._get_category_correlation = AsyncMock(return_value=1.0)
    o._get_content_similarity = AsyncMock(return_value=1.0)
    assert asyncio.run(o._estimate_pairwise_correlation(opp("a"), opp("b"))) == 0.70


def test_pairwise_correlation_failure_is_a_small_default(o):
    o._get_category_correlation = AsyncMock(side_effect=RuntimeError("boom"))
    assert asyncio.run(o._estimate_pairwise_correlation(opp("a"), opp("b"))) == 0.1
    assert "boom" in o.logger.error.call_args.args[0]


def test_placeholder_correlations_are_low(o):
    assert asyncio.run(o._get_category_correlation("a", "b")) == 0.1
    assert asyncio.run(o._get_content_similarity(opp("a"), opp("b"))) == 0.1


def test_correlation_penalty_scales_with_the_average_absolute_correlation(o):
    matrix = np.array([[1.0, 0.4, -0.8], [0.4, 1.0, 0.0], [-0.8, 0.0, 1.0]])
    source = {"a": 0.1, "b": 0.2, "c": 0.4}
    out = o._apply_correlation_adjustments(source, matrix)
    assert out["a"] == pytest.approx(0.1 * (1 - 0.6 * 0.5))
    assert out["b"] == pytest.approx(0.2 * (1 - 0.2 * 0.5))
    assert out["c"] == pytest.approx(0.4 * (1 - 0.4 * 0.5))
    assert source == {"a": 0.1, "b": 0.2, "c": 0.4}


def test_correlation_adjustment_failure_keeps_the_fractions(o):
    out = o._apply_correlation_adjustments({"a": 0.1, "b": 0.2}, np.eye(1))
    assert out == {"a": 0.1, "b": 0.2}
    assert o.logger.error.called


@pytest.mark.xfail(strict=True, reason="np.mean of no other markets is NaN, so one market's fraction "
                                       "becomes NaN and sizing falls to the fallback")
def test_one_market_has_no_correlation_penalty(o):
    assert o._apply_correlation_adjustments({"a": 0.1}, np.eye(1)) == {"a": 0.1}


def test_positive_semidefinite_lifts_a_negative_eigenvalue(o):
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])     # eigenvalues 3 and -1
    fixed = o._ensure_positive_semidefinite(matrix)
    assert np.linalg.eigvalsh(fixed).min() == pytest.approx(0.001)
    assert np.linalg.eigvalsh(fixed).max() == pytest.approx(3.0)


def test_positive_semidefinite_failure_is_identity(o, monkeypatch):
    monkeypatch.setattr(optimizer_module.np.linalg, "eigh", MagicMock(side_effect=ValueError("boom")))
    assert o._ensure_positive_semidefinite(np.ones((2, 2))).tolist() == [[1.0, 0.0], [0.0, 1.0]]


# ------------------------------------------------------------------ multi-objective step
class FakeMinimize:
    """Stands in for scipy's minimize so its inputs can be inspected."""

    def __init__(self, x=None, success=True, error=None):
        self.x, self.success, self.error = x, success, error
        self.kwargs = None

    def __call__(self, fun, x0, **kwargs):
        self.fun, self.x0, self.kwargs = fun, x0, kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(x=x0 if self.x is None else np.array(self.x),
                               success=self.success, message="stubbed")


def install(monkeypatch, **kw):
    fake = FakeMinimize(**kw)
    monkeypatch.setattr(optimizer_module, "minimize", fake)
    return fake


def fallback_spy(o):
    o._simple_allocation_fallback = MagicMock(return_value={"fallback": 0.25})
    return o._simple_allocation_fallback


KELLY = {"m0": 0.1, "m1": 0.2}


def test_nothing_to_optimize_is_an_empty_dict(o):
    assert o._multi_objective_optimization([], {}, np.eye(1)) == {}


def test_tiny_starting_weights_use_the_fallback(o, monkeypatch):
    fake = install(monkeypatch)
    spy = fallback_spy(o)
    batch = many(2)
    out = o._multi_objective_optimization(batch, {"m0": 0.0004, "m1": 0.0005}, np.eye(2))
    assert out == {"fallback": 0.25}
    spy.assert_called_once_with(batch)
    assert fake.kwargs is None


def test_starting_weights_at_the_tiny_threshold_are_optimized(o, monkeypatch):
    fake = install(monkeypatch)
    o._multi_objective_optimization(many(2), {"m0": 0.0005, "m1": 0.0005}, np.eye(2))
    assert fake.kwargs is not None


def test_starting_weights_above_one_are_normalized(o, monkeypatch):
    fake = install(monkeypatch)
    o._multi_objective_optimization(many(2), {"m0": 1.0, "m1": 3.0}, np.eye(2))
    assert fake.x0.tolist() == [0.25, 0.75]


def test_starting_weights_under_one_are_kept(o, monkeypatch):
    fake = install(monkeypatch)
    o._multi_objective_optimization(many(2), {"m0": 0.3, "m1": 0.3}, np.eye(2))
    assert fake.x0.tolist() == [0.3, 0.3]


def test_a_market_missing_from_kelly_starts_at_zero(o, monkeypatch):
    fake = install(monkeypatch)
    o._multi_objective_optimization(many(2), {"m1": 0.2}, np.eye(2))
    assert fake.x0.tolist() == [0.0, 0.2]


def test_solver_settings_and_bounds(o, monkeypatch):
    fake = install(monkeypatch)
    o._multi_objective_optimization(many(2), KELLY, np.eye(2))
    assert fake.kwargs["method"] == "SLSQP"
    assert fake.kwargs["bounds"] == [(0, 0.03), (0, 0.03)]
    assert fake.kwargs["options"] == {"maxiter": 500, "ftol": 1e-6}


def test_bounds_never_pass_thirty_percent(o, monkeypatch):
    fake = install(monkeypatch)
    o.max_position_fraction = 0.5
    o._multi_objective_optimization(many(1), {"m0": 0.1}, np.eye(1))
    assert fake.kwargs["bounds"] == [(0, 0.3)]


@pytest.mark.parametrize("weights, ok", [
    ([0.40, 0.40], True), ([0.41, 0.40], False), ([0.03, 0.02], True), ([0.02, 0.02], False),
])
def test_total_stays_between_five_and_eighty_percent(o, monkeypatch, weights, ok):
    fake = install(monkeypatch)
    o._multi_objective_optimization(many(2), KELLY, np.eye(2))
    assert all(c["type"] == "ineq" for c in fake.kwargs["constraints"])
    values = [c["fun"](np.array(weights)) for c in fake.kwargs["constraints"]]
    assert bool(min(values) >= -1e-12) is ok


def test_objective_is_a_weighted_score(o, monkeypatch):
    fake = install(monkeypatch)
    batch = [opp("m0", vol=0.3, expected_return=0.2), opp("m1", vol=0.3, expected_return=0.1)]
    o._multi_objective_optimization(batch, KELLY, np.eye(2))
    w = np.array([0.1, 0.1])
    ret = 0.03
    vol = np.sqrt(2 * 0.01 * 0.09)
    div = 0.06 / (vol + 1e-8)
    sharpe = ret / (vol + 1e-8)
    drawdown = (0.1 * 0.6 + 0.1 * 0.6) * 0.8
    assert fake.fun(w) == pytest.approx(-(10.0 * ret + 2.0 * sharpe + 0.5 * div - 0.2 * vol - 0.1 * drawdown))


def test_objective_uses_the_correlations_in_its_risk_term(o, monkeypatch):
    fake = install(monkeypatch)
    batch = [opp("m0", vol=0.3, expected_return=0.2), opp("m1", vol=0.3, expected_return=0.2)]
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    o._multi_objective_optimization(batch, KELLY, corr)
    w = np.array([0.1, 0.1])
    vol = np.sqrt(0.09 * (0.01 + 0.01 + 2 * 0.6 * 0.01))
    div = 0.06 / (vol + 1e-8)
    sharpe = 0.04 / (vol + 1e-8)
    drawdown = 0.12 * 0.8
    assert fake.fun(w) == pytest.approx(-(10.0 * 0.04 + 2.0 * sharpe + 0.5 * div - 0.2 * vol - 0.1 * drawdown))


@pytest.mark.parametrize("weights, penalized", [
    ([-0.01, 0.1], True), ([0.6, 0.41], True), ([0.6, 0.4], False), ([0.0, 0.1], False),
])
def test_objective_penalizes_invalid_weights(o, monkeypatch, weights, penalized):
    fake = install(monkeypatch)
    o._multi_objective_optimization(many(2), KELLY, np.eye(2))
    assert bool(fake.fun(np.array(weights)) == 1e6) is penalized


def test_objective_penalizes_an_internal_error(o, monkeypatch):
    fake = install(monkeypatch)
    o._estimate_portfolio_max_drawdown = MagicMock(side_effect=RuntimeError("boom"))
    o._multi_objective_optimization(many(2), KELLY, np.eye(2))
    assert fake.fun(np.array([0.1, 0.1])) == 1e6


@pytest.mark.parametrize("kw", [
    dict(success=False), dict(x=[0.0001, 0.0001]), dict(error=ValueError("no")),
], ids=["not-converged", "near-zero-result", "solver-raised"])
def test_solver_trouble_uses_the_fallback(o, monkeypatch, kw):
    install(monkeypatch, **kw)
    spy = fallback_spy(o)
    batch = many(2)
    assert o._multi_objective_optimization(batch, KELLY, np.eye(2)) == {"fallback": 0.25}
    spy.assert_called_once_with(batch)
    assert o.logger.warning.called


def test_solver_result_at_the_near_zero_line_uses_the_fallback(o, monkeypatch):
    install(monkeypatch, x=[0.0005, 0.0005])
    fallback_spy(o)
    assert o._multi_objective_optimization(many(2), KELLY, np.eye(2)) == {"fallback": 0.25}


def test_solver_result_drops_dust_and_returns_plain_floats(o, monkeypatch):
    install(monkeypatch, x=[0.02, 0.001])
    out = o._multi_objective_optimization(many(2), KELLY, np.eye(2))
    assert out == {"m0": 0.02}
    assert type(out["m0"]) is float


def test_broken_input_uses_the_fallback(o, monkeypatch):
    spy = fallback_spy(o)
    batch = many(2)
    assert o._multi_objective_optimization(batch, KELLY, np.eye(3)) == {"fallback": 0.25}
    spy.assert_called_once_with(batch)
    assert o.logger.error.called


# ------------------------------------------------------------------ fallback allocation
def test_fallback_with_nothing_is_empty(o):
    assert o._simple_allocation_fallback([]) == {}


def ranked(*scores):
    """Opportunities whose fallback score (return x confidence x |edge|) is the given value."""
    return [opp(f"m{i}", conf=1.0, edge=1.0, expected_return=s) for i, s in enumerate(scores)]


def test_fallback_gives_more_to_higher_scores_up_to_eighty_percent(o):
    out = o._simple_allocation_fallback(ranked(0.1, 0.5, 0.4, 0.3, 0.2))
    assert out == pytest.approx({"m1": 0.25, "m2": 0.22, "m3": 0.19, "m4": 0.14})


def test_fallback_score_multiplies_return_confidence_and_edge_size(o):
    batch = [
        opp("a", conf=0.5, edge=0.5, expected_return=0.5),      # 0.125
        opp("b", conf=1.0, edge=-0.4, expected_return=0.5),     # 0.2, a NO edge counts by size
        opp("c", conf=0.5, edge=0.5, expected_return=1.0),      # 0.25
    ]
    out = o._simple_allocation_fallback(batch)
    assert out == pytest.approx({"c": 0.25, "b": 0.22, "a": 0.19})


def test_fallback_stops_at_eighty_percent_however_many_are_offered(o):
    out = o._simple_allocation_fallback(ranked(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3))
    assert set(out) == {"m0", "m1", "m2", "m3"}
    assert sum(out.values()) == pytest.approx(0.8)


def test_fallback_skips_scores_at_or_below_zero(o):
    assert o._simple_allocation_fallback(ranked(0.0, -0.5)) == {}
    assert o._simple_allocation_fallback(ranked(0.0, 0.5)) == {"m1": 0.25}


def test_fallback_failure_is_empty(o):
    assert o._simple_allocation_fallback([SimpleNamespace(market_id="broken")]) == {}
    assert o.logger.error.called


def test_fallback_logs_what_it_did(o):
    o._simple_allocation_fallback(ranked(0.5))
    assert "1 positions, 25.0% capital" in o.logger.info.call_args.args[0]


# ------------------------------------------------------------------ risk constraints
def test_constraints_cap_each_position(o):
    assert o._apply_risk_constraints({"a": 0.25}, [opp("a")]) == {"a": 0.03}


def test_constraints_drop_a_position_under_the_minimum_size(o):
    out = o._apply_risk_constraints({"a": 0.0004, "b": 0.0005, "c": 0.01},
                                    [opp("a"), opp("b"), opp("c")])
    assert out == {"b": 0.0005, "c": 0.01}


def test_constraints_drop_a_market_with_no_opportunity(o):
    assert o._apply_risk_constraints({"ghost": 0.02, "a": 0.02}, [opp("a")]) == {"a": 0.02}


def test_constraints_scale_the_last_position_to_fit_capital(o):
    o.max_position_fraction = 1.0
    out = o._apply_risk_constraints({"a": 0.7, "b": 0.6, "c": 0.2}, [opp("a"), opp("b"), opp("c")])
    assert out == pytest.approx({"a": 0.7, "b": 0.3})


def test_constraints_allow_exactly_all_capital(o):
    o.max_position_fraction = 1.0
    out = o._apply_risk_constraints({"a": 0.5, "b": 0.5}, [opp("a"), opp("b")])
    assert out == {"a": 0.5, "b": 0.5}


def test_constraints_stop_when_no_capital_remains(o):
    o.max_position_fraction = 1.0
    out = o._apply_risk_constraints({"a": 1.0, "b": 0.5, "c": 0.5}, [opp("a"), opp("b"), opp("c")])
    assert out == {"a": 1.0}


def test_constraints_stop_at_the_first_position_that_cannot_fit(o):
    o.max_position_fraction = 1.0
    o.min_position_size = 0
    out = o._apply_risk_constraints({"a": 0.9995, "b": 0.001, "c": 0.0001}, [opp("a"), opp("b"), opp("c")])
    assert out == {"a": 0.9995}


def test_constraints_failure_returns_the_allocation_unchanged(o):
    source = {"a": 0.25}
    assert o._apply_risk_constraints(source, None) is source
    assert o.logger.error.called


# ------------------------------------------------------------------ metrics
def test_metrics_for_no_allocation_are_empty(o):
    assert o._calculate_portfolio_metrics({}, [], np.eye(1)) == EMPTY_METRICS


def test_metrics_for_one_position(o):
    target = opp("a", vol=0.4, expected_return=0.2, kelly_fraction=0.5, max_drawdown_contribution=0.8)
    m = o._calculate_portfolio_metrics({"a": 0.1}, [target], np.eye(1))
    assert m["total_capital_used"] == pytest.approx(1000.0)
    assert m["expected_portfolio_return"] == pytest.approx(0.02)
    assert m["portfolio_volatility"] == pytest.approx(0.04)
    assert m["portfolio_sharpe"] == pytest.approx(0.5, rel=1e-5)
    assert m["diversification_ratio"] == pytest.approx(1.0, rel=1e-5)
    assert m["portfolio_var_95"] == pytest.approx(0.04 * 1.645)
    assert m["portfolio_cvar_95"] == pytest.approx(0.04 * 1.645 * 1.2)
    assert m["aggregate_kelly_fraction"] == pytest.approx(0.05)
    assert m["portfolio_growth_rate"] == pytest.approx(0.02 - 0.04 ** 2 / 2)
    assert m["max_portfolio_drawdown"] == pytest.approx(0.1 * 0.8 * 0.8)


def test_metrics_for_two_positions_use_their_own_correlation(o):
    batch = [opp("skipped", vol=0.9), opp("a", vol=0.3, expected_return=0.2), opp("b", vol=0.3, expected_return=0.1)]
    corr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]])
    m = o._calculate_portfolio_metrics({"a": 0.1, "b": 0.1}, batch, corr)
    expected = np.sqrt(0.09 * (0.01 + 0.01 + 2 * 0.5 * 0.01))
    assert m["portfolio_volatility"] == pytest.approx(expected)
    assert m["expected_portfolio_return"] == pytest.approx(0.03)
    assert m["diversification_ratio"] == pytest.approx(0.06 / expected, rel=1e-5)


def test_metrics_drop_cross_terms_when_the_matrix_is_too_small(o):
    batch = [opp("a", vol=0.3), opp("b", vol=0.3)]
    m = o._calculate_portfolio_metrics({"a": 0.1, "b": 0.1}, batch, np.eye(1))
    assert m["portfolio_volatility"] == pytest.approx(np.sqrt(2 * 0.01 * 0.09))


def test_metrics_failure_is_empty(o):
    assert o._calculate_portfolio_metrics({"a": 0.1}, [SimpleNamespace(market_id="a")], np.eye(1)) == EMPTY_METRICS
    assert o.logger.error.called


def test_drawdown_estimate_takes_a_diversification_discount(o):
    batch = [opp("a", max_drawdown_contribution=0.5), opp("b", max_drawdown_contribution=0.25)]
    assert o._estimate_portfolio_max_drawdown(np.array([0.2, 0.4]), batch) == pytest.approx(0.2 * 0.8)


# ------------------------------------------------------------------ runner
@pytest.fixture
def runner_parts(monkeypatch):
    built = MagicMock()
    built._empty_allocation.return_value = "empty"
    built.optimize_portfolio = AsyncMock(
        return_value=SimpleNamespace(allocations={"a": 0.03}, total_capital_used=300.0))
    cls = MagicMock(return_value=built)
    monkeypatch.setattr(runner_module, "AdvancedPortfolioOptimizer", cls)
    make = AsyncMock(return_value=["opportunity"])
    monkeypatch.setattr(runner_module, "create_market_opportunities_from_markets", make)
    db = MagicMock()
    db.get_eligible_markets = AsyncMock(return_value=["market"])
    return SimpleNamespace(built=built, cls=cls, make=make, db=db, kalshi=MagicMock(), xai=MagicMock())


def run_runner(p):
    return asyncio.run(runner_module.run_portfolio_optimization(p.db, p.kalshi, p.xai))


def test_runner_optimizes_what_it_found(runner_parts):
    p = runner_parts
    assert run_runner(p) is p.built.optimize_portfolio.return_value
    p.cls.assert_called_once_with(p.db, p.kalshi, p.xai)
    p.db.get_eligible_markets.assert_awaited_once_with(volume_min=20000, max_days_to_expiry=365)
    p.make.assert_awaited_once_with(["market"], p.xai, p.kalshi, None, 0)
    p.built.optimize_portfolio.assert_awaited_once_with(["opportunity"])


def test_runner_never_passes_a_db_so_batch_mode_cannot_trade(runner_parts):
    p = runner_parts
    run_runner(p)
    assert p.make.await_args.args[3] is None


def test_runner_with_no_markets_is_empty(runner_parts):
    p = runner_parts
    p.db.get_eligible_markets.return_value = []
    assert run_runner(p) == "empty"
    p.make.assert_not_awaited()


def test_runner_with_no_opportunities_is_empty(runner_parts):
    p = runner_parts
    p.make.return_value = []
    assert run_runner(p) == "empty"
    p.built.optimize_portfolio.assert_not_awaited()


def test_runner_failure_is_empty(runner_parts):
    p = runner_parts
    p.db.get_eligible_markets.side_effect = RuntimeError("boom")
    assert run_runner(p) == "empty"
