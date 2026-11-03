"""src/utils/edge_filter.py: edge/EV math, fees, side symmetry, thresholds, bad inputs.

Pure functions: no network, no Kalshi client, no credentials. The one test that
drives a caller (portfolio/immediate.py) gives it a recording fake that raises if
any order method is called, and asserts nothing was called.

Who calls EdgeFilter (grep):
  * src/jobs/decide.py:336-360       make_decision_for_market -> `cli.py run` (BeastModeBot).
                                     LIVE on the `run` path.
  * src/strategies/portfolio/immediate.py:89,148  `cli.py run` via unified_trading_system.
  * src/strategies/market_making.py:164           `cli.py run` via unified_trading_system.
  * NOT on `cli.py daily`: cmd_daily -> RiskGovernor -> SafeCompounder, which has its own math
    and never imports edge_filter.
  * filter_opportunities / get_edge_summary / the module-level helpers have no caller.

Strict xfails pin real defects; each reason is file:line plus a concrete example. When a
bug is fixed the xfail flips to XPASS(strict) and fails: delete the marker then.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.settings import settings
from src.utils.edge_filter import (
    EdgeFilter,
    EdgeFilterResult,
    calculate_edge,
    get_minimum_edge_for_confidence,
    passes_edge_filter,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from aaa_pricer import taker_fee_order, taker_fee_per_contract  # noqa: E402

NAN = float("nan")
EF = EdgeFilter


def order_fee_dollars(count: int, price_cents: int) -> float:
    """Kalshi taker fee, hand-rolled in exact integer arithmetic.

    fee_cents = ceil(0.07 * count * P * (1-P) * 100) with P = price_cents/100
              = ceil(7 * count * c * (100-c) / 10000)   (c in cents)
    Rounded up ONCE for the whole order.
    """
    num = 7 * count * price_cents * (100 - price_cents)
    return -(-num // 10_000) / 100


# ------------------------------------------------------------ the fee reference
@pytest.mark.parametrize("count,cents,expected", [
    (1, 50, 0.02),      # 17500/10000 = 1.75c -> 2c
    (10, 50, 0.18),     # 17.5c -> 18c   (per-contract rounding would say 0.20)
    (100, 50, 1.75),    # 175.0c exactly, no round-up
    (388, 76, 4.96),    # 7*388*76*24 = 4,953,984 -> 495.4c -> 496c
    (1, 5, 0.01),       # 0.3325c -> 1c
    (100, 5, 0.34),     # 33.25c -> 34c
])
def test_hand_fee_reference_matches_aaa_pricer_order_level(count, cents, expected):
    assert order_fee_dollars(count, cents) == pytest.approx(expected)
    assert taker_fee_order(count, cents / 100) == pytest.approx(expected)


def test_per_contract_rounding_overstates_and_is_confined_to_scripts():
    """Issue #102. 100 contracts @50c really cost $1.75; per-contract rounding says $2.00."""
    assert order_fee_dollars(100, 50) == pytest.approx(1.75)
    assert taker_fee_per_contract(0.50) * 100 == pytest.approx(2.00)
    # Flag any per-contract rounding in src/: nothing under src/ may use it.
    offenders = []
    for path in (REPO / "src").rglob("*.py"):
        text = path.read_text()
        if "taker_fee_per_contract" in text or "aaa_pricer" in text:
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], f"per-contract fee rounding leaked into src: {offenders}"


def test_src_has_no_fee_model_at_all():
    """Records the gap the fee xfail below depends on: edge_filter never sees a fee."""
    src = (REPO / "src/utils/edge_filter.py").read_text().lower()
    assert "fee" not in src


# ------------------------------------------------------------ edge / EV math
@pytest.mark.parametrize("ai,mkt,side,edge_pct", [
    (0.62, 0.50, "YES", 0.12),
    (0.30, 0.45, "NO", 0.15),
    (0.90, 0.10, "YES", 0.80),
])
def test_edge_matches_hand_computed_ev_per_contract(ai, mkt, side, edge_pct):
    """Buy YES at `mkt`: EV = ai - mkt. Buy NO at 1-mkt: EV = (1-ai) - (1-mkt) = mkt - ai."""
    r = EF.calculate_edge(ai, mkt, 0.9)
    assert r.side == side
    assert r.edge_percentage == pytest.approx(edge_pct)
    yes_ev, no_ev = ai - mkt, mkt - ai
    assert r.edge_magnitude == pytest.approx(yes_ev)
    assert r.edge_percentage == pytest.approx(yes_ev if side == "YES" else no_ev)
    assert r.confidence_adjusted_edge == pytest.approx(edge_pct * 0.9)


def test_zero_edge_goes_to_no_side_and_is_rejected():
    r = EF.calculate_edge(0.5, 0.5, 0.9)
    assert r.edge_magnitude == 0 and r.side == "NO" and not r.passes_filter


def test_reason_strings_name_the_failing_gate():
    low_conf = EF.calculate_edge(0.9, 0.5, 0.30)
    assert "below minimum" in low_conf.reason and "30.0%" in low_conf.reason
    thin = EF.calculate_edge(0.52, 0.50, 0.9)
    assert "below required 3.0%" in thin.reason
    ok = EF.calculate_edge(0.60, 0.50, 0.9)
    assert ok.reason.startswith("Meets requirements")


def test_confidence_gate_is_checked_before_edge_gate():
    r = EF.calculate_edge(0.99, 0.01, 0.34)   # huge edge, confidence under the floor
    assert not r.passes_filter and "Confidence" in r.reason


# ------------------------------------------------------------ side symmetry
GRID = [(a, m, c) for a in (0.05, 0.38, 0.47, 0.53, 0.62, 0.95)
        for m in (0.10, 0.50, 0.90) for c in (0.35, 0.59, 0.6, 0.79, 0.8, 0.99)]


@pytest.mark.parametrize("ai,mkt,conf", GRID)
def test_yes_no_mirror_is_symmetric(ai, mkt, conf):
    """Evaluating (p, m) must equal evaluating the complement market (1-p, 1-m), flipped."""
    a = EF.calculate_edge(ai, mkt, conf)
    b = EF.calculate_edge(1 - ai, 1 - mkt, conf)
    assert a.passes_filter == b.passes_filter
    assert a.edge_percentage == pytest.approx(b.edge_percentage, abs=1e-12)
    assert a.edge_magnitude == pytest.approx(-b.edge_magnitude, abs=1e-12)
    if a.edge_percentage > 1e-12:
        assert {a.side, b.side} == {"YES", "NO"}


def test_symmetry_holds_at_the_exact_threshold():
    a = EF.calculate_edge(0.53, 0.50, 0.8)
    b = EF.calculate_edge(0.47, 0.50, 0.8)
    assert a.passes_filter and b.passes_filter
    assert (a.side, b.side) == ("YES", "NO")


# ------------------------------------------------------------ thresholds
def test_constants_pin_the_current_ladder():
    assert (EF.HIGH_CONFIDENCE_EDGE, EF.MEDIUM_CONFIDENCE_EDGE, EF.LOW_CONFIDENCE_EDGE) == (0.03, 0.05, 0.08)
    assert EF.MIN_CONFIDENCE_FOR_TRADE == 0.35
    assert EF.MIN_EDGE_REQUIREMENT == 0.04


@pytest.mark.parametrize("conf,expected", [
    (0.99, 0.03), (0.80, 0.03), (0.7999, 0.05), (0.60, 0.05), (0.5999, 0.08), (0.35, 0.08), (0.0, 0.08),
])
def test_required_edge_ladder_boundaries(conf, expected):
    assert get_minimum_edge_for_confidence(conf) == expected


@pytest.mark.parametrize("conf", [0.99, 0.8, 0.7999, 0.7, 0.6, 0.5999, 0.5, 0.35])
def test_helper_ladder_agrees_with_calculate_edge_ladder(conf):
    """The ladder is written twice (calculate_edge and the helper); they must not drift."""
    need = get_minimum_edge_for_confidence(conf)
    just_over = EF.calculate_edge(0.5 + need + 0.0005, 0.5, conf)
    well_under = EF.calculate_edge(0.5 + need - 0.01, 0.5, conf)
    assert just_over.passes_filter
    assert not well_under.passes_filter


@pytest.mark.parametrize("conf,edge,passes", [
    (0.80, 0.03, True),     # at the HIGH threshold
    (0.80, 0.0285, False),  # clearly under it
    (0.7999, 0.04, False),  # drops to the MEDIUM rung (needs ~5%)
    (0.7999, 0.05, True),
    (0.60, 0.05, True),
    (0.5999, 0.05, False),  # drops to the LOW rung (needs ~8%)
    (0.5999, 0.08, True),
    (0.35, 0.08, True),     # at the confidence floor
    (0.3499, 0.50, False),  # under the floor, however big the edge
])
def test_threshold_boundaries(conf, edge, passes):
    assert EF.calculate_edge(0.5 + edge, 0.5, conf).passes_filter is passes


def test_confidence_floor_is_inclusive():
    assert EF.calculate_edge(0.9, 0.5, 0.35).passes_filter
    assert not EF.calculate_edge(0.9, 0.5, 0.3499999).passes_filter


def test_module_helpers_use_default_confidence_0_7():
    r = calculate_edge(0.56, 0.50)
    assert r.passes_filter and r.confidence_adjusted_edge == pytest.approx(0.06 * 0.7)
    assert passes_edge_filter(0.56, 0.50) is True
    assert passes_edge_filter(0.52, 0.50) is False
    assert isinstance(r, EdgeFilterResult)


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:97 `edge_percentage > required_edge - 0.001`: the 'floating point' "
    "tolerance is 0.1 point (float noise is ~1e-16), so conf 0.8 with a 2.95% edge passes a 3% "
    "rule; at conf 0.5999 a 7.95% edge passes the 8% rule."))
def test_edge_just_under_threshold_is_rejected():
    assert not EF.calculate_edge(0.5295, 0.50, 0.8).passes_filter


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:40,87: MIN_EDGE_REQUIREMENT=4% is declared but never read; "
    "HIGH_CONFIDENCE_EDGE=3% lets conf 0.9 trade a 3.0% edge, under the stated 4% minimum."))
def test_nothing_passes_below_the_declared_minimum_edge():
    r = EF.calculate_edge(0.53, 0.50, 0.9)
    assert not r.passes_filter or r.edge_percentage >= EF.MIN_EDGE_REQUIREMENT


# ------------------------------------------------------------ agreement with settings.py
def test_settings_confidence_floor_is_not_looser_than_edge_filter_floor_on_the_run_path():
    """decide.py applies settings.min_confidence_to_trade (0.45) BEFORE EdgeFilter, so the
    stricter settings value wins there. This pins the relationship that makes that true."""
    assert settings.trading.min_confidence_to_trade >= EF.MIN_CONFIDENCE_FOR_TRADE


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:42 MIN_CONFIDENCE_FOR_TRADE=0.35 < settings.py:81 "
    "min_confidence_to_trade=0.45: portfolio/immediate.py:90 and market_making.py:167 use only "
    "EdgeFilter, so a conf 0.40 opportunity (edge 10%) passes there but is refused in decide.py."))
def test_edge_filter_confidence_floor_agrees_with_settings():
    assert EF.MIN_CONFIDENCE_FOR_TRADE >= settings.trading.min_confidence_to_trade


@pytest.mark.xfail(strict=True, reason=(
    "src/config/settings.py:201 min_trade_edge=0.08 is not honoured: edge_filter.py:40-43 trade at "
    "3-8% depending on confidence, and min_trade_edge has no reader anywhere in src."))
def test_edge_filter_never_trades_below_settings_min_trade_edge():
    import src.config.settings as s
    r = EF.calculate_edge(0.53, 0.50, 0.9)
    assert not r.passes_filter or r.edge_percentage >= s.min_trade_edge


@pytest.mark.xfail(strict=True, reason=(
    "src/strategies/portfolio/immediate.py:156-159 hard-codes min_volume=1000 and "
    "max_time_to_expiry=365 and reads volume via getattr(opportunity,'volume',1000) -- "
    "MarketOpportunity (portfolio/models.py:8) has no volume field, so the default 1000 always "
    "equals min_volume and the liquidity gate can never fail; settings.py:78 allows 14 days, not 365."))
def test_immediate_path_passes_settings_liquidity_and_expiry_limits_to_the_filter(monkeypatch):
    import asyncio
    from src.strategies.portfolio import immediate

    seen = {}

    def fake_should_trade(ai_probability, market_probability, confidence, additional_filters=None):
        seen.update(additional_filters or {})
        return False, "recorded", EF.calculate_edge(ai_probability, market_probability, confidence)

    monkeypatch.setattr(EF, "should_trade_market", staticmethod(fake_should_trade))

    class RecordingKalshi:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            async def rec(*a, **k):
                self.calls.append(name)
                raise AssertionError(f"order/IO method {name} must not be called")
            return rec

    kalshi = RecordingKalshi()
    opp = SimpleNamespace(predicted_probability=0.7, market_probability=0.5, confidence=0.8,
                          expected_return=0.16, time_to_expiry=30.0, market_id="KXT-1")
    asyncio.run(immediate._evaluate_immediate_trade(opp, db_manager=None, kalshi_client=kalshi,
                                                    total_capital=1000))
    assert kalshi.calls == []
    assert seen["max_time_to_expiry"] == settings.trading.max_time_to_expiry_days
    assert seen["min_volume"] == settings.trading.min_volume


# ------------------------------------------------------------ fees
def test_gross_edge_at_threshold_is_below_threshold_net_of_order_fee():
    """Hand computation behind the fee xfail: 100 contracts @50c, conf 0.8, AI 53%."""
    fee_per_contract = order_fee_dollars(100, 50) / 100          # 1.75c
    gross = 0.53 - 0.50
    assert fee_per_contract == pytest.approx(0.0175)
    assert gross - fee_per_contract == pytest.approx(0.0125)
    assert gross - fee_per_contract < EF.HIGH_CONFIDENCE_EDGE


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:93-101: edge is gross of Kalshi's taker fee. 100 contracts @50c, "
    "AI 53% conf 0.8: gross 3.0% passes, but fee $1.75 (1.75c/contract) leaves a 1.25% net edge, "
    "under the 3% rule. EdgeFilter takes no price/count so it cannot be fee-aware."))
def test_edge_is_fee_aware():
    gross = EF.calculate_edge(0.53, 0.50, 0.8)
    net = gross.edge_percentage - order_fee_dollars(100, 50) / 100
    assert gross.passes_filter is (net >= EF.HIGH_CONFIDENCE_EDGE)


# ------------------------------------------------------------ bad inputs: fail closed or open?
@pytest.mark.parametrize("ai,mkt", [(0.0, 0.0), (1.0, 1.0), (0.99, 0.99), (0.01, 0.01)])
def test_equal_probabilities_at_the_rails_have_no_edge(ai, mkt):
    assert not EF.calculate_edge(ai, mkt, 0.9).passes_filter


def test_probabilities_are_clamped_to_one_to_ninety_nine_percent():
    r = EF.calculate_edge(1.0, 0.0, 0.9)
    assert r.edge_magnitude == pytest.approx(0.98) and r.side == "YES"
    r = EF.calculate_edge(0.0, 1.0, 0.9)
    assert r.edge_magnitude == pytest.approx(-0.98) and r.side == "NO"


@pytest.mark.parametrize("ai,mkt,conf", [(None, 0.5, 0.9), (0.6, None, 0.9)])
def test_none_probability_raises_rather_than_passing(ai, mkt, conf):
    with pytest.raises(TypeError):
        EF.calculate_edge(ai, mkt, conf)


def test_nan_confidence_is_rejected():
    r = EF.calculate_edge(0.9, 0.5, NAN)
    assert not r.passes_filter


def test_negative_confidence_is_rejected():
    assert not EF.calculate_edge(0.9, 0.5, -0.5).passes_filter


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:71 `max(0.01, min(0.99, nan))` returns 0.99, so ai_probability=NaN "
    "becomes 99%: vs a 50c market it reports a 49% YES edge and PASSES (conf 0.9)."))
def test_nan_ai_probability_is_rejected():
    assert not EF.calculate_edge(NAN, 0.5, 0.9).passes_filter


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:72 NaN market_probability also clamps to 0.99: ai 0.5 vs a NaN "
    "price reports a 49% NO edge and PASSES."))
def test_nan_market_probability_is_rejected():
    assert not EF.calculate_edge(0.5, NAN, 0.9).passes_filter


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:73 `confidence or 0.7`: confidence 0.0 (and None) is replaced by 0.7, "
    "so a zero-confidence call with a 40% edge PASSES at 'medium' confidence."))
@pytest.mark.parametrize("conf", [0.0, None])
def test_zero_or_missing_confidence_is_rejected(conf):
    assert not EF.calculate_edge(0.9, 0.5, conf).passes_filter


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:71 out-of-range probabilities are silently clamped, not rejected: "
    "ai_probability=5.0 (a percent-vs-fraction unit slip) becomes 0.99 and passes with a 49% edge."))
def test_out_of_range_probability_is_rejected():
    assert not EF.calculate_edge(5.0, 0.5, 0.9).passes_filter


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:73,98 confidence is never range-checked: confidence=1.5 (150%) "
    "passes as 'high' and is echoed in the reason string."))
def test_out_of_range_confidence_is_rejected():
    assert not EF.calculate_edge(0.55, 0.5, 1.5).passes_filter


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:72 market price 0.0 / 1.0 (no quote on a side) is clamped to 0.01/0.99 "
    "and treated as buyable: ai 0.5 vs market 1.0 reports a 49% NO edge and passes."))
@pytest.mark.parametrize("mkt", [0.0, 1.0])
def test_unquoted_market_price_is_rejected(mkt):
    assert not EF.calculate_edge(0.5, mkt, 0.9).passes_filter


# ------------------------------------------------------------ should_trade_market
def _stm(vol=1000, tte=30, conf=0.9, ai=0.60, mkt=0.50, min_vol=1000, max_t=365):
    return EF.should_trade_market(ai, mkt, conf, {"volume": vol, "min_volume": min_vol,
                                                  "time_to_expiry_days": tte, "max_time_to_expiry": max_t})


def test_should_trade_approves_with_prefix_and_returns_edge_result():
    ok, reason, res = _stm()
    assert ok and reason.startswith("TRADE APPROVED: ") and isinstance(res, EdgeFilterResult)


def test_edge_rejection_short_circuits_before_liquidity():
    ok, reason, _ = _stm(ai=0.51, vol=0)
    assert not ok and "Edge" in reason and "Volume" not in reason


@pytest.mark.parametrize("vol,ok", [(999, False), (1000, True), (1001, True)])
def test_volume_boundary_is_inclusive(vol, ok):
    assert _stm(vol=vol)[0] is ok


@pytest.mark.parametrize("tte,ok", [(364.9, True), (365, True), (365.1, False)])
def test_expiry_boundary_is_inclusive(tte, ok):
    assert _stm(tte=tte)[0] is ok


def test_rejection_messages_carry_the_numbers():
    assert _stm(vol=300, min_vol=500)[1] == "Volume 300 below minimum 500"
    assert _stm(tte=20, max_t=14)[1] == "Time to expiry 20 days exceeds maximum 14"


def test_missing_volume_key_defaults_to_zero_and_is_rejected():
    ok, reason, _ = EF.should_trade_market(0.6, 0.5, 0.9, {"min_volume": 100})
    assert not ok and "Volume 0" in reason


def test_missing_min_volume_defaults_to_1000():
    assert EF.should_trade_market(0.6, 0.5, 0.9, {"volume": 1000})[0]
    assert not EF.should_trade_market(0.6, 0.5, 0.9, {"volume": 999})[0]


@pytest.mark.parametrize("filters", [None, {}])
def test_no_additional_filters_means_no_liquidity_or_expiry_gate(filters):
    ok, _, _ = EF.should_trade_market(0.6, 0.5, 0.9, filters)
    assert ok


def test_none_volume_raises_rather_than_passing():
    with pytest.raises(TypeError):
        EF.should_trade_market(0.6, 0.5, 0.9, {"volume": None})


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:195 `volume < min_volume` is False for NaN, so a NaN volume PASSES "
    "the liquidity gate."))
def test_nan_volume_is_rejected():
    assert not _stm(vol=NAN)[0]


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:201 `time_to_expiry > max_time` is False for NaN, so a NaN "
    "time-to-expiry PASSES the horizon gate."))
def test_nan_time_to_expiry_is_rejected():
    assert not _stm(tte=NAN)[0]


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:201 only an upper bound on time_to_expiry: -5 days (already expired) "
    "passes the horizon gate."))
def test_negative_time_to_expiry_is_rejected():
    assert not _stm(tte=-5)[0]


# ------------------------------------------------------------ filter_opportunities / summary
def _opp(**kw):
    base = {"predicted_probability": 0.60, "market_probability": 0.50, "confidence": 0.9}
    base.update(kw)
    return base


def test_filter_opportunities_keeps_passers_and_annotates_them():
    good, thin = _opp(), _opp(predicted_probability=0.51)
    out = EF.filter_opportunities([good, thin])
    assert out == [good]
    assert good["recommended_side"] == "YES"
    assert good["edge_percentage"] == pytest.approx(0.10)
    assert good["filtered_edge"] == pytest.approx(0.10)
    assert isinstance(good["edge_filter_result"], EdgeFilterResult)
    assert "recommended_side" not in thin


def test_filter_opportunities_bypass_flag_returns_input_untouched():
    opps = [_opp(predicted_probability=0.51)]
    assert EF.filter_opportunities(opps, require_edge_filter=False) is opps


def test_filter_opportunities_missing_fields_default_to_no_edge_and_are_rejected():
    assert EF.filter_opportunities([{}, {"confidence": 0.9}]) == []


def test_filter_opportunities_no_side_flips_for_no_trades():
    o = _opp(predicted_probability=0.30, market_probability=0.45)
    assert EF.filter_opportunities([o]) == [o] and o["recommended_side"] == "NO"


def test_filter_opportunities_one_none_probability_aborts_the_batch():
    with pytest.raises(TypeError):
        EF.filter_opportunities([_opp(), _opp(predicted_probability=None)])


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:140 opp.get('confidence', 0.7) then edge_filter.py:73 `or 0.7`: an "
    "opportunity with confidence=None/0 is graded as 70% and passes a 5% edge. (No caller today.)"))
def test_filter_opportunities_rejects_missing_confidence():
    assert EF.filter_opportunities([_opp(confidence=None, predicted_probability=0.55)]) == []


def test_edge_summary_empty():
    assert EF.get_edge_summary([]) == {"total": 0, "passed": 0, "pass_rate": 0.0}


def test_edge_summary_math():
    rs = [EF.calculate_edge(0.60, 0.50, 0.9),    # YES pass, edge .10, adj .09
          EF.calculate_edge(0.30, 0.50, 0.9),    # NO pass, edge .20, adj .18
          EF.calculate_edge(0.51, 0.50, 0.9)]    # YES fail, edge .01, adj .009
    s = EF.get_edge_summary(rs)
    assert (s["total_opportunities"], s["passed_filter"], s["filtered_out"]) == (3, 2, 1)
    assert s["pass_rate"] == pytest.approx(2 / 3)
    assert s["average_edge"] == pytest.approx((0.10 + 0.20 + 0.01) / 3)
    assert s["average_confidence_adjusted_edge"] == pytest.approx((0.09 + 0.18 + 0.009) / 3)
    assert (s["yes_positions"], s["no_positions"]) == (2, 1)
    assert s["side_balance"] == pytest.approx(2 / 3)


@pytest.mark.xfail(strict=True, reason=(
    "src/utils/edge_filter.py:217 empty summary uses keys total/passed, the non-empty one uses "
    "total_opportunities/passed_filter: a caller reading ['passed_filter'] KeyErrors on no data."))
def test_edge_summary_has_the_same_keys_when_empty():
    assert set(EF.get_edge_summary([])) >= {"total_opportunities", "passed_filter"}


def test_valid_grid_inputs_never_produce_nan():
    for ai, mkt, conf in GRID:
        r = EF.calculate_edge(ai, mkt, conf)
        assert not math.isnan(r.edge_percentage) and not math.isnan(r.confidence_adjusted_edge)
