"""AAA data + pricer: parsing, stitching, streak model, fees, strike parsing."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import aaa_data  # noqa: E402
from aaa_pricer import (  # noqa: E402
    cdf_from_samples,
    fee_aware_edges,
    fit_model,
    forecast_cdf,
    parse_strike,
    streak_runs,
    taker_fee_order,
    taker_fee_per_contract,
)
from aaa_futures import convergence  # noqa: E402

PAGE = """
<div class="tblwrap"><table class="table-mob">
<tr><td>Current Avg.</td><td>$4.4874</td><td>$4.9989</td><td>$5.3859</td><td>$6.4839</td><td>$3.4472</td></tr>
<tr><td>Yesterday Avg.</td><td>$4.4918</td><td>$5.0027</td><td>$5.3934</td><td>$6.5019</td><td>$3.4641</td></tr>
</table></div>
"""

PAGE_NO_YDAY = """
<tr><td>Current Avg.</td><td>$3.1000</td><td>$3.2000</td><td>$3.3000</td><td>$4.1000</td></tr>
"""


def test_parse_aaa_page():
    p = aaa_data.parse_aaa_page(PAGE)
    assert p is not None
    assert p["cur_die"] == 6.4839
    assert p["cur_reg"] == 4.4874
    assert p["yes_die"] == 6.5019


def test_parse_aaa_page_without_yesterday():
    p = aaa_data.parse_aaa_page(PAGE_NO_YDAY)
    assert p is not None
    assert p["cur_die"] == 4.1
    assert "yes_die" not in p


def test_parse_aaa_page_garbage():
    assert aaa_data.parse_aaa_page("<html>nothing here</html>") is None


def _rows(pairs):
    """pairs: list of (date, cur, yes) -> csv-shaped rows dict."""
    rows = {}
    for date, cur, yes in pairs:
        rows[date] = {
            "date": date,
            "diesel": cur,
            "diesel_yes": yes if yes is not None else "",
            "regular": cur,
            "regular_yes": yes if yes is not None else "",
            "src": "test",
            "snapshot_ts": date.replace("-", "") + "120000",
        }
    return rows


def test_build_print_series_stitches_pairs():
    rows = _rows([
        ("2026-01-01", 3.00, None),
        ("2026-01-02", 3.05, 3.00),
        ("2026-01-03", 3.02, 3.05),
    ])
    seq = aaa_data.build_print_series(rows)
    assert [s["delta_cents"] for s in seq] == [5.0, -3.0]


def test_streak_runs():
    d = [1.0, -0.5, -0.6, -0.7, 0.2, -0.5, -0.6, 0.0]
    runs = streak_runs(d)
    assert runs == [(1, 3), (5, 2)]


def test_fit_model_blend_stays_in_unit_interval():
    d = [0.5, -0.6, -0.7, 0.3, -0.5, -0.6, 0.2, -1.0, -0.5]
    m = fit_model(d, current_streak=2)
    assert 0.0 <= m["p_extend"] <= 1.0
    for x in (-10.0, -0.4, 0.0, 10.0):
        p = forecast_cdf(m, x)
        assert 0.0 <= p <= 1.0
    # cdf is monotone
    assert forecast_cdf(m, -1.0) <= forecast_cdf(m, 0.0) <= forecast_cdf(m, 1.0)


def test_forecast_cdf_no_streak_falls_back_to_unconditional():
    d = [0.5, 0.3, -0.2, 0.1, -0.1]
    m = fit_model(d, current_streak=0)
    assert m["p_extend"] == 0.0
    assert forecast_cdf(m, 0.0) == pytest.approx(cdf_from_samples(m["uncond"], 0.0))


def test_taker_fee_matches_kalshi_formula():
    # fee_cents = ceil(0.07 * P * (1-P) * 100); maker pays nothing
    assert taker_fee_per_contract(0.05) == pytest.approx(0.01)  # 0.33c -> ceil 1c
    assert taker_fee_per_contract(0.50) == pytest.approx(0.02)  # 1.75c -> ceil 2c
    assert taker_fee_per_contract(0.99) == pytest.approx(0.01)


def test_taker_fee_order_rounds_once_per_order():
    assert taker_fee_order(388.19, 0.76) == pytest.approx(4.96)  # 495.7c -> ceil 496c
    assert taker_fee_order(500, 0.50) == pytest.approx(8.75)  # exact 875c, no round-up
    assert taker_fee_order(1, 0.50) == pytest.approx(0.02)  # 1.75c -> ceil 2c
    assert taker_fee_order(1, 0.05) == pytest.approx(0.01)  # 0.33c -> ceil 1c


def test_fee_aware_edges():
    book = {"yes_bid": 0.10, "yes_ask": 0.11}
    e = fee_aware_edges(fair_yes=0.40, book=book)
    assert e["buy_yes_cost"] == pytest.approx(0.12)  # 0.11 + 1c fee
    assert e["buy_yes_edge"] == pytest.approx(0.28)
    # buying NO at (1-0.10)=0.90 + 1c fee => edge = 0.60 - 0.91
    assert e["buy_no_edge"] == pytest.approx(round(0.60 - 0.91, 4))


def test_parse_strike():
    assert parse_strike("KXDIESELD-26SEP27-T6.480") == 6.48
    assert parse_strike("KXAAAGASD-26SEP27-4.5250") == 4.525


def test_convergence_excess_gap(tmp_path):
    # retail 3.00 on 2026-06-01..06-03; futures flat 1.80 -> gap 1.20 = baseline;
    # latest retail 3.20 with futures 1.80 -> excess 20c
    fut = tmp_path / "heating_oil.csv"
    fut.write_text("date,close\n" + "".join(
        f"2026-{m:02d}-{d:02d},1.80\n" for m, d in
        [(3, dd) for dd in range(20, 32)] + [(4, dd) for dd in range(1, 29)]
        + [(5, dd) for dd in range(1, 29)] + [(6, dd) for dd in range(1, 4)]))
    import aaa_futures as F
    orig = F.FUT_DIR
    F.FUT_DIR = tmp_path
    try:
        rows = _rows(
            [("2026-05-10", 3.00, 3.00)]
            + [(f"2026-04-{dd:02d}", 3.00, 3.00) for dd in range(1, 15)]
            + [("2026-06-01", 3.00, 3.00), ("2026-06-02", 3.20, 3.00)]
        )
        out = convergence(rows, "HO=F")
        assert out["excess_gap_cents"] == pytest.approx(20.0, abs=1.0)
    finally:
        F.FUT_DIR = orig
