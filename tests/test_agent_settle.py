"""Tests for settlement_pnl — realized outcome from Kalshi's authoritative
/portfolio/settlements records. won = held side matched market_result;
pnl = revenue - cost - fees. Records where no contracts were held are skipped."""
import pytest

from src.agent.settle import settlement_pnl


def test_no_won_profit():
    rec = {"ticker": "X", "event_ticker": "E", "market_result": "no",
           "no_count_fp": "16.00", "yes_count_fp": "0.00",
           "no_total_cost_dollars": "15.20", "yes_total_cost_dollars": "0.0",
           "revenue": 1600, "fee_cost": "0.0"}
    out = settlement_pnl(rec)
    assert out["held_side"] == "no" and out["count"] == 16
    assert out["won"] is True
    assert abs(out["pnl"] - 0.80) < 1e-9  # $16.00 revenue - $15.20 cost


def test_no_lost_full_loss():
    rec = {"ticker": "X", "market_result": "yes",
           "no_count_fp": "16.00", "yes_count_fp": "0.00",
           "no_total_cost_dollars": "15.20", "yes_total_cost_dollars": "0.0",
           "revenue": 0, "fee_cost": "0.0"}
    out = settlement_pnl(rec)
    assert out["won"] is False
    assert abs(out["pnl"] - (-15.20)) < 1e-9


def test_fees_subtracted():
    rec = {"ticker": "X", "market_result": "no",
           "no_count_fp": "100.00", "yes_count_fp": "0.00",
           "no_total_cost_dollars": "95.00", "yes_total_cost_dollars": "0.0",
           "revenue": 10000, "fee_cost": "0.50"}
    out = settlement_pnl(rec)
    assert abs(out["pnl"] - (100.0 - 95.0 - 0.50)) < 1e-9


def test_no_position_skipped():
    rec = {"ticker": "X", "market_result": "yes",
           "no_count_fp": "0.00", "yes_count_fp": "0.00",
           "revenue": 0, "fee_cost": "0.0"}
    assert settlement_pnl(rec) is None


# ---------------------------------------------------------------------------
# settlement -> journal-shaped record (feeds the Edge Policy from real outcomes)
# ---------------------------------------------------------------------------
from src.agent.settle import series_category, settlement_to_record


def test_series_category_is_the_kalshi_series_prefix():
    assert series_category("KXCPI-26JUN-3.2") == "KXCPI"
    assert series_category("KXNBA-26-SAS") == "KXNBA"
    assert series_category("KXMAKEMARMAD-26-DUKE") == "KXMAKEMARMAD"
    assert series_category("") == ""
    assert series_category(None) == ""


def test_settlement_to_record_carries_side_category_and_outcome():
    s = {"ticker": "KXCPI-26JUN-3.2", "held_side": "yes", "count": 10,
         "won": False, "result": "no", "cost": 3.0, "revenue": 0.0,
         "fee": 0.0, "pnl": -3.0, "settled_time": "2026-06-20T00:00:00Z"}
    rec = settlement_to_record(s)
    assert rec["side"] == "yes"
    assert rec["category"] == "KXCPI"
    assert rec["outcome"] == {"won": False, "pnl": -3.0,
                              "settled_time": "2026-06-20T00:00:00Z"}
    assert rec["est_prob"] is None            # settlements carry no prediction
    assert rec["source"] == "settlement"      # provenance, NOT a blockable method
    assert "method" not in rec                # must not masquerade as a research method
    assert abs(rec["price"] - 0.30) < 1e-9     # cost/count = avg entry price


def test_settlement_to_record_skips_rows_without_a_held_side():
    assert settlement_to_record({"ticker": "X", "held_side": None,
                                 "result": "no", "won": False, "pnl": 0.0}) is None


def test_fractional_settlement_count_is_kept_and_prices_the_record():
    s = {"ticker": "KXCPI-26JUN-3.2", "held_side": "yes", "count": 2.5,
         "result": "yes", "cost": 1.0, "revenue": 250.0, "fee": 0.0, "pnl": 1.5}
    rec = settlement_to_record(s)
    assert rec["count"] == 2.5
    assert rec["price"] == pytest.approx(0.4)


def test_integral_settlement_count_stays_int():
    s = {"ticker": "KXCPI-26JUN-3.2", "held_side": "yes", "count": 3,
         "result": "yes", "cost": 1.2, "revenue": 300.0, "fee": 0.0, "pnl": 1.8}
    count = settlement_to_record(s)["count"]
    assert count == 3 and isinstance(count, int)


# ---------------------------------------------------------------------------
# fractional fixed-point counts from the settlements API ("0.40", "2.50")
# ---------------------------------------------------------------------------


def test_sub_contract_yes_position_is_kept():
    rec = {"ticker": "X", "market_result": "yes",
           "yes_count_fp": "0.40", "no_count_fp": "0.00",
           "yes_total_cost_dollars": "0.16", "no_total_cost_dollars": "0.0",
           "revenue": 40, "fee_cost": "0"}
    out = settlement_pnl(rec)
    assert out["held_side"] == "yes"
    assert out["count"] == pytest.approx(0.4)
    assert out["won"] is True
    assert out["pnl"] == pytest.approx(0.24)


def test_fractional_yes_count_prices_the_record():
    rec = {"ticker": "X", "market_result": "yes",
           "yes_count_fp": "2.50", "no_count_fp": "0.00",
           "yes_total_cost_dollars": "1.00", "no_total_cost_dollars": "0.0",
           "revenue": 250, "fee_cost": "0"}
    out = settlement_pnl(rec)
    assert out["count"] == pytest.approx(2.5)
    row = settlement_to_record(out)
    assert row["price"] == pytest.approx(0.40)
    assert row["count"] == pytest.approx(2.5)


def test_integral_no_count_stays_int():
    rec = {"ticker": "X", "market_result": "no",
           "yes_count_fp": "0.00", "no_count_fp": "16.00",
           "yes_total_cost_dollars": "0.0", "no_total_cost_dollars": "15.20",
           "revenue": 1600, "fee_cost": "0"}
    count = settlement_pnl(rec)["count"]
    assert count == 16 and isinstance(count, int)


def test_sub_contract_no_lost_full_loss():
    rec = {"ticker": "X", "market_result": "yes",
           "yes_count_fp": "0.00", "no_count_fp": "0.30",
           "yes_total_cost_dollars": "0.0", "no_total_cost_dollars": "0.15",
           "revenue": 0, "fee_cost": "0"}
    out = settlement_pnl(rec)
    assert out["held_side"] == "no"
    assert out["won"] is False
    assert out["count"] == pytest.approx(0.3)
    assert out["pnl"] == pytest.approx(-0.15)


# ---------------------------------------------------------------------------
# held side is the LARGER count, not merely "any yes"
# ---------------------------------------------------------------------------


def test_held_side_is_larger_no_count_when_both_sides_held():
    rec = {"ticker": "X", "market_result": "no",
           "yes_count_fp": "0.30", "no_count_fp": "2.00",
           "no_total_cost_dollars": "1.00", "yes_total_cost_dollars": "0.10",
           "revenue": 200, "fee_cost": "0"}
    out = settlement_pnl(rec)
    assert out["held_side"] == "no"
    assert out["count"] == 2
    assert out["won"] is True


def test_held_side_is_larger_yes_count_when_both_sides_held():
    rec = {"ticker": "X", "market_result": "yes",
           "yes_count_fp": "3.00", "no_count_fp": "1.00",
           "yes_total_cost_dollars": "1.50", "no_total_cost_dollars": "0.40",
           "revenue": 300, "fee_cost": "0"}
    out = settlement_pnl(rec)
    assert out["held_side"] == "yes"
    assert out["count"] == 3
    assert out["won"] is True
