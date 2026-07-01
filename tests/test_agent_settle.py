"""Tests for settlement_pnl — realized outcome from Kalshi's authoritative
/portfolio/settlements records. won = held side matched market_result;
pnl = revenue - cost - fees. Records where no contracts were held are skipped."""
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
    assert rec["outcome"] == {"won": False, "pnl": -3.0}
    assert rec["est_prob"] is None            # settlements carry no prediction
    assert rec["source"] == "settlement"      # provenance, NOT a blockable method
    assert "method" not in rec                # must not masquerade as a research method
    assert abs(rec["price"] - 0.30) < 1e-9     # cost/count = avg entry price


def test_settlement_to_record_skips_rows_without_a_held_side():
    assert settlement_to_record({"ticker": "X", "held_side": None,
                                 "result": "no", "won": False, "pnl": 0.0}) is None
