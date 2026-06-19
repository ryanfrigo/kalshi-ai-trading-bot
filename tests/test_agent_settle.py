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
