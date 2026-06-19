"""Tests for the local backtest data collector (snapshots, never committed)."""
from src.data.collector import build_snapshot


def test_build_snapshot_core_fields():
    bal = {"balance": 63889, "portfolio_value": 16076}
    pos = {"market_positions": [
        {"ticker": "A", "position_fp": "-49.00", "market_exposure_dollars": "39.59",
         "total_traded_dollars": "39.59", "realized_pnl_dollars": "0"},
        {"ticker": "B", "position_fp": "0.00"},  # settled/flat -> excluded
    ]}
    snap = build_snapshot(bal, pos, tag="daily_open",
                          ts="2026-06-18T00:00:00Z", meta={"foo": "bar"})
    assert snap["tag"] == "daily_open"
    assert snap["ts"] == "2026-06-18T00:00:00Z"
    assert snap["cash_cents"] == 63889
    assert snap["portfolio_cents"] == 16076
    assert snap["equity_cents"] == 79965
    assert len(snap["positions"]) == 1  # only non-zero positions captured
    p = snap["positions"][0]
    assert p["ticker"] == "A" and p["side"] == "NO" and p["count"] == 49
    assert snap["meta"] == {"foo": "bar"}


def test_build_snapshot_yes_position_sign():
    bal = {"balance": 100, "portfolio_value": 0}
    pos = {"market_positions": [{"ticker": "Y", "position_fp": "100.00"}]}
    snap = build_snapshot(bal, pos, tag="t", ts="t", meta={})
    assert snap["positions"][0]["side"] == "YES"
    assert snap["positions"][0]["count"] == 100
