"""Paper-mode P&L is not net of Kalshi fees, so every user-facing label says so.

No network, no .env, no KalshiClient: the DB is a tmp SQLite file.
"""
import pytest

import paper_trader
from src.paper import tracker
from src.paper.dashboard import generate_html


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "DB_PATH", str(tmp_path / "paper.db"))


def test_empty_dashboard_labels_pnl_as_before_fees():
    html = generate_html()
    assert "Total P&amp;L (before fees)" in html
    assert "excludes Kalshi trading fees" in html


def test_dashboard_win_rate_label_and_total_after_one_win():
    i = tracker.log_signal("MKT-A", "Title MKT-A", "YES", 0.40, 0.7, "why")
    tracker.settle_signal(i, 1.0)
    html = generate_html()
    assert "Win Rate (before fees)" in html
    assert "$0.60" in html


def test_print_stats_labels_pnl_as_before_fees(capsys):
    paper_trader.print_stats()
    out = capsys.readouterr().out
    assert "Total P&L (before fees)" in out
    assert "excludes Kalshi trading fees" in out
