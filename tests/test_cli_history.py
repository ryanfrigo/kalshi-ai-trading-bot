"""`cli.py history` on a fresh / partially-initialised DB prints a friendly message."""
import argparse
import sqlite3

import cli


def _run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "cli.py"))
    cli.cmd_history(argparse.Namespace(limit=10))
    return capsys.readouterr().out


def test_history_empty_db_no_tables(tmp_path, monkeypatch, capsys):
    sqlite3.connect(tmp_path / "trading_system.db").close()
    out = _run(tmp_path, monkeypatch, capsys)
    assert "No trade history yet." in out
    con = sqlite3.connect(tmp_path / "trading_system.db")
    assert con.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0  # no tables created


def test_history_empty_trade_logs_without_blocked_trades(tmp_path, monkeypatch, capsys):
    con = sqlite3.connect(tmp_path / "trading_system.db")
    con.execute("CREATE TABLE trade_logs (id INTEGER)")
    con.commit(); con.close()
    out = _run(tmp_path, monkeypatch, capsys)
    assert "No trade history yet." in out
