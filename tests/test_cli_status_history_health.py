"""`cli.py status`, `history` and `health` on fakes. No real client is built.

These tests don't read a .env themselves. Importing settings may, as for the whole suite.
"""
import argparse
import sqlite3
from types import SimpleNamespace

import pytest

import cli
import src.clients.kalshi_client as kalshi_module

NS = argparse.Namespace


def _fake_client(monkeypatch, balance=None, positions=None, balance_error=None):
    """Patch KalshiClient with a fake that has read methods only, so an order would raise."""
    state = SimpleNamespace(closed=0)

    class Fake:
        async def get_balance(self):
            if balance_error:
                raise balance_error
            return balance if balance is not None else {"balance": 12_345, "portfolio_value": 5_000}

        async def get_positions(self, **_):
            return positions if positions is not None else {"event_positions": []}

        async def close(self):
            state.closed += 1

    monkeypatch.setattr(kalshi_module, "KalshiClient", Fake)
    return state


# --- status ---------------------------------------------------------------------------

def test_status_shows_cash_positions_and_total(monkeypatch, capsys):
    state = _fake_client(monkeypatch)
    cli.cmd_status(NS())
    out = capsys.readouterr().out
    assert "$    123.45" in out
    assert "$     50.00" in out
    assert "$    173.45" in out
    assert "Total Exposure" not in out
    assert state.closed == 1


def test_status_lists_only_positions_with_exposure_and_totals_them(monkeypatch, capsys):
    _fake_client(monkeypatch, positions={"event_positions": [
        {"event_ticker": "EV-A", "event_exposure_dollars": "10.00", "total_cost_dollars": "9.00",
         "realized_pnl_dollars": "1.50", "fees_paid_dollars": "0.25"},
        {"event_ticker": "EV-B", "event_exposure_dollars": "5.00", "total_cost_dollars": "4.00",
         "realized_pnl_dollars": "-0.50", "fees_paid_dollars": "0.10"},
        {"event_ticker": "EV-FLAT", "event_exposure_dollars": "0"},
    ]})
    cli.cmd_status(NS())
    out = capsys.readouterr().out
    assert "EV-A" in out and "EV-B" in out
    assert "EV-FLAT" not in out
    assert "Total Exposure:     $     15.00" in out
    assert "Total Realized P&L: $      1.00" in out
    assert "Total Fees Paid:    $      0.35" in out


def test_status_failure_exits_1_and_closes_the_client(monkeypatch, capsys):
    state = _fake_client(monkeypatch, balance_error=ConnectionError("down"))
    with pytest.raises(SystemExit) as e:
        cli.cmd_status(NS())
    assert e.value.code == 1
    assert "Error fetching status: down" in capsys.readouterr().out
    assert state.closed == 1


def _paged_client(monkeypatch, pages):
    """Fake whose get_positions serves `pages` by cursor; an Exception value is raised."""
    calls = []

    class Fake:
        async def get_balance(self):
            return {"balance": 0, "portfolio_value": 0}

        async def get_positions(self, cursor=None, **_):
            calls.append(cursor)
            reply = pages[cursor]
            if isinstance(reply, Exception):
                raise reply
            return reply

        async def close(self):
            pass

    monkeypatch.setattr(kalshi_module, "KalshiClient", Fake)
    return calls


def _row(ticker, exposure="1.00"):
    return {"event_ticker": ticker, "event_exposure_dollars": exposure}


def test_status_follows_the_positions_cursor_and_totals_every_page(monkeypatch, capsys):
    calls = _paged_client(monkeypatch, {
        None: {"event_positions": [_row("P1")], "cursor": "next"},
        "next": {"event_positions": [_row("P2", "2.50")], "cursor": ""},
    })
    cli.cmd_status(NS())
    out = capsys.readouterr().out
    assert calls == [None, "next"]
    assert "P1" in out and "P2" in out
    assert "Active Positions:            2" in out
    assert "Total Exposure:     $      3.50" in out


def test_status_shows_nothing_when_a_later_page_fails(monkeypatch, capsys):
    _paged_client(monkeypatch, {
        None: {"event_positions": [_row("P1")], "cursor": "next"},
        "next": ConnectionError("down"),
    })
    with pytest.raises(SystemExit) as e:
        cli.cmd_status(NS())
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "Error fetching status: down" in out
    assert "Active Positions" not in out  # a partial count would read as the whole book


@pytest.mark.parametrize("reply", [{}, {"event_positions": "x"}, [], None])
def test_status_refuses_an_unreadable_positions_reply(monkeypatch, capsys, reply):
    _paged_client(monkeypatch, {None: reply})
    with pytest.raises(SystemExit) as e:
        cli.cmd_status(NS())
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "malformed event_positions page" in out
    assert "Active Positions" not in out


def test_status_treats_a_null_list_as_no_positions(monkeypatch, capsys):
    _paged_client(monkeypatch, {None: {"event_positions": None}})
    cli.cmd_status(NS())
    assert "Active Positions:            0" in capsys.readouterr().out


# --- history --------------------------------------------------------------------------

TRADE_LOGS = """CREATE TABLE trade_logs (id INTEGER PRIMARY KEY, market_id TEXT, side TEXT,
    entry_price REAL, exit_price REAL, quantity INTEGER, pnl REAL, entry_timestamp TEXT,
    exit_timestamp TEXT, rationale TEXT, strategy TEXT)"""


def _db(tmp_path, monkeypatch, trades=(), blocked=None):
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "cli.py"))
    con = sqlite3.connect(tmp_path / "trading_system.db")
    con.execute(TRADE_LOGS)
    for i, (market, pnl, strategy) in enumerate(trades):
        con.execute(
            "INSERT INTO trade_logs (market_id, side, entry_price, exit_price, quantity, pnl,"
            " entry_timestamp, strategy) VALUES (?,?,?,?,?,?,?,?)",
            (market, "YES", 0.40, 0.50, 10, pnl, f"2026-01-{i + 1:02d}T00:00:00", strategy),
        )
    if blocked is not None:
        con.execute("CREATE TABLE blocked_trades (id INTEGER)")
        for _ in range(blocked):
            con.execute("INSERT INTO blocked_trades (id) VALUES (NULL)")
    con.commit()
    con.close()


def _history(limit=50):
    cli.cmd_history(NS(limit=limit))


def test_history_without_a_database_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "cli.py"))
    _history()
    assert "No trading database found." in capsys.readouterr().out
    assert not (tmp_path / "trading_system.db").exists()


def test_history_shows_trades_when_blocked_trades_was_never_created(tmp_path, monkeypatch, capsys):
    # Nothing in the repo builds PortfolioEnforcer, the only creator of blocked_trades.
    _db(tmp_path, monkeypatch, trades=[("M-A", 2.0, "edge"), ("M-B", -1.0, "edge")])
    _history()
    out = capsys.readouterr().out
    assert "No trade history yet." not in out
    assert "Total Trades:  2" in out
    assert "M-A" in out and "M-B" in out
    assert "blocked" not in out


def test_history_with_an_empty_ledger_says_so(tmp_path, monkeypatch, capsys):
    _db(tmp_path, monkeypatch)
    _history()
    out = capsys.readouterr().out
    assert "No trade history yet." in out
    assert "Total Trades" not in out


def test_history_overview_and_category_breakdown(tmp_path, monkeypatch, capsys):
    _db(tmp_path, monkeypatch, trades=[("M-A", 3.0, "edge"), ("M-B", -1.0, "edge"), ("M-C", 1.0, None)])
    _history()
    out = capsys.readouterr().out
    assert "Win Rate:      66.7%" in out
    assert "Total P&L:     $3.00" in out
    assert "Avg per trade: $1.00" in out
    assert out.index("edge") < out.index("unknown")  # ordered by P&L, a null strategy reads unknown
    assert "50%" in out


def test_history_limit_keeps_the_newest(tmp_path, monkeypatch, capsys):
    _db(tmp_path, monkeypatch, trades=[("OLD", 1.0, "s"), ("NEW", 1.0, "s")])
    _history(limit=1)
    out = capsys.readouterr().out
    assert "NEW" in out
    assert "OLD" not in out
    assert "Total Trades:  2" in out


def test_history_counts_blocked_trades_when_the_table_exists(tmp_path, monkeypatch, capsys):
    _db(tmp_path, monkeypatch, trades=[("M", 1.0, "s")], blocked=3)
    _history()
    out = capsys.readouterr().out
    assert "3 trades blocked by portfolio enforcer" in out
    assert "cli.py health" not in out  # health has no blocked-trade details


def test_history_read_failure_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "cli.py"))
    (tmp_path / "trading_system.db").write_text("this is not a database, not even close")
    with pytest.raises(SystemExit) as e:
        _history()
    assert e.value.code == 1
    assert "Error fetching history" in capsys.readouterr().out


# --- health ---------------------------------------------------------------------------

@pytest.fixture
def health_env(tmp_path, monkeypatch):
    """Point health at tmp_path and stub the load_dotenv call inside cmd_health."""
    import dotenv

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "cli.py"))
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("KALSHI_API_KEY", "a-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "a-test-key")
    (tmp_path / ".env").write_text("")
    _fake_client(monkeypatch)
    return tmp_path


def test_health_all_pass_exits_0(health_env, capsys):
    cli.cmd_health(NS())
    out = capsys.readouterr().out
    assert "[FAIL]" not in out
    assert "balance=$123.45" in out
    assert "6/6 checks passed" in out
    assert "All systems operational." in out


def test_health_missing_env_file_fails(health_env, capsys):
    (health_env / ".env").unlink()
    with pytest.raises(SystemExit) as e:
        cli.cmd_health(NS())
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL] .env file missing" in out
    assert "5/6 checks passed" in out
    assert "1 issue(s) need attention" in out


@pytest.mark.parametrize("var,placeholder", [
    ("KALSHI_API_KEY", "your_kalshi_api_key_here"),
    ("OPENROUTER_API_KEY", "your_openrouter_api_key_here"),
])
def test_health_rejects_the_template_placeholder_and_an_empty_key(health_env, monkeypatch, capsys, var, placeholder):
    for value in (placeholder, ""):
        monkeypatch.setenv(var, value)
        with pytest.raises(SystemExit):
            cli.cmd_health(NS())
        assert f"[FAIL] {var} is missing or placeholder" in capsys.readouterr().out


@pytest.mark.parametrize("var", ["KALSHI_API_KEY", "OPENROUTER_API_KEY"])
def test_health_placeholders_match_the_template(var):
    # If env.template's placeholder drifts, a copied-but-unfilled .env would pass health.
    template = (cli.Path(__file__).resolve().parents[1] / "env.template").read_text()
    source = (cli.Path(__file__).resolve().parents[1] / "cli.py").read_text()
    value = next(line.split("=", 1)[1] for line in template.splitlines() if line.startswith(f"{var}="))
    assert f'("{var}", "{value}")' in source


def test_health_api_key_is_never_printed(health_env, monkeypatch, capsys):
    monkeypatch.setenv("KALSHI_API_KEY", "super-secret-value")
    cli.cmd_health(NS())
    assert "super-secret-value" not in capsys.readouterr().out


def test_health_401_explains_the_key_pair(health_env, monkeypatch, capsys):
    state = _fake_client(monkeypatch, balance_error=RuntimeError("HTTP 401 Unauthorized"))
    with pytest.raises(SystemExit):
        cli.cmd_health(NS())
    out = capsys.readouterr().out
    assert "[FAIL] Kalshi API connection -- HTTP 401 Unauthorized" in out
    assert "almost always means" in out
    assert state.closed == 1


def test_health_other_api_errors_skip_the_key_advice(health_env, monkeypatch, capsys):
    _fake_client(monkeypatch, balance_error=ConnectionError("dns failure"))
    with pytest.raises(SystemExit):
        cli.cmd_health(NS())
    out = capsys.readouterr().out
    assert "[FAIL] Kalshi API connection -- dns failure" in out
    assert "almost always means" not in out


def test_health_client_that_cannot_be_built_is_a_failure(health_env, monkeypatch, capsys):
    def boom():
        raise ValueError("no private key")

    monkeypatch.setattr(kalshi_module, "KalshiClient", boom)
    with pytest.raises(SystemExit):
        cli.cmd_health(NS())
    assert "[FAIL] Kalshi API connection -- no private key" in capsys.readouterr().out


def test_health_database_failure_is_reported(health_env, monkeypatch, capsys):
    import src.utils.database as database

    class Broken:
        def __init__(self):
            raise OSError("disk full")

    monkeypatch.setattr(database, "DatabaseManager", Broken)
    with pytest.raises(SystemExit):
        cli.cmd_health(NS())
    assert "[FAIL] Database initialization -- disk full" in capsys.readouterr().out


def test_health_names_the_database_it_really_opened(health_env, tmp_path, monkeypatch, capsys):
    # DatabaseManager's default path is relative to the working directory, not to cli.py.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cli.cmd_health(NS())
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "Database initialization" in ln)
    assert str(tmp_path / "trading_system.db") not in line
    assert (elsewhere / "trading_system.db").exists()


def test_health_old_python_fails(health_env, monkeypatch, capsys):
    class Old(tuple):
        major, minor, micro = 3, 11, 0

    monkeypatch.setattr(cli.sys, "version_info", Old((3, 11, 0)))
    with pytest.raises(SystemExit):
        cli.cmd_health(NS())
    assert "[FAIL] Python version -- requires >=3.12" in capsys.readouterr().out
