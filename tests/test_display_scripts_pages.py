"""The five display scripts read every page of positions, and a bad page stops the
figures. Fakes only, with read methods alone, so an order would raise. No .env is loaded."""
import asyncio
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.database import DatabaseManager, TradeLog

ROOT = Path(__file__).resolve().parent.parent


def _row(ticker, fp="5.00"):
    return {"ticker": ticker, "position_fp": fp, "market_exposure_dollars": "2.00"}


class Fake:
    def __init__(self, pages):
        self.pages = pages
        self.closed = 0

    async def get_positions(self, cursor=None, **_):
        reply = self.pages[cursor]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def get_balance(self):
        return {"balance": 10_000}

    async def get_market(self, ticker):
        return {"market": {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.44",
                           "no_bid_dollars": "0.56", "no_ask_dollars": "0.60", "title": ticker}}

    async def close(self):
        self.closed += 1


TWO_PAGES = {None: {"market_positions": [_row("A")], "cursor": "next"},
             "next": {"market_positions": [_row("B", "-3.00"), _row("FLAT", "0")], "cursor": ""}}
BAD_LATER = {None: {"market_positions": [_row("A")], "cursor": "next"},
             "next": ConnectionError("down")}


def _load(relpath, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script(tmp_path, monkeypatch):
    """Load a script with a fake Kalshi client and a temp database."""
    monkeypatch.chdir(tmp_path)

    def install(relpath, pages):
        mod = _load(relpath, "display_script_under_test")
        fake = Fake(pages)
        monkeypatch.setattr(mod, "KalshiClient", lambda: fake)
        monkeypatch.setattr(mod, "DatabaseManager",
                            lambda: DatabaseManager(db_path=str(tmp_path / "t.db")))
        return mod, fake

    return install


@pytest.fixture
def dashboard(script, monkeypatch):
    """Streamlit and plotly are stand-ins: the charts aren't under test, and neither
    package is in the dev install."""
    for name in ("streamlit", "plotly", "plotly.express", "plotly.graph_objects", "plotly.subplots"):
        monkeypatch.setitem(sys.modules, name, MagicMock(name=name))
    sys.modules["streamlit"].cache_data = lambda **_: lambda fn: fn
    yield lambda pages: script("scripts/trading_dashboard.py", pages)
    asyncio.set_event_loop(None)


async def test_get_positions_counts_every_page(script, capsys):
    mod, fake = script("scripts/get_positions.py", TWO_PAGES)
    await mod.get_positions()
    out = capsys.readouterr().out
    assert "Total positions: 2" in out
    assert "A: 5.0 contracts" in out and "B: -3.0 contracts" in out
    assert "FLAT" not in out
    assert fake.closed == 1


async def test_get_positions_raises_on_a_bad_later_page(script, capsys):
    mod, fake = script("scripts/get_positions.py", BAD_LATER)
    with pytest.raises(ConnectionError):
        await mod.get_positions()
    assert "Total positions" not in capsys.readouterr().out
    assert fake.closed == 1


async def test_performance_analysis_counts_every_page(script, capsys):
    mod, fake = script("scripts/performance_analysis.py", TWO_PAGES)
    await mod.comprehensive_analysis()
    out = capsys.readouterr().out
    assert "Active Kalshi Positions: 2 markets" in out
    assert "Total Contracts Held: 8" in out
    assert fake.closed == 1


async def test_performance_analysis_raises_on_a_bad_later_page(script, capsys):
    mod, fake = script("scripts/performance_analysis.py", BAD_LATER)
    with pytest.raises(ConnectionError):
        await mod.comprehensive_analysis()
    assert "Active Kalshi Positions" not in capsys.readouterr().out
    assert fake.closed == 1


async def test_health_check_values_every_page(script):
    mod, fake = script("scripts/portfolio_health_check.py", TWO_PAGES)
    health = await mod.get_portfolio_health()
    assert [p["ticker"] for p in health["position_details"]] == ["A", "B"]
    assert health["positions_count"] == 2
    assert health["position_value"] == pytest.approx(5 * 0.42 + 3 * 0.58)
    assert fake.closed == 1


async def test_health_check_raises_on_a_bad_later_page(script):
    mod, fake = script("scripts/portfolio_health_check.py", BAD_LATER)
    with pytest.raises(ConnectionError):
        await mod.get_portfolio_health()
    assert fake.closed == 1


def _quick(script, monkeypatch, pages):
    mod, fake = script("scripts/quick_performance_analysis.py", pages)
    xai = MagicMock(get_completion=AsyncMock(return_value="advice"), close=AsyncMock())
    monkeypatch.setattr(mod, "XAIClient", lambda **_: xai)
    return mod, fake, xai


async def _settled_trade(mod):
    db = mod.DatabaseManager()
    await db.initialize()
    now = datetime.now()
    await db.add_trade_log(TradeLog(market_id="OLD", side="YES", entry_price=0.5, exit_price=0.6,
                                    quantity=1, pnl=0.1, entry_timestamp=now, exit_timestamp=now,
                                    rationale="t", strategy="s"))


async def test_quick_analysis_prompt_counts_every_page(script, monkeypatch):
    mod, fake, xai = _quick(script, monkeypatch, TWO_PAGES)
    await _settled_trade(mod)
    await mod.run_quick_analysis()
    prompt = str(xai.get_completion.await_args)
    assert "Active Positions: 2 markets" in prompt
    assert "Total Contracts: 8" in prompt
    assert fake.closed == 1


async def test_quick_analysis_makes_no_paid_call_on_a_bad_later_page(script, monkeypatch):
    mod, fake, xai = _quick(script, monkeypatch, BAD_LATER)
    with pytest.raises(ConnectionError):
        await mod.run_quick_analysis()
    xai.get_completion.assert_not_awaited()
    assert fake.closed == 1


async def test_quick_analysis_makes_no_paid_call_on_an_empty_ledger(script, monkeypatch, capsys):
    mod, fake, xai = _quick(script, monkeypatch, TWO_PAGES)
    await mod.run_quick_analysis()
    xai.get_completion.assert_not_awaited()
    assert "No completed trades" in capsys.readouterr().out
    assert fake.closed == 1
    xai.close.assert_awaited_once()


async def test_quick_analysis_prompt_states_no_fixed_facts(script, monkeypatch):
    mod, _, xai = _quick(script, monkeypatch, TWO_PAGES)
    await _settled_trade(mod)
    await mod.run_quick_analysis()
    prompt = str(xai.get_completion.await_args)
    for fixed in ("91.4", "16 active", "manual trades", "unrealized"):
        assert fixed not in prompt


async def test_quick_analysis_fallback_prints_no_fixed_recommendations(script, monkeypatch, capsys):
    mod, _, xai = _quick(script, monkeypatch, TWO_PAGES)
    xai.get_completion = AsyncMock(side_effect=RuntimeError("no"))
    await _settled_trade(mod)
    await mod.run_quick_analysis()
    out = capsys.readouterr().out
    assert "FALLBACK ANALYSIS" in out
    assert "RECOMMENDATIONS" not in out
    assert "live trading" not in out


def test_dashboard_positions_tab_lists_every_page(dashboard):
    mod, _ = dashboard(TWO_PAGES)
    _, positions = mod.load_performance_data()
    assert [(p["market_id"], p["side"], p["quantity"]) for p in positions] == [("A", "YES", 5), ("B", "NO", 3)]
    mod.st.error.assert_not_called()


def test_dashboard_positions_tab_shows_an_error_on_a_bad_later_page(dashboard):
    mod, _ = dashboard(BAD_LATER)
    assert mod.load_performance_data() == ({}, [])
    mod.st.error.assert_called_once()


def test_dashboard_health_counts_and_values_every_page(dashboard):
    mod, _ = dashboard(TWO_PAGES)
    health = mod.load_system_health()
    assert health["positions_count"] == 2
    assert health["position_value"] == pytest.approx(5 * 0.42 + 3 * 0.58)
    assert health["total_portfolio_value"] == pytest.approx(100 + 5 * 0.42 + 3 * 0.58)
    mod.st.error.assert_not_called()


def test_dashboard_health_shows_an_error_on_a_bad_later_page(dashboard):
    mod, _ = dashboard(BAD_LATER)
    assert mod.load_system_health()["positions_count"] == 0
    mod.st.error.assert_called_once()
