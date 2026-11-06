"""The five display scripts read every page of positions, and a bad page stops the
figures. Fakes only, with read methods alone, so an order would raise. No .env is loaded."""
import asyncio
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.database import DatabaseManager, TradeLog, Position

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


async def test_quick_analysis_makes_no_paid_call_with_no_capital(script, monkeypatch, capsys):
    mod, fake, xai = _quick(script, monkeypatch, {None: {"market_positions": [], "cursor": ""}})
    fake.get_balance = AsyncMock(return_value={"balance": 0})
    await _settled_trade(mod)
    await mod.run_quick_analysis()
    xai.get_completion.assert_not_awaited()
    assert "No cash and no positions" in capsys.readouterr().out
    assert fake.closed == 1


async def test_quick_analysis_still_calls_with_cash_and_no_positions(script, monkeypatch):
    mod, _, xai = _quick(script, monkeypatch, {None: {"market_positions": [], "cursor": ""}})
    await _settled_trade(mod)
    await mod.run_quick_analysis()
    xai.get_completion.assert_awaited_once()
    assert "current capital utilization appropriate" in str(xai.get_completion.await_args)


async def test_quick_analysis_still_calls_with_positions_and_no_cash(script, monkeypatch):
    mod, fake, xai = _quick(script, monkeypatch, TWO_PAGES)
    fake.get_balance = AsyncMock(return_value={"balance": 0})
    await _settled_trade(mod)
    await mod.run_quick_analysis()
    xai.get_completion.assert_awaited_once()


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


THREE = {None: {"market_positions": [_row("A"), _row("B", "-3.00"), _row("C")], "cursor": ""}}


class FakeBadMarket(Fake):
    async def get_market(self, ticker):
        if ticker == "A":
            raise ConnectionError("down")
        if ticker == "B":
            return {}
        return await super().get_market(ticker)


async def test_health_check_names_positions_it_could_not_value(script, monkeypatch):
    mod, _ = script("scripts/portfolio_health_check.py", TWO_PAGES)
    monkeypatch.setattr(mod, "KalshiClient", lambda: FakeBadMarket(THREE))
    health = await mod.get_portfolio_health()
    assert health["unvalued_tickers"] == ["A", "B"]
    assert health["position_value"] == pytest.approx(5 * 0.42)


async def test_health_check_summary_marks_partial_totals(script, monkeypatch, capsys):
    mod, _ = script("scripts/portfolio_health_check.py", TWO_PAGES)
    monkeypatch.setattr(mod, "KalshiClient", lambda: FakeBadMarket(THREE))
    health = await mod.get_portfolio_health()
    mod.print_portfolio_summary(health)
    assert "PARTIAL: could not value A, B" in capsys.readouterr().out


async def test_health_check_summary_has_no_mark_when_all_valued(script, capsys):
    mod, _ = script("scripts/portfolio_health_check.py", TWO_PAGES)
    health = await mod.get_portfolio_health()
    mod.print_portfolio_summary(health)
    assert "PARTIAL" not in capsys.readouterr().out
    assert health["unvalued_tickers"] == []


def test_dashboard_health_names_positions_it_could_not_value(dashboard, monkeypatch):
    mod, _ = dashboard(TWO_PAGES)
    monkeypatch.setattr(mod, "KalshiClient", lambda: FakeBadMarket(THREE))
    health = mod.load_system_health()
    assert health["unvalued_tickers"] == ["A", "B"]
    assert health["position_value"] == pytest.approx(5 * 0.42)


NO_TICKER = {None: {"market_positions": [_row("A"), _row("", "2.00"), _row(None, "-1.00"), _row("", "0")],
                    "cursor": ""}}


async def test_health_check_names_a_position_with_no_ticker(script):
    mod, _ = script("scripts/portfolio_health_check.py", NO_TICKER)
    health = await mod.get_portfolio_health()
    assert health["unvalued_tickers"] == ["(unnamed position)", "(unnamed position)"]
    assert health["position_value"] == pytest.approx(5 * 0.42)


def test_dashboard_health_names_a_position_with_no_ticker(dashboard):
    mod, _ = dashboard(NO_TICKER)
    health = mod.load_system_health()
    assert health["unvalued_tickers"] == ["(unnamed position)", "(unnamed position)"]
    assert health["position_value"] == pytest.approx(5 * 0.42)


def test_dashboard_warns_on_partial_totals(dashboard):
    mod, _ = dashboard(TWO_PAGES)
    mod.warn_if_partial({"unvalued_tickers": ["A", "B"]})
    mod.st.warning.assert_called_once()
    assert "A, B" in mod.st.warning.call_args.args[0]


def test_dashboard_stays_quiet_when_all_valued(dashboard):
    mod, _ = dashboard(TWO_PAGES)
    mod.warn_if_partial({"unvalued_tickers": []})
    mod.st.warning.assert_not_called()


def test_dashboard_main_calls_the_warning_with_the_health_data(dashboard, monkeypatch):
    mod, _ = dashboard(TWO_PAGES)
    mod.st.columns.return_value = (MagicMock(), MagicMock())
    mod.st.button.return_value = False
    health = {"unvalued_tickers": ["A"]}
    monkeypatch.setattr(mod, "load_performance_data", lambda: ({}, []))
    monkeypatch.setattr(mod, "load_llm_data", lambda: ([], {}))
    monkeypatch.setattr(mod, "load_system_health", lambda: health)

    class Reached(Exception):
        pass

    seen = []

    def stop_here(data):
        seen.append(data)
        raise Reached

    monkeypatch.setattr(mod, "warn_if_partial", stop_here)
    with pytest.raises(Reached):
        mod.main()
    assert seen == [health]


EMPTY = {None: {"market_positions": [], "cursor": ""}}


async def _analysis(script, pages, monkeypatch=None, client=None, balance=None, rows=(), trades=()):
    mod, fake = script("scripts/performance_analysis.py", pages)
    if client:
        monkeypatch.setattr(mod, "KalshiClient", lambda: client)
        fake = client
    if balance is not None:
        fake.get_balance = AsyncMock(return_value={"balance": balance})
    db = mod.DatabaseManager()
    await db.initialize()
    now = datetime.now()
    for market_id, side, quantity, live in rows:
        await db.add_position(Position(market_id=market_id, side=side, entry_price=0.5, quantity=quantity,
                                       timestamp=now, rationale="t", confidence=0.5, live=live,
                                       status="open", strategy="s"))
    for pnl in trades:
        await db.add_trade_log(TradeLog(market_id="OLD", side="YES", entry_price=0.5, exit_price=0.6,
                                        quantity=1, pnl=pnl, entry_timestamp=now, exit_timestamp=now,
                                        rationale="t", strategy="s"))
    await mod.comprehensive_analysis()
    return fake


async def test_performance_analysis_survives_no_capital(script, capsys):
    await _analysis(script, EMPTY, balance=0)
    assert "Capital Utilization: n/a" in capsys.readouterr().out


async def test_performance_analysis_still_shows_utilization_with_capital(script, capsys):
    await _analysis(script, EMPTY)
    assert "Capital Utilization: 0.0%" in capsys.readouterr().out


async def test_performance_analysis_names_what_it_could_not_value(script, monkeypatch, capsys):
    await _analysis(script, TWO_PAGES, monkeypatch, client=FakeBadMarket(THREE))
    out = capsys.readouterr().out
    assert "PARTIAL: could not value A, B" in out
    assert "Position Value: $2.10" in out
    assert "Total Portfolio: $102.10" in out


async def test_performance_analysis_names_a_position_with_no_ticker(script, capsys):
    await _analysis(script, NO_TICKER)
    out = capsys.readouterr().out
    assert "PARTIAL: could not value (unnamed position), (unnamed position)" in out
    assert "Position Value: $2.10" in out


async def test_performance_analysis_has_no_partial_mark_when_all_valued(script, capsys):
    await _analysis(script, TWO_PAGES)
    assert "PARTIAL" not in capsys.readouterr().out


async def test_performance_analysis_flags_a_live_row_with_no_kalshi_position(script, capsys):
    await _analysis(script, EMPTY, rows=[("GONE", "YES", 5, True)])
    out = capsys.readouterr().out
    assert "GONE: DB=5, Kalshi=no position" in out
    assert "Found 1 position discrepancies" in out
    assert "All database positions match" not in out


async def test_performance_analysis_leaves_out_a_paper_row_with_no_kalshi_position(script, capsys):
    await _analysis(script, EMPTY, rows=[("PAPER", "YES", 5, False)])
    assert "All database positions match" in capsys.readouterr().out


async def test_performance_analysis_passes_a_row_that_matches(script, capsys):
    await _analysis(script, TWO_PAGES, rows=[("A", "YES", 5, True), ("B", "NO", 3, True)])
    assert "All database positions match" in capsys.readouterr().out


@pytest.mark.parametrize("pages", [EMPTY, TWO_PAGES])
@pytest.mark.parametrize("pnl", [0.1, -0.1])
async def test_performance_analysis_prints_no_verdicts(script, capsys, pages, pnl):
    await _analysis(script, pages, trades=[pnl])
    out = capsys.readouterr().out
    for verdict in ("conservative", "Conservative", "diversification", "Suggestion",
                    "performing well", "consider adjusting"):
        assert verdict not in out


async def test_performance_analysis_prints_the_win_rate_with_its_count(script, capsys):
    await _analysis(script, EMPTY, trades=[0.1, -0.1, 0.2, -0.2])
    assert "Historical Win Rate: 50.0% (settled trades: 4)" in capsys.readouterr().out


async def test_performance_analysis_says_when_nothing_has_settled(script, capsys):
    await _analysis(script, EMPTY)
    assert "No settled trades yet" in capsys.readouterr().out


class NoBalance(Fake):
    async def get_balance(self):
        return {}


async def test_health_check_raises_on_a_balance_reply_with_no_balance(script, monkeypatch):
    mod, _ = script("scripts/portfolio_health_check.py", EMPTY)
    client = NoBalance(EMPTY)
    monkeypatch.setattr(mod, "KalshiClient", lambda: client)
    with pytest.raises(KeyError):
        await mod.get_portfolio_health()
    assert client.closed == 1


async def test_health_check_still_reads_a_real_zero_balance(script):
    mod, fake = script("scripts/portfolio_health_check.py", EMPTY)
    fake.get_balance = AsyncMock(return_value={"balance": 0})
    health = await mod.get_portfolio_health()
    assert health["available_cash"] == 0


async def test_performance_analysis_raises_on_a_balance_reply_with_no_balance(script, monkeypatch, capsys):
    with pytest.raises(KeyError):
        await _analysis(script, EMPTY, monkeypatch, client=NoBalance(EMPTY))
    assert "Available Cash" not in capsys.readouterr().out


async def test_quick_analysis_makes_no_paid_call_on_a_balance_reply_with_no_balance(script, monkeypatch):
    mod, fake, xai = _quick(script, monkeypatch, EMPTY)
    fake.get_balance = AsyncMock(return_value={})
    await _settled_trade(mod)
    with pytest.raises(KeyError):
        await mod.run_quick_analysis()
    xai.get_completion.assert_not_awaited()
    assert fake.closed == 1


def test_dashboard_health_shows_an_error_on_a_balance_reply_with_no_balance(dashboard, monkeypatch):
    mod, _ = dashboard(EMPTY)
    monkeypatch.setattr(mod, "KalshiClient", lambda: NoBalance(EMPTY))
    mod.load_system_health()
    mod.st.error.assert_called_once()


def test_dashboard_health_still_reads_a_real_zero_balance(dashboard):
    mod, fake = dashboard(EMPTY)
    fake.get_balance = AsyncMock(return_value={"balance": 0})
    assert mod.load_system_health()["available_cash"] == 0
    mod.st.error.assert_not_called()
