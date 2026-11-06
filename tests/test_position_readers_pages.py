"""The risk report and both performance analyzers read every page of positions.
Fakes only, with read methods alone, so an order would raise. No .env is loaded."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts import portfolio_risk
from src.jobs.automated_performance_analyzer import AutomatedPerformanceAnalyzer
from src.jobs.performance_analyzer import TradingPerformanceAnalyzer


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
        return {"market": {"yes_bid_dollars": "0.40", "no_bid_dollars": "0.56"}}

    async def close(self):
        self.closed += 1


TWO_PAGES = {None: {"market_positions": [_row("A")], "cursor": "next"},
             "next": {"market_positions": [_row("B", "-3.00"), _row("FLAT", "0")], "cursor": ""}}
BAD_LATER = {None: {"market_positions": [_row("A")], "cursor": "next"},
             "next": ConnectionError("down")}


@pytest.fixture
def risk_client(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    def install(pages):
        fake = Fake(pages)
        monkeypatch.setattr("src.clients.kalshi_client.KalshiClient", lambda: fake)
        return fake

    return install


async def test_risk_report_counts_every_page(risk_client):
    fake = risk_client(TWO_PAGES)
    rows = await portfolio_risk.collect()
    assert [r["ticker"] for r in rows] == ["A", "B"]
    assert fake.closed == 1


@pytest.mark.parametrize("pages", [BAD_LATER, {None: {}}])
async def test_risk_report_refuses_a_partial_book(risk_client, pages):
    fake = risk_client(pages)
    with pytest.raises((ConnectionError, RuntimeError)):
        await portfolio_risk.collect()
    assert fake.closed == 1


async def test_automated_analyzer_counts_every_page():
    analyzer = AutomatedPerformanceAnalyzer()
    analyzer.kalshi_client = Fake(TWO_PAGES)
    data = await analyzer._gather_portfolio_data()
    assert data["active_positions"] == 2
    assert data["total_contracts"] == 8


async def test_automated_analyzer_raises_on_a_bad_later_page():
    analyzer = AutomatedPerformanceAnalyzer()
    analyzer.kalshi_client = Fake(BAD_LATER)
    with pytest.raises(ConnectionError):
        await analyzer._gather_portfolio_data()


async def test_performance_analyzer_portfolio_block_counts_every_page(tmp_path):
    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    analyzer = TradingPerformanceAnalyzer(db, Fake(TWO_PAGES), None)
    data = await analyzer._gather_performance_data()
    assert data["portfolio"]["active_positions"] == 2


async def test_performance_analyzer_leaves_the_block_empty_on_a_bad_later_page(tmp_path):
    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    analyzer = TradingPerformanceAnalyzer(db, Fake(BAD_LATER), None)
    data = await analyzer._gather_performance_data()
    assert data["portfolio"] == {}


async def test_performance_analyzer_skips_the_paid_call_when_the_portfolio_read_fails(tmp_path):
    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    xai = MagicMock(get_completion=AsyncMock(return_value="advice"))
    report = await TradingPerformanceAnalyzer(db, Fake(BAD_LATER), xai).run_comprehensive_analysis()
    xai.get_completion.assert_not_awaited()
    assert report["ai_insights"] == {"error": "portfolio read failed; AI analysis skipped"}
    assert report["action_items"] == []
    assert report["performance_data"]["portfolio"] == {}


async def test_performance_analyzer_still_makes_the_call_when_the_read_works(tmp_path):
    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    xai = MagicMock(get_completion=AsyncMock(return_value="advice"))
    report = await TradingPerformanceAnalyzer(db, Fake(TWO_PAGES), xai).run_comprehensive_analysis()
    xai.get_completion.assert_awaited_once()
    assert report["ai_insights"]["raw_analysis"] == "advice"


async def test_performance_analyzer_fallback_survives_an_empty_ledger(tmp_path):
    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    xai = MagicMock(get_completion=AsyncMock(side_effect=RuntimeError("no")))
    report = await TradingPerformanceAnalyzer(db, Fake(TWO_PAGES), xai).run_comprehensive_analysis()
    assert report["ai_insights"]["error"] == "no"
    assert "FALLBACK ANALYSIS" in report["ai_insights"]["fallback_analysis"]


async def test_performance_analyzer_reports_no_made_up_action_items(tmp_path):
    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    xai = MagicMock(get_completion=AsyncMock(return_value="advice"))
    report = await TradingPerformanceAnalyzer(db, Fake(TWO_PAGES), xai).run_comprehensive_analysis()
    assert report["action_items"] == []


def test_fallback_calls_a_low_win_rate_critical_only_with_trades():
    analyzer = TradingPerformanceAnalyzer(None, None, None)

    def text(**stats):
        return analyzer._generate_fallback_analysis({
            "performance": {"overall_stats": stats},
            "portfolio": {"available_cash": 100.0},
        })

    assert "Win rate" not in text(total_trades=0, win_rate=0)
    assert "Win rate is 25.0%" in text(total_trades=4, win_rate=0.25)


class NoBalance(Fake):
    async def get_balance(self):
        return {}


ONE_POSITION = {None: {"market_positions": [{"ticker": "A", "position_fp": "5.00"}], "cursor": ""}}


async def test_performance_analyzer_skips_the_paid_call_on_a_balance_reply_with_no_balance(tmp_path):
    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    xai = MagicMock(get_completion=AsyncMock(return_value="advice"))
    report = await TradingPerformanceAnalyzer(db, NoBalance(ONE_POSITION), xai).run_comprehensive_analysis()
    xai.get_completion.assert_not_awaited()
    assert report["performance_data"]["portfolio"] == {}


async def test_performance_analyzer_still_reads_a_real_zero_balance(tmp_path):
    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    fake = Fake(ONE_POSITION)
    fake.get_balance = AsyncMock(return_value={"balance": 0})
    data = await TradingPerformanceAnalyzer(db, fake, None)._gather_performance_data()
    assert data["portfolio"]["available_cash"] == 0
    assert data["portfolio"]["active_positions"] == 1
