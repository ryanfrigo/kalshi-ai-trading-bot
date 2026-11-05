"""The risk report and both performance analyzers read every page of positions.
Fakes only, with read methods alone, so an order would raise. No .env is loaded."""
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
