"""AI spend cap fails closed. Fully offline: no network, no LLM calls."""
import asyncio
import pickle
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.clients.llm_types import DailyUsageTracker
from src.clients.openrouter_client import OpenRouterClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.jobs.decide import make_decision_for_market

TODAY = datetime.now().strftime("%Y-%m-%d")
YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def _make(cls, tmp_path, **kw):
    c = cls(api_key="test-key", **kw) if cls is OpenRouterClient else cls(api_key="test-key", **kw)
    c.usage_file = str(tmp_path / "usage.pkl")
    return c


@pytest.fixture(params=[OpenRouterClient, XAIClient])
def client(request, tmp_path):
    return _make(request.param, tmp_path)


def test_missing_file_is_fresh_day(client):
    assert not client._load_daily_tracker().is_exhausted


def test_corrupt_file_fails_closed(client):
    with open(client.usage_file, "wb") as fh:
        fh.write(b"\x00not a pickle")
    t = client._load_daily_tracker()
    assert t.is_exhausted and t.date == TODAY
    assert t.daily_limit == getattr(settings.trading, "daily_ai_cost_limit", t.daily_limit)


def test_wrong_type_fails_closed(client):
    with open(client.usage_file, "wb") as fh:
        pickle.dump({"date": TODAY}, fh)
    assert client._load_daily_tracker().is_exhausted


def test_old_date_file_is_fresh(client):
    with open(client.usage_file, "wb") as fh:
        pickle.dump(DailyUsageTracker(date=YESTERDAY, total_cost=99.0, is_exhausted=True), fh)
    t = client._load_daily_tracker()
    assert not t.is_exhausted and t.total_cost == 0.0 and t.date == TODAY


def test_same_date_file_unchanged(client):
    with open(client.usage_file, "wb") as fh:
        pickle.dump(DailyUsageTracker(date=TODAY, total_cost=1.25, request_count=3), fh)
    t = client._load_daily_tracker()
    assert t.total_cost == 1.25 and t.request_count == 3 and not t.is_exhausted


@pytest.mark.asyncio
async def test_exhausted_blocks_calls(client):
    with open(client.usage_file, "wb") as fh:
        fh.write(b"garbage")
    assert await client._check_daily_limits() is False


@pytest.mark.asyncio
async def test_update_daily_cost_records_to_db(tmp_path):
    db = MagicMock()
    db.upsert_daily_cost = AsyncMock()
    c = _make(OpenRouterClient, tmp_path, db_manager=db)
    c._update_daily_cost(0.5)
    await asyncio.gather(*list(c._db_cost_tasks))
    db.upsert_daily_cost.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_db_failure_does_not_skip_pickle_update(tmp_path):
    db = MagicMock()
    db.upsert_daily_cost = AsyncMock(side_effect=RuntimeError("db down"))
    c = _make(OpenRouterClient, tmp_path, db_manager=db)
    c._update_daily_cost(0.5)
    await asyncio.gather(*list(c._db_cost_tasks))
    assert c.daily_tracker.total_cost == 0.5
    with open(c.usage_file, "rb") as fh:
        assert pickle.load(fh).total_cost == 0.5


@pytest.mark.asyncio
async def test_decide_blocks_when_db_budget_exceeded():
    db = MagicMock()
    db.get_daily_ai_cost = AsyncMock(return_value=settings.trading.daily_ai_budget)
    xai = MagicMock()
    xai.get_completion = AsyncMock()
    xai.get_trading_decision = AsyncMock()
    kalshi = MagicMock()
    kalshi.get_balance = AsyncMock()
    market = MagicMock(market_id="M", title="t")

    assert await make_decision_for_market(market, db, xai, kalshi) is None
    xai.get_completion.assert_not_called()
    xai.get_trading_decision.assert_not_called()
    kalshi.get_balance.assert_not_called()


@pytest.mark.parametrize("cls", [OpenRouterClient, XAIClient])
@pytest.mark.asyncio
async def test_corrupt_refuses_across_instances_until_file_removed(cls, tmp_path):
    first = _make(cls, tmp_path)
    with open(first.usage_file, "wb") as fh:
        fh.write(b"corrupt")
    assert await first._check_daily_limits() is False  # call 1 refuses

    second = _make(cls, tmp_path)  # fresh instance, same file
    assert await second._check_daily_limits() is False  # still refuses
    with open(second.usage_file, "rb") as fh:
        persisted = pickle.load(fh)
    assert persisted.is_exhausted and persisted.date == TODAY

    import os
    os.remove(second.usage_file)
    third = _make(cls, tmp_path)
    assert await third._check_daily_limits() is True  # missing = fresh day


def test_update_daily_cost_without_event_loop_logs_and_keeps_pickle(tmp_path):
    db = MagicMock()
    db.upsert_daily_cost = AsyncMock()
    c = _make(OpenRouterClient, tmp_path, db_manager=db)
    c.logger = MagicMock()

    c._update_daily_cost(0.75)  # sync caller, no running loop

    assert c.daily_tracker.total_cost == 0.75
    with open(c.usage_file, "rb") as fh:
        assert pickle.load(fh).total_cost == 0.75
    c.logger.error.assert_called_once()
    assert "DB cost update" in c.logger.error.call_args[0][0]
    db.upsert_daily_cost.assert_not_called()


# --- The cap binds on recorded spend, not only on the stored flag ---


def _store(client, **fields):
    defaults = dict(date=TODAY, daily_limit=50.0, total_cost=0.0)
    with open(client.usage_file, "wb") as fh:
        pickle.dump(DailyUsageTracker(**{**defaults, **fields}), fh)


def _stored(client):
    with open(client.usage_file, "rb") as fh:
        return pickle.load(fh)


@pytest.mark.asyncio
async def test_limit_lowered_below_todays_spend_refuses_and_persists(client, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 20.0)
    _store(client, total_cost=30.0)  # flag unset: $30 was under the old $50 limit
    assert await client._check_daily_limits() is False
    assert client.daily_tracker.last_exhausted_time is not None
    saved = _stored(client)  # src/jobs/evaluate.py reads this file for trading_paused
    assert saved.is_exhausted is True
    assert saved.total_cost == 30.0
    assert saved.daily_limit == 20.0


@pytest.mark.asyncio
async def test_spend_at_cap_with_the_flag_unset_refuses(client, monkeypatch):
    """A crash between the two saves in _update_daily_cost leaves exactly this on disk."""
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 20.0)
    _store(client, daily_limit=20.0, total_cost=20.0)
    assert await client._check_daily_limits() is False


@pytest.mark.asyncio
async def test_spend_just_under_the_cap_is_allowed_and_not_flagged(client, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 20.0)
    _store(client, daily_limit=20.0, total_cost=19.99)
    assert await client._check_daily_limits() is True
    assert _stored(client).is_exhausted is False


@pytest.mark.asyncio
async def test_limit_raised_above_spend_clears_the_flag(client, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 50.0)
    _store(client, daily_limit=20.0, total_cost=20.0, is_exhausted=True)
    assert await client._check_daily_limits() is True
    assert client.daily_tracker.daily_limit == 50.0
    assert client.daily_tracker.is_exhausted is False


@pytest.mark.asyncio
async def test_limit_raised_to_exactly_spend_stays_exhausted(client, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 20.0)
    _store(client, daily_limit=10.0, total_cost=20.0, is_exhausted=True)
    assert await client._check_daily_limits() is False


@pytest.mark.asyncio
async def test_zero_limit_refuses_even_with_no_budget_file(client, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 0.0)
    assert await client._check_daily_limits() is False


@pytest.mark.asyncio
async def test_new_day_clears_yesterdays_spend_under_a_lower_limit(client, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 20.0)
    _store(client, date=YESTERDAY, total_cost=30.0, is_exhausted=True)
    assert await client._check_daily_limits() is True
    assert client.daily_tracker.total_cost == 0.0


@pytest.mark.asyncio
async def test_unwritable_budget_file_still_refuses(client, tmp_path):
    """Load and the fail-closed save both fail on a directory; the refusal must survive that."""
    client.usage_file = str(tmp_path)
    assert await client._check_daily_limits() is False


def test_unwritable_budget_file_does_not_stop_spend_being_counted(client, tmp_path):
    client.usage_file = str(tmp_path)
    before = client.daily_tracker.total_cost
    client._update_daily_cost(0.5)
    assert client.daily_tracker.total_cost == before + 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("spent", [0.0, 99.0])
async def test_non_finite_limit_refuses_and_persists(client, monkeypatch, limit, spent):
    """float() accepts these from DAILY_AI_COST_LIMIT; a NaN limit never tripped the cap at all."""
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", limit)
    _store(client, daily_limit=limit, total_cost=spent)
    assert await client._check_daily_limits() is False
    assert _stored(client).is_exhausted is True


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [float("nan"), float("inf")])
async def test_non_finite_limit_refuses_with_no_budget_file(client, monkeypatch, limit):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", limit)
    assert await client._check_daily_limits() is False


@pytest.mark.asyncio
async def test_non_finite_limit_refuses_a_stale_day_file(client, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", float("nan"))
    _store(client, date=YESTERDAY, total_cost=5.0)
    assert await client._check_daily_limits() is False


@pytest.mark.asyncio
async def test_infinite_limit_does_not_reopen_a_spent_day(client, monkeypatch):
    """_load_daily_tracker clears the flag because 30 < inf; the guard has to set it again."""
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", float("inf"))
    _store(client, daily_limit=20.0, total_cost=30.0, is_exhausted=True)
    assert await client._check_daily_limits() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("spent", [float("nan"), float("inf")])
async def test_non_finite_spend_refuses(client, monkeypatch, spent):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 20.0)
    _store(client, daily_limit=20.0, total_cost=spent)
    assert await client._check_daily_limits() is False
    assert _stored(client).is_exhausted is True


@pytest.mark.asyncio
async def test_fixing_a_bad_limit_reopens_the_same_day(client, monkeypatch):
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", float("nan"))
    _store(client, daily_limit=float("nan"), total_cost=5.0)
    assert await client._check_daily_limits() is False
    monkeypatch.setattr(settings.trading, "daily_ai_cost_limit", 20.0)
    assert await client._check_daily_limits() is True
    assert client.daily_tracker.total_cost == 5.0
