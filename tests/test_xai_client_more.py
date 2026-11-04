"""XAIClient (the shim behind every strategy and decide.py): daily-cost mirror, budget gate,
delegation to OpenRouterClient, failure contract seen by decide.py.

No network: the OpenRouter delegate is a recording fake; cwd is a tmp dir so the pickle budget file
never touches the real ``logs/``. Dummy key only.

Strict xfails mark real defects (reason = file:line + dollar example).
"""
import asyncio
import pickle
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.clients import xai_client as xc
from src.clients.llm_types import DailyUsageTracker, TradingDecision
from src.clients.openrouter_client import OpenRouterClient
from src.clients.xai_client import XAIClient
from src.jobs.decide import make_decision_for_market
from tests.test_decide_paths import CASH, LIMITS, Db, Kalshi, LimitsManager, _market

DUMMY_KEY = "test-dummy-key"  # secret-scan:allow
TODAY = lambda: datetime.now().strftime("%Y-%m-%d")  # noqa: E731


class FakeOR:
    """Stand-in OpenRouterClient. Mimics what the real one exposes: it does NOT set
    ``_last_request_cost`` (grep openrouter_client.py), but ``spend`` lets a test pretend a
    call cost money by bumping its own total_cost like the real client does."""

    def __init__(self, completion="text", decision=None, raises=None, spend=0.0):
        self.completion, self.decision, self.raises, self.spend = completion, decision, raises, spend
        self.total_cost = 0.0
        self.calls = []
        self.closed = False

    async def get_completion(self, **kw):
        self.calls.append(("completion", kw))
        if self.raises:
            raise self.raises
        self.total_cost += self.spend
        return self.completion

    async def get_trading_decision(self, **kw):
        self.calls.append(("decision", kw))
        if self.raises:
            raise self.raises
        self.total_cost += self.spend
        return self.decision

    async def close(self):
        self.closed = True


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _make(delegate=None, limit=10.0):
        monkeypatch.setattr(xc.settings.trading, "daily_ai_cost_limit", limit, raising=False)
        c = XAIClient(api_key=DUMMY_KEY)
        c._openrouter_client = delegate
        return c
    return _make


def _decision():
    return TradingDecision(action="BUY", side="YES", confidence=0.8, limit_price=55)


# --------------------------------------------------------------------------- shim contract
def test_reexports_are_the_canonical_llm_types():
    assert xc.TradingDecision is TradingDecision and xc.DailyUsageTracker is DailyUsageTracker


async def test_completion_delegates_with_primary_model_and_defaults(make):
    d = FakeOR("answer")
    c = make(d)
    assert await c.get_completion("p", strategy="s", query_type="q", market_id="M") == "answer"
    kind, kw = d.calls[0]
    assert kw["prompt"] == "p" and kw["model"] == c.primary_model
    assert kw["max_tokens"] == c.max_tokens and kw["temperature"] == c.temperature
    assert (kw["strategy"], kw["query_type"], kw["market_id"]) == ("s", "q", "M")


async def test_explicit_zero_temperature_is_preserved_and_zero_max_tokens_falls_back(make):
    d = FakeOR()
    c = make(d)
    c.temperature = 0.7          # non-zero default so `x or default` would be detectable
    await c.get_completion("p", temperature=0.0, max_tokens=0)
    kw = d.calls[0][1]
    assert kw["temperature"] == 0.0
    assert kw["max_tokens"] == c.max_tokens


async def test_trading_decision_delegates_and_returns_the_decision(make):
    d = FakeOR(decision=_decision())
    c = make(d)
    out = await c.get_trading_decision({"title": "t"}, {"cash": 1}, "news")
    assert out is d.decision
    kw = d.calls[0][1]
    assert kw["market_data"] == {"title": "t"} and kw["news_summary"] == "news"
    assert kw["model"] == c.primary_model


async def test_search_is_a_stub_that_never_calls_openrouter(make):
    d = FakeOR()
    out = await make(d).search("who wins" * 30)
    assert out.startswith("[Search unavailable") and d.calls == []


async def test_close_closes_the_delegate_only_if_it_was_created(make):
    d = FakeOR()
    await make(d).close()
    assert d.closed
    await make(None).close()           # no delegate: must not raise


def test_lazy_delegate_construction_failure_returns_none(make):
    c = make(None)
    with patch("src.clients.openrouter_client.OpenRouterClient", side_effect=RuntimeError("boom")):
        assert c._get_openrouter_client() is None


def test_lazy_delegate_is_built_once_with_the_db_manager(make):
    c = make(None)
    c.db_manager = "DB"
    with patch("src.clients.openrouter_client.OpenRouterClient", return_value=FakeOR()) as ctor:
        a, b = c._get_openrouter_client(), c._get_openrouter_client()
    assert a is b and ctor.call_count == 1 and ctor.call_args.kwargs == {"db_manager": "DB"}


# --------------------------------------------------------------------------- failure contract
@pytest.mark.parametrize("exc", [RuntimeError("503"), asyncio.TimeoutError(), ValueError("bad")])
async def test_delegate_exception_becomes_none_and_records_no_spend(make, exc):
    c = make(FakeOR(raises=exc))
    assert await c.get_completion("p") is None
    assert await c.get_trading_decision({}, {}) is None
    assert c.total_cost == 0 and c.request_count == 0 and c.daily_tracker.request_count == 0


async def test_cancellation_propagates(make):
    c = make(FakeOR(raises=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await c.get_completion("p")


async def test_none_from_delegate_is_passed_through_unrecorded(make):
    c = make(FakeOR(completion=None, decision=None))
    assert await c.get_completion("p") is None
    assert await c.get_trading_decision({}, {}) is None
    assert c.request_count == 0


async def test_missing_delegate_returns_none_without_spend(make):
    c = make(None)
    with patch.object(c, "_get_openrouter_client", return_value=None):
        assert await c.get_completion("p") is None
        assert await c.get_trading_decision({}, {}) is None


# --------------------------------------------------------------------------- budget gate
async def test_exhausted_xai_budget_blocks_before_the_delegate_is_called(make):
    d = FakeOR()
    c = make(d, limit=1.0)
    c.daily_tracker.total_cost = 1.0
    c._update_daily_cost(0.0)
    assert await c.get_completion("p") is None
    assert await c.get_trading_decision({}, {}) is None
    assert d.calls == []


async def test_budget_is_reread_from_disk_before_each_call(make):
    d = FakeOR()
    c = make(d, limit=2.0)
    with open(c.usage_file, "wb") as fh:
        pickle.dump(DailyUsageTracker(date=TODAY(), total_cost=2.0, daily_limit=2.0, is_exhausted=True), fh)
    assert await c.get_completion("p") is None and d.calls == []


@pytest.mark.parametrize("payload", [b"", b"junk", pickle.dumps(["x"])])
async def test_unreadable_budget_file_fails_closed(make, payload):
    d = FakeOR()
    c = make(d, limit=4.0)
    with open(c.usage_file, "wb") as fh:
        fh.write(payload)
    assert await c.get_completion("p") is None
    assert d.calls == []
    with open(c.usage_file, "rb") as fh:
        t = pickle.load(fh)
    assert t.is_exhausted and t.total_cost == 4.0


async def test_new_day_resets_an_exhausted_tracker(make):
    c = make(FakeOR(), limit=3.0)
    old = DailyUsageTracker(date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                            total_cost=3.0, daily_limit=3.0, is_exhausted=True)
    with open(c.usage_file, "wb") as fh:
        pickle.dump(old, fh)
    assert await c.get_completion("p") == "text"


def test_raising_the_limit_unexhausts_but_a_corrupt_marker_cannot(make):
    c = make(limit=5.0)
    with open(c.usage_file, "wb") as fh:
        pickle.dump(DailyUsageTracker(date=TODAY(), total_cost=5.0, daily_limit=5.0, is_exhausted=True), fh)
    c2 = make(limit=9.0)                       # operator raised the cap
    assert c2._load_daily_tracker().is_exhausted is False
    with open(c.usage_file, "wb") as fh:
        fh.write(b"corrupt")
    c3 = make(limit=9.0)
    assert c3.daily_tracker.is_exhausted and c3.daily_tracker.total_cost == 9.0
    assert c3._load_daily_tracker().is_exhausted      # cost == limit, so the sync cannot clear it


def test_update_daily_cost_exhausts_at_limit_and_persists(make):
    c = make(limit=1.0)
    c._update_daily_cost(0.4)
    assert not c.daily_tracker.is_exhausted
    c._update_daily_cost(0.6)                          # 0.4 + 0.6 == 1.0 -> exhausted
    assert c.daily_tracker.is_exhausted and c.daily_tracker.request_count == 2
    with open(c.usage_file, "rb") as fh:
        t = pickle.load(fh)
    assert t.total_cost == pytest.approx(1.0) and t.is_exhausted


async def test_completion_cost_reaches_the_xai_daily_tracker(make):
    c = make(FakeOR("ok", spend=0.0105))
    await c.get_completion("p")
    assert c.daily_tracker.total_cost == pytest.approx(0.0105)
    assert c.total_cost == pytest.approx(0.0105)


async def test_decision_cost_reaches_the_xai_daily_tracker(make):
    c = make(FakeOR(decision=_decision(), spend=0.012))
    await c.get_trading_decision({}, {})
    assert c.daily_tracker.total_cost == pytest.approx(0.012)


async def test_real_openrouter_client_has_no_last_request_cost_attribute(tmp_path, monkeypatch):
    # Pins the root cause of the two xfails above against the REAL class, not the fake.
    monkeypatch.chdir(tmp_path)
    assert not hasattr(OpenRouterClient(api_key=DUMMY_KEY), "_last_request_cost")


async def test_request_counter_counts_successes_only(make):
    d = FakeOR("ok")
    c = make(d)
    await c.get_completion("a")
    d.completion = None
    await c.get_completion("b")
    assert c.request_count == 1 and c.daily_tracker.request_count == 1


@pytest.mark.xfail(strict=True, reason="xai_client.py:96 default daily limit is 10.0 here but "
                   "openrouter_client.py:171 defaults to 50.0 when settings lacks the field; the two "
                   "trackers (separate pickle files) enforce different caps, 5x apart")
def test_both_clients_default_to_the_same_daily_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.object(xc, "settings", SimpleNamespace(trading=SimpleNamespace(
            primary_model="m", fallback_model="f", ai_temperature=0, ai_max_tokens=1))), \
         patch("src.clients.openrouter_client.settings", SimpleNamespace(
            trading=SimpleNamespace(ai_temperature=0, ai_max_tokens=1),
            api=SimpleNamespace(openrouter_api_key=DUMMY_KEY, openrouter_base_url="https://x.invalid"))):
        assert XAIClient()._load_daily_tracker().daily_limit == \
            OpenRouterClient(api_key=DUMMY_KEY)._load_daily_tracker().daily_limit


# --------------------------------------------------------------------------- what decide.py receives
def _decide(market, delegate, db=None):
    db, kalshi = db or Db(), Kalshi()
    c = XAIClient(api_key=DUMMY_KEY)
    c._openrouter_client = delegate
    c.search = AsyncMock(return_value="news")
    LimitsManager.open_positions = 0
    with patch(LIMITS, AsyncMock(return_value=(True, "ok"))), patch(CASH, AsyncMock(return_value=(True, "ok"))), \
         patch("src.utils.position_limits.PositionLimitsManager", LimitsManager):
        pos = asyncio.run(make_decision_for_market(market, db, c, kalshi))
    kalshi.place_order.assert_not_awaited()
    return pos, db, c


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("delegate", [FakeOR(decision=None), FakeOR(raises=RuntimeError("503")),
                                      FakeOR(raises=asyncio.TimeoutError())])
def test_decide_records_no_decision_and_opens_nothing_when_the_client_fails(cwd, delegate):
    pos, db, _ = _decide(_market(), delegate)
    assert pos is None
    assert [c.args[1] for c in db.record_market_analysis.await_args_list] == ["SKIP"]
    assert db.record_market_analysis.await_args_list[0].args[-1] == "no_decision"


def test_decide_with_exhausted_xai_budget_gets_none_and_skips(cwd):
    d = FakeOR(decision=_decision())
    c_path = "logs/daily_ai_usage.pkl"
    import os
    os.makedirs("logs", exist_ok=True)
    with open(c_path, "wb") as fh:
        pickle.dump(DailyUsageTracker(date=TODAY(), total_cost=10.0, daily_limit=10.0, is_exhausted=True), fh)
    pos, db, _ = _decide(_market(), d)
    assert pos is None and d.calls == []
    assert db.record_market_analysis.await_args_list[0].args[-1] == "no_decision"


def test_decide_with_unreadable_xai_budget_makes_no_llm_call(cwd):
    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/daily_ai_usage.pkl", "wb") as fh:
        fh.write(b"corrupt")
    d = FakeOR(decision=_decision())
    pos, _, _ = _decide(_market(), d)
    assert pos is None and d.calls == []


def test_decide_db_budget_gate_runs_before_the_client_is_touched(cwd):
    d = FakeOR(decision=_decision())
    db = Db()
    db.get_daily_ai_cost = AsyncMock(side_effect=RuntimeError("db down"))   # budget store unreadable
    pos, db, _ = _decide(_market(), d, db=db)
    assert pos is None and d.calls == []                    # fail closed


def test_near_expiry_branch_handles_a_none_decision_like_the_standard_path(cwd):
    m = _market(yes=0.92, no=0.08, days=0.25)
    pos, db, _ = _decide(m, FakeOR(decision=None))
    assert pos is None
    assert db.record_market_analysis.await_args_list[0].args[-1] == "no_decision"
