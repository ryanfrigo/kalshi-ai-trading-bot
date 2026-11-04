"""OpenRouterClient: cost accounting, daily budget, retry/backoff, fallback order, parsing.

No network: ``client.client`` (the AsyncOpenAI SDK object) is replaced by a recording fake and
``asyncio.sleep`` by a recorder, so backoff is observed rather than waited for. A dummy key is used
and the process cwd is a tmp dir so the pickle budget file never touches the real ``logs/``.

Strict xfails mark real defects (reason = file:line + dollar/behaviour example).
"""
import asyncio
import math
import pickle
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.clients import openrouter_client as orc
from src.clients.llm_types import DailyUsageTracker
from src.clients.openrouter_client import (
    DEFAULT_FALLBACK_ORDER, MODEL_PRICING, OpenRouterClient)

DUMMY_KEY = "test-dummy-key"  # secret-scan:allow
SONNET = "anthropic/claude-sonnet-4"       # $0.003 in / $0.015 out per 1k
GPT = "openai/gpt-4.1"                      # $0.002 / $0.008
GEMINI = "google/gemini-2.5-pro-preview"    # $0.00125 / $0.01
R1 = "deepseek/deepseek-r1"                 # $0.0008 / $0.002
GOOD_JSON = '{"action": "BUY", "side": "YES", "limit_price": 55, "confidence": 0.72}'


def resp(content="hello", prompt=1000, completion=500, usage=True, choices=True):
    msg = SimpleNamespace(content=content)
    u = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion) if usage else None
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)] if choices else [], usage=u)


class FakeCompletions:
    """Scripted ``chat.completions.create``: per-model queue of responses/exceptions."""

    def __init__(self, script):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls = []          # (model, kwargs)

    async def create(self, **kw):
        self.calls.append(kw)
        queue = self.script.get(kw["model"])
        if not queue:
            raise AssertionError(f"unscripted call to {kw['model']}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def models(self):
        return [c["model"] for c in self.calls]


class FakeDb:
    def __init__(self, fail_upsert=False):
        self.costs = []
        self.queries = []
        self.fail_upsert = fail_upsert

    async def upsert_daily_cost(self, cost):
        if self.fail_upsert:
            raise RuntimeError("db down")
        self.costs.append(cost)

    async def log_llm_query(self, q):
        self.queries.append(q)


@pytest.fixture
def sleeps(monkeypatch):
    rec = []

    async def fake_sleep(d):
        rec.append(d)

    monkeypatch.setattr(orc.asyncio, "sleep", fake_sleep)
    return rec


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _make(script=None, db=None, limit=10.0):
        monkeypatch.setattr(orc.settings.trading, "daily_ai_cost_limit", limit, raising=False)
        c = OpenRouterClient(api_key=DUMMY_KEY, db_manager=db)
        c.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(script or {})),
                                   close=_async_noop)
        return c
    return _make


async def _async_noop():
    return None


def calls(c):
    return c.client.chat.completions


# --------------------------------------------------------------------------- cost arithmetic
@pytest.mark.parametrize("model,i,o,expected", [
    (SONNET, 1000, 500, 0.003 + 0.0075),            # 0.0105
    (GPT, 2000, 1000, 0.004 + 0.008),               # 0.012
    (GEMINI, 4000, 200, 0.005 + 0.002),             # 0.007
    (R1, 10_000, 3_000, 0.008 + 0.006),             # 0.014
    (SONNET, 0, 0, 0.0),
])
def test_calculate_cost_matches_hand_computed(make, model, i, o, expected):
    assert make()._calculate_cost(model, i, o) == pytest.approx(expected, abs=1e-12)


def test_unknown_model_uses_flat_per_token_rate(make):
    # (1000 + 500) * 1e-5 = $0.015, not zero
    assert make()._calculate_cost("mystery/model", 1000, 500) == pytest.approx(0.015)


def test_every_registered_model_is_priced_input_and_output(make):
    c = make()
    for m, p in MODEL_PRICING.items():
        assert c._calculate_cost(m, 1000, 0) == pytest.approx(p["input_per_1k"])
        assert c._calculate_cost(m, 0, 1000) == pytest.approx(p["output_per_1k"])


async def test_completion_records_cost_exactly_once_everywhere(make, sleeps):
    db = FakeDb()
    c = make({SONNET: [resp("ok", 1000, 500)]}, db=db)
    assert await c.get_completion("hi", model=SONNET) == "ok"
    expected = 0.0105
    assert c.total_cost == pytest.approx(expected)
    assert c.request_count == 1
    assert c.model_costs[SONNET].total_cost == pytest.approx(expected)
    assert c.model_costs[SONNET].request_count == 1
    assert c.model_costs[SONNET].input_tokens == 1000 and c.model_costs[SONNET].output_tokens == 500
    assert c.daily_tracker.total_cost == pytest.approx(expected)
    assert c.daily_tracker.request_count == 1
    assert db.costs == [pytest.approx(expected)]           # mirrored to DB once
    assert len(calls(c).calls) == 1
    # persisted to disk too, so another process sees it
    with open(c.usage_file, "rb") as fh:
        assert pickle.load(fh).total_cost == pytest.approx(expected)


async def test_retried_call_that_finally_succeeds_is_billed_once(make, sleeps):
    c = make({SONNET: [RuntimeError("503 overloaded"), RuntimeError("502 bad"), resp("ok", 1000, 500)]})
    assert await c.get_completion("hi", model=SONNET) == "ok"
    assert len(calls(c).calls) == 3
    assert c.request_count == 1 and c.daily_tracker.request_count == 1
    assert c.total_cost == pytest.approx(0.0105)
    assert c.model_costs[SONNET].error_count == 2


async def test_fallback_after_failure_bills_only_the_model_that_answered(make, sleeps):
    c = make({SONNET: [RuntimeError("bad request")], GPT: [resp("ok", 2000, 1000)]})
    assert await c.get_completion("hi", model=SONNET) == "ok"
    assert c.model_costs[SONNET].total_cost == 0 and c.model_costs[SONNET].request_count == 0
    assert c.model_costs[GPT].total_cost == pytest.approx(0.012)
    assert c.total_cost == pytest.approx(0.012)


async def test_trading_decision_bills_before_parsing_and_once_per_billed_call(make, sleeps):
    # Sonnet answers with garbage (billed), GPT answers with good JSON (billed).
    c = make({SONNET: [resp("no json here", 1000, 500)], GPT: [resp(GOOD_JSON, 2000, 1000)]})
    d = await c.get_trading_decision({"title": "t"}, {"cash": 100}, model=SONNET)
    assert d is not None and d.action == "BUY"
    assert c.request_count == 2
    assert c.total_cost == pytest.approx(0.0105 + 0.012)
    assert c.daily_tracker.total_cost == pytest.approx(0.0225)


async def test_daily_tracker_and_db_agree_after_several_calls(make, sleeps):
    db = FakeDb()
    c = make({SONNET: [resp("a", 1000, 500)]}, db=db)
    for _ in range(3):
        await c.get_completion("hi", model=SONNET)
    assert c.daily_tracker.total_cost == pytest.approx(0.0315)
    assert sum(db.costs) == pytest.approx(0.0315)
    assert len(db.costs) == 3


async def test_db_cost_write_failure_does_not_break_the_call(make, sleeps):
    c = make({SONNET: [resp("ok")]}, db=FakeDb(fail_upsert=True))
    assert await c.get_completion("hi", model=SONNET) == "ok"
    assert c.daily_tracker.total_cost == pytest.approx(0.0105)   # local tracker still bound


async def test_missing_usage_block_is_billed_nonzero(make, sleeps):
    # A reply with tokens but no usage object is billed $0 -> budget never binds.
    c = make({SONNET: [resp("ok", usage=False)]})
    await c.get_completion("hi", model=SONNET)
    assert c.total_cost > 0


# --------------------------------------------------------------------------- billed-but-failed
async def test_billed_empty_reply_still_records_its_cost(make, sleeps):
    c = make({SONNET: [resp("", 1000, 500)], GPT: [resp("ok", 0, 0)]})
    await c.get_completion("hi", model=SONNET)
    assert c.total_cost >= 0.0105


# --------------------------------------------------------------------------- budget gate
async def test_exhausted_budget_blocks_before_any_http_call(make, sleeps):
    c = make({SONNET: [resp()]}, limit=1.0)
    c.daily_tracker.total_cost = 1.0
    c._update_daily_cost(0.0)               # persist as exhausted
    assert await c.get_completion("hi") is None
    assert await c.get_trading_decision({}, {}) is None
    assert calls(c).calls == []


async def test_spend_that_crosses_the_limit_exhausts_and_blocks_the_next_call(make, sleeps):
    c = make({SONNET: [resp("ok", 1000, 500)]}, limit=0.01)     # one call costs 0.0105
    assert await c.get_completion("a") == "ok"
    assert c.daily_tracker.is_exhausted
    assert await c.get_completion("b") is None
    assert len(calls(c).calls) == 1


async def test_budget_is_reloaded_from_disk_each_call(make, sleeps):
    c = make({SONNET: [resp()]}, limit=5.0)
    other = DailyUsageTracker(date=datetime.now().strftime("%Y-%m-%d"), total_cost=5.0,
                              daily_limit=5.0, is_exhausted=True)
    with open(c.usage_file, "wb") as fh:
        pickle.dump(other, fh)
    assert await c.get_completion("hi") is None      # sibling process spent it
    assert calls(c).calls == []


@pytest.mark.parametrize("payload", [b"", b"garbage", pickle.dumps({"not": "a tracker"})])
def test_unreadable_budget_store_fails_closed_and_persists(make, payload):
    c = make(limit=7.0)
    with open(c.usage_file, "wb") as fh:
        fh.write(payload)
    t = c._load_daily_tracker()
    assert t.is_exhausted and t.total_cost == 7.0
    with open(c.usage_file, "rb") as fh:
        assert pickle.load(fh).is_exhausted        # persisted so siblings refuse too


async def test_corrupt_budget_file_means_no_http_call(make, sleeps):
    c = make({SONNET: [resp()]})
    with open(c.usage_file, "wb") as fh:
        fh.write(b"\x00\x01corrupt")
    assert await c.get_completion("hi") is None
    assert calls(c).calls == []


async def test_stale_yesterday_tracker_resets(make, sleeps):
    c = make({SONNET: [resp()]}, limit=5.0)
    old = DailyUsageTracker(date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                            total_cost=5.0, daily_limit=5.0, is_exhausted=True)
    with open(c.usage_file, "wb") as fh:
        pickle.dump(old, fh)
    assert await c.get_completion("hi") == "hello"


@pytest.mark.xfail(strict=True, reason="openrouter_client.py:172 os.makedirs in _load_daily_tracker is "
                   "outside any try: an unwritable logs/ dir makes get_completion/get_trading_decision "
                   "raise PermissionError instead of returning None (no spend, but strategies that "
                   "call get_completion bare crash)")
async def test_unwritable_budget_dir_does_not_raise_into_the_caller(make, sleeps, monkeypatch):
    # decide.py / strategies call get_completion without a try for budget-store errors.
    c = make({SONNET: [resp()]})
    monkeypatch.setattr(orc.os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(PermissionError("ro")))
    try:
        out = await c.get_completion("hi")
    except PermissionError:
        pytest.fail("budget-store error escaped get_completion")
    assert out is None          # and, failing closed, made no call
    assert calls(c).calls == []


async def test_budget_rechecked_between_fallback_models(make, sleeps):
    c = make({SONNET: [resp("no json", 1000, 500)], GPT: [resp(GOOD_JSON)], GEMINI: [resp(GOOD_JSON)],
              R1: [resp(GOOD_JSON)]}, limit=0.01)
    await c.get_trading_decision({"title": "t"}, {"cash": 1}, model=SONNET)
    assert len(calls(c).calls) == 1


async def test_budget_spent_to_exactly_the_limit_exhausts(make, sleeps):
    limit = OpenRouterClient._calculate_cost(None, SONNET, 1000, 500)
    c = make({SONNET: [resp()]}, limit=limit)
    await c.get_completion("hi", model=SONNET)
    assert c.daily_tracker.is_exhausted                  # >= not >
    assert await c.get_completion("again", model=SONNET) is None


# --------------------------------------------------------------------------- retry / backoff
def test_backoff_schedule_is_exponential_and_capped(make):
    c = make()
    assert [c._backoff_delay(i) for i in range(7)] == [1, 2, 4, 8, 16, 30, 30]


@pytest.mark.parametrize("msg,retry", [
    ("Error 429", True), ("rate limit hit", True), ("Too Many Requests", True),
    ("502 bad gateway", True), ("503", True), ("504", True), ("upstream timeout", True),
    ("model overloaded", True), ("Internal Error", True), ("server error", True),
    ("401 invalid api key", False), ("400 bad request", False), ("context length exceeded", False),
])
def test_retryable_classification(make, msg, retry):
    assert OpenRouterClient._is_retryable_error(RuntimeError(msg)) is retry


async def test_retryable_error_is_tried_max_retries_times_with_backoff(make, sleeps):
    c = make({SONNET: [RuntimeError("503 overloaded")]})
    out = await c.get_completion("hi", model=SONNET)
    assert out is None
    per_model = OpenRouterClient.MAX_RETRIES_PER_MODEL
    chain = DEFAULT_FALLBACK_ORDER  # sonnet is first; the others raise AssertionError (unscripted)
    sonnet_calls = [m for m in calls(c).models if m == SONNET]
    assert len(sonnet_calls) == per_model == 3
    assert sleeps[:2] == [1.0, 2.0]          # no sleep after the final attempt
    assert len(sleeps) == 2
    assert c.total_cost == 0 and c.request_count == 0   # failed calls are not billed
    assert c.model_costs[SONNET].error_count == 3
    assert len(calls(c).calls) <= per_model * len(chain)


async def test_rate_limit_doubles_the_delay(make, sleeps):
    c = make({SONNET: [RuntimeError("429 rate limit"), RuntimeError("429 rate limit"), resp("ok")]})
    await c.get_completion("hi", model=SONNET)
    assert sleeps == [2.0, 4.0]


async def test_non_retryable_error_is_tried_once_and_falls_back(make, sleeps):
    c = make({SONNET: [RuntimeError("401 invalid api key")], GPT: [resp("ok")]})
    assert await c.get_completion("hi", model=SONNET) == "ok"
    assert calls(c).models == [SONNET, GPT]
    assert sleeps == []


async def test_total_http_calls_are_bounded_when_everything_fails(make, sleeps):
    script = {m: [RuntimeError("503 overloaded")] for m in DEFAULT_FALLBACK_ORDER}
    c = make(script)
    assert await c.get_completion("hi") is None
    assert len(calls(c).calls) == 3 * len(DEFAULT_FALLBACK_ORDER)      # 12, no more
    assert len(sleeps) == 2 * len(DEFAULT_FALLBACK_ORDER)
    assert max(sleeps) <= OpenRouterClient.MAX_BACKOFF * 2


async def test_empty_reply_is_not_retried_and_falls_through(make, sleeps):
    for bad in (resp(""), resp(None), resp(choices=False)):
        c = make({SONNET: [bad], GPT: [resp("ok")]})
        assert await c.get_completion("hi", model=SONNET) == "ok"
        assert calls(c).models == [SONNET, GPT]


@pytest.mark.xfail(strict=True, reason="openrouter_client.py:340-357 _is_retryable_error matches "
                   "'timeout' but the openai SDK's APITimeoutError str is 'Request timed out.', so a "
                   "120s SDK timeout is never retried and goes straight to the next (billed) model")
async def test_sdk_timeout_is_retried_on_the_same_model(make, sleeps):
    exc = openai.APITimeoutError(request=httpx.Request("POST", "https://openrouter.invalid/x"))
    c = make({SONNET: [exc, resp("ok")]})
    assert await c.get_completion("hi", model=SONNET) == "ok"
    assert calls(c).models == [SONNET, SONNET]


@pytest.mark.xfail(strict=True, reason="openrouter_client.py:340-357 'quota' counts as a rate-limit "
                   "(line 333-338) but not as retryable, so a quota 429-less error skips the backoff "
                   "and burns the next fallback model immediately")
async def test_quota_error_backs_off_before_moving_on(make, sleeps):
    c = make({SONNET: [RuntimeError("quota exceeded"), resp("ok")]})
    await c.get_completion("hi", model=SONNET)
    assert calls(c).models[:2] == [SONNET, SONNET]


def test_client_timeout_and_sdk_retries_are_pinned(make):
    c = make()
    assert c.client is not None
    fresh = OpenRouterClient(api_key=DUMMY_KEY)
    assert fresh.client.timeout == 120.0
    assert fresh.client.max_retries == 0          # our loop owns retries; SDK must not double them


async def test_cancellation_is_not_swallowed_by_the_retry_loop(make, sleeps):
    c = make({SONNET: [asyncio.CancelledError()]})
    with pytest.raises(asyncio.CancelledError):
        await c.get_completion("hi", model=SONNET)


# --------------------------------------------------------------------------- request params
async def test_default_and_explicit_request_parameters(make, sleeps):
    c = make({SONNET: [resp()]})
    await c.get_completion("the prompt", model=SONNET, temperature=0.0, max_tokens=123)
    kw = calls(c).calls[0]
    assert kw["messages"] == [{"role": "user", "content": "the prompt"}]
    assert kw["temperature"] == 0.0 and kw["max_tokens"] == 123
    await c.get_completion("p", model=SONNET)
    kw = calls(c).calls[1]
    assert kw["temperature"] == c.temperature and kw["max_tokens"] == c.max_tokens


async def test_trading_decision_uses_low_temperature_and_4000_cap(make, sleeps):
    c = make({SONNET: [resp(GOOD_JSON)]})
    await c.get_trading_decision({"title": "t"}, {"cash": 10}, model=SONNET)
    kw = calls(c).calls[0]
    assert kw["temperature"] == 0.1 and kw["max_tokens"] == 4000


# --------------------------------------------------------------------------- fallback order
def test_fallback_chain_contents(make):
    c = make()
    assert c._build_fallback_chain(None) == [SONNET, GPT, GEMINI, R1]
    assert c._build_fallback_chain(GEMINI) == [GEMINI, SONNET, GPT, R1]
    assert c._build_fallback_chain("x/unknown") == ["x/unknown", SONNET, GPT, GEMINI, R1]
    assert len(set(c._build_fallback_chain(GPT))) == 4        # no duplicates


async def test_models_are_tried_in_chain_order_and_stop_at_first_success(make, sleeps):
    c = make({SONNET: [RuntimeError("400 no")], GPT: [RuntimeError("400 no")], GEMINI: [resp("ok")],
              R1: [resp("never")]})
    assert await c.get_completion("hi") == "ok"
    assert calls(c).models == [SONNET, GPT, GEMINI]


async def test_all_models_failing_returns_none_and_bills_nothing(make, sleeps):
    c = make({m: [RuntimeError("400 no")] for m in DEFAULT_FALLBACK_ORDER})
    assert await c.get_completion("hi") is None
    assert await c.get_trading_decision({"title": "t"}, {"cash": 1}) is None
    assert c.total_cost == 0 and c.daily_tracker.total_cost == 0


async def test_unparsable_decision_falls_through_every_model_then_none(make, sleeps):
    c = make({m: [resp("I refuse to answer", 100, 100)] for m in DEFAULT_FALLBACK_ORDER})
    assert await c.get_trading_decision({"title": "t"}, {"cash": 1}) is None
    assert len(calls(c).calls) == 4
    assert c.request_count == 4                       # all four were billed and recorded
    assert c.total_cost == pytest.approx(
        sum(c._calculate_cost(m, 100, 100) for m in DEFAULT_FALLBACK_ORDER))


# --------------------------------------------------------------------------- parsing
@pytest.fixture
def parse(make):
    return make()._parse_trading_decision


def test_parse_fenced_and_bare_and_prose_wrapped(parse):
    for text in ("```json\n" + GOOD_JSON + "\n```", GOOD_JSON, "Sure! " + GOOD_JSON + " Hope it helps"):
        d = parse(text)
        assert (d.action, d.side, d.limit_price, d.confidence) == ("BUY", "YES", 55, 0.72)


@pytest.mark.parametrize("text", ["", "   ", "no braces", "{", "}", "[]", "[1,2]", "null", "```\n```"])
def test_parse_garbage_returns_none(parse, text):
    assert parse(text) is None


def test_parse_truncated_without_closing_brace_returns_none(parse):
    # max_tokens cut: no `}` -> no JSON found -> None (safe: no trade)
    assert parse('{"action": "BUY", "side": "NO", "limit_price": 40, "confidence": 0.8, "reasoning": "cut of') is None
    assert parse('```json\n{"action": "BUY", "side": "YES"') is None


def test_parse_truncated_after_nested_object_is_repaired(parse):
    d = parse('{"action":"BUY","side":"NO","limit_price":40,"confidence":0.8,"meta":{"a":1}')
    assert d is not None and (d.side, d.limit_price, d.confidence) == ("NO", 40, 0.8)


def test_parse_missing_confidence_defaults_to_half(parse):
    d = parse('{"action": "BUY", "side": "YES", "limit_price": 55}')
    assert d.confidence == 0.5 and d.limit_price == 55


@pytest.mark.parametrize("raw,norm", [("buy_yes", "BUY"), ("BUY_NO", "BUY"), ("Buy", "BUY"),
                                      ("sell", "SELL"), ("hold", "SKIP"), ("", "SKIP"),
                                      ("LIQUIDATE", "SKIP")])
def test_action_normalisation(parse, raw, norm):
    assert parse('{"action": "%s", "side": "YES", "limit_price": 55, "confidence": 0.7}' % raw).action == norm


def test_parse_non_object_json_inside_fence_returns_none(parse):
    assert parse("```json\n[1, 2]\n```") is None
    assert parse('```json\n"just a string"\n```') is None


def test_parse_null_fields_do_not_crash(parse):
    assert parse('{"action": null, "side": "YES"}') is None            # None.upper() -> caught
    assert parse('{"action": "BUY", "side": null, "confidence": 0.9}') is None
    assert parse('{"action": "BUY", "side": "YES", "confidence": null}') is None


def test_parse_null_limit_price_is_rejected_and_string_price_coerces(parse):
    assert parse('{"action":"BUY","side":"YES","limit_price":null,"confidence":0.7}') is None
    assert parse('{"action":"BUY","side":"YES","limit_price":"55","confidence":"0.7"}').limit_price == 55


def test_parse_nan_price_is_rejected(parse):
    assert parse('{"action":"BUY","side":"YES","limit_price":NaN,"confidence":0.7}') is None
    assert parse('{"action":"BUY","side":"YES","limit_price":Infinity,"confidence":0.7}') is None


def test_parse_nan_confidence_is_rejected(parse):
    d = parse('{"action":"BUY","side":"YES","limit_price":55,"confidence":NaN}')
    assert d is None or not math.isnan(d.confidence)


@pytest.mark.parametrize("conf", ["72", "-5", "1e9", "Infinity"])
def test_parse_out_of_range_confidence_is_rejected(parse, conf):
    d = parse('{"action":"BUY","side":"YES","limit_price":55,"confidence":%s}' % conf)
    assert d is None or 0.0 <= d.confidence <= 1.0


@pytest.mark.parametrize("extra", ['"side":"MAYBE","limit_price":55', '"side":"YES","limit_price":150',
                                   '"side":"YES","limit_price":-3', '"side":"YES","limit_price":0'])
def test_parse_buy_with_invalid_side_or_price_is_rejected(parse, extra):
    d = parse('{"action":"BUY",%s,"confidence":0.9}' % extra)
    assert d is None or d.action != "BUY"


def test_parse_keeps_reasoning(parse):
    d = parse('{"action":"BUY","side":"YES","limit_price":55,"confidence":0.9,"reasoning":"because"}')
    assert d.reasoning == "because"


@pytest.mark.parametrize("raw", ['123', '""', '"   "', 'null', '["a"]'])
def test_parse_non_string_or_blank_reasoning_is_none(parse, raw):
    d = parse('{"action":"BUY","side":"YES","limit_price":55,"confidence":0.9,"reasoning":%s}' % raw)
    assert d is not None and d.reasoning is None


def test_parse_caps_reasoning_at_1000_chars(parse):
    d = parse('{"action":"BUY","side":"YES","limit_price":55,"confidence":0.9,"reasoning":"%s"}' % ("x" * 5000))
    assert d is not None and d.reasoning == "x" * 1000


@pytest.mark.xfail(strict=True, reason="openrouter_client.py:693-698 greedy `\\{.*\\}` over a reply "
                   "with a second brace group (e.g. a trailing example '{like this}') builds invalid "
                   "JSON; json_repair gives up and the parse returns None, "
                   "discarding a valid first object (a skipped trade, not a wrong one)")
def test_parse_reply_with_trailing_brace_text_uses_first_object(parse):
    d = parse(GOOD_JSON + "\nNote: use the format {action: BUY} always")
    assert d is not None and d.limit_price == 55 and d.confidence == 0.72


async def test_decision_with_nonstring_content_is_handled(make, sleeps):
    # a provider returning list-of-parts content must not crash get_trading_decision
    c = make({m: [resp([{"type": "text", "text": GOOD_JSON}], 10, 10)] for m in DEFAULT_FALLBACK_ORDER})
    assert await c.get_trading_decision({"title": "t"}, {"cash": 1}) is None


# --------------------------------------------------------------------------- misc
def test_trading_prompt_handles_dollar_and_cent_fields_and_truncates_news(make):
    c = make()
    p = c._build_trading_prompt({"title": "T", "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.50",
                                 "no_bid_dollars": "0.50", "no_ask_dollars": "0.60"},
                                {"cash": 100.0, "max_trade_value": 5.0}, "n" * 900)
    assert "YES price: 0.45c" in p and "Max trade value: $5.00" in p
    assert "n" * 800 + "..." in p and "n" * 801 not in p
    legacy = c._build_trading_prompt({"yes_bid": 40, "yes_ask": 50, "no_bid": 50, "no_ask": 60}, {"balance": 10}, "")
    assert "YES price: 45.0c" in legacy and "Unknown Market" in legacy


def test_cost_summary_only_lists_used_models(make):
    c = make()
    c._track_model_cost(GPT, 1000, 1000, 0.01)
    s = c.get_cost_summary()
    assert list(s["per_model"]) == [GPT]
    assert s["per_model"][GPT]["total_cost"] == 0.01


async def test_close_flushes_pending_db_costs(make, sleeps):
    db = FakeDb()
    c = make({SONNET: [resp()]}, db=db)
    await c.get_completion("hi", model=SONNET)
    await c.close()
    assert len(db.costs) == 1


async def test_query_log_is_written_with_truncation(make):
    db = FakeDb()
    c = make({SONNET: [resp("r" * 6000, 100, 50)]}, db=db)
    await c.get_completion("p" * 3000, model=SONNET, strategy="s", market_id="M")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    q = db.queries[0]
    assert len(q.prompt) == 2000 and len(q.response) == 5000 and q.tokens_used == 150
    assert q.cost_usd == pytest.approx(c._calculate_cost(SONNET, 100, 50))


# --------------------------------------------------------------------------- spend-path fixes
async def test_missing_usage_cost_is_prompt_chars_over_4_plus_max_tokens(make, sleeps, caplog):
    c = make({SONNET: [resp("ok", usage=False)]})
    await c.get_completion("x" * 4000, model=SONNET, max_tokens=500)
    # input = 4000/4 = 1000 tokens, output = max_tokens = 500 -> 0.003 + 0.0075
    assert c.total_cost == pytest.approx(0.0105)
    assert c.daily_tracker.total_cost == pytest.approx(0.0105)


async def test_budget_rechecked_before_each_retry(make, sleeps):
    # First attempt is retryable-failing; the budget is spent externally in between.
    c = make({SONNET: [RuntimeError("503 service unavailable"), resp()]}, limit=1.0)
    n = {"i": 0}

    async def gate():
        n["i"] += 1
        return n["i"] == 1        # open for the entry check, closed for the retry
    c._check_daily_limits = gate
    assert await c.get_completion("hi", model=SONNET) is None
    assert len(calls(c).calls) == 1


@pytest.mark.parametrize("side", ["yes", "No", "NO"])
def test_parse_accepts_case_insensitive_side(parse, side):
    d = parse('{"action":"BUY","side":"%s","limit_price":55,"confidence":0.9}' % side)
    assert d is not None and d.side == side.upper()


@pytest.mark.parametrize("price", ["null", '"x"', "100", "0.5"])
def test_parse_buy_with_missing_or_bad_price_is_rejected(parse, price):
    assert parse('{"action":"BUY","side":"YES","limit_price":%s,"confidence":0.9}' % price) is None


def test_parse_buy_without_limit_price_is_rejected(parse):
    assert parse('{"action":"BUY","side":"YES","confidence":0.9}') is None


def test_parse_skip_without_price_still_parses(parse):
    d = parse('{"action":"SKIP","confidence":0.3}')
    assert d is not None and d.action == "SKIP" and d.limit_price is None


# --------------------------------------------------------------------------- security-review pins
async def test_get_completion_budget_rechecked_between_fallback_models(make, sleeps):
    # Empty-but-billed reply is non-retryable, so only the per-fallback guard can stop model 2.
    c = make({SONNET: [resp("", 1000, 500)], GPT: [resp("ok")]}, limit=0.01)
    assert await c.get_completion("hi", model=SONNET) is None
    assert calls(c).models == [SONNET]
    assert c.total_cost == pytest.approx(0.0105)


async def test_billed_empty_reply_is_booked_exactly_once_per_attempt(make, sleeps, monkeypatch):
    c = make({SONNET: [resp("", 1000, 500)]})
    monkeypatch.setattr(c, "_build_fallback_chain", lambda model: [SONNET])
    booked = []
    real = c._record_spend

    async def spy(model, i, o, cost):
        booked.append((model, i, o, cost))
        await real(model, i, o, cost)
    monkeypatch.setattr(c, "_record_spend", spy)

    assert await c.get_completion("hi", model=SONNET) is None
    # "Empty response" is not a retryable error, so exactly one attempt reached the API.
    attempts = len(calls(c).calls)
    assert attempts == 1
    assert len(booked) == attempts
    assert booked[0][:3] == (SONNET, 1000, 500) and booked[0][3] == pytest.approx(0.0105)
    assert c.total_cost == pytest.approx(attempts * 0.0105)
    assert c.daily_tracker.total_cost == pytest.approx(attempts * 0.0105)
    assert c.model_costs[SONNET].request_count == attempts


@pytest.mark.parametrize("body", ['"side":"maybe","limit_price":55', '"side":"YES","limit_price":0',
                                  '"side":"YES","limit_price":150'])
def test_parse_sell_with_bad_side_or_price_is_rejected(parse, body):
    assert parse('{"action":"SELL",%s,"confidence":0.9}' % body) is None
