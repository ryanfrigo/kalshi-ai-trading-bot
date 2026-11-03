"""Behaviour tests for src/data/sentiment_analyzer.py.

No network and no credentials: ``AsyncOpenAI`` is replaced by a fake client that
returns canned chat completions (or raises / hangs), and the NewsAggregator is a
stub. Cost is tracked on ``analyzer.total_cost``; src/jobs/decide.py persists it
via ``upsert_daily_cost`` (covered with a fake analyzer in
test_spend_accounting.py), so here we pin the number decide.py will persist.

Bugs found are pinned as strict xfails at the bottom, with file:line in the reason.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.config.settings import settings
from src.data import sentiment_analyzer as sa
from src.data.news_aggregator import NewsArticle
from src.data.sentiment_analyzer import (
    ArticleSentiment,
    MarketSentiment,
    SentimentAnalyzer,
    SentimentResult,
)


def completion(content, prompt_tokens=1000, completion_tokens=50, usage=True):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if usage
        else None,
    )


def reply(score=0.5, confidence=0.8, reasoning="because"):
    return json.dumps({"score": score, "confidence": confidence, "reasoning": reasoning})


class FakeClient:
    """Stands in for AsyncOpenAI. ``script`` is consumed one entry per call:
    a str (reply content), a ready completion, an Exception (raised), or a
    callable(kwargs) -> any of those. When exhausted the last entry repeats."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if callable(step):
            step = step(kwargs)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, str):
            step = completion(step)
        return step

    async def close(self):
        self.closed = True


class FakeNews:
    def __init__(self, relevant=(), fetch_error=None):
        self.relevant = list(relevant)
        self.fetch_error = fetch_error
        self.fetched = 0
        self.queries = []

    async def fetch_all(self):
        self.fetched += 1
        if self.fetch_error:
            raise self.fetch_error
        return []

    def get_relevant_articles(self, title, max_articles=5):
        self.queries.append((title, max_articles))
        return self.relevant


def art(title="Fed holds rates", summary="Officials kept rates steady."):
    return NewsArticle(title=title, summary=summary, source="wire", published=None, url="u")


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(settings.sentiment, "enabled", True)
    holder = {}

    def _make(script=(reply(),), news=None):
        client = FakeClient(script)
        monkeypatch.setattr(sa, "AsyncOpenAI", lambda **kw: client)
        analyzer = SentimentAnalyzer(news_aggregator=news or FakeNews())
        holder["client"] = client
        return analyzer, client

    return _make


# ----------------------------------------------------- what is sent to the model


async def test_request_prompt_shape(make):
    analyzer, client = make()
    await analyzer.analyze_sentiment("Fed holds. Rates steady.", "Will the Fed cut in March?")

    (call,) = client.calls
    assert call["model"] == settings.sentiment.sentiment_model
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 256
    system, user = call["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    # system prompt demands JSON-only with the three keys the parser reads
    for key in ('"score"', '"confidence"', '"reasoning"', "JSON"):
        assert key in system["content"]
    # user message carries the article and the market question
    assert "Fed holds. Rates steady." in user["content"]
    assert "Will the Fed cut in March?" in user["content"]


async def test_request_without_market_context_omits_the_market_section(make):
    analyzer, client = make()
    await analyzer.analyze_sentiment("Some text")
    user = client.calls[0]["messages"][1]["content"]
    assert "Some text" in user
    assert "Market question" not in user


async def test_market_analysis_sends_title_dot_summary_per_article(make):
    analyzer, client = make()
    await analyzer.analyze_market_sentiment(
        "Will X happen?", [art("Title A", "Body A"), art("Title B", "Body B")]
    )
    users = sorted(c["messages"][1]["content"] for c in client.calls)
    assert len(users) == 2
    assert "Title A. Body A" in users[0] and "Title B. Body B" in users[1]
    assert all("Will X happen?" in u for u in users)


# ----------------------------------------------------- parsing the reply


def parse(content):
    return SentimentAnalyzer._parse_sentiment_response(content)


def test_parse_valid_reply():
    r = parse(reply(-0.4, 0.7, "bad news"))
    assert (r.score, r.confidence, r.reasoning) == (-0.4, 0.7, "bad news")


@pytest.mark.parametrize("fence", ["```json\n{body}\n```", "```\n{body}\n```", "  {body}  "])
def test_parse_strips_code_fences_and_whitespace(fence):
    r = parse(fence.format(body=reply(0.2, 0.9, "ok")))
    assert (r.score, r.confidence) == (0.2, 0.9)


@pytest.mark.parametrize(
    "content",
    ["", "not json", "{'score': 0.5}", '{"score": 0.5', "```json\n```", "Sure! Here you go: {}"],
)
def test_parse_malformed_json_raises_valueerror(content):
    with pytest.raises(ValueError, match="Failed to parse sentiment model response"):
        parse(content)


def test_parse_error_message_truncates_raw_content():
    with pytest.raises(ValueError) as e:
        parse("x" * 5000)
    assert len(str(e.value)) < 600


@pytest.mark.parametrize(
    "score,confidence,exp_score,exp_conf",
    [(5, 3, 1.0, 1.0), (-5, -2, -1.0, 0.0), (1.0, 1.0, 1.0, 1.0), (-1.0, 0.0, -1.0, 0.0)],
)
def test_parse_clamps_out_of_range_values(score, confidence, exp_score, exp_conf):
    r = parse(reply(score, confidence))
    assert (r.score, r.confidence) == (exp_score, exp_conf)


def test_parse_missing_keys_default_to_neutral_half_confidence():
    r = parse("{}")
    assert (r.score, r.confidence, r.reasoning) == (0.0, 0.5, "No reasoning provided.")


def test_parse_numeric_strings_are_accepted():
    r = parse('{"score": "0.25", "confidence": "0.5"}')
    assert (r.score, r.confidence) == (0.25, 0.5)


@pytest.mark.parametrize(
    "content,exc",
    [
        ('{"score": "bullish"}', ValueError),
        ('{"score": null}', TypeError),
        ('{"score": [0.5]}', TypeError),
        ("[0.5]", AttributeError),
        ("null", AttributeError),
        ("0.5", AttributeError),
    ],
)
def test_parse_wrong_shapes_raise_rather_than_return_a_value(content, exc):
    with pytest.raises(exc):
        parse(content)


@pytest.mark.parametrize(
    "body",
    ['{"score": Infinity}', '{"score": -Infinity}', '{"score": 0.5, "confidence": Infinity}'],
)
def test_parse_infinite_values_are_rejected(body):
    with pytest.raises(ValueError, match="Non-finite"):
        parse(body)


# ----------------------------------------------------- analyze_sentiment: cache + failure


async def test_analyze_sentiment_caches_by_text_and_context(make):
    analyzer, client = make()
    a = await analyzer.analyze_sentiment("t", "m")
    b = await analyzer.analyze_sentiment("t", "m")
    assert a is b and len(client.calls) == 1
    await analyzer.analyze_sentiment("t", "other market")
    await analyzer.analyze_sentiment("other text", "m")
    assert len(client.calls) == 3


async def test_analyze_sentiment_does_not_cache_failures(make):
    analyzer, client = make([RuntimeError("boom"), reply(0.3)])
    with pytest.raises(RuntimeError, match="boom"):
        await analyzer.analyze_sentiment("t", "m")
    assert (await analyzer.analyze_sentiment("t", "m")).score == 0.3
    assert len(client.calls) == 2


async def test_unparseable_reply_propagates_valueerror_and_is_not_cached(make):
    analyzer, client = make(["garbage", reply(0.1)])
    with pytest.raises(ValueError):
        await analyzer.analyze_sentiment("t", "m")
    assert (await analyzer.analyze_sentiment("t", "m")).score == 0.1


async def test_api_exception_propagates(make):
    analyzer, _ = make([ConnectionError("down")])
    with pytest.raises(ConnectionError):
        await analyzer.analyze_sentiment("t")


async def test_empty_choices_propagates(make):
    analyzer, _ = make([SimpleNamespace(choices=[], usage=None)])
    with pytest.raises(IndexError):
        await analyzer.analyze_sentiment("t")


async def test_null_content_propagates(make):
    analyzer, _ = make([completion(None)])
    with pytest.raises(AttributeError):
        await analyzer.analyze_sentiment("t")


# ----------------------------------------------------- cost accounting


async def test_cost_is_tokens_times_rate_and_counts_requests(make):
    analyzer, _ = make([completion(reply(), prompt_tokens=1000, completion_tokens=500)])
    await analyzer.analyze_sentiment("a")
    await analyzer.analyze_sentiment("b")
    assert analyzer.request_count == 2
    assert analyzer.total_cost == pytest.approx(2 * (1000 * 1e-7 + 500 * 4e-7))


async def test_cache_hit_costs_nothing(make):
    analyzer, _ = make()
    await analyzer.analyze_sentiment("a", "m")
    spent = analyzer.total_cost
    await analyzer.analyze_sentiment("a", "m")
    assert analyzer.total_cost == spent and analyzer.request_count == 1


async def test_missing_usage_records_zero_cost_but_counts_the_request(make):
    analyzer, _ = make([completion(reply(), usage=False)])
    await analyzer.analyze_sentiment("a")
    assert analyzer.total_cost == 0.0 and analyzer.request_count == 1


async def test_spend_is_recorded_even_when_the_reply_cannot_be_parsed(make):
    # The tokens were billed; decide.py persists total_cost in a finally block.
    analyzer, _ = make([completion("garbage", prompt_tokens=2000, completion_tokens=100)])
    with pytest.raises(ValueError):
        await analyzer.analyze_sentiment("a")
    assert analyzer.total_cost == pytest.approx(2000 * 1e-7 + 100 * 4e-7)
    assert analyzer.request_count == 1


async def test_failed_api_call_records_no_cost(make):
    analyzer, _ = make([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        await analyzer.analyze_sentiment("a")
    assert analyzer.total_cost == 0.0 and analyzer.request_count == 0


async def test_analyzer_never_writes_to_the_database_itself(make, monkeypatch):
    # Persistence is decide.py's job (upsert_daily_cost(analyzer.total_cost));
    # the analyzer must not also write, or spend would be counted twice.
    import inspect

    assert "upsert_daily_cost" not in inspect.getsource(sa)


# ----------------------------------------------------- analyze_market_sentiment


async def test_empty_article_list_returns_zero_signal_and_makes_no_calls(make):
    analyzer, client = make()
    ms = await analyzer.analyze_market_sentiment("Q?", [])
    assert ms == MarketSentiment(0.0, [], 0.0, 0)
    assert client.calls == []


async def test_scores_are_averaged_and_relevance_weighted(make):
    # one reply per call, in article order (gather schedules them in order)
    analyzer, _ = make([reply(1.0), reply(-1.0)])
    ms = await analyzer.analyze_market_sentiment(
        "Q?", [art("a"), art("b")], relevance_scores=[0.75, 0.25]
    )
    assert ms.num_articles == 2
    assert ms.overall_score == 0.0
    assert ms.relevance_weighted_score == pytest.approx(0.5)
    assert [s.relevance_score for s in ms.article_sentiments] == [0.75, 0.25]


async def test_default_relevance_is_equal_weighting(make):
    analyzer, _ = make([reply(1.0), reply(0.0)])
    ms = await analyzer.analyze_market_sentiment("Q?", [art("a"), art("b")])
    assert ms.overall_score == ms.relevance_weighted_score == 0.5


async def test_one_failed_article_is_skipped_not_fatal(make):
    analyzer, _ = make([reply(0.8), "garbage", reply(0.4)])
    ms = await analyzer.analyze_market_sentiment("Q?", [art("a"), art("b"), art("c")])
    assert ms.num_articles == 2
    assert ms.overall_score == pytest.approx(0.6)
    assert [s.article.title for s in ms.article_sentiments] == ["a", "c"]


async def test_all_articles_failing_returns_zero_articles(make):
    analyzer, _ = make([RuntimeError("down")])
    ms = await analyzer.analyze_market_sentiment("Q?", [art("a"), art("b")])
    assert (ms.num_articles, ms.overall_score, ms.relevance_weighted_score) == (0, 0.0, 0.0)
    assert ms.article_sentiments == []


async def test_zero_total_relevance_gives_zero_weighted_score(make):
    analyzer, _ = make([reply(1.0)])
    ms = await analyzer.analyze_market_sentiment("Q?", [art("a")], relevance_scores=[0.0])
    assert ms.relevance_weighted_score == 0.0
    assert ms.overall_score == 1.0  # unweighted mean still reports it


async def test_scores_are_rounded_to_four_places(make):
    analyzer, _ = make([reply(1.0), reply(0.0), reply(0.0)])
    ms = await analyzer.analyze_market_sentiment("Q?", [art("a"), art("b"), art("c")])
    assert ms.overall_score == 0.3333


async def test_confidence_is_ignored_by_the_aggregate_today(make):
    # Pins current behaviour; see the strict xfail below for why it is a bug.
    analyzer, _ = make([reply(1.0, 0.0)])
    ms = await analyzer.analyze_market_sentiment("Q?", [art("a")])
    assert ms.overall_score == 1.0


# ----------------------------------------------------- summary for decide.py


async def test_summary_disabled_makes_no_fetch_and_no_call(make, monkeypatch):
    news = FakeNews([(art(), 1.0)])
    analyzer, client = make(news=news)
    monkeypatch.setattr(settings.sentiment, "enabled", False)
    out = await analyzer.get_market_sentiment_summary("Q?")
    assert out == "Sentiment analysis is disabled."
    assert news.fetched == 0 and client.calls == []


async def test_summary_no_relevant_articles_says_no_signal_and_spends_nothing(make):
    news = FakeNews([])
    analyzer, client = make(news=news)
    out = await analyzer.get_market_sentiment_summary("Will X happen?")
    assert "No relevant news articles found for: Will X happen?" in out
    assert "neutral (no signal)" in out
    assert news.fetched == 1 and client.calls == [] and analyzer.total_cost == 0.0


async def test_summary_asks_for_at_most_five_relevant_articles(make):
    news = FakeNews([])
    analyzer, _ = make(news=news)
    await analyzer.get_market_sentiment_summary("Q?")
    assert news.queries == [("Q?", 5)]


async def test_summary_happy_path_labels_and_lists_signals(make):
    news = FakeNews([(art("Rates cut"), 0.9), (art("Jobs strong"), 0.5)])
    analyzer, _ = make([reply(0.8, 0.9, "good"), reply(0.6, 0.9, "fine")], news=news)
    out = await analyzer.get_market_sentiment_summary("Will rates fall?")
    assert "News Sentiment for 'Will rates fall?'" in out
    assert "Overall sentiment: positive" in out
    assert "Based on 2 relevant article(s)." in out
    assert "Rates cut" in out and "relevance: 90%" in out
    assert "Reasoning: good" in out


@pytest.mark.parametrize(
    "score,label",
    [(0.9, "positive"), (0.2, "slightly positive"), (0.0, "neutral"),
     (-0.2, "slightly negative"), (-0.9, "negative")],
)
async def test_summary_label_thresholds(make, score, label):
    news = FakeNews([(art(), 1.0)])
    analyzer, _ = make([reply(score)], news=news)
    out = await analyzer.get_market_sentiment_summary("Q?")
    assert f"Overall sentiment: {label} " in out


async def test_summary_lists_at_most_three_signals(make):
    news = FakeNews([(art(f"Story{i}"), 1.0) for i in range(5)])
    analyzer, _ = make([reply(0.5)], news=news)
    out = await analyzer.get_market_sentiment_summary("Q?")
    assert out.count("    - [") == 3


async def test_fetch_failure_returns_unavailable_string_not_exception(make):
    analyzer, client = make(news=FakeNews(fetch_error=RuntimeError("feeds down")))
    out = await analyzer.get_market_sentiment_summary("Q?")
    assert out.startswith("Sentiment analysis unavailable (error: feeds down)")
    assert "Proceed without sentiment signal" in out
    assert client.calls == []


async def test_summary_swallows_unexpected_internal_errors(make, monkeypatch):
    analyzer, _ = make(news=FakeNews([(art(), 1.0)]))

    async def boom(*a, **k):
        raise KeyError("x")

    monkeypatch.setattr(analyzer, "analyze_market_sentiment", boom)
    out = await analyzer.get_market_sentiment_summary("Q?")
    assert out.startswith("Sentiment analysis unavailable")


async def test_hung_model_is_cut_off_by_deciders_timeout(make):
    # decide.py wraps the summary in wait_for(timeout=30). The analyzer adds no
    # timeout of its own beyond the OpenAI client's, so a hang surfaces there.
    analyzer, client = make(news=FakeNews([(art(), 1.0)]))

    async def hang(**kwargs):
        await asyncio.sleep(3600)

    client.chat.completions.create = hang
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(analyzer.get_market_sentiment_summary("Q?"), timeout=0.05)


async def test_close_closes_the_client(make):
    analyzer, client = make()
    await analyzer.close()
    assert client.closed


def test_format_summary_zero_articles():
    out = SentimentAnalyzer._format_summary("Q?", MarketSentiment(0.0, [], 0.0, 0))
    assert out == "No relevant news articles found for: Q?. Sentiment: neutral (no signal)."


# ============================================================================
# Bugs found. Fixed ones are plain tests; the two open design calls stay as
# strict xfails that flip to a failure (XPASS) when fixed.
# ============================================================================


@pytest.mark.parametrize(
    "body",
    ['{"score": NaN, "confidence": NaN}', '{"score": NaN}', '{"score": 0.5, "confidence": NaN}'],
)
def test_nan_reply_is_rejected_not_turned_into_max_confidence_bullish(body):
    with pytest.raises(ValueError, match="Non-finite"):
        parse(body)


async def test_nan_reply_is_a_parse_failure_that_is_skipped_and_not_cached(make):
    analyzer, _ = make(['{"score": NaN, "confidence": NaN}', reply(0.2)])
    ms = await analyzer.analyze_market_sentiment("Q?", [art("a")])
    assert ms.num_articles == 0
    assert (await analyzer.analyze_sentiment("a. Officials kept rates steady.", "Q?")).score == 0.2


async def test_all_model_calls_failing_returns_none_so_decide_falls_back(make):
    news = FakeNews([(art("Rates cut"), 0.9), (art("Jobs"), 0.8)])
    analyzer, _ = make([RuntimeError("401 bad key")], news=news)
    # None is what decide.py treats as "no news_summary": it falls back to xAI search.
    assert await analyzer.get_market_sentiment_summary("Q?") is None


async def test_neutral_article_is_not_labelled_negative(make):
    news = FakeNews([(art("Meh"), 1.0)])
    analyzer, _ = make([reply(0.0)], news=news)
    out = await analyzer.get_market_sentiment_summary("Q?")
    assert "[neutral +0.00]" in out and "[negative" not in out and "[positive" not in out


@pytest.mark.xfail(strict=True, reason=(
    "sentiment_analyzer.py:155-176 the aggregate ignores each SentimentResult.confidence, "
    "so an article the model scored with confidence 0.0 still drives the summary to "
    "'positive (+1.00)' that decide.py puts in the trading prompt"))
async def test_zero_confidence_article_does_not_produce_a_confident_summary(make):
    news = FakeNews([(art("Rumour"), 1.0)])
    analyzer, _ = make([reply(1.0, 0.0, "no idea")], news=news)
    out = await analyzer.get_market_sentiment_summary("Q?")
    assert "Overall sentiment: positive" not in out


@pytest.mark.xfail(strict=True, reason=(
    "sentiment_analyzer.py:172-176 weights are normalised by their own sum "
    "(weighted = sum(s*r)/sum(r)), so a single barely-relevant article (relevance 0.05) "
    "yields the same 'positive (+1.00)' as a perfectly relevant one; absolute relevance "
    "is lost"))
async def test_single_barely_relevant_article_is_not_reported_as_strong_signal(make):
    news = FakeNews([(art("Tangent"), 0.05)])
    analyzer, _ = make([reply(1.0, 0.9)], news=news)
    out = await analyzer.get_market_sentiment_summary("Q?")
    assert "Overall sentiment: positive" not in out


async def test_mismatched_relevance_scores_length_is_an_error(make):
    analyzer, client = make([reply(0.5)])
    with pytest.raises(ValueError, match="relevance_scores"):
        await analyzer.analyze_market_sentiment(
            "Q?", [art("a"), art("b"), art("c")], relevance_scores=[1.0]
        )
    assert client.calls == []  # raised before any (billed) model call
    with pytest.raises(ValueError):  # too many is just as wrong
        await analyzer.analyze_market_sentiment("Q?", [art("a")], relevance_scores=[1.0, 1.0])


async def test_cache_entries_expire_after_cache_ttl(make, monkeypatch):
    import time as real_time

    analyzer, client = make()
    now = real_time.time()
    monkeypatch.setattr(real_time, "time", lambda: now)
    monkeypatch.setattr(real_time, "monotonic", lambda: now)
    await analyzer.analyze_sentiment("t", "m")
    later = now + settings.sentiment.cache_ttl_minutes * 60 + 1
    monkeypatch.setattr(real_time, "time", lambda: later)
    monkeypatch.setattr(real_time, "monotonic", lambda: later)
    await analyzer.analyze_sentiment("t", "m")
    assert len(client.calls) == 2


async def test_cache_entry_is_still_served_just_inside_the_ttl(make, monkeypatch):
    import time as real_time

    analyzer, client = make()
    now = real_time.monotonic()
    monkeypatch.setattr(real_time, "monotonic", lambda: now)
    await analyzer.analyze_sentiment("t", "m")
    monkeypatch.setattr(
        real_time, "monotonic", lambda: now + settings.sentiment.cache_ttl_minutes * 60 - 1
    )
    await analyzer.analyze_sentiment("t", "m")
    assert len(client.calls) == 1
