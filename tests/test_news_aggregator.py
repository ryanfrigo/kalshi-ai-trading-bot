"""Behaviour tests for src/data/news_aggregator.py.

No network: ``httpx.get`` is replaced by a fake that serves canned RSS text (or
raises) per feed URL. The real feedparser still parses the bytes, so parsing
behaviour is the real thing.
"""

from types import SimpleNamespace

import httpx
import pytest

from src.config.settings import settings
from src.data import news_aggregator as na
from src.data.news_aggregator import NewsAggregator, NewsArticle



def rss(items, title="Wire One"):
    body = "".join(items)
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        f"<title>{title}</title>{body}</channel></rss>"
    )


def item(title, desc="", link="http://x/1", date=None, cat=None):
    s = f"<item><title>{title}</title><link>{link}</link>"
    if desc:
        s += f"<description>{desc}</description>"
    if date:
        s += f"<pubDate>{date}</pubDate>"
    if cat:
        s += f"<category>{cat}</category>"
    return s + "</item>"


class Clock:
    def __init__(self):
        self.t = 100_000.0  # host up ~28h; see test_fetch_feed_works_shortly_after_boot

    def monotonic(self):
        return self.t


@pytest.fixture
def world(monkeypatch):
    """feeds: url -> rss text | Exception. calls: list of parsed urls."""
    feeds, calls, kwargs = {}, [], []

    def fake_get(url, **kw):
        calls.append(url)
        kwargs.append(kw)
        v = feeds[url]
        if isinstance(v, Exception):
            raise v
        return SimpleNamespace(content=v.encode(), raise_for_status=lambda: None)

    monkeypatch.setattr(na.httpx, "get", fake_get)
    clock = Clock()
    monkeypatch.setattr(na, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(settings.sentiment, "cache_ttl_minutes", 30)
    monkeypatch.setattr(settings.sentiment, "max_articles_per_source", 10)
    monkeypatch.setattr(settings.sentiment, "relevance_threshold", 0.3)

    def make(*urls):
        monkeypatch.setattr(settings.sentiment, "rss_feeds", list(urls))
        return NewsAggregator()

    return SimpleNamespace(feeds=feeds, calls=calls, kwargs=kwargs, clock=clock, make=make)


def art(title, summary=""):
    return NewsArticle(title=title, summary=summary, source="s", published=None, url="u")


# ---------------------------------------------------------------- parsing


async def test_parse_maps_fields_and_strips_html(world):
    world.feeds["f1"] = rss(
        [item("  Fed holds rates  ", "<p>Powell said <b>no</b> cut</p>",
              link="http://x/a", date="Mon, 03 Nov 2026 12:30:45 GMT", cat="Economy")],
        title="Wire One",
    )
    [a] = await world.make("f1").fetch_all()
    assert a.title == "Fed holds rates"
    assert a.summary == "Powell said no cut"
    assert a.source == "Wire One"
    assert a.url == "http://x/a"
    assert a.category == "Economy"
    assert (a.published.year, a.published.month, a.published.day) == (2026, 11, 3)
    assert (a.published.hour, a.published.minute) == (12, 30)
    assert a.published.utcoffset().total_seconds() == 0


async def test_missing_optional_fields_default(world):
    world.feeds["f1"] = rss([item("Only a title")])
    [a] = await world.make("f1").fetch_all()
    assert a.summary == ""
    assert a.published is None
    assert a.category == ""


async def test_source_falls_back_to_url_when_feed_untitled(world):
    world.feeds["f1"] = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        + item("Untitled feed story") + "</channel></rss>"
    )
    [a] = await world.make("f1").fetch_all()
    assert a.source == "f1"


async def test_max_articles_per_source_caps_each_feed(world, monkeypatch):
    monkeypatch.setattr(settings.sentiment, "max_articles_per_source", 2)
    world.feeds["f1"] = rss([item(f"story {i}") for i in range(5)])
    agg = world.make("f1")
    got = await agg.fetch_all()
    assert [a.title for a in got] == ["story 0", "story 1"]


async def test_bad_pubdate_gives_none_not_crash(world):
    world.feeds["f1"] = rss([item("Story", date="not a date at all")])
    [a] = await world.make("f1").fetch_all()
    assert a.published is None


def test_parse_published_date_rejects_impossible_tuple():
    assert NewsAggregator._parse_published_date({"published_parsed": (2026, 13, 45, 0, 0, 0)}) is None
    assert NewsAggregator._parse_published_date({}) is None


# ---------------------------------------------------------------- dedupe


async def test_dedupe_across_feeds_ignores_case_and_whitespace_first_wins(world):
    world.feeds["f1"] = rss([item("Fed Holds Rates", link="http://one")], title="One")
    world.feeds["f2"] = rss([item("  fed holds rates ", link="http://two"),
                             item("Different story", link="http://three")], title="Two")
    got = await world.make("f1", "f2").fetch_all()
    assert [(a.title, a.source) for a in got] == [
        ("Fed Holds Rates", "One"), ("Different story", "Two")]


def test_dedupe_keeps_distinct_titles_in_order():
    out = NewsAggregator._deduplicate([art("b"), art("a"), art("B"), art("c")])
    assert [a.title for a in out] == ["b", "a", "c"]


# ------------------------------------------------- failure isolation / bad feeds


async def test_fetch_error_in_one_feed_does_not_poison_others(world):
    world.feeds["bad"] = RuntimeError("connection reset")
    world.feeds["good"] = rss([item("Survivor story")])
    got = await world.make("bad", "good").fetch_all()
    assert [a.title for a in got] == ["Survivor story"]


async def test_malformed_feed_with_no_entries_is_dropped_others_kept(world):
    world.feeds["junk"] = "<<<this is not xml at all"
    world.feeds["good"] = rss([item("Still here")])
    got = await world.make("junk", "good").fetch_all()
    assert [a.title for a in got] == ["Still here"]


async def test_empty_feed_yields_no_articles_and_no_error(world):
    world.feeds["empty"] = rss([])
    assert await world.make("empty").fetch_all() == []


async def test_all_feeds_failing_returns_empty_and_is_retried_next_call(world):
    world.feeds["bad"] = RuntimeError("down")
    agg = world.make("bad")
    assert await agg.fetch_all() == []
    world.feeds["bad"] = rss([item("Back online")])
    assert [a.title for a in await agg.fetch_all()] == ["Back online"]


async def test_partially_malformed_feed_keeps_its_good_entries(world):
    # Truncated XML: feedparser flags bozo but still returns the entries it read.
    xml = rss([item("Readable one"), item("Readable two")])
    world.feeds["f1"] = xml.replace("</channel></rss>", "<broken")
    got = await world.make("f1").fetch_all()
    assert {a.title for a in got} >= {"Readable one"}


# ---------------------------------------------------------------- caching


async def test_fetch_all_serves_cache_within_ttl_without_refetching(world):
    world.feeds["f1"] = rss([item("v1")])
    agg = world.make("f1")
    first = await agg.fetch_all()
    world.feeds["f1"] = rss([item("v2")])
    world.clock.t += 29 * 60
    again = await agg.fetch_all()
    assert [a.title for a in first] == [a.title for a in again] == ["v1"]


async def test_fetch_all_refetches_after_ttl(world):
    world.feeds["f1"] = rss([item("v1")])
    agg = world.make("f1")
    await agg.fetch_all()
    world.feeds["f1"] = rss([item("v2")])
    world.clock.t += 30 * 60
    assert [a.title for a in await agg.fetch_all()] == ["v2"]


async def test_ttl_comes_from_settings(world, monkeypatch):
    monkeypatch.setattr(settings.sentiment, "cache_ttl_minutes", 1)
    world.feeds["f1"] = rss([item("v1")])
    agg = world.make("f1")
    await agg.fetch_all()
    world.feeds["f1"] = rss([item("v2")])
    world.clock.t += 61
    assert [a.title for a in await agg.fetch_all()] == ["v2"]


async def test_fetch_feed_is_rate_limited_after_a_successful_fetch(world):
    world.feeds["f1"] = rss([item("a")])
    agg = world.make("f1")
    assert [a.title for a in await agg.fetch_feed("f1")] == ["a"]
    assert await agg.fetch_feed("f1") == []
    world.clock.t += 31 * 60
    assert [a.title for a in await agg.fetch_feed("f1")] == ["a"]


async def test_fetch_feed_does_not_populate_relevance_cache(world):
    world.feeds["f1"] = rss([item("Fed holds interest rates")])
    agg = world.make("f1")
    await agg.fetch_feed("f1")
    assert agg.get_relevant_articles("Will the Fed hold interest rates") == []


async def test_failed_feed_is_not_rate_limited(world):
    world.feeds["f1"] = RuntimeError("boom")
    agg = world.make("f1")
    with pytest.raises(RuntimeError):
        await agg.fetch_feed("f1")
    world.feeds["f1"] = rss([item("ok now")])
    assert [a.title for a in await agg.fetch_feed("f1")] == ["ok now"]


# ------------------------------------------------------------ relevance


def test_extract_keywords_drops_stopwords_and_short_tokens():
    kws = NewsAggregator._extract_keywords("Will the Fed cut rates in 2026?")
    assert kws == ["fed", "cut", "rates", "2026"]


def test_extract_keywords_empty_and_all_stopwords():
    assert NewsAggregator._extract_keywords("") == []
    assert NewsAggregator._extract_keywords("will the of and") == []


def test_score_is_fraction_of_keywords_found_in_title_and_summary():
    a = art("Fed holds", "rates steady")
    assert NewsAggregator._score_relevance(a, ["fed", "rates", "tariff", "oil"]) == 0.5
    assert NewsAggregator._score_relevance(a, ["tariff"]) == 0.0
    assert NewsAggregator._score_relevance(a, []) == 0.0


def _agg_with_cache(world, *articles):
    agg = world.make()
    agg._cache = list(articles)
    return agg


def test_relevant_articles_filter_by_threshold_sort_desc_and_cap(world):
    agg = _agg_with_cache(
        world,
        art("Fed hold"),                                  # 1/4 = .25 -> below .3
        art("Fed rates", "unrelated"),                    # 2/4
        art("Fed cut rates 2026"),                        # 4/4
        art("Fed cut rates"),                             # 3/4
    )
    got = agg.get_relevant_articles("Fed cut rates 2026")
    assert [(a.title, s) for a, s in got] == [
        ("Fed cut rates 2026", 1.0), ("Fed cut rates", 0.75), ("Fed rates", 0.5)]
    assert len(agg.get_relevant_articles("Fed cut rates 2026", max_articles=1)) == 1


def test_threshold_is_inclusive_and_read_from_settings(world, monkeypatch):
    agg = _agg_with_cache(world, art("alpha bravo"))
    q = "alpha bravo charlie delta"                       # exactly 0.5
    monkeypatch.setattr(settings.sentiment, "relevance_threshold", 0.5)
    assert len(agg.get_relevant_articles(q)) == 1
    monkeypatch.setattr(settings.sentiment, "relevance_threshold", 0.51)
    assert agg.get_relevant_articles(q) == []


def test_no_keywords_or_empty_cache_returns_empty(world):
    agg = _agg_with_cache(world, art("anything"))
    assert agg.get_relevant_articles("will the of") == []
    assert _agg_with_cache(world).get_relevant_articles("Fed rates") == []


async def test_end_to_end_fetch_then_relevance(world):
    world.feeds["f1"] = rss([item("Fed holds interest rates", "Powell speaks"),
                             item("Local bake sale raises money")])
    agg = world.make("f1")
    await agg.fetch_all()
    got = agg.get_relevant_articles("Will the Fed hold interest rates?")
    assert [a.title for a, _ in got] == ["Fed holds interest rates"]


def test_whole_word_match_scores_zero_for_substring_only():
    a = art("Software company ships federal compliance update")
    assert NewsAggregator._score_relevance(a, ["war", "fed"]) == 0.0


def test_whole_word_match_still_finds_real_words_and_punctuation_edges():
    a = art("Fed, again: rates hold.", "(war) ends")
    assert NewsAggregator._score_relevance(a, ["fed", "rates", "war"]) == 1.0


def test_regex_metacharacters_in_keywords_are_literal():
    assert NewsAggregator._score_relevance(art("plain text"), ["a+b"]) == 0.0




def test_distinct_untitled_articles_are_not_merged():
    a = NewsArticle("", "story about oil", "s", None, "http://a")
    b = NewsArticle("", "story about gas", "s", None, "http://b")
    assert len(NewsAggregator._deduplicate([a, b])) == 2


def test_articles_with_neither_title_nor_link_are_never_merged():
    a = NewsArticle("", "story about oil", "s", None, "")
    b = NewsArticle("  ", "story about gas", "s", None, "")
    assert len(NewsAggregator._deduplicate([a, b])) == 2


def test_untitled_articles_sharing_a_link_are_merged():
    a = NewsArticle("", "x", "s", None, "http://same")
    b = NewsArticle("", "y", "s", None, "http://same")
    assert len(NewsAggregator._deduplicate([a, b])) == 1


async def test_fetch_feed_works_shortly_after_boot(world):
    world.clock.t = 120.0
    world.feeds["f1"] = rss([item("first ever fetch")])
    assert [a.title for a in await world.make("f1").fetch_feed("f1")] == ["first ever fetch"]


# ------------------------------------------------- formerly open bugs, now fixed


def test_two_letter_acronyms_are_kept_as_keywords():
    kws = NewsAggregator._extract_keywords("Will the EU ban AI")
    assert "eu" in kws and "ai" in kws


def test_repeated_query_word_counts_once():
    a = art("Tariff news")
    # 1 of 3 distinct keywords (the xfail's old expected 0.25 was itself wrong:
    # it is neither the buggy 2/4 nor the correct 1/3)
    kws = NewsAggregator._extract_keywords("tariff tariff oil gas")
    assert NewsAggregator._score_relevance(a, kws) == pytest.approx(1 / 3)


async def test_partial_failure_does_not_pin_cache_for_full_ttl(world):
    world.feeds["bad"] = RuntimeError("blip")
    world.feeds["good"] = rss([item("from good")])
    agg = world.make("bad", "good")
    await agg.fetch_all()
    world.feeds["bad"] = rss([item("from bad, recovered")])
    world.clock.t += 60
    assert "from bad, recovered" in [a.title for a in await agg.fetch_all()]


def test_two_letter_stopwords_are_still_not_keywords():
    kws = NewsAggregator._extract_keywords("Is it up or in the US vs EU am I")
    assert kws == ["eu"]  # "us" is a stop word today; see report


def test_keyword_dedupe_keeps_first_seen_order():
    assert NewsAggregator._extract_keywords("oil gas oil ai gas") == ["oil", "gas", "ai"]


async def test_partial_result_is_trusted_only_briefly_and_not_refetched_each_call(world):
    world.feeds["bad"] = RuntimeError("blip")
    world.feeds["good"] = rss([item("from good")])
    agg = world.make("bad", "good")
    await agg.fetch_all()
    n = len(world.calls)
    world.clock.t += na.PARTIAL_RETRY_SECONDS - 1  # inside the retry window: cached
    await agg.fetch_all()
    assert len(world.calls) == n
    world.clock.t += 2  # past it: retried
    await agg.fetch_all()
    assert len(world.calls) == n + 2


async def test_complete_result_is_still_cached_for_the_full_ttl(world):
    world.feeds["a"] = rss([item("one")])
    agg = world.make("a")
    await agg.fetch_all()
    world.clock.t += 29 * 60
    await agg.fetch_all()
    assert world.calls == ["a"]
    world.clock.t += 2 * 60
    await agg.fetch_all()
    assert world.calls == ["a", "a"]


async def test_feed_fetch_passes_a_finite_timeout(world):
    world.feeds["a"] = rss([item("one")])
    await world.make("a").fetch_all()
    t = world.kwargs[0]["timeout"]
    assert t is not None and 0 < t <= 60


async def test_hung_feed_times_out_and_is_isolated(world, monkeypatch):
    world.feeds["good"] = rss([item("from good")])
    real_fake = na.httpx.get

    hung_timeouts = []

    def hanging_get(url, **kw):
        if url == "hung":
            # A real hang would block for ever; a bounded client raises instead.
            hung_timeouts.append(kw.get("timeout"))
            if not kw.get("timeout"):
                raise AssertionError("would hang: no timeout")
            raise httpx.ReadTimeout("hung")
        return real_fake(url, **kw)

    monkeypatch.setattr(na.httpx, "get", hanging_get)
    agg = world.make("hung", "good")
    assert [a.title for a in await agg.fetch_all()] == ["from good"]
    assert hung_timeouts and all(hung_timeouts)  # fetched with a timeout, which fired


async def test_http_error_status_is_a_failed_feed(world, monkeypatch):
    def get_404(url, **kw):
        def boom():
            raise httpx.HTTPStatusError("404", request=None, response=None)
        return SimpleNamespace(content=b"<html>not found</html>", raise_for_status=boom)

    monkeypatch.setattr(na.httpx, "get", get_404)
    with pytest.raises(httpx.HTTPStatusError):
        await world.make("a").fetch_feed("a")
