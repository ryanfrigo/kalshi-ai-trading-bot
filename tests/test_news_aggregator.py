"""Behaviour tests for src/data/news_aggregator.py.

No network: ``feedparser.parse`` is replaced by a fake that serves canned RSS
text (or raises) per feed URL. The real feedparser still parses the canned text,
so parsing behaviour is the real thing.
"""

from types import SimpleNamespace

import pytest

# feedparser is imported by the module but not declared in pyproject.toml (see
# tests/test_news_aggregator_deps.py). Skip rather than error where it is absent.
pytest.importorskip("feedparser")

from src.config.settings import settings  # noqa: E402
from src.data import news_aggregator as na  # noqa: E402
from src.data.news_aggregator import NewsAggregator, NewsArticle  # noqa: E402

_real_parse = na.feedparser.parse


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
        self.t = 100_000.0  # host up ~28h; see the early-uptime xfail below

    def monotonic(self):
        return self.t


@pytest.fixture
def world(monkeypatch):
    """feeds: url -> rss text | Exception. calls: list of parsed urls."""
    feeds, calls = {}, []

    def fake_parse(url, *a, **kw):
        calls.append(url)
        v = feeds[url]
        if isinstance(v, Exception):
            raise v
        return _real_parse(v)

    monkeypatch.setattr(na.feedparser, "parse", fake_parse)
    clock = Clock()
    monkeypatch.setattr(na, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(settings.sentiment, "cache_ttl_minutes", 30)
    monkeypatch.setattr(settings.sentiment, "max_articles_per_source", 10)
    monkeypatch.setattr(settings.sentiment, "relevance_threshold", 0.3)

    def make(*urls):
        monkeypatch.setattr(settings.sentiment, "rss_feeds", list(urls))
        return NewsAggregator()

    return SimpleNamespace(feeds=feeds, calls=calls, clock=clock, make=make)


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


# ----------------------------------------------------- suspected real bugs
# Each is pinned strict-xfail with the behaviour a correct implementation has.


@pytest.mark.xfail(strict=True, reason=(
    "news_aggregator.py:268 uses substring `kw in article_text`, so 'fed' matches "
    "'federal', 'war' matches 'software'; unrelated articles score as relevant"))
def test_relevance_matches_whole_words_not_substrings():
    a = art("Software company ships federal compliance update")
    assert NewsAggregator._score_relevance(a, ["war", "fed"]) == 0.0


@pytest.mark.xfail(strict=True, reason=(
    "news_aggregator.py:255 `len(t) > 2` discards 2-letter tokens like 'ai', 'eu', "
    "'uk', 'un', which are decisive terms in market titles"))
def test_two_letter_acronyms_are_kept_as_keywords():
    kws = NewsAggregator._extract_keywords("Will the EU ban AI")
    assert "eu" in kws and "ai" in kws


@pytest.mark.xfail(strict=True, reason=(
    "news_aggregator.py:134-141 keywords are not de-duplicated, so a repeated word in "
    "the market title is counted twice and skews the score"))
def test_repeated_query_word_counts_once():
    a = art("Tariff news")
    assert NewsAggregator._score_relevance(a, NewsAggregator._extract_keywords("tariff tariff oil gas")) == 0.25


@pytest.mark.xfail(strict=True, reason=(
    "news_aggregator.py:225-227 dedupes on title only, so distinct untitled items "
    "(empty title) collapse into one article"))
def test_distinct_untitled_articles_are_not_merged():
    a = NewsArticle("", "story about oil", "s", None, "http://a")
    b = NewsArticle("", "story about gas", "s", None, "http://b")
    assert len(NewsAggregator._deduplicate([a, b])) == 2


@pytest.mark.xfail(strict=True, reason=(
    "news_aggregator.py:67,95 a partially failed fetch_all caches the survivors for "
    "the whole TTL, so a transiently failed feed is not retried for 30 minutes"))
async def test_partial_failure_does_not_pin_cache_for_full_ttl(world):
    world.feeds["bad"] = RuntimeError("blip")
    world.feeds["good"] = rss([item("from good")])
    agg = world.make("bad", "good")
    await agg.fetch_all()
    world.feeds["bad"] = rss([item("from bad, recovered")])
    world.clock.t += 60
    assert "from bad, recovered" in [a.title for a in await agg.fetch_all()]


@pytest.mark.xfail(strict=True, reason=(
    "news_aggregator.py:110,112 a never-fetched feed defaults to last_fetch=0.0, so "
    "while time.monotonic() < cache_ttl (host/container up under 30 min) fetch_feed "
    "treats it as rate limited and returns []"))
async def test_fetch_feed_works_shortly_after_boot(world):
    world.clock.t = 120.0
    world.feeds["f1"] = rss([item("first ever fetch")])
    assert [a.title for a in await world.make("f1").fetch_feed("f1")] == ["first ever fetch"]
