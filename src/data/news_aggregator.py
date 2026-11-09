"""
Multi-source news aggregator for the Kalshi trading system.
Fetches articles from configured RSS feeds, deduplicates them, and provides
relevance-scored article retrieval for market sentiment analysis.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import feedparser
import httpx

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin

# feedparser.parse(url) has no timeout, so one hung feed would hang fetch_all.
# Fetch the bytes ourselves with a bound and hand them to feedparser.
FEED_TIMEOUT_SECONDS = 10.0

# A fetch_all that lost a feed is only trusted this long, not the full TTL.
PARTIAL_RETRY_SECONDS = 30


@dataclass
class NewsArticle:
    """Represents a single news article fetched from an RSS feed."""
    title: str
    summary: str
    source: str
    published: Optional[datetime]
    url: str
    category: str = ""

    @property
    def normalized_title(self) -> str:
        """Return a lowercase, stripped version of the title for deduplication."""
        return self.title.lower().strip()


class NewsAggregator(TradingLoggerMixin):
    """
    Multi-source news fetcher and aggregator.
    Fetches articles from configured RSS feeds in parallel, deduplicates them,
    and provides relevance-scored article lookup for market analysis.
    """

    def __init__(self) -> None:
        self._cache: List[NewsArticle] = []
        self._cache_timestamp: float = 0.0
        self._feed_timestamps: Dict[str, float] = {}
        self._cache_ttl_seconds: int = settings.sentiment.cache_ttl_minutes * 60
        self._max_articles_per_source: int = settings.sentiment.max_articles_per_source
        self._feeds: List[str] = settings.sentiment.rss_feeds

        self.logger.info(
            "NewsAggregator initialized",
            num_feeds=len(self._feeds),
            cache_ttl_minutes=settings.sentiment.cache_ttl_minutes,
            max_articles_per_source=self._max_articles_per_source,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_all(self) -> List[NewsArticle]:
        """
        Fetch articles from every configured RSS feed in parallel.
        Returns cached results if the cache is still valid.
        """
        now = time.monotonic()

        if self._cache and (now - self._cache_timestamp) < self._cache_ttl_seconds:
            self.logger.debug(
                "Returning cached articles",
                num_articles=len(self._cache),
                cache_age_seconds=round(now - self._cache_timestamp, 1),
            )
            return self._cache

        self.logger.info("Fetching articles from all RSS feeds", num_feeds=len(self._feeds))

        tasks = [self._fetch_feed_async(url) for url in self._feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: List[NewsArticle] = []
        any_failed = False
        for url, result in zip(self._feeds, results):
            if isinstance(result, Exception):
                any_failed = True
                self.logger.warning(
                    "Feed fetch failed, continuing with remaining feeds",
                    feed_url=url,
                    error=str(result),
                )
                continue
            all_articles.extend(result)

        deduplicated = self._deduplicate(all_articles)
        self._cache = deduplicated
        # Don't let a partial result stand for the whole TTL: backdate the stamp
        # so the failed feed is retried after PARTIAL_RETRY_SECONDS.
        self._cache_timestamp = now - max(
            0, self._cache_ttl_seconds - PARTIAL_RETRY_SECONDS
        ) if any_failed else now

        self.logger.info(
            "Feed fetch complete",
            total_raw=len(all_articles),
            after_dedup=len(deduplicated),
        )
        return deduplicated

    async def fetch_feed(self, url: str) -> List[NewsArticle]:
        """
        Fetch and parse a single RSS feed.
        Respects per-feed rate limiting based on cache TTL.
        """
        now = time.monotonic()
        last_fetch = self._feed_timestamps.get(url)

        if last_fetch is not None and (now - last_fetch) < self._cache_ttl_seconds:
            self.logger.debug("Skipping feed (rate limited)", feed_url=url)
            return []

        return await self._fetch_feed_async(url)

    def get_relevant_articles(
        self,
        market_title: str,
        max_articles: int = 5,
    ) -> List[Tuple[NewsArticle, float]]:
        """
        Return articles from the cache scored by relevance to *market_title*.

        Uses keyword overlap scoring: extract significant terms from the market
        title, then score each article by the fraction of those terms that
        appear in the article's title + summary.

        Returns:
            List of (NewsArticle, relevance_score) tuples sorted by score desc.
            Only articles meeting the configured relevance_threshold are included.
        """
        keywords = self._extract_keywords(market_title)
        if not keywords:
            return []

        scored: List[Tuple[NewsArticle, float]] = []
        for article in self._cache:
            score = self._score_relevance(article, keywords)
            if score >= settings.sentiment.relevance_threshold:
                scored.append((article, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:max_articles]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_feed_async(self, url: str) -> List[NewsArticle]:
        """Run the synchronous feedparser in a thread to avoid blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._parse_feed, url)

    def _parse_feed(self, url: str) -> List[NewsArticle]:
        """Parse a single RSS feed URL and return a list of NewsArticle objects."""
        try:
            resp = httpx.get(
                url,
                timeout=FEED_TIMEOUT_SECONDS,
                follow_redirects=True,
                headers={"User-Agent": feedparser.USER_AGENT},
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)

            if feed.bozo and not feed.entries:
                raise ValueError(
                    f"Feed parsing error: {feed.bozo_exception}"
                )

            source_name = feed.feed.get("title", url)
            articles: List[NewsArticle] = []

            for entry in feed.entries[: self._max_articles_per_source]:
                published = self._parse_published_date(entry)
                category = ""
                if hasattr(entry, "tags") and entry.tags:
                    category = entry.tags[0].get("term", "")

                summary = entry.get("summary", entry.get("description", ""))
                # Strip HTML tags from summary
                summary = re.sub(r"<[^>]+>", "", summary).strip()

                articles.append(
                    NewsArticle(
                        title=entry.get("title", "").strip(),
                        summary=summary,
                        source=source_name,
                        published=published,
                        url=entry.get("link", ""),
                        category=category,
                    )
                )

            self._feed_timestamps[url] = time.monotonic()

            self.logger.debug(
                "Parsed feed successfully",
                feed_url=url,
                source=source_name,
                num_articles=len(articles),
            )
            return articles

        except Exception as exc:
            self.logger.error(
                "Error parsing RSS feed",
                feed_url=url,
                error=str(exc),
            )
            raise

    @staticmethod
    def _parse_published_date(entry: dict) -> Optional[datetime]:
        """Attempt to extract a datetime from a feedparser entry."""
        published_parsed = entry.get("published_parsed")
        if published_parsed:
            try:
                return datetime(*published_parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _deduplicate(articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Remove duplicate articles based on normalized title. An untitled
        article is keyed by its link instead; one with neither is never merged.
        """
        seen: set = set()
        unique: List[NewsArticle] = []
        for article in articles:
            key = article.normalized_title or article.url.strip()
            if not key:
                unique.append(article)
            elif key not in seen:
                seen.add(key)
                unique.append(article)
        return unique

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """
        Extract significant keywords from text for relevance matching.
        Filters out common stop words and one-character tokens. Two-letter
        tokens are kept (\"ai\", \"eu\", \"uk\") unless they are stop words, and
        repeated words count once.
        """
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "to", "for", "with", "on", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "about",
            "up", "down", "and", "but", "or", "if", "while", "what",
            "which", "who", "whom", "this", "that", "these", "those",
            "it", "its", "they", "them", "their", "we", "us", "our",
            "you", "your", "he", "him", "his", "she", "her", "my",
            "am", "vs",
        }
        # Tokenize: lowercase, keep only alphanumeric
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        # dict.fromkeys de-duplicates while keeping first-seen order
        return list(dict.fromkeys(
            t for t in tokens if t not in stop_words and len(t) >= 2
        ))

    @staticmethod
    def _score_relevance(article: NewsArticle, keywords: List[str]) -> float:
        """
        Score an article's relevance to a set of keywords.
        Returns a float between 0.0 and 1.0 representing the fraction of
        keywords found in the article's title + summary.
        """
        if not keywords:
            return 0.0

        article_text = f"{article.title} {article.summary}".lower()
        matches = sum(
            1 for kw in keywords
            if re.search(rf"\b{re.escape(kw)}\b", article_text)
        )
        return matches / len(keywords)
