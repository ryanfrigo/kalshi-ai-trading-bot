"""
Market Ingestion Job

This job fetches active markets from the Kalshi API, transforms them into a structured format,
and upserts them into the database.
"""
import asyncio
import math
import time
from datetime import datetime
from typing import Optional, List

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.utils.market_prices import is_tradeable_market

# Page caps shared by the events and /markets loops.
MAX_PAGES = 20


def _parse_quotes(market_data: dict):
    """Return (yes_bid, yes_ask, no_bid, no_ask) in dollars, or raise ValueError.

    All four fields must be present, non-None, parseable and finite; a missing
    field is a malformed row, not a 0 quote. An explicit 0 is a real quote.
    """
    if "yes_bid_dollars" in market_data:
        keys = ("yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars")
        scale = 1.0
    else:
        # Legacy API: values in cents (0-100)
        keys = ("yes_bid", "yes_ask", "no_bid", "no_ask")
        scale = 100.0
    quotes = []
    for k in keys:
        raw = market_data.get(k)
        if raw is None:
            raise ValueError(f"missing price field {k}")
        quote = float(raw) / scale
        if not math.isfinite(quote):
            raise ValueError(f"non-finite {k}={raw!r}")
        quotes.append(quote)
    return tuple(quotes)


def _build_market(market_data: dict, existing_position_market_ids: set, logger):
    """Transform one raw market row into a Market, or None to skip it.

    Raises KeyError/TypeError/ValueError on a malformed row; the caller skips it.
    """
    yes_bid, yes_ask, no_bid, no_ask = _parse_quotes(market_data)

    # A zero ask with a bid (or an empty book) would make the mid half the real
    # quote, so skip instead of inventing a price. A zero bid with a positive ask
    # (typical longshot, e.g. bid 0 / ask 0.03) is a real state and keeps the
    # bid/ask mid.
    if yes_ask <= 0 or no_ask <= 0 or yes_bid < 0 or no_bid < 0:
        logger.warning(
            f"Skipping {market_data['ticker']}: no two-sided quote "
            f"(yes {yes_bid:.4f}/{yes_ask:.4f}, no {no_bid:.4f}/{no_ask:.4f})"
        )
        return None
    yes_price = (yes_bid + yes_ask) / 2
    no_price = (no_bid + no_ask) / 2

    # Skip collection/aggregate tickers: both yes_ask and no_ask near $1.00
    # (e.g. KXMVECROSSCATEGORY-*, KXMVESPORTSMULTIGAMEEXTENDED-*).
    _COLLECTION_THRESHOLD = 0.98
    if yes_ask >= _COLLECTION_THRESHOLD and no_ask >= _COLLECTION_THRESHOLD:
        logger.warning(
            f"Skipping collection/aggregate market {market_data['ticker']}: "
            f"yes_ask={yes_ask:.4f}, no_ask={no_ask:.4f} "
            f"(both >= {_COLLECTION_THRESHOLD}, not a tradeable binary market)"
        )
        return None

    # Kalshi API v2 uses volume_fp (string/float) instead of volume (int)
    volume = int(float(market_data.get("volume_fp", 0) or market_data.get("volume", 0) or 0))

    has_position = market_data["ticker"] in existing_position_market_ids

    # Centralised in is_tradeable_market() so execute.py can reuse the same guard (issue #42).
    if not is_tradeable_market(market_data):
        logger.debug(f"Skipping collection ticker {market_data['ticker']} (YES_ask={yes_ask}, NO_ask={no_ask})")
        return None

    return Market(
        market_id=market_data["ticker"],
        title=market_data["title"],
        yes_price=yes_price,
        no_price=no_price,
        volume=volume,
        expiration_ts=int(
            datetime.fromisoformat(
                market_data["expiration_time"].replace("Z", "+00:00")
            ).timestamp()
        ),
        category=market_data.get("category") or "unknown",
        status=market_data["status"],
        last_updated=datetime.now(),
        has_position=has_position,
    )


async def process_and_queue_markets(
    markets_data: List[dict],
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    existing_position_market_ids: set,
    logger,
):
    """
    Transforms market data, upserts to DB, and puts eligible markets on the queue.
    """
    markets_to_upsert = []
    for market_data in markets_data:
        try:
            market = _build_market(market_data, existing_position_market_ids, logger)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as err:
            ticker = market_data.get("ticker") if isinstance(market_data, dict) else None
            logger.warning(f"Skipping malformed market row {ticker!r}: {type(err).__name__}: {err}")
            continue
        if market is not None:
            markets_to_upsert.append(market)

    if markets_to_upsert:
        await db_manager.upsert_markets(markets_to_upsert)
        logger.info(f"Successfully upserted {len(markets_to_upsert)} markets.")

        # Primary filtering criteria - MORE PERMISSIVE FOR MORE OPPORTUNITIES!
        min_volume: float = 100.0  # DECREASED: Much lower volume threshold (was 100, keeping low)
        min_volume_for_ai_analysis: float = 150.0  # DECREASED: Lower volume for AI analysis (was 200, now 150)  
        preferred_categories: List[str] = []  # Empty = all categories allowed
        excluded_categories: List[str] = []  # Empty = no categories excluded

        # Enhanced filtering for better opportunities - MORE PERMISSIVE FOR MORE TRADES
        min_price_movement: float = 0.015  # DECREASED: Even lower minimum range (was 0.02, now 1.5¢)
        max_bid_ask_spread: float = 0.20   # INCREASED: Allow even wider spreads (was 0.15, now 20¢)
        min_confidence_for_long_term: float = 0.40  # DECREASED: Lower confidence required (was 0.5, now 40%)

        eligible_markets = [
            m
            for m in markets_to_upsert
            if m.volume >= min_volume
            # REMOVED TIME RESTRICTION - we can now trade markets with ANY deadline!
            # Dynamic exit strategies will handle timing automatically
            and (
                not settings.trading.preferred_categories
                or m.category in settings.trading.preferred_categories
            )
            and m.category not in settings.trading.excluded_categories
        ]

        logger.info(
            f"Found {len(eligible_markets)} eligible markets to process in this batch."
        )
        for market in eligible_markets:
            await queue.put(market)

    else:
        logger.info("No new markets to upsert in this batch.")


async def run_ingestion(
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    market_ticker: Optional[str] = None,
):
    """
    Main function for the market ingestion job.

    Args:
        db_manager: DatabaseManager instance.
        queue: asyncio.Queue to put ingested markets into.
        market_ticker: Optional specific market ticker to ingest.
    """
    logger = get_trading_logger("market_ingestion")
    logger.info("Starting market ingestion job.", market_ticker=market_ticker)

    kalshi_client = KalshiClient()

    try:
        # Get all market IDs with existing positions
        existing_position_market_ids = await db_manager.get_markets_with_positions()

        if market_ticker:
            logger.info(f"Fetching single market: {market_ticker}")
            market_response = await kalshi_client.get_market(ticker=market_ticker)
            single = market_response.get("market") if market_response else None
            if single and single.get("status") != "active":
                logger.warning(
                    f"Market {market_ticker} is not active (status={single.get('status')!r}); skipping."
                )
            elif single:
                await process_and_queue_markets(
                    [single],
                    db_manager,
                    queue,
                    existing_position_market_ids,
                    logger,
                )
            else:
                logger.warning(f"Could not find market with ticker: {market_ticker}")
        else:
            # Primary: fetch via events API (Kalshi migrated all tickers to KXMVE*,
            # so /markets only returns parlay tickers. Real markets live under events.)
            logger.info("Fetching markets via events API (with nested markets).")
            seen_tickers = set()
            cursor = None
            events_page = 0
            try:
                while True:
                    params = {"status": "open", "limit": 100, "with_nested_markets": "true"}
                    if cursor:
                        params["cursor"] = cursor
                    
                    resp = await kalshi_client._make_authenticated_request(
                        "GET", "/trade-api/v2/events", params=params
                    )
                    events = resp.get("events", [])
                    if not events:
                        break
                    
                    batch = []
                    for event in events:
                        if not isinstance(event, dict):
                            continue
                        # Kalshi v2 carries `category` on the event only, not on nested markets.
                        event_category = event.get("category")
                        for m in event.get("markets") or []:
                            if not isinstance(m, dict):
                                continue
                            ticker = m.get("ticker", "")
                            if ticker and ticker not in seen_tickers and m.get("status") == "active":
                                seen_tickers.add(ticker)
                                if event_category and not m.get("category"):
                                    m = {**m, "category": event_category}
                                batch.append(m)
                    
                    if batch:
                        logger.info(f"Fetched {len(batch)} active markets from events page {events_page}.")
                        await process_and_queue_markets(
                            batch,
                            db_manager,
                            queue,
                            existing_position_market_ids,
                            logger,
                        )
                    
                    cursor = resp.get("cursor")
                    if not cursor:
                        break
                    events_page += 1
                    # Cap at MAX_PAGES pages (~2000 events, ~5000+ markets) to avoid
                    # 16+ minute ingestion that blocks the trading cycle.
                    # Most high-volume tradeable markets appear in the first pages.
                    if events_page >= MAX_PAGES:
                        logger.info(f"Reached page limit ({MAX_PAGES}), stopping ingestion with {len(seen_tickers)} markets.")
                        break
                    
                    await asyncio.sleep(0.1)
            except Exception as events_err:
                logger.warning(f"Events API failed, falling back to /markets: {events_err}")
            
            # Fallback: also check /markets for anything missed
            if len(seen_tickers) < 100:
                logger.info(f"Few markets from events ({len(seen_tickers)}), also fetching /markets.")
                cursor = None
                seen_cursors = set()
                fallback_pages = 0
                while True:
                    response = await kalshi_client.get_markets(limit=100, cursor=cursor)
                    fallback_pages += 1
                    markets_page = response.get("markets") or []

                    active_markets = [m for m in markets_page if isinstance(m, dict)
                                     and m.get("status") == "active"
                                     and m.get("ticker", "") not in seen_tickers]
                    if active_markets:
                        for m in active_markets:
                            seen_tickers.add(m.get("ticker", ""))
                        logger.info(
                            f"Fetched {len(markets_page)} markets, {len(active_markets)} new active."
                        )
                        await process_and_queue_markets(
                            active_markets,
                            db_manager,
                            queue,
                            existing_position_market_ids,
                            logger,
                        )

                    cursor = response.get("cursor")
                    if not cursor:
                        break
                    if cursor in seen_cursors:
                        logger.warning(f"/markets repeated cursor {cursor!r}, stopping.")
                        break
                    seen_cursors.add(cursor)
                    if fallback_pages >= MAX_PAGES:
                        logger.info(f"Reached /markets page limit ({MAX_PAGES}), stopping.")
                        break
            
            logger.info(f"Total unique markets ingested: {len(seen_tickers)}")

    except Exception as e:
        logger.error(
            "An error occurred during market ingestion.", error=str(e), exc_info=True
        )
    finally:
        await kalshi_client.close()
        logger.info("Market ingestion job finished.")
