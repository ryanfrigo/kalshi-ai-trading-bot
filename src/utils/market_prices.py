"""
Market price normalization utilities.

Kalshi API v2 changed price fields from cent-based integers
(yes_ask, yes_bid, no_ask, no_bid) to dollar-based floats
(yes_ask_dollars, yes_bid_dollars, no_ask_dollars, no_bid_dollars).

This module provides a single helper that normalizes both formats
to dollar values (0.0–1.0), plus a guard for non-tradeable
collection/aggregate tickers (issue #42).
"""
from typing import Dict, Any, Tuple

# Threshold above which both sides signal a collection/aggregate ticker
# that is not directly tradeable on Kalshi (e.g. KXMVECROSSCATEGORY-*).
_COLLECTION_TICKER_THRESHOLD = 0.99


def is_tradeable_market(market_info: Dict[str, Any]) -> bool:
    """
    Return False if this market looks like a collection/aggregate ticker.

    Collection tickers (e.g. KXMVECROSSCATEGORY-*, KXMVESPORTSMULTIGAMEEXTENDED-*)
    are not directly tradeable.  The Kalshi API returns yes_ask == no_ask == $1.00
    for these markets.  Attempting to place an order against them results in an
    HTTP 400 ``invalid_price`` error.

    A market is considered non-tradeable when BOTH ask prices are at or above
    the collection-ticker threshold (``>= 0.99``).

    Args:
        market_info: Raw market dict from the Kalshi API.

    Returns:
        True  – market is tradeable.
        False – market is a collection/aggregate ticker; skip it.
    """
    _, yes_ask, _, no_ask = get_market_prices(market_info)
    if yes_ask >= _COLLECTION_TICKER_THRESHOLD and no_ask >= _COLLECTION_TICKER_THRESHOLD:
        return False
    return True


def get_market_prices(market_info: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Extract and normalize market prices from a Kalshi API market object.

    Supports both:
      - API v2: yes_bid_dollars / yes_ask_dollars / no_bid_dollars / no_ask_dollars
                (float, already in dollars, e.g. 0.52 = $0.52)
      - Legacy: yes_bid / yes_ask / no_bid / no_ask
                (integer cents, e.g. 52 = $0.52)

    Returns:
        (yes_bid, yes_ask, no_bid, no_ask) all as dollar floats in [0.0, 1.0].
    """
    if "yes_bid_dollars" in market_info:
        yes_bid = float(market_info.get("yes_bid_dollars", 0) or 0)
        yes_ask = float(market_info.get("yes_ask_dollars", 0) or 0)
        no_bid  = float(market_info.get("no_bid_dollars",  0) or 0)
        no_ask  = float(market_info.get("no_ask_dollars",  0) or 0)
    else:
        # Legacy API: values in cents (0–100)
        yes_bid = (market_info.get("yes_bid", 0) or 0) / 100
        yes_ask = (market_info.get("yes_ask", 0) or 0) / 100
        no_bid  = (market_info.get("no_bid",  0) or 0) / 100
        no_ask  = (market_info.get("no_ask",  0) or 0) / 100

    return yes_bid, yes_ask, no_bid, no_ask


def signed_position(row: Dict[str, Any]) -> float:
    """
    Signed contract count from a Kalshi position row.

    API v2 rows carry a signed string ``position_fp`` ("-40.00" = 40 NO,
    "+" = YES) and no ``position`` key. Returns 0.0 when the field is absent.
    """
    return float(row.get("position_fp", 0) or 0)


def held_side_mid(market_info: Dict[str, Any], signed: float) -> float:
    """
    Mid price in dollars of the side a signed position holds.

    ``signed > 0`` is YES, otherwise NO. ``market_info`` is the unwrapped
    market object (the ``"market"`` value of ``get_market``).
    """
    yes_bid, yes_ask, no_bid, no_ask = get_market_prices(market_info)
    if signed > 0:
        return (yes_bid + yes_ask) / 2
    return (no_bid + no_ask) / 2
