"""
Market price normalization utilities.

Kalshi API v2 changed price fields from cent-based integers
(yes_ask, yes_bid, no_ask, no_bid) to dollar-based floats
(yes_ask_dollars, yes_bid_dollars, no_ask_dollars, no_bid_dollars).

This module provides a single helper that normalizes both formats
to dollar values (0.0–1.0).
"""
from typing import Dict, Any, Tuple


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
