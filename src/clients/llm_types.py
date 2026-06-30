"""Neutral LLM data types shared across clients.

These dataclasses used to live in ``xai_client``, which made the live LLM
gateway (``openrouter_client``) depend on the legacy xAI shim just to borrow two
plain types. They carry no behavior and no heavy dependencies, so they live here
instead — letting ``openrouter_client`` (and the agent-native trade/verify paths
that use it) import them without dragging in the legacy stack. ``xai_client``
re-exports them for backward compatibility with the older callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TradingDecision:
    """Represents an AI trading decision."""
    action: str           # "buy", "sell", "hold"
    side: str             # "yes", "no"
    confidence: float     # 0.0 to 1.0
    limit_price: Optional[int] = None   # limit price in cents
    reasoning: Optional[str] = None


@dataclass
class DailyUsageTracker:
    """Track daily AI usage and costs."""
    date: str
    total_cost: float = 0.0
    request_count: int = 0
    daily_limit: float = 10.0  # Default $10/day (override via DAILY_AI_COST_LIMIT env var)
    is_exhausted: bool = False
    last_exhausted_time: Optional[datetime] = None
