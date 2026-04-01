"""
Unified model routing layer for the Kalshi AI Trading Bot.

Routes ALL requests through OpenRouter — single API key, full model fleet.
Provides aggregate cost tracking, daily budget enforcement, and transparent
fallback across the full model fleet.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.clients.xai_client import TradingDecision, DailyUsageTracker
from src.clients.openrouter_client import OpenRouterClient, MODEL_PRICING
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


# ---------------------------------------------------------------------------
# Capability-to-model mapping — all via OpenRouter (April 2026 models)
# ---------------------------------------------------------------------------

CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [
        ("x-ai/grok-4.1-fast", "openrouter"),
        ("google/gemini-3.1-pro", "openrouter"),
    ],
    "cheap": [
        ("deepseek/deepseek-v3.2", "openrouter"),
        ("google/gemini-3.1-pro", "openrouter"),
    ],
    "reasoning": [
        ("anthropic/claude-sonnet-4.5", "openrouter"),
        ("openai/gpt-5.4", "openrouter"),
        ("google/gemini-3.1-pro", "openrouter"),
    ],
    "balanced": [
        ("anthropic/claude-sonnet-4.5", "openrouter"),
        ("openai/gpt-5.4", "openrouter"),
        ("x-ai/grok-4.1-fast", "openrouter"),
    ],
}

# Full fleet: ordered by quality/priority for fallback chains.
FULL_FLEET: List[Tuple[str, str]] = [
    ("anthropic/claude-sonnet-4.5", "openrouter"),
    ("google/gemini-3.1-pro", "openrouter"),
    ("openai/gpt-5.4", "openrouter"),
    ("deepseek/deepseek-v3.2", "openrouter"),
    ("x-ai/grok-4.1-fast", "openrouter"),
]


# ---------------------------------------------------------------------------
# Per-model health tracking
# ---------------------------------------------------------------------------

@dataclass
class ModelHealth:
    """Tracks success/failure rates for a single model."""
    model: str
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_latency: float = 0.0  # cumulative seconds

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0  # Assume healthy until proven otherwise
        return self.successful_requests / self.total_requests

    @property
    def avg_latency(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests

    @property
    def is_healthy(self) -> bool:
        """
        A model is considered unhealthy if it has 5+ consecutive failures
        and the last failure was within the past 5 minutes.
        """
        if self.consecutive_failures < 5:
            return True
        if self.last_failure_time is None:
            return True
        cooldown = timedelta(minutes=5)
        return datetime.now() - self.last_failure_time > cooldown

    def record_success(self, latency: float) -> None:
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        self.total_latency += latency

    def record_failure(self) -> None:
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter(TradingLoggerMixin):
    """
    Unified routing layer that dispatches ALL AI requests through OpenRouter.

    Usage::

        router = ModelRouter()
        # By capability
        text = await router.get_completion("prompt", capability="fast")
        # By explicit model
        text = await router.get_completion("prompt", model="openai/gpt-5.4")
        # Trading decision
        decision = await router.get_trading_decision(market, portfolio)
    """

    def __init__(
        self,
        openrouter_client: Optional[OpenRouterClient] = None,
        db_manager: Any = None,
        # xai_client param accepted for backward compat but ignored — all routing
        # now goes through OpenRouter.
        xai_client: Any = None,
    ):
        self.db_manager = db_manager

        # Single provider: OpenRouter
        self.openrouter_client: Optional[OpenRouterClient] = openrouter_client

        # Daily cost tracking (persisted via pickle, shared with OpenRouterClient)
        self.daily_tracker: DailyUsageTracker = self._load_daily_tracker()

        # Build health trackers for the full fleet
        self.model_health: Dict[str, ModelHealth] = {}
        for model_name, provider in FULL_FLEET:
            key = self._model_key(model_name, provider)
            self.model_health[key] = ModelHealth(model=model_name, provider=provider)

        self.logger.info(
            "ModelRouter initialized — all models via OpenRouter",
            openrouter_available=self.openrouter_client is not None,
            fleet_size=len(FULL_FLEET),
        )

    # ------------------------------------------------------------------
    # Daily cost tracking
    # ------------------------------------------------------------------

    def _load_daily_tracker(self) -> DailyUsageTracker:
        """Load or create daily usage tracker (shared with OpenRouterClient)."""
        import os
        import pickle

        today = datetime.now().strftime("%Y-%m-%d")
        usage_file = "logs/daily_ai_usage.pkl"
        daily_limit = getattr(settings.trading, "daily_ai_cost_limit", 10.0)

        os.makedirs("logs", exist_ok=True)

        try:
            if os.path.exists(usage_file):
                with open(usage_file, "rb") as f:
                    tracker = pickle.load(f)
                if tracker.date != today:
                    tracker = DailyUsageTracker(date=today, daily_limit=daily_limit)
                else:
                    tracker.daily_limit = daily_limit
                    if tracker.is_exhausted and tracker.total_cost < daily_limit:
                        tracker.is_exhausted = False
                return tracker
        except Exception as e:
            self.logger.warning(f"Failed to load daily tracker: {e}")

        return DailyUsageTracker(date=today, daily_limit=daily_limit)

    def _save_daily_tracker(self) -> None:
        import os
        import pickle

        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/daily_ai_usage.pkl", "wb") as f:
                pickle.dump(self.daily_tracker, f)
        except Exception as e:
            self.logger.error(f"Failed to save daily tracker: {e}")

    def _update_daily_cost(self, cost: float) -> None:
        """Update daily cost tracking."""
        self.daily_tracker.total_cost += cost
        self.daily_tracker.request_count += 1
        self._save_daily_tracker()

        if self.daily_tracker.total_cost >= self.daily_tracker.daily_limit:
            self.daily_tracker.is_exhausted = True
            self.daily_tracker.last_exhausted_time = datetime.now()
            self._save_daily_tracker()
            self.logger.warning(
                "Daily AI cost limit reached — trading paused until tomorrow.",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit,
            )

    async def check_daily_limits(self) -> bool:
        """
        Returns True if we can proceed with AI calls, False if daily limit reached.
        Beast-mode-bot calls this before each trading cycle.
        """
        # Reload to catch changes from other processes
        self.daily_tracker = self._load_daily_tracker()

        if self.daily_tracker.is_exhausted:
            now = datetime.now()
            if self.daily_tracker.date != now.strftime("%Y-%m-%d"):
                # New day — reset
                self.daily_tracker = DailyUsageTracker(
                    date=now.strftime("%Y-%m-%d"),
                    daily_limit=self.daily_tracker.daily_limit,
                )
                self._save_daily_tracker()
                self.logger.info("New day — daily AI limits reset")
                return True

            self.logger.info(
                "Daily AI limit reached — request skipped",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_key(model: str, provider: str) -> str:
        return f"{provider}::{model}"

    def _ensure_openrouter(self) -> OpenRouterClient:
        """Return the OpenRouter client, creating it on first use if needed."""
        if self.openrouter_client is None:
            self.openrouter_client = OpenRouterClient(db_manager=self.db_manager)
            self.logger.info("Lazily initialized OpenRouterClient")
        return self.openrouter_client

    def _infer_provider(self, model: str) -> str:
        """All models route through OpenRouter."""
        return "openrouter"

    def _resolve_targets(
        self,
        model: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Produce an ordered list of (model, provider) tuples to attempt.

        Priority:
        1. Explicit *model* (+ fallback chain).
        2. *capability* mapping (+ fallback chain).
        3. Full fleet sorted by health / success rate.
        """
        targets: List[Tuple[str, str]] = []

        if model is not None:
            provider = self._infer_provider(model)
            targets.append((model, provider))
        elif capability is not None:
            cap_targets = CAPABILITY_MAP.get(capability, [])
            targets.extend(cap_targets)
        else:
            targets = list(FULL_FLEET)

        # Append remaining fleet members not yet in the list
        seen = set(targets)
        for entry in FULL_FLEET:
            if entry not in seen:
                targets.append(entry)
                seen.add(entry)

        # Filter out unhealthy models (keep at least 2)
        healthy = [t for t in targets if self._is_model_healthy(t[0], t[1])]
        if len(healthy) >= 2:
            targets = healthy

        return targets

    def _is_model_healthy(self, model: str, provider: str) -> bool:
        key = self._model_key(model, provider)
        health = self.model_health.get(key)
        if health is None:
            return True
        return health.is_healthy

    def _record_success(self, model: str, provider: str, latency: float) -> None:
        key = self._model_key(model, provider)
        health = self.model_health.get(key)
        if health:
            health.record_success(latency)

    def _record_failure(self, model: str, provider: str) -> None:
        key = self._model_key(model, provider)
        health = self.model_health.get(key)
        if health:
            health.record_failure()

    # ------------------------------------------------------------------
    # Dispatch helpers — all through OpenRouter
    # ------------------------------------------------------------------

    async def _dispatch_completion(
        self,
        prompt: str,
        model: str,
        provider: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        market_id: Optional[str] = None,
    ) -> Optional[str]:
        """Send a completion request through OpenRouter."""
        client = self._ensure_openrouter()
        result = await client.get_completion(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            strategy=strategy,
            query_type=query_type,
            market_id=market_id,
        )
        # Update router-level daily tracker from openrouter client's last cost
        if result is not None and hasattr(client, "_last_request_cost"):
            self._update_daily_cost(client._last_request_cost)
        return result

    async def _dispatch_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
        model: str,
        provider: str,
    ) -> Optional[TradingDecision]:
        """Request a trading decision through OpenRouter."""
        client = self._ensure_openrouter()
        decision = await client.get_trading_decision(
            market_data=market_data,
            portfolio_data=portfolio_data,
            news_summary=news_summary,
            model=model,
        )
        if decision is not None and hasattr(client, "_last_request_cost"):
            self._update_daily_cost(client._last_request_cost)
        return decision

    # ------------------------------------------------------------------
    # Public API: get_completion
    # ------------------------------------------------------------------

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        capability: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        market_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a completion routed to the best available model via OpenRouter.

        Args:
            prompt: The user/system prompt.
            model: Explicit model identifier (e.g. ``"anthropic/claude-sonnet-4.5"``).
            capability: Capability hint — ``"fast"``, ``"reasoning"``,
                ``"balanced"``, or ``"cheap"``.  Ignored if *model* is given.
            temperature: Sampling temperature override.
            max_tokens: Max output tokens override.
            strategy: Strategy label for logging.
            query_type: Query type label for logging.
            market_id: Optional Kalshi market id for logging.

        Returns:
            Response text, or ``None`` if all models fail.
        """
        targets = self._resolve_targets(model=model, capability=capability)

        for target_model, provider in targets:
            start = time.time()
            try:
                result = await self._dispatch_completion(
                    prompt=prompt,
                    model=target_model,
                    provider=provider,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    strategy=strategy,
                    query_type=query_type,
                    market_id=market_id,
                )

                if result is not None:
                    self._record_success(target_model, provider, time.time() - start)
                    self.logger.debug(
                        "Completion routed successfully",
                        model=target_model,
                        provider=provider,
                        latency=round(time.time() - start, 2),
                    )
                    return result

                self._record_failure(target_model, provider)
                self.logger.warning(
                    "Model returned None, trying next",
                    model=target_model,
                    provider=provider,
                )

            except Exception as exc:
                self._record_failure(target_model, provider)
                self.logger.warning(
                    "Model failed during completion routing",
                    model=target_model,
                    provider=provider,
                    error=str(exc),
                )
                continue

        self.logger.error(
            "All models exhausted for get_completion",
            targets_tried=[(m, p) for m, p in targets],
        )
        return None

    # ------------------------------------------------------------------
    # Public API: get_trading_decision
    # ------------------------------------------------------------------

    async def get_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str = "",
        model: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> Optional[TradingDecision]:
        """
        Get a trading decision from the best available model via OpenRouter.

        Args:
            market_data: Market information dict.
            portfolio_data: Portfolio / balance information dict.
            news_summary: Optional news context.
            model: Explicit model identifier.
            capability: Capability hint (see ``get_completion``).

        Returns:
            A ``TradingDecision`` or ``None`` if all models fail.
        """
        targets = self._resolve_targets(model=model, capability=capability)

        for target_model, provider in targets:
            start = time.time()
            try:
                decision = await self._dispatch_trading_decision(
                    market_data=market_data,
                    portfolio_data=portfolio_data,
                    news_summary=news_summary,
                    model=target_model,
                    provider=provider,
                )

                if decision is not None:
                    self._record_success(target_model, provider, time.time() - start)
                    self.logger.info(
                        "Trading decision routed successfully",
                        model=target_model,
                        provider=provider,
                        action=decision.action,
                        confidence=decision.confidence,
                        latency=round(time.time() - start, 2),
                    )
                    return decision

                self._record_failure(target_model, provider)
                self.logger.warning(
                    "Model returned no decision, trying next",
                    model=target_model,
                    provider=provider,
                )

            except Exception as exc:
                self._record_failure(target_model, provider)
                self.logger.warning(
                    "Model failed during trading decision routing",
                    model=target_model,
                    provider=provider,
                    error=str(exc),
                )
                continue

        self.logger.error(
            "All models exhausted for get_trading_decision",
            targets_tried=[(m, p) for m, p in targets],
        )
        return None

    # ------------------------------------------------------------------
    # Aggregate cost tracking
    # ------------------------------------------------------------------

    def get_total_cost(self) -> float:
        """Return aggregate cost across all providers."""
        total = 0.0
        if self.openrouter_client:
            total += self.openrouter_client.total_cost
        return total

    def get_total_requests(self) -> int:
        """Return aggregate request count across all providers."""
        total = 0
        if self.openrouter_client:
            total += self.openrouter_client.request_count
        return total

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Return a comprehensive cost and health summary.
        """
        summary: Dict[str, Any] = {
            "total_cost": round(self.get_total_cost(), 6),
            "total_requests": self.get_total_requests(),
            "providers": {},
            "model_health": {},
            "daily": {
                "cost": round(self.daily_tracker.total_cost, 6),
                "limit": self.daily_tracker.daily_limit,
                "requests": self.daily_tracker.request_count,
                "is_exhausted": self.daily_tracker.is_exhausted,
            },
        }

        if self.openrouter_client:
            summary["providers"]["openrouter"] = self.openrouter_client.get_cost_summary()

        for key, health in self.model_health.items():
            if health.total_requests > 0:
                summary["model_health"][key] = {
                    "model": health.model,
                    "provider": health.provider,
                    "total_requests": health.total_requests,
                    "success_rate": round(health.success_rate, 4),
                    "avg_latency": round(health.avg_latency, 3),
                    "consecutive_failures": health.consecutive_failures,
                    "is_healthy": health.is_healthy,
                }

        return summary

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down all provider clients."""
        tasks = []
        if self.openrouter_client:
            tasks.append(self.openrouter_client.close())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info(
            "ModelRouter closed",
            total_cost=round(self.get_total_cost(), 6),
            total_requests=self.get_total_requests(),
        )
