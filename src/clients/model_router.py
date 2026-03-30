"""
Unified model routing layer for the Kalshi AI Trading Bot.

Routes requests to the appropriate AI provider (XAI / OpenRouter / direct OpenAI)
based on explicit model selection, capability requirements, or automatic load
balancing.  Provides aggregate cost tracking and transparent fallback across
the full model fleet.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.clients.xai_client import TradingDecision, XAIClient
from src.clients.openrouter_client import OpenRouterClient, MODEL_PRICING
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


# ---------------------------------------------------------------------------
# Capability-to-model mapping
# ---------------------------------------------------------------------------

# Each capability maps to an ordered preference list of (model, provider).
# The router will try them in order until one succeeds.
CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [
        ("google/gemini-2.5-flash-preview", "openrouter"),
        ("grok-4-1-fast-reasoning", "xai"),
    ],
    "cheap": [
        ("deepseek/deepseek-r1", "openrouter"),
        ("google/gemini-2.5-flash-preview", "openrouter"),
    ],
    "reasoning": [
        ("openai/o3", "openrouter"),
        ("google/gemini-2.5-pro-preview", "openrouter"),
        ("anthropic/claude-sonnet-4", "openrouter"),
    ],
    "balanced": [
        ("anthropic/claude-sonnet-4", "openrouter"),
        ("openai/gpt-4.1", "openrouter"),
        ("grok-4-1-fast-reasoning", "xai"),
    ],
}

# Full fleet: used when we need a fallback chain that spans all providers.
FULL_FLEET: List[Tuple[str, str]] = [
    ("grok-4-1-fast-reasoning", "xai"),
    ("anthropic/claude-sonnet-4", "openrouter"),
    ("openai/gpt-4.1", "openrouter"),
    ("google/gemini-2.5-pro-preview", "openrouter"),
    ("deepseek/deepseek-r1", "openrouter"),
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
    Unified routing layer that dispatches AI requests to the best available
    provider and model.

    Usage::

        router = ModelRouter()
        # By capability
        text = await router.get_completion("prompt", capability="fast")
        # By explicit model
        text = await router.get_completion("prompt", model="openai/gpt-4o")
        # Trading decision
        decision = await router.get_trading_decision(market, portfolio)
    """

    def __init__(
        self,
        xai_client: Optional[XAIClient] = None,
        openrouter_client: Optional[OpenRouterClient] = None,
        db_manager: Any = None,
    ):
        self.db_manager = db_manager

        # Lazily initialise provider clients
        self.xai_client: Optional[XAIClient] = xai_client
        self.openrouter_client: Optional[OpenRouterClient] = openrouter_client

        # Build health trackers for the full fleet
        self.model_health: Dict[str, ModelHealth] = {}
        for model_name, provider in FULL_FLEET:
            key = self._model_key(model_name, provider)
            self.model_health[key] = ModelHealth(model=model_name, provider=provider)

        self.logger.info(
            "ModelRouter initialized",
            xai_available=self.xai_client is not None,
            openrouter_available=self.openrouter_client is not None,
            fleet_size=len(FULL_FLEET),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_key(model: str, provider: str) -> str:
        return f"{provider}::{model}"

    def _ensure_xai(self) -> XAIClient:
        """Return the XAI client, creating it on first use if needed."""
        if self.xai_client is None:
            self.xai_client = XAIClient(db_manager=self.db_manager)
            self.logger.info("Lazily initialized XAIClient")
        return self.xai_client

    def _ensure_openrouter(self) -> OpenRouterClient:
        """Return the OpenRouter client, creating it on first use if needed."""
        if self.openrouter_client is None:
            self.openrouter_client = OpenRouterClient(db_manager=self.db_manager)
            self.logger.info("Lazily initialized OpenRouterClient")
        return self.openrouter_client

    def _infer_provider(self, model: str) -> str:
        """Determine which provider owns a model string."""
        # OpenRouter models use a slash-delimited namespace
        if "/" in model:
            return "openrouter"
        # Grok models go through XAI
        if model.startswith("grok"):
            return "xai"
        # Default: try openrouter (it can proxy many models)
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
            # No preference -- use full fleet ordered by success rate
            targets = list(FULL_FLEET)

        # Append remaining fleet members that are not yet in the list
        seen = set(targets)
        for entry in FULL_FLEET:
            if entry not in seen:
                targets.append(entry)
                seen.add(entry)

        # Filter out unhealthy models (but keep at least 2 options)
        healthy = [t for t in targets if self._is_model_healthy(t[0], t[1])]
        if len(healthy) >= 2:
            targets = healthy
        # else: keep the full list -- better to try an unhealthy model than nothing

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
    # Dispatch helpers
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
        """
        Send a completion request to the specified (model, provider) pair.
        Returns the response text or raises on failure.
        """
        if provider == "xai":
            client = self._ensure_xai()
            return await client.get_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy,
                query_type=query_type,
                market_id=market_id,
            )
        else:
            # openrouter (and anything else)
            client = self._ensure_openrouter()
            return await client.get_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy,
                query_type=query_type,
                market_id=market_id,
            )

    async def _dispatch_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
        model: str,
        provider: str,
    ) -> Optional[TradingDecision]:
        """
        Request a trading decision from the specified (model, provider) pair.
        """
        if provider == "xai":
            client = self._ensure_xai()
            return await client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary,
            )
        else:
            client = self._ensure_openrouter()
            return await client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary,
                model=model,
            )

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
        Get a completion routed to the best available model.

        Args:
            prompt: The user/system prompt.
            model: Explicit model identifier (e.g. ``"openai/gpt-4o"``).
            capability: Capability hint -- ``"fast"``, ``"reasoning"``,
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

                # result is None -- model returned nothing (e.g. daily limit)
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
        Get a trading decision from the best available model.

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

                # Decision was None -- parsing failed or model declined
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
        if self.xai_client:
            total += self.xai_client.total_cost
        if self.openrouter_client:
            total += self.openrouter_client.total_cost
        return total

    def get_total_requests(self) -> int:
        """Return aggregate request count across all providers."""
        total = 0
        if self.xai_client:
            total += self.xai_client.request_count
        if self.openrouter_client:
            total += self.openrouter_client.request_count
        return total

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Return a comprehensive cost and health summary.

        Includes per-provider cost breakdowns and per-model health stats.
        """
        summary: Dict[str, Any] = {
            "total_cost": round(self.get_total_cost(), 6),
            "total_requests": self.get_total_requests(),
            "providers": {},
            "model_health": {},
        }

        # XAI provider summary
        if self.xai_client:
            summary["providers"]["xai"] = {
                "total_cost": round(self.xai_client.total_cost, 6),
                "total_requests": self.xai_client.request_count,
                "daily_cost": round(self.xai_client.daily_tracker.total_cost, 6),
                "daily_limit": self.xai_client.daily_tracker.daily_limit,
            }

        # OpenRouter provider summary
        if self.openrouter_client:
            summary["providers"]["openrouter"] = self.openrouter_client.get_cost_summary()

        # Model health
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
        if self.xai_client:
            tasks.append(self.xai_client.close())
        if self.openrouter_client:
            tasks.append(self.openrouter_client.close())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info(
            "ModelRouter closed",
            total_cost=round(self.get_total_cost(), 6),
            total_requests=self.get_total_requests(),
        )
