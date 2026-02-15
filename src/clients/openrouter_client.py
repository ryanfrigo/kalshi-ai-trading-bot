"""
OpenRouter client for multi-model AI-powered trading decisions.
Routes requests through OpenRouter's unified API to access Claude, GPT-4o,
Gemini, DeepSeek, and other frontier models for market analysis.
"""

import asyncio
import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from json_repair import repair_json
from openai import AsyncOpenAI

from src.clients.xai_client import TradingDecision, DailyUsageTracker
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin, log_error_with_context


# ---------------------------------------------------------------------------
# Model registry: pricing per 1K tokens (USD)
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "anthropic/claude-sonnet-4.5": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
    },
    "openai/o3": {
        "input_per_1k": 0.002,
        "output_per_1k": 0.008,
    },
    "google/gemini-3-pro-preview": {
        "input_per_1k": 0.002,
        "output_per_1k": 0.012,
    },
    "google/gemini-3-flash-preview": {
        "input_per_1k": 0.0005,
        "output_per_1k": 0.003,
    },
    "deepseek/deepseek-v3.2": {
        "input_per_1k": 0.00025,
        "output_per_1k": 0.00038,
    },
}

# Ordered fallback chain -- if the requested model fails, try the next one.
DEFAULT_FALLBACK_ORDER: List[str] = [
    "anthropic/claude-sonnet-4.5",
    "openai/o3",
    "google/gemini-3-pro-preview",
    "deepseek/deepseek-v3.2",
]


# ---------------------------------------------------------------------------
# Per-model cost accumulator
# ---------------------------------------------------------------------------

@dataclass
class ModelCostTracker:
    """Accumulated cost data for a single model."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None


# ---------------------------------------------------------------------------
# OpenRouterClient
# ---------------------------------------------------------------------------

class OpenRouterClient(TradingLoggerMixin):
    """
    Async client that accesses multiple frontier models through OpenRouter.

    Provides the same interface as XAIClient (``get_completion`` and
    ``get_trading_decision``) so callers can swap providers transparently.

    Features:
        * Per-model cost tracking with model-specific pricing
        * Automatic fallback across models on failure
        * Exponential-backoff retry logic with rate-limit awareness
        * Daily cost tracking (mirrors DailyUsageTracker from xai_client)
    """

    # Maximum number of retries for a single model before moving to fallback
    MAX_RETRIES_PER_MODEL: int = 3
    # Base delay (seconds) for exponential backoff
    BASE_BACKOFF: float = 1.0
    # Cap on backoff delay (seconds)
    MAX_BACKOFF: float = 30.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "anthropic/claude-sonnet-4",
        db_manager: Any = None,
    ):
        self.api_key = api_key or settings.api.openrouter_api_key
        self.base_url = settings.api.openrouter_base_url
        self.default_model = default_model
        self.db_manager = db_manager

        # OpenAI-compatible async client pointed at OpenRouter
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,
            max_retries=0,  # We handle retries ourselves
        )

        # Default generation parameters
        self.temperature = settings.trading.ai_temperature
        self.max_tokens = settings.trading.ai_max_tokens

        # Per-model cost trackers
        self.model_costs: Dict[str, ModelCostTracker] = {
            m: ModelCostTracker(model=m) for m in MODEL_PRICING
        }

        # Aggregate cost tracking
        self.total_cost: float = 0.0
        self.request_count: int = 0

        # Daily usage tracker (same pattern as XAIClient)
        self.usage_file = "logs/daily_openrouter_usage.pkl"
        self.daily_tracker: DailyUsageTracker = self._load_daily_tracker()

        self.logger.info(
            "OpenRouter client initialized",
            default_model=self.default_model,
            available_models=list(MODEL_PRICING.keys()),
            daily_limit=self.daily_tracker.daily_limit,
            today_cost=self.daily_tracker.total_cost,
            today_requests=self.daily_tracker.request_count,
        )

    # ------------------------------------------------------------------
    # Daily usage persistence (mirrors XAIClient)
    # ------------------------------------------------------------------

    def _load_daily_tracker(self) -> DailyUsageTracker:
        """Load or create a daily usage tracker from disk."""
        today = datetime.now().strftime("%Y-%m-%d")
        os.makedirs("logs", exist_ok=True)

        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, "rb") as fh:
                    tracker: DailyUsageTracker = pickle.load(fh)
                if tracker.date != today:
                    tracker = DailyUsageTracker(
                        date=today,
                        daily_limit=tracker.daily_limit,
                    )
                return tracker
        except Exception as exc:
            self.logger.warning(f"Failed to load daily tracker: {exc}")

        daily_limit = getattr(settings.trading, "daily_ai_cost_limit", 50.0)
        return DailyUsageTracker(date=today, daily_limit=daily_limit)

    def _save_daily_tracker(self) -> None:
        """Persist the daily tracker to disk."""
        try:
            os.makedirs("logs", exist_ok=True)
            with open(self.usage_file, "wb") as fh:
                pickle.dump(self.daily_tracker, fh)
        except Exception as exc:
            self.logger.error(f"Failed to save daily tracker: {exc}")

    def _update_daily_cost(self, cost: float) -> None:
        """Add *cost* to the daily tracker and check the limit."""
        self.daily_tracker.total_cost += cost
        self.daily_tracker.request_count += 1
        self._save_daily_tracker()

        if self.daily_tracker.total_cost >= self.daily_tracker.daily_limit:
            self.daily_tracker.is_exhausted = True
            self.daily_tracker.last_exhausted_time = datetime.now()
            self._save_daily_tracker()
            self.logger.warning(
                "Daily OpenRouter cost limit reached",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit,
                requests_today=self.daily_tracker.request_count,
            )

    async def _check_daily_limits(self) -> bool:
        """Return True if we are within the daily spending limit."""
        self.daily_tracker = self._load_daily_tracker()

        if self.daily_tracker.is_exhausted:
            now = datetime.now()
            if self.daily_tracker.date != now.strftime("%Y-%m-%d"):
                # New day -- reset
                self.daily_tracker = DailyUsageTracker(
                    date=now.strftime("%Y-%m-%d"),
                    daily_limit=self.daily_tracker.daily_limit,
                )
                self._save_daily_tracker()
                self.logger.info(
                    "New day -- OpenRouter daily limits reset",
                    daily_limit=self.daily_tracker.daily_limit,
                )
                return True

            self.logger.info(
                "OpenRouter daily limit reached -- request skipped",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Return the USD cost for the given token counts."""
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            # Unknown model -- use a conservative default
            return (input_tokens + output_tokens) * 0.00001

        input_cost = (input_tokens / 1000.0) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000.0) * pricing["output_per_1k"]
        return input_cost + output_cost

    def _track_model_cost(
        self, model: str, input_tokens: int, output_tokens: int, cost: float
    ) -> None:
        """Update the per-model cost tracker."""
        tracker = self.model_costs.get(model)
        if tracker is None:
            tracker = ModelCostTracker(model=model)
            self.model_costs[model] = tracker

        tracker.input_tokens += input_tokens
        tracker.output_tokens += output_tokens
        tracker.total_cost += cost
        tracker.request_count += 1
        tracker.last_used = datetime.now()

    # ------------------------------------------------------------------
    # Rate-limit / error helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        error_str = str(exc).lower()
        return any(
            indicator in error_str
            for indicator in ["rate limit", "429", "too many requests", "quota"]
        )

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        error_str = str(exc).lower()
        return any(
            indicator in error_str
            for indicator in [
                "rate limit",
                "429",
                "too many requests",
                "timeout",
                "502",
                "503",
                "504",
                "server error",
                "internal error",
                "overloaded",
            ]
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay for *attempt* (0-based)."""
        delay = self.BASE_BACKOFF * (2 ** attempt)
        return min(delay, self.MAX_BACKOFF)

    # ------------------------------------------------------------------
    # Fallback chain helper
    # ------------------------------------------------------------------

    def _build_fallback_chain(self, requested_model: Optional[str] = None) -> List[str]:
        """
        Return an ordered list of models to try.  The *requested_model* is
        first, followed by the remaining models from DEFAULT_FALLBACK_ORDER.
        """
        first = requested_model or self.default_model
        chain = [first]
        for model in DEFAULT_FALLBACK_ORDER:
            if model not in chain:
                chain.append(model)
        return chain

    # ------------------------------------------------------------------
    # Core completion request (single model, with retries)
    # ------------------------------------------------------------------

    async def _request_single_model(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, float, int, int]:
        """
        Make a completion request to a single model with retries.

        Returns:
            (response_text, cost, input_tokens, output_tokens)

        Raises:
            Exception -- if all retries are exhausted for this model.
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        last_exc: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES_PER_MODEL):
            try:
                start = time.time()

                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                elapsed = time.time() - start

                # Validate response
                if (
                    not response.choices
                    or not response.choices[0].message
                    or not response.choices[0].message.content
                ):
                    raise ValueError(
                        f"Empty response from {model} on attempt {attempt + 1}"
                    )

                content = response.choices[0].message.content

                # Token usage
                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                cost = self._calculate_cost(model, input_tokens, output_tokens)

                self.logger.debug(
                    "OpenRouter completion succeeded",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=round(cost, 6),
                    processing_time=round(elapsed, 2),
                    attempt=attempt + 1,
                )

                return content, cost, input_tokens, output_tokens

            except Exception as exc:
                last_exc = exc

                # Track the error on the model
                tracker = self.model_costs.get(model)
                if tracker:
                    tracker.error_count += 1

                is_retryable = self._is_retryable_error(exc)

                self.logger.warning(
                    "OpenRouter request failed",
                    model=model,
                    attempt=attempt + 1,
                    max_retries=self.MAX_RETRIES_PER_MODEL,
                    retryable=is_retryable,
                    error=str(exc),
                )

                if is_retryable and attempt < self.MAX_RETRIES_PER_MODEL - 1:
                    delay = self._backoff_delay(attempt)
                    if self._is_rate_limit_error(exc):
                        delay *= 2  # Extra patience for rate limits
                    await asyncio.sleep(delay)
                elif not is_retryable:
                    # Non-retryable error -- bail immediately
                    break

        # All retries exhausted for this model
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API: get_completion
    # ------------------------------------------------------------------

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        market_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a completion from the best available OpenRouter model.

        Falls back through the model chain if the requested model fails.
        Returns None if daily limits are exceeded or all models fail.
        """
        if not await self._check_daily_limits():
            return None

        messages = [{"role": "user", "content": prompt}]
        fallback_chain = self._build_fallback_chain(model)

        for candidate_model in fallback_chain:
            try:
                content, cost, input_tok, output_tok = await self._request_single_model(
                    messages=messages,
                    model=candidate_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Update tracking
                self._track_model_cost(candidate_model, input_tok, output_tok, cost)
                self.total_cost += cost
                self.request_count += 1
                self._update_daily_cost(cost)

                # Log query if db_manager is available
                await self._log_query(
                    strategy=strategy,
                    query_type=query_type,
                    prompt=prompt,
                    response=content,
                    market_id=market_id,
                    tokens_used=input_tok + output_tok,
                    cost_usd=cost,
                )

                return content

            except Exception as exc:
                self.logger.warning(
                    "Model failed, trying next in fallback chain",
                    failed_model=candidate_model,
                    error=str(exc),
                )
                continue

        self.logger.error(
            "All OpenRouter models failed for get_completion",
            models_tried=fallback_chain,
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
    ) -> Optional[TradingDecision]:
        """
        Obtain a structured trading decision from an OpenRouter model.

        The method builds a prompt, queries the model (with fallback), and
        parses the JSON response into a ``TradingDecision`` object.
        """
        if not await self._check_daily_limits():
            return None

        prompt = self._build_trading_prompt(market_data, portfolio_data, news_summary)
        messages = [{"role": "user", "content": prompt}]
        fallback_chain = self._build_fallback_chain(model)

        for candidate_model in fallback_chain:
            try:
                content, cost, input_tok, output_tok = await self._request_single_model(
                    messages=messages,
                    model=candidate_model,
                    temperature=0.1,  # Low temperature for structured output
                    max_tokens=4000,
                )

                # Update tracking
                self._track_model_cost(candidate_model, input_tok, output_tok, cost)
                self.total_cost += cost
                self.request_count += 1
                self._update_daily_cost(cost)

                decision = self._parse_trading_decision(content)

                if decision is not None:
                    # Log the successful query
                    await self._log_query(
                        strategy="openrouter",
                        query_type="trading_decision",
                        prompt=prompt,
                        response=content,
                        tokens_used=input_tok + output_tok,
                        cost_usd=cost,
                        confidence_extracted=decision.confidence,
                        decision_extracted=decision.action,
                    )
                    return decision

                self.logger.warning(
                    "Failed to parse trading decision from model response",
                    model=candidate_model,
                    response_preview=content[:200] if content else "EMPTY",
                )

            except Exception as exc:
                self.logger.warning(
                    "Model failed for trading decision, trying fallback",
                    failed_model=candidate_model,
                    error=str(exc),
                )
                continue

        self.logger.error(
            "All OpenRouter models failed for get_trading_decision",
            models_tried=fallback_chain,
        )
        return None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_trading_prompt(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
    ) -> str:
        """Build a concise trading-decision prompt."""
        title = market_data.get("title", "Unknown Market")
        yes_price = (market_data.get("yes_bid", 0) + market_data.get("yes_ask", 100)) / 2
        no_price = (market_data.get("no_bid", 0) + market_data.get("no_ask", 100)) / 2
        volume = market_data.get("volume", 0)
        days_to_expiry = market_data.get("days_to_expiry", "Unknown")
        rules = market_data.get("rules", "No specific rules provided")

        cash = portfolio_data.get("cash", portfolio_data.get("balance", 1000))
        max_trade_value = portfolio_data.get(
            "max_trade_value",
            cash * settings.trading.max_position_size_pct / 100,
        )

        truncated_news = (
            news_summary[:800] + "..." if len(news_summary) > 800 else news_summary
        )

        return f"""Analyze this prediction market and provide a trading decision.

Market: {title}
Rules: {rules}
YES price: {yes_price}c | NO price: {no_price}c | Volume: ${volume:,.0f}
Days to expiry: {days_to_expiry}

Available cash: ${cash:,.2f} | Max trade value: ${max_trade_value:,.2f}

News/Context:
{truncated_news}

Instructions:
- Estimate the true probability of the event.
- Only trade if your estimated edge (|your_probability - market_price/100|) exceeds 10%.
- Confidence must be >60% to recommend a trade.
- Return ONLY a JSON object in the following format (no markdown, no extra text):

{{"action": "BUY", "side": "YES", "limit_price": 55, "confidence": 0.72, "reasoning": "brief explanation"}}

If you do not recommend trading, use action "SKIP":

{{"action": "SKIP", "side": "YES", "limit_price": 0, "confidence": 0.40, "reasoning": "insufficient edge"}}
"""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_trading_decision(self, response_text: str) -> Optional[TradingDecision]:
        """
        Extract a TradingDecision from model output.

        Handles JSON wrapped in markdown code fences or bare JSON objects.
        Falls back to json_repair for malformed output.
        """
        try:
            # Try markdown code fence first
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Bare JSON object
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    self.logger.warning(
                        "No JSON found in trading decision response",
                        response_preview=response_text[:300],
                    )
                    return None

            # Attempt standard parse first, then repair
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                repaired = repair_json(json_str)
                if repaired:
                    data = json.loads(repaired)
                else:
                    self.logger.warning("JSON repair returned empty result")
                    return None

            # Normalise action
            action = data.get("action", "SKIP").upper()
            if action in ("BUY_YES", "BUY_NO", "BUY"):
                action = "BUY"
            elif action in ("SELL",):
                action = "SELL"
            else:
                action = "SKIP"

            side = data.get("side", "YES").upper()
            confidence = float(data.get("confidence", 0.5))
            limit_price = int(data.get("limit_price", 50)) if data.get("limit_price") is not None else None

            return TradingDecision(
                action=action,
                side=side,
                confidence=confidence,
                limit_price=limit_price,
            )

        except Exception as exc:
            self.logger.error(
                f"Error parsing trading decision: {exc}",
                response_preview=response_text[:500] if response_text else "EMPTY",
            )
            return None

    # ------------------------------------------------------------------
    # Query logging
    # ------------------------------------------------------------------

    async def _log_query(
        self,
        strategy: str,
        query_type: str,
        prompt: str,
        response: str,
        market_id: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        confidence_extracted: Optional[float] = None,
        decision_extracted: Optional[str] = None,
    ) -> None:
        """Persist a query record if a database manager is available."""
        if not self.db_manager:
            return
        try:
            from src.utils.database import LLMQuery

            llm_query = LLMQuery(
                timestamp=datetime.now(),
                strategy=strategy,
                query_type=query_type,
                market_id=market_id,
                prompt=prompt[:2000],
                response=response[:5000],
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                confidence_extracted=confidence_extracted,
                decision_extracted=decision_extracted,
            )
            asyncio.create_task(self.db_manager.log_llm_query(llm_query))
        except Exception as exc:
            self.logger.error(f"Failed to log LLM query: {exc}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_cost_summary(self) -> Dict[str, Any]:
        """Return a summary of costs across all models."""
        per_model = {}
        for model, tracker in self.model_costs.items():
            if tracker.request_count > 0 or tracker.error_count > 0:
                per_model[model] = {
                    "requests": tracker.request_count,
                    "errors": tracker.error_count,
                    "input_tokens": tracker.input_tokens,
                    "output_tokens": tracker.output_tokens,
                    "total_cost": round(tracker.total_cost, 6),
                    "last_used": tracker.last_used.isoformat() if tracker.last_used else None,
                }
        return {
            "total_cost": round(self.total_cost, 6),
            "total_requests": self.request_count,
            "daily_cost": round(self.daily_tracker.total_cost, 6),
            "daily_limit": self.daily_tracker.daily_limit,
            "daily_exhausted": self.daily_tracker.is_exhausted,
            "per_model": per_model,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        try:
            await self.client.close()
        except Exception:
            pass
        self.logger.info(
            "OpenRouter client closed",
            total_cost=round(self.total_cost, 6),
            total_requests=self.request_count,
        )
