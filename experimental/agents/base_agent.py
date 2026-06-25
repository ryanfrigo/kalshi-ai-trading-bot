"""
Base agent class for the multi-agent ensemble decision engine.

All specialized agents (forecaster, news analyst, bull/bear researchers,
risk manager, trader) inherit from this base class which provides:
- Standard prompt formatting helpers
- JSON response parsing with repair
- Error handling wrapper for the analyze method
- Logging integration
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from json_repair import repair_json

from src.utils.logging_setup import get_trading_logger


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the ensemble.

    Each agent wraps a specific LLM model and role. The agent does NOT
    import or hold a reference to any client; instead, a ``get_completion``
    callable is passed into ``analyze`` so that the caller controls routing.

    Subclasses must:
        1. Set ``AGENT_NAME``, ``AGENT_ROLE``, and ``SYSTEM_PROMPT`` class attrs.
        2. Implement ``_build_prompt(market_data, context) -> str``.
        3. Implement ``_parse_result(raw_json) -> dict``.
    """

    # ------------------------------------------------------------------
    # Subclass configuration (override in each concrete agent)
    # ------------------------------------------------------------------
    AGENT_NAME: str = "base_agent"
    AGENT_ROLE: str = "base"
    SYSTEM_PROMPT: str = ""
    DEFAULT_MODEL: str = ""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            model_name: Override the default model for this agent.
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._logger = get_trading_logger(f"agent.{self.AGENT_NAME}")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Human-readable agent name."""
        return self.AGENT_NAME

    @property
    def role(self) -> str:
        """Agent role in the ensemble (e.g. forecaster, bull_researcher)."""
        return self.AGENT_ROLE

    @property
    def model_name(self) -> str:
        """Model identifier used for completions."""
        return self._model_name

    @property
    def logger(self):
        return self._logger

    # ------------------------------------------------------------------
    # Core public interface
    # ------------------------------------------------------------------
    async def analyze(
        self,
        market_data: dict,
        context: dict,
        get_completion: Callable,
    ) -> dict:
        """
        Run the agent's analysis on the given market.

        Args:
            market_data: Dict with keys like title, yes_price, no_price, volume,
                         days_to_expiry, rules, news_summary, etc.
            context:     Additional context (e.g. other agents' outputs, portfolio).
            get_completion: An async callable with signature
                ``async def get_completion(prompt: str) -> str``
                that returns the raw model response text.

        Returns:
            A dict whose shape depends on the concrete agent.  On failure
            an ``{"error": "..."}`` dict is returned so callers never get None.
        """
        start_time = time.time()
        try:
            prompt = self._build_user_prompt(market_data, context)
            self.logger.debug(
                "Agent sending prompt",
                agent=self.name,
                model=self.model_name,
                prompt_length=len(prompt),
            )

            raw_response = await get_completion(prompt)

            if raw_response is None:
                return self._error_result("Model returned None (API limit or failure)")

            elapsed = time.time() - start_time
            self.logger.info(
                "Agent received response",
                agent=self.name,
                model=self.model_name,
                response_length=len(raw_response),
                elapsed_seconds=round(elapsed, 2),
            )

            parsed = self._extract_json(raw_response)
            if parsed is None:
                return self._error_result(
                    f"Failed to extract JSON from response: {raw_response[:300]}"
                )

            result = self._parse_result(parsed)
            result["_agent"] = self.name
            result["_model"] = self.model_name
            result["_elapsed_seconds"] = round(elapsed, 2)
            return result

        except Exception as exc:
            elapsed = time.time() - start_time
            self.logger.error(
                "Agent analysis failed",
                agent=self.name,
                error=str(exc),
                elapsed_seconds=round(elapsed, 2),
                exc_info=True,
            )
            return self._error_result(str(exc))

    # ------------------------------------------------------------------
    # Prompt building helpers
    # ------------------------------------------------------------------
    def _build_user_prompt(self, market_data: dict, context: dict) -> str:
        """
        Build the full prompt sent to the model.

        Combines the class-level SYSTEM_PROMPT with the subclass-specific
        user prompt returned by ``_build_prompt``.
        """
        user_section = self._build_prompt(market_data, context)

        if self.SYSTEM_PROMPT:
            return f"{self.SYSTEM_PROMPT}\n\n{user_section}"
        return user_section

    @abstractmethod
    def _build_prompt(self, market_data: dict, context: dict) -> str:
        """
        Build the agent-specific portion of the prompt.

        Subclasses implement this to inject market data, portfolio info,
        and any context (e.g. other agents' outputs) into their prompt.
        """
        ...

    @abstractmethod
    def _parse_result(self, raw_json: dict) -> dict:
        """
        Validate and normalise the parsed JSON into the agent's output schema.

        Subclasses implement this to enforce required keys, clamp ranges, etc.
        """
        ...

    # ------------------------------------------------------------------
    # JSON extraction and repair
    # ------------------------------------------------------------------
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract a JSON object from model output.

        Handles:
        - JSON wrapped in ```json ... ``` code blocks
        - Bare JSON objects
        - Slightly malformed JSON via json_repair
        """
        # Strategy 1: Look for ```json ... ``` code block
        code_block_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            result = self._try_parse_json(candidate)
            if result is not None:
                return result

        # Strategy 2: Look for any ``` ... ``` code block
        code_block_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            result = self._try_parse_json(candidate)
            if result is not None:
                return result

        # Strategy 3: Find the outermost { ... } in the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0).strip()
            result = self._try_parse_json(candidate)
            if result is not None:
                return result

        # Strategy 4: Try repair on the entire text
        result = self._try_parse_json(text)
        if result is not None:
            return result

        self.logger.warning(
            "All JSON extraction strategies failed",
            agent=self.name,
            text_preview=text[:200],
        )
        return None

    def _try_parse_json(self, candidate: str) -> Optional[dict]:
        """Attempt to parse *candidate* as JSON, falling back to repair."""
        # Direct parse
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Repair attempt
        try:
            repaired = repair_json(candidate, return_objects=False)
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                self.logger.debug("JSON repair succeeded", agent=self.name)
                return parsed
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def format_market_summary(market_data: dict) -> str:
        """Return a concise human-readable market summary for prompts."""
        title = market_data.get("title", "Unknown Market")
        yes_price = market_data.get("yes_price", "?")
        no_price = market_data.get("no_price", "?")
        volume = market_data.get("volume", 0)
        days = market_data.get("days_to_expiry", "?")
        rules = market_data.get("rules", "")
        news = market_data.get("news_summary", "")

        lines = [
            f"Market: {title}",
            f"Rules: {rules}" if rules else "",
            f"YES Price: {yes_price}c | NO Price: {no_price}c",
            f"Volume: ${volume:,.0f}" if isinstance(volume, (int, float)) else f"Volume: {volume}",
            f"Days to Expiry: {days}",
        ]
        if news:
            lines.append(f"Recent News: {news[:500]}")

        return "\n".join(line for line in lines if line)

    @staticmethod
    def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """Clamp a numeric value between lo and hi."""
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return lo

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------
    def _error_result(self, message: str) -> dict:
        """Return a standardised error dict."""
        self.logger.warning("Agent returning error result", agent=self.name, error=message)
        return {
            "error": message,
            "_agent": self.name,
            "_model": self.model_name,
        }
