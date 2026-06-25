"""
Bull vs Bear debate protocol with final trader decision.

Orchestrates a structured 4-step debate:
    Step 1: Bull researcher presents the YES case
    Step 2: Bear researcher presents the NO case
    Step 3: Risk manager evaluates both sides
    Step 4: Trader makes the final call with full context

Returns a TradingDecision-compatible dict with the full debate transcript
embedded in the reasoning field.
"""

import asyncio
import time
from typing import Callable, Dict, Optional

from experimental.agents.base_agent import BaseAgent
from experimental.agents.bull_researcher import BullResearcher
from experimental.agents.bear_researcher import BearResearcher
from experimental.agents.risk_manager_agent import RiskManagerAgent
from experimental.agents.trader_agent import TraderAgent
from experimental.agents.forecaster_agent import ForecasterAgent
from experimental.agents.news_analyst_agent import NewsAnalystAgent
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("debate")


class DebateRunner:
    """
    Orchestrates the Bull-vs-Bear debate protocol.

    Usage::

        runner = DebateRunner()
        decision = await runner.run_debate(
            market_data=market_data,
            get_completions={
                "forecaster": forecaster_fn,
                "news_analyst": news_fn,
                "bull_researcher": bull_fn,
                "bear_researcher": bear_fn,
                "risk_manager": risk_fn,
                "trader": trader_fn,
            },
            context={"portfolio": {...}},
        )
    """

    def __init__(
        self,
        agents: Optional[Dict[str, BaseAgent]] = None,
    ):
        """
        Args:
            agents: Mapping of role -> agent instance.  If None, defaults are
                    created.
        """
        self.agents: Dict[str, BaseAgent] = agents or self._default_agents()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run_debate(
        self,
        market_data: dict,
        get_completions: Dict[str, Callable],
        context: Optional[dict] = None,
    ) -> dict:
        """
        Run the full debate protocol.

        Args:
            market_data:     Standard market data dict.
            get_completions: Mapping of role -> async get_completion callable.
            context:         Shared context (portfolio info, etc.).

        Returns:
            A TradingDecision-compatible dict:
            {
                "action": "BUY" | "SELL" | "SKIP",
                "side": "YES" | "NO",
                "limit_price": int,
                "confidence": float,
                "position_size_pct": float,
                "reasoning": str,  # includes full debate transcript
                "debate_transcript": str,
                "step_results": dict,
                "elapsed_seconds": float,
                "error": str | None,
            }
        """
        start = time.time()
        context = dict(context or {})
        transcript_parts = []
        step_results = {}

        market_title = market_data.get("title", "Unknown")[:80]
        logger.info("Debate starting", market=market_title)

        # ==================================================================
        # Pre-step: Optionally gather forecaster + news in parallel
        # ==================================================================
        pre_results = await self._run_pre_analysis(
            market_data, context, get_completions
        )
        if pre_results.get("forecaster_result"):
            context["forecaster_result"] = pre_results["forecaster_result"]
            step_results["forecaster"] = pre_results["forecaster_result"]
            transcript_parts.append(
                self._format_step(
                    "PRE-ANALYSIS: Forecaster",
                    pre_results["forecaster_result"],
                )
            )
        if pre_results.get("news_result"):
            context["news_result"] = pre_results["news_result"]
            step_results["news_analyst"] = pre_results["news_result"]
            transcript_parts.append(
                self._format_step(
                    "PRE-ANALYSIS: News Analyst",
                    pre_results["news_result"],
                )
            )

        # ==================================================================
        # Step 1: Bull researcher presents YES case
        # ==================================================================
        bull_result = await self._run_step(
            "bull_researcher", market_data, context, get_completions, "Step 1: BULL CASE"
        )
        step_results["bull_researcher"] = bull_result
        context["bull_result"] = bull_result
        transcript_parts.append(self._format_step("STEP 1 -- BULL CASE", bull_result))

        # ==================================================================
        # Step 2: Bear researcher presents NO case
        # ==================================================================
        bear_result = await self._run_step(
            "bear_researcher", market_data, context, get_completions, "Step 2: BEAR CASE"
        )
        step_results["bear_researcher"] = bear_result
        context["bear_result"] = bear_result
        transcript_parts.append(self._format_step("STEP 2 -- BEAR CASE", bear_result))

        # ==================================================================
        # Step 3: Risk manager evaluates both sides
        # ==================================================================
        risk_result = await self._run_step(
            "risk_manager", market_data, context, get_completions, "Step 3: RISK ASSESSMENT"
        )
        step_results["risk_manager"] = risk_result
        context["risk_result"] = risk_result
        transcript_parts.append(self._format_step("STEP 3 -- RISK ASSESSMENT", risk_result))

        # ==================================================================
        # Step 4: Trader makes final call
        # ==================================================================
        trader_result = await self._run_step(
            "trader", market_data, context, get_completions, "Step 4: FINAL DECISION"
        )
        step_results["trader"] = trader_result
        transcript_parts.append(self._format_step("STEP 4 -- FINAL DECISION", trader_result))

        elapsed = time.time() - start
        transcript = "\n\n".join(transcript_parts)

        # Build final output (TradingDecision-compatible)
        if "error" in trader_result:
            # Trader failed -- fall back to a conservative SKIP
            logger.warning(
                "Trader agent failed; defaulting to SKIP",
                error=trader_result.get("error"),
            )
            return self._skip_decision(
                reasoning=f"Trader agent failed: {trader_result.get('error')}",
                transcript=transcript,
                step_results=step_results,
                elapsed=elapsed,
            )

        # Merge trader reasoning with debate transcript
        trader_reasoning = trader_result.get("reasoning", "")
        full_reasoning = f"{trader_reasoning}\n\n--- DEBATE TRANSCRIPT ---\n{transcript}"

        result = {
            "action": trader_result.get("action", "SKIP"),
            "side": trader_result.get("side", "YES"),
            "limit_price": trader_result.get("limit_price", 50),
            "confidence": trader_result.get("confidence", 0.0),
            "position_size_pct": trader_result.get("position_size_pct", 0.0),
            "reasoning": full_reasoning,
            "debate_transcript": transcript,
            "step_results": step_results,
            "elapsed_seconds": round(elapsed, 2),
            "error": None,
        }

        logger.info(
            "Debate complete",
            action=result["action"],
            side=result["side"],
            confidence=result["confidence"],
            elapsed=round(elapsed, 2),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _run_pre_analysis(
        self,
        market_data: dict,
        context: dict,
        get_completions: Dict[str, Callable],
    ) -> dict:
        """
        Run forecaster and news analyst in parallel as pre-analysis.
        These are optional; failures are tolerated.
        """
        results = {}
        tasks = {}

        if "forecaster" in self.agents and "forecaster" in get_completions:
            tasks["forecaster_result"] = asyncio.create_task(
                self._run_agent_safe(
                    "forecaster", market_data, context, get_completions["forecaster"]
                )
            )
        if "news_analyst" in self.agents and "news_analyst" in get_completions:
            tasks["news_result"] = asyncio.create_task(
                self._run_agent_safe(
                    "news_analyst", market_data, context, get_completions["news_analyst"]
                )
            )

        if tasks:
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, res in zip(tasks.keys(), done):
                if isinstance(res, Exception):
                    logger.warning(f"Pre-analysis {key} failed", error=str(res))
                elif "error" not in res:
                    results[key] = res
                else:
                    logger.warning(f"Pre-analysis {key} returned error", error=res.get("error"))

        return results

    async def _run_step(
        self,
        role: str,
        market_data: dict,
        context: dict,
        get_completions: Dict[str, Callable],
        step_label: str,
    ) -> dict:
        """
        Run a single debate step.  If the agent or completion is missing,
        return an error dict instead of failing.
        """
        if role not in self.agents:
            logger.warning(f"{step_label}: no agent for role '{role}'")
            return {"error": f"No agent registered for role '{role}'", "_agent": role}

        if role not in get_completions:
            logger.warning(f"{step_label}: no completion callable for role '{role}'")
            return {"error": f"No completion callable for role '{role}'", "_agent": role}

        logger.info(f"{step_label}: running", role=role)
        result = await self._run_agent_safe(
            role, market_data, context, get_completions[role]
        )

        if "error" in result:
            logger.warning(f"{step_label}: agent returned error", error=result["error"])
        else:
            logger.info(f"{step_label}: completed", role=role)

        return result

    async def _run_agent_safe(
        self,
        role: str,
        market_data: dict,
        context: dict,
        get_completion: Callable,
    ) -> dict:
        """Run a single agent with full error handling."""
        agent = self.agents.get(role)
        if agent is None:
            return {"error": f"No agent for role '{role}'", "_agent": role}
        try:
            return await agent.analyze(market_data, context, get_completion)
        except Exception as exc:
            logger.error("Agent failed in debate", role=role, error=str(exc), exc_info=True)
            return {"error": str(exc), "_agent": role}

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_step(label: str, result: dict) -> str:
        """Format a single step result into a readable transcript block."""
        if "error" in result:
            return f"[{label}]\nERROR: {result['error']}"

        agent_name = result.get("_agent", "unknown")
        model = result.get("_model", "unknown")
        elapsed = result.get("_elapsed_seconds", "?")

        # Pick the most informative fields to display
        lines = [f"[{label}] (agent={agent_name}, model={model}, {elapsed}s)"]

        for key in ("probability", "probability_floor", "probability_ceiling",
                     "confidence", "sentiment", "relevance", "risk_score",
                     "ev_estimate", "recommended_size_pct", "should_trade",
                     "action", "side", "limit_price"):
            if key in result:
                lines.append(f"  {key}: {result[key]}")

        for key in ("key_arguments", "key_factors", "catalysts", "risk_factors"):
            if key in result and result[key]:
                items = result[key][:5]
                lines.append(f"  {key}: {'; '.join(str(i) for i in items)}")

        reasoning = result.get("reasoning", "")
        if reasoning:
            # Truncate for transcript readability
            lines.append(f"  reasoning: {reasoning[:500]}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _skip_decision(
        reasoning: str,
        transcript: str,
        step_results: dict,
        elapsed: float,
    ) -> dict:
        return {
            "action": "SKIP",
            "side": "YES",
            "limit_price": 50,
            "confidence": 0.0,
            "position_size_pct": 0.0,
            "reasoning": reasoning,
            "debate_transcript": transcript,
            "step_results": step_results,
            "elapsed_seconds": round(elapsed, 2),
            "error": "Trader agent failed; defaulting to SKIP",
        }

    # ------------------------------------------------------------------
    # Default agent factory
    # ------------------------------------------------------------------
    @staticmethod
    def _default_agents() -> Dict[str, BaseAgent]:
        """Create the default set of agents for the debate."""
        return {
            "forecaster": ForecasterAgent(),
            "news_analyst": NewsAnalystAgent(),
            "bull_researcher": BullResearcher(),
            "bear_researcher": BearResearcher(),
            "risk_manager": RiskManagerAgent(),
            "trader": TraderAgent(),
        }
