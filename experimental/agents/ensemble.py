"""
Multi-model probability ensemble.

Runs multiple agents in parallel, collects their YES probability estimates,
computes a weighted average with confidence-adjusted weights, detects
disagreement, and optionally tracks calibration over time.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from experimental.agents.base_agent import BaseAgent
from experimental.agents.forecaster_agent import ForecasterAgent
from experimental.agents.news_analyst_agent import NewsAnalystAgent
from experimental.agents.bull_researcher import BullResearcher
from experimental.agents.bear_researcher import BearResearcher
from experimental.agents.risk_manager_agent import RiskManagerAgent
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("ensemble")

# Default weights from EnsembleConfig; keyed by agent role
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "forecaster": 0.30,
    "news_analyst": 0.20,
    "bull_researcher": 0.20,
    "bear_researcher": 0.15,
    "risk_manager": 0.15,
}

# Path where calibration records are stored
_CALIBRATION_FILE = Path("logs/ensemble_calibration.json")


class EnsembleRunner:
    """
    Orchestrates a multi-model probability ensemble.

    Usage::

        runner = EnsembleRunner()
        result = await runner.run_ensemble(
            market_data=market_data,
            get_completions={
                "forecaster": forecaster_get_completion,
                "news_analyst": news_get_completion,
                ...
            },
        )
    """

    def __init__(
        self,
        agents: Optional[Dict[str, BaseAgent]] = None,
        weights: Optional[Dict[str, float]] = None,
        min_models: Optional[int] = None,
        disagreement_threshold: Optional[float] = None,
    ):
        """
        Args:
            agents:  Mapping of role -> agent instance. If None, default agents
                     are created.
            weights: Mapping of role -> weight for the weighted average.
            min_models: Minimum number of successful agent results required
                        to produce a consensus. Defaults to
                        ``settings.ensemble.min_models_for_consensus``.
            disagreement_threshold: Std-dev above this triggers a low-confidence
                        flag. Defaults to
                        ``settings.ensemble.disagreement_threshold``.
        """
        self.agents: Dict[str, BaseAgent] = agents or self._default_agents()
        self.weights = weights or dict(_DEFAULT_WEIGHTS)
        self.min_models = min_models or settings.ensemble.min_models_for_consensus
        self.disagreement_threshold = (
            disagreement_threshold
            if disagreement_threshold is not None
            else settings.ensemble.disagreement_threshold
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run_ensemble(
        self,
        market_data: dict,
        get_completions: Dict[str, Callable],
        context: Optional[dict] = None,
    ) -> dict:
        """
        Run the ensemble: invoke agents in parallel, aggregate results.

        Args:
            market_data:      Standard market data dict.
            get_completions:  Mapping of agent role -> async get_completion callable.
                              Only agents whose role appears here will run.
            context:          Shared context dict passed to all agents.

        Returns:
            {
                "probability": float,   # weighted-average YES probability
                "confidence": float,    # aggregate confidence (lowered by disagreement)
                "disagreement": float,  # std dev of model probabilities
                "model_results": [...], # per-model result dicts
                "num_models_used": int,
                "error": str | None,
            }
        """
        start = time.time()
        context = context or {}
        model_results: List[dict] = []

        # Determine which agents to run (intersection of available agents and
        # provided completions)
        roles_to_run = [
            role for role in self.agents if role in get_completions
        ]

        if not roles_to_run:
            return self._error("No matching agents for provided completions")

        logger.info(
            "Ensemble starting",
            roles=roles_to_run,
            market=market_data.get("title", "?")[:60],
        )

        # ------------------------------------------------------------------
        # Launch agents in parallel
        # ------------------------------------------------------------------
        if settings.ensemble.parallel_requests:
            tasks = {
                role: asyncio.create_task(
                    self._run_agent_safe(role, market_data, context, get_completions[role])
                )
                for role in roles_to_run
            }
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            results_map: Dict[str, dict] = {}
            for role, result in zip(tasks.keys(), done):
                if isinstance(result, Exception):
                    logger.warning("Agent raised exception", role=role, error=str(result))
                    results_map[role] = {"error": str(result), "_agent": role}
                else:
                    results_map[role] = result
        else:
            # Sequential fallback
            results_map = {}
            for role in roles_to_run:
                results_map[role] = await self._run_agent_safe(
                    role, market_data, context, get_completions[role]
                )

        # ------------------------------------------------------------------
        # Collect successful probability estimates
        # ------------------------------------------------------------------
        probabilities: List[Tuple[str, float, float]] = []  # (role, prob, confidence)
        for role, result in results_map.items():
            model_results.append(result)
            if "error" in result:
                continue

            prob = self._extract_probability(role, result)
            conf = result.get("confidence", 0.5)
            if prob is not None:
                probabilities.append((role, prob, conf))

        if len(probabilities) < self.min_models:
            elapsed = time.time() - start
            logger.warning(
                "Not enough models for consensus",
                successful=len(probabilities),
                required=self.min_models,
                elapsed=round(elapsed, 2),
            )
            return {
                "probability": None,
                "confidence": 0.0,
                "disagreement": None,
                "model_results": model_results,
                "num_models_used": len(probabilities),
                "elapsed_seconds": round(elapsed, 2),
                "error": (
                    f"Only {len(probabilities)} models succeeded; "
                    f"need {self.min_models} for consensus"
                ),
            }

        # ------------------------------------------------------------------
        # Weighted average (confidence-adjusted weights)
        # ------------------------------------------------------------------
        weighted_prob, raw_confidence, disagreement = self._aggregate(probabilities)

        # If disagreement is high, discount confidence
        if disagreement > self.disagreement_threshold:
            penalty = min(1.0, disagreement / self.disagreement_threshold) * 0.3
            adjusted_confidence = max(0.0, raw_confidence - penalty)
            logger.info(
                "High disagreement detected -- confidence penalised",
                disagreement=round(disagreement, 4),
                threshold=self.disagreement_threshold,
                raw_confidence=round(raw_confidence, 4),
                adjusted_confidence=round(adjusted_confidence, 4),
            )
        else:
            adjusted_confidence = raw_confidence

        elapsed = time.time() - start
        logger.info(
            "Ensemble complete",
            probability=round(weighted_prob, 4),
            confidence=round(adjusted_confidence, 4),
            disagreement=round(disagreement, 4),
            models_used=len(probabilities),
            elapsed=round(elapsed, 2),
        )

        # Optionally record for calibration tracking
        if settings.ensemble.calibration_tracking:
            self._record_calibration(
                market_data=market_data,
                probability=weighted_prob,
                confidence=adjusted_confidence,
                disagreement=disagreement,
                model_results=model_results,
            )

        return {
            "probability": round(weighted_prob, 4),
            "confidence": round(adjusted_confidence, 4),
            "disagreement": round(disagreement, 4),
            "model_results": model_results,
            "num_models_used": len(probabilities),
            "elapsed_seconds": round(elapsed, 2),
            "error": None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _run_agent_safe(
        self,
        role: str,
        market_data: dict,
        context: dict,
        get_completion: Callable,
    ) -> dict:
        """Run a single agent, catching all errors."""
        agent = self.agents.get(role)
        if agent is None:
            return {"error": f"No agent registered for role '{role}'", "_agent": role}
        try:
            return await agent.analyze(market_data, context, get_completion)
        except Exception as exc:
            logger.error("Agent failed", role=role, error=str(exc), exc_info=True)
            return {"error": str(exc), "_agent": role}

    def _extract_probability(self, role: str, result: dict) -> Optional[float]:
        """
        Extract a YES probability from an agent result.

        Different agents store probability differently:
        - forecaster, bull, bear: ``probability`` key directly.
        - news_analyst: derive from sentiment + relevance heuristic.
        - risk_manager: does not provide a probability per se; skip.
        """
        if role in ("forecaster", "bull_researcher", "bear_researcher"):
            val = result.get("probability")
            if val is not None:
                return float(val)

        if role == "news_analyst":
            # Convert sentiment (-1..1) into a probability adjustment centred on 0.5
            sentiment = result.get("sentiment", 0.0)
            relevance = result.get("relevance", 0.5)
            # Scale: sentiment * relevance gives -1..1 adjustment, map to 0..1
            prob = 0.5 + (sentiment * relevance * 0.5)
            return max(0.0, min(1.0, prob))

        if role == "risk_manager":
            # Risk manager may optionally include a probability; if not, skip
            val = result.get("probability")
            if val is not None:
                return float(val)
            return None

        # Fallback: look for a "probability" key
        val = result.get("probability")
        if val is not None:
            return float(val)
        return None

    def _aggregate(
        self, probabilities: List[Tuple[str, float, float]]
    ) -> Tuple[float, float, float]:
        """
        Compute weighted average probability, aggregate confidence, and
        disagreement (standard deviation).

        Weights are base weight * agent confidence so that more confident
        agents contribute more.
        """
        import math

        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0

        for role, prob, conf in probabilities:
            base_w = self.weights.get(role, 0.1)
            adjusted_w = base_w * max(conf, 0.1)  # Floor conf at 0.1 to avoid zero
            weighted_sum += prob * adjusted_w
            confidence_sum += conf * base_w
            total_weight += adjusted_w

        if total_weight == 0:
            return 0.5, 0.0, 1.0

        avg_prob = weighted_sum / total_weight

        # Normalise confidence by total base weight
        total_base = sum(self.weights.get(r, 0.1) for r, _, _ in probabilities)
        avg_conf = confidence_sum / total_base if total_base > 0 else 0.5

        # Standard deviation of probabilities (unweighted for simplicity)
        probs = [p for _, p, _ in probabilities]
        mean = sum(probs) / len(probs)
        variance = sum((p - mean) ** 2 for p in probs) / len(probs)
        std_dev = math.sqrt(variance)

        return avg_prob, avg_conf, std_dev

    # ------------------------------------------------------------------
    # Calibration tracking
    # ------------------------------------------------------------------
    def _record_calibration(
        self,
        market_data: dict,
        probability: float,
        confidence: float,
        disagreement: float,
        model_results: list,
    ) -> None:
        """
        Append a calibration record to the JSON calibration file.

        The file is a JSON array of objects.  Resolution outcome can be
        backfilled later to compute calibration curves.
        """
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "market_title": market_data.get("title", "")[:200],
            "market_ticker": market_data.get("ticker", ""),
            "yes_price": market_data.get("yes_price"),
            "ensemble_probability": probability,
            "ensemble_confidence": confidence,
            "disagreement": disagreement,
            "num_models": len([r for r in model_results if "error" not in r]),
            "model_probabilities": {
                r.get("_agent", "?"): r.get("probability")
                for r in model_results
                if "error" not in r and r.get("probability") is not None
            },
            "resolved_yes": None,  # To be backfilled after market resolves
        }

        try:
            _CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            existing: list = []
            if _CALIBRATION_FILE.exists():
                try:
                    existing = json.loads(_CALIBRATION_FILE.read_text())
                    if not isinstance(existing, list):
                        existing = []
                except (json.JSONDecodeError, OSError):
                    existing = []

            existing.append(record)

            # Keep a reasonable cap (last 5000 records)
            if len(existing) > 5000:
                existing = existing[-5000:]

            _CALIBRATION_FILE.write_text(json.dumps(existing, indent=2))
        except Exception as exc:
            logger.warning("Failed to write calibration record", error=str(exc))

    # ------------------------------------------------------------------
    # Default agent factory
    # ------------------------------------------------------------------
    @staticmethod
    def _default_agents() -> Dict[str, BaseAgent]:
        """Create the default set of agents from EnsembleConfig."""
        return {
            "forecaster": ForecasterAgent(),
            "news_analyst": NewsAnalystAgent(),
            "bull_researcher": BullResearcher(),
            "bear_researcher": BearResearcher(),
            "risk_manager": RiskManagerAgent(),
        }

    @staticmethod
    def _error(msg: str) -> dict:
        return {
            "probability": None,
            "confidence": 0.0,
            "disagreement": None,
            "model_results": [],
            "num_models_used": 0,
            "error": msg,
        }
