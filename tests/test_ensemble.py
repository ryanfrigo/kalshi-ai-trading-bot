"""Tests for the multi-model ensemble system."""

import asyncio
import pytest
from unittest.mock import MagicMock

from experimental.agents.ensemble import EnsembleRunner
from experimental.agents.debate import DebateRunner


# --- Helper: mock completion factories ---

async def _mock_completion(prob, conf, side="yes"):
    """Create a mock completion function that returns a fixed agent response."""
    async def fn(prompt):
        return f'''```json
{{"probability": {prob}, "confidence": {conf}, "base_rate": 0.5, "side": "{side}", "reasoning": "mock"}}
```'''
    return fn


async def _mock_news_completion(prompt):
    return '''```json
{"sentiment": 0.4, "relevance": 0.7, "key_factors": ["test"], "impact_direction": "up", "reasoning": "mock"}
```'''


async def _mock_risk_completion(prompt):
    return '''```json
{"risk_score": 3.5, "recommended_size_pct": 2.5, "ev_estimate": 0.15, "max_loss_pct": 100, "edge_durability_hours": 48, "should_trade": true, "reasoning": "mock"}
```'''


async def _mock_trader_completion(prompt):
    return '''```json
{"action": "BUY", "side": "YES", "limit_price": 42, "confidence": 0.75, "position_size_pct": 2.5, "reasoning": "consensus"}
```'''


SAMPLE_MARKET = {
    "title": "Will test event happen?",
    "yes_price": 45,
    "no_price": 55,
    "volume": 10000,
    "days_to_expiry": 14,
    "rules": "Resolves YES if event happens.",
    "news_summary": "No news.",
}


class TestEnsembleRunner:
    """Tests for the EnsembleRunner."""

    def test_init_default_agents(self):
        """Ensemble initializes with default agents."""
        runner = EnsembleRunner()
        assert "forecaster" in runner.agents
        assert "bull_researcher" in runner.agents
        assert "bear_researcher" in runner.agents

    @pytest.mark.asyncio
    async def test_ensemble_with_agreeing_models(self):
        """Test ensemble with models that broadly agree."""
        runner = EnsembleRunner(min_models=2)
        completions = {
            "forecaster": await _mock_completion(0.70, 0.90),
            "bull_researcher": await _mock_completion(0.75, 0.85),
            "bear_researcher": await _mock_completion(0.60, 0.70),
        }
        result = await runner.run_ensemble(SAMPLE_MARKET, completions)

        assert result["error"] is None
        assert result["num_models_used"] >= 2
        assert 0.0 <= result["probability"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0
        # Models roughly agree, so disagreement should be moderate
        assert result["disagreement"] < 0.5

    @pytest.mark.asyncio
    async def test_ensemble_with_disagreeing_models(self):
        """Test that high disagreement lowers confidence."""
        runner = EnsembleRunner(min_models=2)
        completions = {
            "forecaster": await _mock_completion(0.90, 0.90),
            "bull_researcher": await _mock_completion(0.95, 0.85),
            "bear_researcher": await _mock_completion(0.10, 0.80),
        }
        result = await runner.run_ensemble(SAMPLE_MARKET, completions)

        assert result["error"] is None
        # High disagreement should be detected
        assert result["disagreement"] > 0.2

    @pytest.mark.asyncio
    async def test_ensemble_handles_failures(self):
        """Test that ensemble works even if some models fail."""
        runner = EnsembleRunner(min_models=2)

        async def failing_completion(prompt):
            raise RuntimeError("Model unavailable")

        completions = {
            "forecaster": await _mock_completion(0.65, 0.85),
            "bull_researcher": failing_completion,
            "bear_researcher": await _mock_completion(0.55, 0.70),
        }
        result = await runner.run_ensemble(SAMPLE_MARKET, completions)

        # Should still succeed with 2 models
        assert result["error"] is None
        assert result["num_models_used"] >= 2


class TestDebateRunner:
    """Tests for the DebateRunner (bull vs bear debate)."""

    def test_init_all_agents(self):
        """Debate runner has all required agents."""
        runner = DebateRunner()
        assert "bull_researcher" in runner.agents
        assert "bear_researcher" in runner.agents
        assert "risk_manager" in runner.agents
        assert "trader" in runner.agents

    @pytest.mark.asyncio
    async def test_full_debate(self):
        """Test a complete debate produces a valid decision."""
        runner = DebateRunner()
        completions = {
            "forecaster": await _mock_completion(0.70, 0.90),
            "news_analyst": _mock_news_completion,
            "bull_researcher": await _mock_completion(0.80, 0.85),
            "bear_researcher": await _mock_completion(0.40, 0.70),
            "risk_manager": _mock_risk_completion,
            "trader": _mock_trader_completion,
        }

        result = await runner.run_debate(
            SAMPLE_MARKET,
            completions,
            context={"portfolio": {"cash": 1000}},
        )

        assert result["error"] is None
        assert result["action"] in ("BUY", "SELL", "SKIP")
        assert result["side"] in ("YES", "NO")
        assert 0.0 <= result["confidence"] <= 1.0
        assert "debate_transcript" in result
        assert "step_results" in result

    @pytest.mark.asyncio
    async def test_debate_with_trader_failure_defaults_to_skip(self):
        """If the trader agent fails, debate should return SKIP."""
        runner = DebateRunner()

        async def failing_trader(prompt):
            raise RuntimeError("Trader unavailable")

        completions = {
            "forecaster": await _mock_completion(0.70, 0.90),
            "news_analyst": _mock_news_completion,
            "bull_researcher": await _mock_completion(0.80, 0.85),
            "bear_researcher": await _mock_completion(0.40, 0.70),
            "risk_manager": _mock_risk_completion,
            "trader": failing_trader,
        }

        result = await runner.run_debate(
            SAMPLE_MARKET, completions, context={}
        )

        # Should default to SKIP, not crash
        assert result["action"] == "SKIP"
