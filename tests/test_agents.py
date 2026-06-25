"""Tests for the agent system (base agent, individual agents, JSON extraction)."""

import asyncio
import pytest
from unittest.mock import MagicMock

from experimental.agents.base_agent import BaseAgent
from experimental.agents.forecaster_agent import ForecasterAgent
from experimental.agents.news_analyst_agent import NewsAnalystAgent
from experimental.agents.bull_researcher import BullResearcher
from experimental.agents.bear_researcher import BearResearcher
from experimental.agents.risk_manager_agent import RiskManagerAgent
from experimental.agents.trader_agent import TraderAgent


SAMPLE_MARKET = {
    "title": "Will Bitcoin reach $100k by end of 2025?",
    "yes_price": 45,
    "no_price": 55,
    "volume": 50000,
    "days_to_expiry": 30,
    "rules": "Resolves YES if BTC >= $100,000 USD.",
    "news_summary": "Bitcoin rallying recently.",
}


class TestBaseAgent:
    """Tests for the base agent JSON extraction and utilities."""

    def test_extract_json_from_code_block(self):
        """Extract JSON from markdown code blocks."""
        agent = ForecasterAgent()
        text = '''Here is my analysis:
```json
{"probability": 0.65, "confidence": 0.8, "base_rate": 0.5, "side": "yes", "reasoning": "test"}
```
Done.'''
        result = agent._extract_json(text)
        assert result is not None
        assert result["probability"] == 0.65

    def test_extract_bare_json(self):
        """Extract bare JSON without code blocks."""
        agent = ForecasterAgent()
        text = '{"probability": 0.4, "confidence": 0.6, "base_rate": 0.3, "side": "no", "reasoning": "bare"}'
        result = agent._extract_json(text)
        assert result is not None
        assert result["probability"] == 0.4

    def test_extract_json_repair(self):
        """Test JSON repair for malformed responses."""
        agent = ForecasterAgent()
        text = '{probability: 0.7, confidence: 0.5, base_rate: 0.4, side: "yes", reasoning: "repaired"}'
        result = agent._extract_json(text)
        assert result is not None

    def test_clamp_values(self):
        """Test value clamping utility."""
        agent = ForecasterAgent()
        assert agent.clamp(1.5, 0.0, 1.0) == 1.0
        assert agent.clamp(-0.5, 0.0, 1.0) == 0.0
        assert agent.clamp(0.5, 0.0, 1.0) == 0.5

    def test_format_market_summary(self):
        """Test market data formatting."""
        summary = BaseAgent.format_market_summary(SAMPLE_MARKET)
        assert "Bitcoin" in summary
        assert "45" in summary
        assert "50000" in str(SAMPLE_MARKET["volume"])


class TestAgentProperties:
    """Test that each agent has correct properties."""

    def test_forecaster_properties(self):
        agent = ForecasterAgent()
        assert agent.name == "forecaster"
        assert agent.role == "forecaster"
        assert agent.model_name == "grok-4-1-fast-reasoning"

    def test_news_analyst_properties(self):
        agent = NewsAnalystAgent()
        assert agent.name == "news_analyst"
        assert agent.role == "news_analyst"
        assert "claude-sonnet-4.5" in agent.model_name or "anthropic" in agent.model_name

    def test_bull_researcher_properties(self):
        agent = BullResearcher()
        assert agent.name == "bull_researcher"
        assert agent.role == "bull_researcher"
        assert "o3" in agent.model_name or "openai" in agent.model_name

    def test_bear_researcher_properties(self):
        agent = BearResearcher()
        assert agent.name == "bear_researcher"
        assert agent.role == "bear_researcher"
        assert "gemini-3" in agent.model_name or "google" in agent.model_name

    def test_risk_manager_properties(self):
        agent = RiskManagerAgent()
        assert agent.name == "risk_manager"
        assert agent.role == "risk_manager"
        assert "deepseek-v3.2" in agent.model_name or "deepseek" in agent.model_name

    def test_trader_properties(self):
        agent = TraderAgent()
        assert agent.name == "trader"
        assert agent.role == "trader"
        assert agent.model_name == "grok-4-1-fast-reasoning"


class TestAgentAnalyze:
    """Test agent analyze method with mock completions."""

    @pytest.mark.asyncio
    async def test_forecaster_analyze(self):
        """Forecaster returns probability estimate."""
        agent = ForecasterAgent()

        async def mock_completion(prompt):
            return '''```json
{"probability": 0.72, "confidence": 0.85, "base_rate": 0.50, "side": "yes", "reasoning": "Strong trend."}
```'''

        result = await agent.analyze(SAMPLE_MARKET, {}, mock_completion)
        assert "error" not in result
        assert result["probability"] == 0.72
        assert result["confidence"] == 0.85
        assert result["_agent"] == "forecaster"

    @pytest.mark.asyncio
    async def test_bull_researcher_analyze(self):
        """Bull researcher returns bullish case."""
        agent = BullResearcher()

        async def mock_completion(prompt):
            return '''```json
{"probability": 0.80, "probability_floor": 0.60, "confidence": 0.85, "key_arguments": ["momentum", "adoption"], "catalysts": ["ETF"], "reasoning": "Bullish."}
```'''

        result = await agent.analyze(SAMPLE_MARKET, {}, mock_completion)
        assert "error" not in result
        assert result["probability"] == 0.80
        assert result["_agent"] == "bull_researcher"

    @pytest.mark.asyncio
    async def test_bear_researcher_analyze(self):
        """Bear researcher returns bearish case."""
        agent = BearResearcher()

        async def mock_completion(prompt):
            return '''```json
{"probability": 0.30, "probability_ceiling": 0.50, "confidence": 0.70, "key_arguments": ["regulation", "crash risk"], "risk_factors": ["macro"], "reasoning": "Bearish."}
```'''

        result = await agent.analyze(SAMPLE_MARKET, {}, mock_completion)
        assert "error" not in result
        assert result["probability"] == 0.30
        assert result["_agent"] == "bear_researcher"

    @pytest.mark.asyncio
    async def test_risk_manager_analyze(self):
        """Risk manager returns risk assessment."""
        agent = RiskManagerAgent()

        async def mock_completion(prompt):
            return '''```json
{"risk_score": 4.0, "recommended_size_pct": 2.5, "ev_estimate": 0.18, "max_loss_pct": 100, "edge_durability_hours": 48, "should_trade": true, "reasoning": "Acceptable risk."}
```'''

        result = await agent.analyze(SAMPLE_MARKET, {}, mock_completion)
        assert "error" not in result
        assert result["risk_score"] == 4.0
        assert result["should_trade"] is True

    @pytest.mark.asyncio
    async def test_agent_handles_completion_failure(self):
        """Agent returns error dict when completion fails."""
        agent = ForecasterAgent()

        async def failing_completion(prompt):
            raise RuntimeError("API down")

        result = await agent.analyze(SAMPLE_MARKET, {}, failing_completion)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_agent_handles_malformed_json(self):
        """Agent handles non-JSON response gracefully."""
        agent = ForecasterAgent()

        async def bad_completion(prompt):
            return "I'm not sure about this market. It could go either way."

        result = await agent.analyze(SAMPLE_MARKET, {}, bad_completion)
        # Should either return error or attempt JSON repair
        assert isinstance(result, dict)


class TestEventBus:
    """Tests for the event bus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test basic publish/subscribe flow."""
        from src.events.event_bus import EventBus

        bus = EventBus.get_instance()
        bus.reset_instance()
        bus = EventBus.get_instance()

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("test_event", handler)
        await bus.publish("test_event", {"value": 42})

        assert len(received) == 1
        assert received[0].data["value"] == 42

        bus.unsubscribe_all()

    @pytest.mark.asyncio
    async def test_ticker_filtering(self):
        """Test that ticker filtering works."""
        from src.events.event_bus import EventBus

        bus = EventBus.get_instance()
        bus.reset_instance()
        bus = EventBus.get_instance()

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("price_update", handler, ticker="AAPL")
        await bus.publish("price_update", {"ticker": "AAPL", "price": 100})
        await bus.publish("price_update", {"ticker": "GOOG", "price": 200})

        assert len(received) == 1
        assert received[0].data["ticker"] == "AAPL"

        bus.unsubscribe_all()
