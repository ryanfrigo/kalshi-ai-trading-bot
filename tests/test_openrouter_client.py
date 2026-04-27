"""Tests for the OpenRouter multi-model client."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.clients.openrouter_client import OpenRouterClient, MODEL_PRICING


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""

    def test_model_pricing_registry(self):
        """Verify all expected models are in the pricing registry."""
        expected_models = [
            "anthropic/claude-sonnet-4.5",
            "openai/o3",
            "google/gemini-3-pro-preview",
            "deepseek/deepseek-v3.2",
        ]
        for model in expected_models:
            assert model in MODEL_PRICING, f"Missing model pricing: {model}"
            pricing = MODEL_PRICING[model]
            # Check pricing has input and output keys (may use different naming)
            has_input = any("input" in k for k in pricing)
            has_output = any("output" in k for k in pricing)
            assert has_input, f"Missing input pricing for {model}: {pricing}"
            assert has_output, f"Missing output pricing for {model}: {pricing}"

    def test_client_initialization(self):
        """Test client initializes with correct settings."""
        with patch("src.clients.openrouter_client.settings") as mock_settings:
            mock_settings.api.openrouter_api_key = "test-key"
            mock_settings.api.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.trading.daily_ai_cost_limit = 50.0
            client = OpenRouterClient()
            assert client.total_cost == 0.0
            assert client.request_count == 0

    def test_parse_trading_decision_valid_json(self):
        """Test parsing a valid JSON trading decision."""
        client = OpenRouterClient.__new__(OpenRouterClient)
        client._logger = MagicMock()

        response = '''Here is my analysis:
```json
{"action": "BUY", "side": "YES", "limit_price": 45, "confidence": 0.78, "reasoning": "Good edge"}
```'''
        decision = client._parse_trading_decision(response)
        assert decision is not None
        assert decision.action == "BUY"
        assert decision.side == "YES"
        assert decision.confidence == 0.78
        assert decision.limit_price == 45

    def test_parse_trading_decision_skip(self):
        """Test parsing a SKIP decision."""
        client = OpenRouterClient.__new__(OpenRouterClient)
        client._logger = MagicMock()

        response = '{"action": "SKIP", "side": "YES", "limit_price": 50, "confidence": 0.3, "reasoning": "No edge"}'
        decision = client._parse_trading_decision(response)
        assert decision is not None
        assert decision.action == "SKIP"

    def test_parse_trading_decision_invalid(self):
        """Test parsing invalid response returns None."""
        client = OpenRouterClient.__new__(OpenRouterClient)
        client._logger = MagicMock()

        decision = client._parse_trading_decision("This is not JSON at all")
        assert decision is None

    def test_fallback_chain_ordering(self):
        """Test that fallback chain has the requested model first."""
        client = OpenRouterClient.__new__(OpenRouterClient)
        client._logger = MagicMock()

        chain = client._build_fallback_chain("openai/o3")
        assert chain[0] == "openai/o3"
        # Should contain other models as fallbacks
        assert len(chain) > 1

    def test_cost_summary(self):
        """Test cost summary generation."""
        client = OpenRouterClient.__new__(OpenRouterClient)
        client._logger = MagicMock()
        client.total_cost = 0.05
        client.request_count = 10
        client.model_costs = {}
        client.daily_tracker = MagicMock()
        client.daily_tracker.total_cost = 0.05
        client.daily_tracker.daily_limit = 50.0
        client.daily_tracker.is_exhausted = False

        summary = client.get_cost_summary()
        assert summary["total_cost"] == 0.05
        assert summary["total_requests"] == 10


