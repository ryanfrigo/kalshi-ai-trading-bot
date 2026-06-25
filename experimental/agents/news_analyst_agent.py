"""
News Analyst Agent -- analyses news sentiment and relevance to a market.

Uses Claude Sonnet 4 (via OpenRouter) by default.  Focuses on:
- News impact assessment
- Sentiment scoring (-1 to 1)
- Relevance to specific market
"""

from experimental.agents.base_agent import BaseAgent


class NewsAnalystAgent(BaseAgent):
    """Analyses recent news for sentiment and relevance to a market."""

    AGENT_NAME = "news_analyst"
    AGENT_ROLE = "news_analyst"
    DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

    SYSTEM_PROMPT = (
        "You are an expert financial news analyst specialising in prediction "
        "markets. Your role is to assess how recent news and current events "
        "affect the probability of a specific market outcome.\n\n"
        "Your analysis should cover:\n"
        "1. SENTIMENT -- Is the overall news flow positive or negative for the "
        "YES outcome? Score from -1.0 (strongly negative) to +1.0 (strongly "
        "positive).\n"
        "2. RELEVANCE -- How directly does the available news relate to this "
        "specific market? Score from 0.0 (unrelated) to 1.0 (directly about "
        "this exact topic).\n"
        "3. KEY FACTORS -- List the 2-5 most important news-driven factors.\n"
        "4. IMPACT DIRECTION -- Does the news push the probability UP or DOWN "
        "relative to the current market price?\n\n"
        "Return your analysis as a JSON object (inside a ```json``` code block) "
        "with the following keys:\n"
        '  "sentiment": float (-1.0 to 1.0),\n'
        '  "relevance": float (0.0 to 1.0),\n'
        '  "key_factors": list of strings,\n'
        '  "impact_direction": "up" or "down" or "neutral",\n'
        '  "reasoning": string (your detailed analysis)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)
        news = market_data.get("news_summary", "")
        extra_news = context.get("additional_news", "")

        news_section = ""
        if news or extra_news:
            combined = f"{news}\n{extra_news}".strip()
            news_section = f"\n\n--- NEWS FEED ---\n{combined[:2000]}\n--- END NEWS ---"
        else:
            news_section = "\n\n[No recent news available. Analyse based on general knowledge.]"

        return (
            f"Assess the news sentiment and relevance for the following "
            f"prediction market.\n\n"
            f"{summary}{news_section}\n\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        sentiment = self.clamp(raw_json.get("sentiment", 0.0), lo=-1.0, hi=1.0)
        relevance = self.clamp(raw_json.get("relevance", 0.5))
        key_factors = raw_json.get("key_factors", [])
        if not isinstance(key_factors, list):
            key_factors = [str(key_factors)]
        # Ensure all entries are strings
        key_factors = [str(f) for f in key_factors][:10]

        impact = str(raw_json.get("impact_direction", "neutral")).lower()
        if impact not in ("up", "down", "neutral"):
            impact = "neutral"

        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "sentiment": sentiment,
            "relevance": relevance,
            "key_factors": key_factors,
            "impact_direction": impact,
            "reasoning": reasoning,
        }
