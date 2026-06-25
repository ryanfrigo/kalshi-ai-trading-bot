"""
Trader Agent -- final decision synthesiser.

Uses Grok-4 (via xAI) by default.  Takes all other agents' outputs and
makes the final BUY/SKIP decision, producing a TradingDecision-compatible dict.
"""

from experimental.agents.base_agent import BaseAgent


class TraderAgent(BaseAgent):
    """
    Synthesises all agents' outputs into a final BUY/SELL/SKIP decision.

    The output dict is compatible with the TradingDecision dataclass in
    ``src.clients.xai_client``.
    """

    AGENT_NAME = "trader"
    AGENT_ROLE = "trader"
    DEFAULT_MODEL = "grok-4-1-fast-reasoning"

    SYSTEM_PROMPT = (
        "You are the head trader at an AI-powered prediction market fund. "
        "You receive analysis from a team of specialist agents and make the "
        "FINAL trading decision.\n\n"
        "Your decision framework:\n"
        "1. CONSENSUS CHECK -- Do the agents broadly agree? High disagreement "
        "   should lower your confidence.\n"
        "2. EV THRESHOLD -- Only trade if expected value is strongly positive "
        "   (at least 10% edge over market price).\n"
        "3. RISK CHECK -- Does the risk manager approve? Respect position sizing "
        "   recommendations.\n"
        "4. PRICE SETTING -- Set a limit price that gives you edge. Never chase "
        "   the market.\n"
        "5. CONVICTION -- You need high conviction from multiple sources to act. "
        "   When in doubt, SKIP.\n\n"
        "Return your decision as a JSON object (inside a ```json``` code block) "
        "with the following keys:\n"
        '  "action": "BUY" | "SELL" | "SKIP",\n'
        '  "side": "YES" | "NO",\n'
        '  "limit_price": int (cents, 1-99),\n'
        '  "confidence": float (0.0-1.0),\n'
        '  "position_size_pct": float (percent of capital to risk),\n'
        '  "reasoning": string (detailed justification referencing the agents)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)

        # Assemble each agent's results into a structured briefing
        briefing_parts = []

        if context.get("forecaster_result"):
            fc = context["forecaster_result"]
            briefing_parts.append(
                f"FORECASTER (model: {fc.get('_model', 'unknown')}):\n"
                f"  YES probability: {fc.get('probability', '?')}\n"
                f"  Confidence: {fc.get('confidence', '?')}\n"
                f"  Base rate: {fc.get('base_rate', '?')}\n"
                f"  Side: {fc.get('side', '?')}\n"
                f"  Reasoning: {fc.get('reasoning', 'N/A')[:300]}"
            )

        if context.get("news_result"):
            news = context["news_result"]
            factors = ", ".join(news.get("key_factors", [])[:5])
            briefing_parts.append(
                f"NEWS ANALYST (model: {news.get('_model', 'unknown')}):\n"
                f"  Sentiment: {news.get('sentiment', '?')}\n"
                f"  Relevance: {news.get('relevance', '?')}\n"
                f"  Impact direction: {news.get('impact_direction', '?')}\n"
                f"  Key factors: {factors}\n"
                f"  Reasoning: {news.get('reasoning', 'N/A')[:300]}"
            )

        if context.get("bull_result"):
            bull = context["bull_result"]
            args = "; ".join(bull.get("key_arguments", [])[:5])
            briefing_parts.append(
                f"BULL RESEARCHER (model: {bull.get('_model', 'unknown')}):\n"
                f"  YES probability: {bull.get('probability', '?')}\n"
                f"  Probability floor: {bull.get('probability_floor', '?')}\n"
                f"  Confidence: {bull.get('confidence', '?')}\n"
                f"  Key arguments: {args}\n"
                f"  Reasoning: {bull.get('reasoning', 'N/A')[:300]}"
            )

        if context.get("bear_result"):
            bear = context["bear_result"]
            args = "; ".join(bear.get("key_arguments", [])[:5])
            briefing_parts.append(
                f"BEAR RESEARCHER (model: {bear.get('_model', 'unknown')}):\n"
                f"  YES probability: {bear.get('probability', '?')}\n"
                f"  Probability ceiling: {bear.get('probability_ceiling', '?')}\n"
                f"  Confidence: {bear.get('confidence', '?')}\n"
                f"  Key arguments: {args}\n"
                f"  Reasoning: {bear.get('reasoning', 'N/A')[:300]}"
            )

        if context.get("risk_result"):
            risk = context["risk_result"]
            briefing_parts.append(
                f"RISK MANAGER (model: {risk.get('_model', 'unknown')}):\n"
                f"  Risk score: {risk.get('risk_score', '?')}/10\n"
                f"  Recommended size: {risk.get('recommended_size_pct', '?')}%\n"
                f"  EV estimate: {risk.get('ev_estimate', '?')}\n"
                f"  Should trade: {risk.get('should_trade', '?')}\n"
                f"  Reasoning: {risk.get('reasoning', 'N/A')[:300]}"
            )

        briefing = "\n\n".join(briefing_parts) if briefing_parts else "[No agent analyses available]"

        # Portfolio context
        portfolio_note = ""
        if context.get("portfolio"):
            pf = context["portfolio"]
            portfolio_note = (
                f"\n\nPortfolio: cash=${pf.get('cash', 0):,.2f}, "
                f"max_risk=${pf.get('max_trade_value', 0):,.2f}, "
                f"max_position_pct={pf.get('max_position_pct', 5)}%"
            )

        return (
            f"Review the following market and your team's analysis, then make "
            f"a final trading decision.\n\n"
            f"=== MARKET ===\n{summary}\n\n"
            f"=== TEAM ANALYSIS ===\n{briefing}{portfolio_note}\n\n"
            f"Make your final BUY/SELL/SKIP decision. Be decisive but disciplined.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        action = str(raw_json.get("action", "SKIP")).upper()
        if action not in ("BUY", "SELL", "SKIP"):
            action = "SKIP"

        side = str(raw_json.get("side", "YES")).upper()
        if side not in ("YES", "NO"):
            side = "YES"

        confidence = self.clamp(raw_json.get("confidence", 0.5))

        # Limit price: integer cents 1-99
        try:
            limit_price = int(raw_json.get("limit_price", 50))
            limit_price = max(1, min(99, limit_price))
        except (TypeError, ValueError):
            limit_price = 50

        position_size_pct = self.clamp(
            raw_json.get("position_size_pct", 1.0), lo=0.0, hi=100.0
        )

        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "action": action,
            "side": side,
            "limit_price": limit_price,
            "confidence": confidence,
            "position_size_pct": position_size_pct,
            "reasoning": reasoning,
        }
