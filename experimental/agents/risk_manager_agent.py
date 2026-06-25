"""
Risk Manager Agent -- evaluates risk/reward and recommends position sizing.

Uses DeepSeek R1 (via OpenRouter) by default.  Focuses on:
- Expected value calculation
- Position sizing recommendation
- Risk assessment (1-10 scale)
"""

from experimental.agents.base_agent import BaseAgent


class RiskManagerAgent(BaseAgent):
    """Evaluates risk/reward profile and recommends position sizing."""

    AGENT_NAME = "risk_manager"
    AGENT_ROLE = "risk_manager"
    DEFAULT_MODEL = "deepseek/deepseek-v3.2"

    SYSTEM_PROMPT = (
        "You are a quantitative risk manager for a prediction-market trading "
        "desk. Your job is to evaluate whether a proposed trade has acceptable "
        "risk/reward and to recommend position sizing.\n\n"
        "You must consider:\n"
        "1. EXPECTED VALUE (EV) -- Calculate EV = (estimated probability * payout) "
        "   - cost. Only trades with positive EV should be taken.\n"
        "2. RISK SCORE -- Rate the overall risk from 1 (very safe) to 10 (very "
        "   risky). Consider: liquidity, time to expiry, volatility, information "
        "   quality, and model disagreement.\n"
        "3. POSITION SIZE -- Recommend what percentage of available capital to "
        "   allocate (0-100%). Use fractional Kelly criterion logic: higher EV "
        "   and lower risk = larger position.\n"
        "4. WORST CASE -- What is the maximum loss, and is it acceptable?\n"
        "5. EDGE DURABILITY -- How long will the informational edge last?\n\n"
        "Return your analysis as a JSON object (inside a ```json``` code block) "
        "with the following keys:\n"
        '  "risk_score": float (1.0-10.0),\n'
        '  "recommended_size_pct": float (0.0-100.0, percent of capital),\n'
        '  "ev_estimate": float (expected value as a decimal, e.g. 0.15 = 15%),\n'
        '  "max_loss_pct": float (worst case loss as percent of position),\n'
        '  "edge_durability_hours": float (estimated hours the edge lasts),\n'
        '  "should_trade": boolean (true if trade meets risk criteria),\n'
        '  "reasoning": string (detailed risk analysis)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)

        # Build context from other agents' outputs
        agents_section = ""
        pieces = []

        if context.get("forecaster_result"):
            fc = context["forecaster_result"]
            pieces.append(
                f"Forecaster: YES prob={fc.get('probability', '?')}, "
                f"confidence={fc.get('confidence', '?')}"
            )

        if context.get("bull_result"):
            bull = context["bull_result"]
            pieces.append(
                f"Bull Researcher: YES prob={bull.get('probability', '?')}, "
                f"floor={bull.get('probability_floor', '?')}"
            )

        if context.get("bear_result"):
            bear = context["bear_result"]
            pieces.append(
                f"Bear Researcher: YES prob={bear.get('probability', '?')}, "
                f"ceiling={bear.get('probability_ceiling', '?')}"
            )

        if context.get("news_result"):
            news = context["news_result"]
            pieces.append(
                f"News Analyst: sentiment={news.get('sentiment', '?')}, "
                f"relevance={news.get('relevance', '?')}, "
                f"direction={news.get('impact_direction', '?')}"
            )

        if pieces:
            agents_section = (
                "\n\n--- OTHER AGENTS' ASSESSMENTS ---\n"
                + "\n".join(f"- {p}" for p in pieces)
                + "\n--- END ASSESSMENTS ---"
            )

        # Portfolio context
        portfolio_section = ""
        if context.get("portfolio"):
            pf = context["portfolio"]
            portfolio_section = (
                f"\n\nPortfolio: cash=${pf.get('cash', 0):,.2f}, "
                f"max_position_pct={pf.get('max_position_pct', 5)}%, "
                f"existing_positions={pf.get('existing_positions', 0)}"
            )

        return (
            f"Evaluate the risk/reward for the following prediction market "
            f"trade.\n\n"
            f"{summary}{agents_section}{portfolio_section}\n\n"
            f"Calculate EV, assess risk, and recommend position sizing.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        risk_score = self.clamp(raw_json.get("risk_score", 5.0), lo=1.0, hi=10.0)
        recommended_size_pct = self.clamp(
            raw_json.get("recommended_size_pct", 1.0), lo=0.0, hi=100.0
        )
        ev_estimate = float(raw_json.get("ev_estimate", 0.0))
        max_loss_pct = self.clamp(
            raw_json.get("max_loss_pct", 100.0), lo=0.0, hi=100.0
        )
        edge_durability = max(0.0, float(raw_json.get("edge_durability_hours", 24.0)))
        should_trade = bool(raw_json.get("should_trade", False))
        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "risk_score": risk_score,
            "recommended_size_pct": recommended_size_pct,
            "ev_estimate": ev_estimate,
            "max_loss_pct": max_loss_pct,
            "edge_durability_hours": edge_durability,
            "should_trade": should_trade,
            "reasoning": reasoning,
        }
