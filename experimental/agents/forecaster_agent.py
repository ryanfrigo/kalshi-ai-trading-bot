"""
Forecaster Agent -- estimates the true YES probability for a market.

Uses Grok-4 (via xAI) by default.  Focuses on:
- Base rate analysis
- Current conditions assessment
- Probability estimation with calibration
"""

from experimental.agents.base_agent import BaseAgent


class ForecasterAgent(BaseAgent):
    """Estimates the true YES probability for a prediction market."""

    AGENT_NAME = "forecaster"
    AGENT_ROLE = "forecaster"
    DEFAULT_MODEL = "grok-4-1-fast-reasoning"

    SYSTEM_PROMPT = (
        "You are a world-class probability forecaster specialising in prediction "
        "markets. Your job is to estimate the TRUE probability that a market "
        "resolves YES, independent of the current market price.\n\n"
        "You reason like a superforecaster:\n"
        "1. Start with the BASE RATE -- how often do events of this type occur?\n"
        "2. Update based on CURRENT CONDITIONS -- what specific evidence shifts "
        "the probability up or down?\n"
        "3. Apply CALIBRATION -- are you overconfident? Adjust toward the base "
        "rate if uncertain.\n"
        "4. State your confidence in your own estimate (how sure are you that "
        "your probability is well-calibrated?).\n\n"
        "Return your analysis as a JSON object (inside a ```json``` code block) "
        "with the following keys:\n"
        '  "probability": float (0.0-1.0, your estimated TRUE YES probability),\n'
        '  "confidence": float (0.0-1.0, confidence in your own estimate),\n'
        '  "base_rate": float (0.0-1.0, the base rate you started from),\n'
        '  "side": "yes" or "no" (which side has positive EV at current prices),\n'
        '  "reasoning": string (detailed reasoning including base rate, updates, '
        "and calibration notes)"
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)
        portfolio_note = ""
        if context.get("portfolio"):
            portfolio_note = (
                f"\nPortfolio cash: ${context['portfolio'].get('cash', 0):,.2f}"
            )

        return (
            f"Analyse the following prediction market and estimate the TRUE "
            f"YES probability.\n\n"
            f"{summary}{portfolio_note}\n\n"
            f"Think step-by-step: base rate -> current conditions -> calibration.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        probability = self.clamp(raw_json.get("probability", 0.5))
        confidence = self.clamp(raw_json.get("confidence", 0.5))
        base_rate = self.clamp(raw_json.get("base_rate", 0.5))
        side = str(raw_json.get("side", "yes")).lower()
        if side not in ("yes", "no"):
            side = "yes" if probability >= 0.5 else "no"
        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "probability": probability,
            "confidence": confidence,
            "base_rate": base_rate,
            "side": side,
            "reasoning": reasoning,
        }
