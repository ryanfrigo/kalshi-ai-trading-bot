"""
Bull Researcher Agent -- makes the strongest possible case for YES.

Uses GPT-4o (via OpenRouter) by default.  Focuses on:
- Arguments FOR the event happening
- Supporting evidence
- Probability floor estimate (lower bound for YES probability)
"""

from experimental.agents.base_agent import BaseAgent


class BullResearcher(BaseAgent):
    """Researches and presents the bullish (YES) case for a market."""

    AGENT_NAME = "bull_researcher"
    AGENT_ROLE = "bull_researcher"
    DEFAULT_MODEL = "openai/o3"

    SYSTEM_PROMPT = (
        "You are a conviction-driven research analyst whose job is to make the "
        "STRONGEST possible case that this event WILL happen (the YES outcome).\n\n"
        "Your goal is adversarial truth-seeking: you are the bull in a bull-vs-bear "
        "debate. Another analyst will argue the opposite side.\n\n"
        "Structure your analysis:\n"
        "1. THESIS -- State your bullish thesis in one sentence.\n"
        "2. KEY ARGUMENTS -- List 3-5 concrete reasons the event is likely.\n"
        "   Each argument should cite specific evidence, data, or precedent.\n"
        "3. PROBABILITY FLOOR -- What is the MINIMUM reasonable YES probability, "
        "   even accounting for bear arguments? This is the floor, not your "
        "   point estimate.\n"
        "4. CATALYSTS -- What near-term events could push the probability higher?\n\n"
        "Be specific and evidence-based. Avoid vague hand-waving.\n\n"
        "Return your analysis as a JSON object (inside a ```json``` code block) "
        "with the following keys:\n"
        '  "probability": float (0.0-1.0, your YES probability estimate),\n'
        '  "probability_floor": float (0.0-1.0, minimum reasonable YES probability),\n'
        '  "confidence": float (0.0-1.0, confidence in your analysis),\n'
        '  "key_arguments": list of strings (3-5 arguments for YES),\n'
        '  "catalysts": list of strings (near-term bullish catalysts),\n'
        '  "reasoning": string (your detailed bull thesis)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)

        # Include forecaster output if available from prior analysis
        forecaster_note = ""
        if context.get("forecaster_result"):
            fc = context["forecaster_result"]
            forecaster_note = (
                f"\n\nThe Forecaster estimated a YES probability of "
                f"{fc.get('probability', '?')} with confidence "
                f"{fc.get('confidence', '?')}."
            )

        return (
            f"Make the STRONGEST possible case that the following market "
            f"resolves YES.\n\n"
            f"{summary}{forecaster_note}\n\n"
            f"Be specific, evidence-based, and persuasive.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        probability = self.clamp(raw_json.get("probability", 0.6))
        probability_floor = self.clamp(raw_json.get("probability_floor", 0.3))
        confidence = self.clamp(raw_json.get("confidence", 0.5))

        key_arguments = raw_json.get("key_arguments", [])
        if not isinstance(key_arguments, list):
            key_arguments = [str(key_arguments)]
        key_arguments = [str(a) for a in key_arguments][:10]

        catalysts = raw_json.get("catalysts", [])
        if not isinstance(catalysts, list):
            catalysts = [str(catalysts)]
        catalysts = [str(c) for c in catalysts][:10]

        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "probability": probability,
            "probability_floor": probability_floor,
            "confidence": confidence,
            "key_arguments": key_arguments,
            "catalysts": catalysts,
            "reasoning": reasoning,
        }
