"""
Bear Researcher Agent -- makes the strongest possible case for NO.

Uses Gemini (via OpenRouter) by default.  Focuses on:
- Arguments AGAINST the event happening
- Risk factors and counterarguments
- Probability ceiling estimate (upper bound for YES probability)
"""

from experimental.agents.base_agent import BaseAgent


class BearResearcher(BaseAgent):
    """Researches and presents the bearish (NO) case for a market."""

    AGENT_NAME = "bear_researcher"
    AGENT_ROLE = "bear_researcher"
    DEFAULT_MODEL = "google/gemini-3-pro-preview"

    SYSTEM_PROMPT = (
        "You are a sceptical risk analyst whose job is to make the STRONGEST "
        "possible case that this event will NOT happen (the NO outcome).\n\n"
        "Your goal is adversarial truth-seeking: you are the bear in a "
        "bull-vs-bear debate. Another analyst will argue the opposite side.\n\n"
        "Structure your analysis:\n"
        "1. COUNTER-THESIS -- State your bearish thesis in one sentence.\n"
        "2. KEY ARGUMENTS -- List 3-5 concrete reasons the event is unlikely.\n"
        "   Each argument should cite specific evidence, data, or precedent.\n"
        "3. PROBABILITY CEILING -- What is the MAXIMUM reasonable YES probability, "
        "   even accounting for bull arguments? This is the ceiling, not your "
        "   point estimate.\n"
        "4. RISK FACTORS -- What could go wrong for YES holders?\n"
        "5. HISTORICAL PRECEDENT -- Have similar events failed before?\n\n"
        "Be rigorous and evidence-based. Challenge every assumption.\n\n"
        "Return your analysis as a JSON object (inside a ```json``` code block) "
        "with the following keys:\n"
        '  "probability": float (0.0-1.0, your YES probability estimate -- '
        "typically lower than the bull's),\n"
        '  "probability_ceiling": float (0.0-1.0, maximum reasonable YES probability),\n'
        '  "confidence": float (0.0-1.0, confidence in your analysis),\n'
        '  "key_arguments": list of strings (3-5 arguments for NO),\n'
        '  "risk_factors": list of strings (risks for YES holders),\n'
        '  "reasoning": string (your detailed bear thesis)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)

        # Include bull researcher output if available
        bull_note = ""
        if context.get("bull_result"):
            bull = context["bull_result"]
            bull_args = bull.get("key_arguments", [])
            args_str = "; ".join(bull_args[:5]) if bull_args else "N/A"
            bull_note = (
                f"\n\nThe Bull Researcher estimated YES probability at "
                f"{bull.get('probability', '?')} and argued: {args_str}\n"
                f"Counter their arguments directly."
            )

        # Include forecaster output if available
        forecaster_note = ""
        if context.get("forecaster_result"):
            fc = context["forecaster_result"]
            forecaster_note = (
                f"\n\nThe Forecaster estimated a YES probability of "
                f"{fc.get('probability', '?')}."
            )

        return (
            f"Make the STRONGEST possible case that the following market "
            f"resolves NO.\n\n"
            f"{summary}{forecaster_note}{bull_note}\n\n"
            f"Challenge every assumption. Be rigorous.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        probability = self.clamp(raw_json.get("probability", 0.4))
        probability_ceiling = self.clamp(raw_json.get("probability_ceiling", 0.7))
        confidence = self.clamp(raw_json.get("confidence", 0.5))

        key_arguments = raw_json.get("key_arguments", [])
        if not isinstance(key_arguments, list):
            key_arguments = [str(key_arguments)]
        key_arguments = [str(a) for a in key_arguments][:10]

        risk_factors = raw_json.get("risk_factors", [])
        if not isinstance(risk_factors, list):
            risk_factors = [str(risk_factors)]
        risk_factors = [str(r) for r in risk_factors][:10]

        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "probability": probability,
            "probability_ceiling": probability_ceiling,
            "confidence": confidence,
            "key_arguments": key_arguments,
            "risk_factors": risk_factors,
            "reasoning": reasoning,
        }
