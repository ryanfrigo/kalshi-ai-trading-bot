"""
Jev (TypeSafe's calibrated decisions model) as an independent forecaster for the verify gate.

Jev is served on OpenRouter as ``~typesafe/jev-latest`` through the alpha Decisions
endpoint (NOT chat/completions): you send a ``state`` of facts plus typed
``questions`` and get calibrated probabilities back. It does no research of its
own — it reasons over exactly the facts it is handed.

How it plugs into ``verify``: the researcher/skeptic (operator file or LLM) supply
the facts; Jev independently turns those facts into P(YES); the skeptic's
``adjusted_true_yes_pct`` is floored at Jev's number. Jev can only make the gate
STRICTER, never looser. Jev never sees the market price or the operator's
probability estimates — both would anchor it (observed: passing the price pulls
its answer toward the book).
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.agent.verify import RESEARCH_SCHEMA, SKEPTIC_SCHEMA, check_probability

JEV_MODEL = "~typesafe/jev-latest"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

LlmCall = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def build_jev_request(
    question: str,
    research: Dict[str, Any],
    skeptic: Dict[str, Any],
    facts: Optional[List[str]] = None,
    today: Optional[str] = None,
) -> Dict[str, Any]:
    """Decisions-API request asking P(YES) from researched facts. PURE.

    Only qualitative evidence goes in the state — no prices, no probabilities.
    """
    state: Dict[str, Any] = {
        "today": today or date.today().isoformat(),
        "market_question": question,
        "resolution_criteria": research.get("resolution_criteria", ""),
        "catalyst": research.get("catalyst", ""),
        "research_verdict": research.get("verdict", ""),
        "strongest_yes_path": skeptic.get("strongest_yes_path", ""),
    }
    if facts:
        state["facts"] = list(facts)
    return {
        "model": JEV_MODEL,
        "state": state,
        "questions": {
            "resolves_yes": {
                "type": "noul",
                "instructions": (
                    "Given only these facts, will this prediction market resolve YES "
                    "under its exact resolution criteria?"
                ),
                "criteria": {
                    "true": "The resolution criteria are met by the deadline",
                    "false": "They are not met",
                },
            }
        },
    }


def parse_jev_yes_pct(resp: Dict[str, Any]) -> float:
    """Extract P(YES) as a percent 0-100 from a Decisions response. PURE.

    Raises ValueError on a malformed/error response — a missing forecast must
    fail loudly, never silently let the gate through.
    """
    try:
        p = float(resp["answers"]["resolves_yes"]["noul"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed Jev response: {str(resp)[:300]}") from e
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Jev probability out of range: {p}")
    return p * 100.0


async def jev_yes_pct(request: Dict[str, Any], api_key: Optional[str] = None) -> float:
    """POST one Decisions request and return Jev's P(YES) percent."""
    import httpx

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — Jev unavailable")
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            DECISIONS_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=request,
        )
    try:
        body = r.json()
    except ValueError:
        body = {"error": r.text[:300]}
    if r.status_code != 200:
        raise RuntimeError(f"Jev HTTP {r.status_code}: {str(body)[:300]}")
    return parse_jev_yes_pct(body)


def with_jev_floor(
    llm_call: LlmCall,
    question: str,
    forecast: Callable[[Dict[str, Any]], Awaitable[float]] = jev_yes_pct,
    facts: Optional[List[str]] = None,
    record: Optional[Dict[str, Any]] = None,
) -> LlmCall:
    """Wrap an ``llm_call`` so the skeptic's true-YES is floored at Jev's forecast.

    ``forecast`` takes a Decisions request and returns a percent (injected for
    tests). ``record`` (optional dict) receives ``jev_true_yes_pct`` for display.
    """
    captured: Dict[str, Any] = {}

    async def wrapped(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        out = await llm_call(prompt, schema)
        if schema is RESEARCH_SCHEMA:
            captured["research"] = out
            return out
        if schema is SKEPTIC_SCHEMA:
            # Check before flooring, or max() turns a malformed value into a clean number.
            check_probability("adjusted_true_yes_pct", out.get("adjusted_true_yes_pct"))
            req = build_jev_request(question, captured.get("research", {}), out, facts)
            jev_pct = await forecast(req)
            if record is not None:
                record["jev_true_yes_pct"] = jev_pct
            floored = max(float(out["adjusted_true_yes_pct"]), jev_pct)
            return {**out, "adjusted_true_yes_pct": floored}
        return out

    return wrapped


# ---------------------------------------------------------------------------
# Batch text -> rule classification (where Jev measurably earns its keep)
# ---------------------------------------------------------------------------
# Measured 2026-09-26: endorsement posts 206/207, word-form mention rules 92/107,
# ~50k text x rule pairs/min. Use it to FILTER large text volumes, then confirm
# hits yourself. Never ask it to count or compare numbers.

def build_classify_request(
    rule: str, texts: List[str], context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """One Decisions request asking, per text, "does this satisfy the rule?". PURE."""
    state: Dict[str, Any] = {"market_rule": rule}
    if context:
        state["context"] = context
    return {
        "model": JEV_MODEL,
        "state": state,
        "questions": {
            f"q{i}": {
                "type": "noul",
                "instructions": f'Text: """{t[:3000]}"""\nDoes THIS text satisfy the market rule for YES?',
                "criteria": {"true": "Satisfies the rule", "false": "Does not"},
            }
            for i, t in enumerate(texts)
        },
    }


def parse_classify(resp: Dict[str, Any], n: int) -> List[float]:
    """P(satisfies) per text from a classify response. PURE; raises on malformed."""
    try:
        return [float(resp["answers"][f"q{i}"]["noul"]) for i in range(n)]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed Jev classify response: {str(resp)[:300]}") from e


async def classify_texts(
    rule: str,
    texts: List[str],
    context: Optional[Dict[str, Any]] = None,
    batch: int = 20,
    concurrency: int = 8,
    api_key: Optional[str] = None,
) -> List[Optional[float]]:
    """Classify many texts against one rule; returns P per text (None if a batch failed)."""
    import asyncio

    import httpx

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — Jev unavailable")
    out: List[Optional[float]] = [None] * len(texts)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120) as http:
        async def run(start: int) -> None:
            chunk = texts[start:start + batch]
            async with sem:
                for _ in range(3):
                    try:
                        r = await http.post(
                            DECISIONS_URL,
                            headers={"Authorization": f"Bearer {key}"},
                            json=build_classify_request(rule, chunk, context),
                        )
                        out[start:start + len(chunk)] = parse_classify(r.json(), len(chunk))
                        return
                    except (httpx.HTTPError, ValueError):
                        continue
        await asyncio.gather(*(run(i) for i in range(0, len(texts), batch)))
    return out
