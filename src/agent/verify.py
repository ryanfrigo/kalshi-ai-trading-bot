"""Adversarial-verify engine — research a fade, then try hard to REFUTE it.

A single edge candidate (a NO I might fade) is never bought on the strength of
one optimistic take. This module formalizes the research -> adversarial-verify
orchestration the agent runs before committing capital:

  1. ``CatalystResearcher`` — given the live candidate (ticker, question, the
     executable NO ask), produce precise RESOLUTION CRITERIA, an estimate of the
     TRUE yes-probability, and the CATALYST + its DIRECTION (adverse / none /
     positive / live). It carries the discipline this repo paid to learn: a
     liquid Kalshi book is already the sharp price; named-event longshots usually
     have a real catalyst; a NO is a fade only when true-YES sits well below the
     implied YES AND nothing positive/live is driving it.
  2. ``Skeptic`` — an adversary whose only job is to REFUTE the fade. It finds the
     single most plausible path by which YES resolves, stress-tests the
     researcher's number down, and rules whether the fade SURVIVES.
  3. ``Calibrator`` (``aggregate_verdict``) — THE deterministic gate. It does NOT
     trust the LLM's recommendation or size; it recomputes the edge in points,
     applies the hard gating rules (survives, >=5 pts edge, no positive/live
     catalyst, not an election frontrunner, and never overriding a skeptic PASS),
     and recomputes the size hint (``full`` reserved for genuine sub-5%
     structural longshots only).

Design contract (mirrors ``edge``/``learnings``/``settle``): every function here
is PURE — no ``datetime.now``, no file IO, no network. The candidate, the
research dict, and the skeptic dict are all passed IN. The ONLY thing that is not
pure is ``run_verify``, and it stays pure-by-injection: the caller hands it an
async ``llm_call``. This module never imports an LLM client — the IO/LLM adapter
lives only in the CLI layer (``cli.py verify``).
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 0. JSON schemas — the structured outputs each LLM role must return
# ---------------------------------------------------------------------------

RESEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "resolution_criteria": {"type": "string"},
        "true_yes_pct": {"type": "number"},
        "catalyst": {"type": "string"},
        "direction": {"type": "string", "enum": ["adverse", "none", "positive", "live"]},
        "is_frontrunner": {"type": "boolean"},
        "verdict": {"type": "string"},
    },
    "required": ["resolution_criteria", "true_yes_pct", "catalyst", "direction", "is_frontrunner", "verdict"],
}

SKEPTIC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "strongest_yes_path": {"type": "string"},
        "adjusted_true_yes_pct": {"type": "number"},
        "survives": {"type": "boolean"},
        "recommend": {"type": "string", "enum": ["BUY_NO", "PASS"]},
        "size_hint": {"type": "string", "enum": ["full", "half", "quarter"]},
    },
    "required": ["strongest_yes_path", "adjusted_true_yes_pct", "survives", "recommend", "size_hint"],
}


# ---------------------------------------------------------------------------
# 1. Prompt builders — pure string assembly (the only "templating" here)
# ---------------------------------------------------------------------------

def build_research_prompt(candidate: Dict[str, Any]) -> str:
    """Prompt for the CatalystResearcher.

    Frames the candidate (ticker, question, executable NO ask and the implied
    YES it backs out to), states the discipline this repo learned the hard way,
    and asks for the precise RESOLUTION CRITERIA, the TRUE yes-probability as a
    percent, the CATALYST + DIRECTION, and a one-line VERDICT — returned ONLY as
    a JSON object matching ``RESEARCH_SCHEMA``. Pure.
    """
    ticker = candidate["ticker"]
    question = candidate["question"]
    no_ask = float(candidate["no_ask"])
    implied_yes = round((1.0 - no_ask) * 100.0)
    return f"""You are the CatalystResearcher for a Kalshi NO-fade decision.

MARKET
  ticker:   {ticker}
  question: {question}
  NO ask:   {no_ask:.2f}  (the executable price to BUY NO)
  implied YES = {implied_yes}%  (the price the book is charging for YES)

We are considering FADING this market by buying NO. Research it.

DISCIPLINE (this account paid to learn these — do not violate them):
  - a liquid Kalshi book is already the sharp price; large apparent edge on a
    deep, tight book is a red flag (stale / live-vs-pre-match data), not a gift.
  - named-event longshots usually have a real catalyst — find it before fading.
  - a NO is a fade only if true-YES is 5+ points below the implied YES AND there
    is no positive or live catalyst pushing YES.
  - for an election leg, a longshot NO is a fade only if this is NOT the
    frontrunner (a frontrunner longshot-NO is never a fade).

DELIVER
  - resolution_criteria: the PRECISE criteria under which this market resolves YES.
  - true_yes_pct: your best estimate of the TRUE yes-probability, as a percent 0-100.
  - catalyst: the catalyst/driver behind YES, or "none" if there is genuinely none.
  - direction: one of adverse | none | positive | live
        adverse  = the catalyst helps NO / hurts YES
        none     = no catalyst either way
        positive = a catalyst helps YES (do NOT fade into this)
        live     = the event is live / in-progress (do NOT fade a live event)
  - is_frontrunner: true if this is an election leg AND this is the frontrunner;
        false otherwise (false if it is not an election).
  - verdict: a one-line research verdict.

Return ONLY a JSON object with these keys: resolution_criteria, true_yes_pct,
catalyst, direction, is_frontrunner, verdict. No markdown, no prose."""


def build_skeptic_prompt(candidate: Dict[str, Any], research: Dict[str, Any]) -> str:
    """Prompt for the Skeptic/adversary.

    Echoes the candidate and the researcher's numbers back so the skeptic can
    attack them, then demands the single most plausible YES path, a stress-tested
    adjusted true-YES, and a SURVIVES ruling — returned ONLY as a JSON object
    matching ``SKEPTIC_SCHEMA``. Pure.
    """
    ticker = candidate["ticker"]
    question = candidate["question"]
    no_ask = float(candidate["no_ask"])
    implied_yes = round((1.0 - no_ask) * 100.0)
    true_yes_pct = research.get("true_yes_pct")
    catalyst = research.get("catalyst")
    direction = research.get("direction")
    return f"""You are the Skeptic. Your job is to REFUTE the fade. Default to skepticism.

MARKET
  ticker:   {ticker}
  question: {question}
  NO ask:   {no_ask:.2f}   ->   implied YES = {implied_yes}%

THE RESEARCHER CLAIMS (attack these)
  true_yes_pct: {true_yes_pct}%
  catalyst:     {catalyst}
  direction:    {direction}

Find the single most plausible path by which YES resolves. Assume the book is
sharp and the researcher is too optimistic about the fade. Stress-test the
true-YES estimate UPWARD along that path.

SURVIVAL RULE
  If a credible YES path makes the adjusted true-YES less than 5 points below the
  implied YES, the fade does NOT survive. Only a fade that clears that margin
  AFTER your strongest YES path survives.

DELIVER
  - strongest_yes_path: the single most plausible path by which YES resolves.
  - adjusted_true_yes_pct: your adjusted TRUE yes-probability (0-100) after the
        stress test.
  - survives: true only if the fade survives your strongest YES path.
  - recommend: BUY_NO or PASS.
  - size_hint: full | half | quarter.

Return ONLY a JSON object with these keys: strongest_yes_path,
adjusted_true_yes_pct, survives, recommend, size_hint. No markdown, no prose."""


# ---------------------------------------------------------------------------
# 2. Calibrator — THE deterministic verdict gate
# ---------------------------------------------------------------------------

def aggregate_verdict(
    candidate: Dict[str, Any],
    research: Dict[str, Any],
    skeptic: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministically combine research + skeptic into the final verdict.

    This gate does NOT trust the LLM's ``recommend``/``size_hint`` blindly — it
    recomputes the edge and re-derives both. The edge is the skeptic's implied
    TRUE-NO percent minus the price-implied NO percent, in points::

        edge_pts = round((100 - adjusted_true_yes_pct) - no_ask*100)

    BUY_NO requires ALL of: the fade SURVIVES, edge_pts >= 5, the catalyst
    direction is not ``positive`` or ``live``, this is not an election
    frontrunner, AND the skeptic did not itself say PASS (we never override a
    skeptic PASS into a BUY_NO). Otherwise PASS.

    ``size_hint``: ``none`` on PASS; ``full`` ONLY for a genuine sub-5% adverse/
    none-catalyst structural longshot; else ``half`` when edge_pts >= 10; else
    ``quarter``. Pure / deterministic. ``true_yes`` in the returned dict is the
    skeptic's post-stress-test number — the one the verdict is actually based on.
    """
    no_ask = float(candidate["no_ask"])
    adjusted_true_yes_pct = float(skeptic["adjusted_true_yes_pct"])
    edge_pts = round((100.0 - adjusted_true_yes_pct) - no_ask * 100.0)
    survives = bool(skeptic["survives"])
    direction = research["direction"]
    is_frontrunner = bool(research.get("is_frontrunner", False))
    skeptic_recommend = skeptic.get("recommend")

    blocked_direction = direction in {"positive", "live"}

    recommend = "PASS"
    if (
        survives
        and edge_pts >= 5
        and not blocked_direction
        and not is_frontrunner
        and skeptic_recommend == "BUY_NO"
    ):
        recommend = "BUY_NO"

    if recommend == "PASS":
        size_hint = "none"
    elif adjusted_true_yes_pct < 5.0 and direction in {"adverse", "none"}:
        size_hint = "full"
    elif edge_pts >= 10:
        size_hint = "half"
    else:
        size_hint = "quarter"

    if recommend == "BUY_NO":
        note = (
            f"BUY_NO: fade survives, edge {edge_pts:+d} pts (true-YES "
            f"{adjusted_true_yes_pct:.0f}% vs implied YES {round((1.0 - no_ask) * 100.0)}%), "
            f"direction={direction}, size={size_hint}."
        )
    else:
        reasons: List[str] = []
        if not survives:
            reasons.append("fade does not survive skeptic")
        if edge_pts < 5:
            reasons.append(f"edge {edge_pts:+d} pts < 5")
        if blocked_direction:
            reasons.append(f"{direction} catalyst blocks fade")
        if is_frontrunner:
            reasons.append("election frontrunner — never a fade")
        if skeptic_recommend == "PASS":
            reasons.append("skeptic recommends PASS")
        if not reasons:
            reasons.append("gate not cleared")
        note = (
            f"PASS: {'; '.join(reasons)} "
            f"(edge {edge_pts:+d} pts, survives={survives}, direction={direction})."
        )

    return {
        "ticker": candidate["ticker"],
        "edge_pts": edge_pts,
        "survives": survives,
        "recommend": recommend,
        "size_hint": size_hint,
        "true_yes": adjusted_true_yes_pct,
        "direction": direction,
        "note": note,
    }


def make_operator_llm(
    payload: Dict[str, Any],
) -> Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    """Build an ``llm_call`` that answers from operator-supplied research. PURE.

    ``payload`` is ``{"research": {...RESEARCH_SCHEMA...}, "skeptic":
    {...SKEPTIC_SCHEMA...}}`` — produced by a human or an agent that did the
    research out-of-band (e.g. Claude in-session with live web search). This
    keeps the verify gate usable with no API key, per the repo's agent-native
    design: the operator supplies the *judgments*, but the deterministic
    verdict still recomputes the edge off the live book, so the gate cannot be
    fudged by optimistic pricing.

    Validation is strict: a missing section or missing required key raises
    ``ValueError`` — a half-filled research file must never silently pass the
    gate.
    """
    def _validated(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        section = payload.get(name)
        if not isinstance(section, dict):
            raise ValueError(f"operator research file has no '{name}' object")
        missing = [k for k in schema.get("required", []) if k not in section]
        if missing:
            raise ValueError(f"'{name}' section missing required keys: {missing}")
        return section

    async def llm_call(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if schema is RESEARCH_SCHEMA:
            return _validated("research", RESEARCH_SCHEMA)
        if schema is SKEPTIC_SCHEMA:
            return _validated("skeptic", SKEPTIC_SCHEMA)
        raise ValueError("operator llm_call got an unknown schema")

    return llm_call


# ---------------------------------------------------------------------------
# 3. Orchestrator — wires the three pure pieces with an injected LLM
# ---------------------------------------------------------------------------

async def run_verify(
    candidate: Dict[str, Any],
    llm_call: Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Run research -> skeptic -> deterministic verdict for one candidate.

    ``llm_call`` is an INJECTED async callable ``(prompt, schema) -> dict`` — the
    only IO in this module, and it is supplied by the caller (the CLI wires it to
    the real LLM; tests pass a mock). This keeps ``verify`` import-clean and unit-
    testable: it imports no client. Returns the ``aggregate_verdict`` dict.
    """
    research = await llm_call(build_research_prompt(candidate), RESEARCH_SCHEMA)
    skeptic = await llm_call(build_skeptic_prompt(candidate, research), SKEPTIC_SCHEMA)
    return aggregate_verdict(candidate, research, skeptic)
