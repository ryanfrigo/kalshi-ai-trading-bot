"""Tests for the adversarial-verify engine — research -> skeptic -> verdict.

Pure-function tests with in-memory fixtures (no network, no live Kalshi, no LLM).
They exhaustively cover the deterministic ``aggregate_verdict`` gate (every BUY_NO
/ PASS branch and every size_hint tier), the two prompt builders (the discipline
tokens and echoed research fields tests grep for), and ``run_verify``'s
orchestration against a MOCK ``llm_call`` (research-then-skeptic ordering and the
PASS pass-through).

Every ``edge_pts`` is hand-verified in a comment with the formula
``edge_pts = round((100 - adjusted_true_yes_pct) - no_ask*100)``.
"""
import asyncio

from src.agent.verify import (
    RESEARCH_SCHEMA,
    SKEPTIC_SCHEMA,
    build_research_prompt,
    build_skeptic_prompt,
    aggregate_verdict,
    run_verify,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _candidate(ticker="T", question="Will X happen?", no_ask=0.95):
    return {"ticker": ticker, "question": question, "no_ask": no_ask}


def _research(true_yes_pct=2.0, direction="none", catalyst="none", is_frontrunner=False):
    return {
        "resolution_criteria": "Resolves YES if X.",
        "true_yes_pct": true_yes_pct,
        "catalyst": catalyst,
        "direction": direction,
        "is_frontrunner": is_frontrunner,
        "verdict": "looks like a fade",
    }


def _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="BUY_NO", size_hint="full"):
    return {
        "strongest_yes_path": "some path",
        "adjusted_true_yes_pct": adjusted_true_yes_pct,
        "survives": survives,
        "recommend": recommend,
        "size_hint": size_hint,
    }


# ---------------------------------------------------------------------------
# aggregate_verdict — BUY_NO / PASS gating (every branch)
# ---------------------------------------------------------------------------

def test_buy_no_when_survives_edge_and_adverse_or_none():
    # no_ask=0.90 (implied NO=90), adjusted_true_yes=2, direction none, survives.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8  (>=5) -> BUY_NO
    v = aggregate_verdict(_candidate(no_ask=0.90), _research(direction="none"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="BUY_NO"))
    assert v["edge_pts"] == 8
    assert v["recommend"] == "BUY_NO"


def test_pass_when_does_not_survive():
    # survives=False but edge>=5 and direction none.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8 -> still PASS (no survive)
    v = aggregate_verdict(_candidate(no_ask=0.90), _research(direction="none"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=False, recommend="BUY_NO"))
    assert v["edge_pts"] == 8
    assert v["recommend"] == "PASS"


def test_pass_when_edge_below_5():
    # no_ask=0.95 (implied NO=95), adjusted_true_yes=2, survives, direction adverse.
    # edge_pts = round((100 - 2) - 0.95*100) = round(98 - 95) = 3  (<5) -> PASS
    v = aggregate_verdict(_candidate(no_ask=0.95), _research(direction="adverse"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="BUY_NO"))
    assert v["edge_pts"] == 3
    assert v["recommend"] == "PASS"


def test_pass_when_positive_catalyst():
    # direction positive blocks the fade even with a big edge.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8 -> PASS (positive)
    v = aggregate_verdict(_candidate(no_ask=0.90), _research(direction="positive"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="BUY_NO"))
    assert v["edge_pts"] == 8
    assert v["recommend"] == "PASS"


def test_pass_when_live_catalyst():
    # direction live blocks the fade.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8 -> PASS (live)
    v = aggregate_verdict(_candidate(no_ask=0.90), _research(direction="live"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="BUY_NO"))
    assert v["edge_pts"] == 8
    assert v["recommend"] == "PASS"


def test_pass_when_frontrunner():
    # adverse + big edge + survives, but an election frontrunner is never a fade.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8 -> PASS (frontrunner)
    v = aggregate_verdict(_candidate(no_ask=0.90),
                          _research(direction="adverse", is_frontrunner=True),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="BUY_NO"))
    assert v["edge_pts"] == 8
    assert v["recommend"] == "PASS"


def test_pass_when_skeptic_recommends_pass():
    # survives, edge>=5, direction none — but the skeptic said PASS: never override.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8 -> PASS (skeptic PASS)
    v = aggregate_verdict(_candidate(no_ask=0.90), _research(direction="none"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="PASS"))
    assert v["edge_pts"] == 8
    assert v["recommend"] == "PASS"


# ---------------------------------------------------------------------------
# aggregate_verdict — size_hint tiers
# ---------------------------------------------------------------------------

def test_size_full_for_sub5_adverse():
    # no_ask=0.90, adjusted_true_yes=2.0 (<5), direction adverse, survives, BUY_NO.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8 (>=5) -> BUY_NO, full
    v = aggregate_verdict(_candidate(no_ask=0.90), _research(direction="adverse"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="BUY_NO"))
    assert v["edge_pts"] == 8
    assert v["recommend"] == "BUY_NO"
    assert v["size_hint"] == "full"


def test_size_half_when_not_sub5_but_edge_ge_10():
    # no_ask=0.30 (implied NO=30), adjusted_true_yes=30 (>=5), direction none.
    # edge_pts = round((100 - 30) - 0.30*100) = round(70 - 30) = 40 (>=10) -> half
    v = aggregate_verdict(_candidate(no_ask=0.30), _research(direction="none"),
                          _skeptic(adjusted_true_yes_pct=30.0, survives=True, recommend="BUY_NO"))
    assert v["edge_pts"] == 40
    assert v["recommend"] == "BUY_NO"
    assert v["size_hint"] == "half"


def test_size_quarter_when_not_sub5_and_edge_5_to_9():
    # no_ask=0.75 (implied NO=75), adjusted_true_yes=20 (>=5), direction none.
    # edge_pts = round((100 - 20) - 0.75*100) = round(80 - 75) = 5 (>=5, <10) -> quarter
    v = aggregate_verdict(_candidate(no_ask=0.75), _research(direction="none"),
                          _skeptic(adjusted_true_yes_pct=20.0, survives=True, recommend="BUY_NO"))
    assert v["edge_pts"] == 5
    assert v["recommend"] == "BUY_NO"
    assert v["size_hint"] == "quarter"


def test_size_none_when_pass():
    # Any PASS case -> size_hint none. survives=False here.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8 -> PASS, none
    v = aggregate_verdict(_candidate(no_ask=0.90), _research(direction="none"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=False, recommend="BUY_NO"))
    assert v["recommend"] == "PASS"
    assert v["size_hint"] == "none"


# ---------------------------------------------------------------------------
# aggregate_verdict — returned dict shape
# ---------------------------------------------------------------------------

def test_verdict_dict_shape():
    # no_ask=0.90, adjusted_true_yes=2.0, direction none, survives, BUY_NO.
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8
    v = aggregate_verdict(_candidate(ticker="ZZZ", no_ask=0.90), _research(direction="none"),
                          _skeptic(adjusted_true_yes_pct=2.0, survives=True, recommend="BUY_NO"))
    assert set(v.keys()) == {
        "ticker", "edge_pts", "survives", "recommend",
        "size_hint", "true_yes", "direction", "note",
    }
    assert v["ticker"] == "ZZZ"
    assert v["true_yes"] == 2.0          # the skeptic's adjusted number
    assert v["direction"] == "none"
    assert v["edge_pts"] == 8
    assert isinstance(v["note"], str) and v["note"]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_research_prompt_includes_key_fields():
    p = build_research_prompt(_candidate(ticker="ABC", question="Will Y?", no_ask=0.93))
    assert "ABC" in p
    assert "Will Y?" in p
    assert "implied" in p.lower()
    # implied YES = round((1 - 0.93) * 100) = 7
    assert "7" in p
    # discipline tokens
    assert "sharp" in p.lower()
    assert "catalyst" in p.lower()
    assert "frontrunner" in p.lower()
    # the fade rule mentions "5" points
    assert "5" in p
    # asks for the structured fields
    assert "resolution" in p.lower()
    assert "direction" in p.lower()
    assert "verdict" in p.lower()


def test_skeptic_prompt_includes_research_and_refute_language():
    research = _research(true_yes_pct=12, direction="adverse", catalyst="downgrade")
    p = build_skeptic_prompt(_candidate(), research)
    assert "refute" in p.lower()
    assert ("skeptic" in p.lower()) or ("skepticism" in p.lower())
    assert ("strongest" in p.lower()) or ("most plausible" in p.lower())
    # echoes the researcher's claims back so the skeptic can attack them
    assert "downgrade" in p
    assert "12" in p            # true_yes echoed
    # the survival "5"-point rule is present
    assert "5" in p


# ---------------------------------------------------------------------------
# run_verify — orchestration against a MOCK llm_call (no network)
# ---------------------------------------------------------------------------

def test_run_verify_orchestrates_with_mock():
    calls = []
    research_out = _research(true_yes_pct=2.0, direction="none")
    skeptic_out = _skeptic(adjusted_true_yes_pct=2.0, survives=True,
                           recommend="BUY_NO", size_hint="full")

    async def fake_llm(prompt, schema):
        calls.append((prompt, schema))
        if schema is RESEARCH_SCHEMA:
            return research_out
        return skeptic_out

    cand = _candidate(no_ask=0.90)
    verdict = asyncio.run(run_verify(cand, fake_llm))
    # two LLM calls, in order: research then skeptic
    assert len(calls) == 2
    assert calls[0][1] is RESEARCH_SCHEMA
    assert calls[1][1] is SKEPTIC_SCHEMA
    # the skeptic prompt must reference the research output (direction echoed)
    assert "none" in calls[1][0].lower()
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8, survives, none, sub-5
    assert verdict["recommend"] == "BUY_NO"
    assert verdict["edge_pts"] == 8
    assert verdict["size_hint"] == "full"
    assert verdict["true_yes"] == 2.0


def test_run_verify_passes_through_skeptic_pass():
    research_out = _research(true_yes_pct=2.0, direction="none")
    skeptic_out = _skeptic(adjusted_true_yes_pct=2.0, survives=False,
                           recommend="PASS", size_hint="full")

    async def fake_llm(prompt, schema):
        if schema is RESEARCH_SCHEMA:
            return research_out
        return skeptic_out

    cand = _candidate(no_ask=0.90)
    verdict = asyncio.run(run_verify(cand, fake_llm))
    # edge_pts = round((100 - 2) - 0.90*100) = round(98 - 90) = 8, but survives=False -> PASS
    assert verdict["recommend"] == "PASS"
    assert verdict["size_hint"] == "none"


# ---------------------------------------------------------------------------
# make_operator_llm — the no-API-key operator research path
# ---------------------------------------------------------------------------

def test_operator_llm_feeds_run_verify_end_to_end():
    from src.agent.verify import make_operator_llm

    payload = {"research": _research(), "skeptic": _skeptic()}
    llm = make_operator_llm(payload)
    verdict = asyncio.run(run_verify(_candidate(no_ask=0.90), llm))
    # Same numbers as the mock-LLM BUY_NO case: edge = round(98 - 90) = 8 pts.
    assert verdict["recommend"] == "BUY_NO"
    assert verdict["edge_pts"] == 8


def test_operator_llm_dispatches_by_schema():
    from src.agent.verify import make_operator_llm

    llm = make_operator_llm({"research": _research(), "skeptic": _skeptic()})
    research = asyncio.run(llm("any prompt", RESEARCH_SCHEMA))
    skeptic = asyncio.run(llm("any prompt", SKEPTIC_SCHEMA))
    assert research["true_yes_pct"] == 2.0
    assert skeptic["recommend"] == "BUY_NO"


def test_operator_llm_missing_section_raises():
    import pytest
    from src.agent.verify import make_operator_llm

    llm = make_operator_llm({"research": _research()})  # no skeptic section
    with pytest.raises(ValueError, match="skeptic"):
        asyncio.run(llm("any prompt", SKEPTIC_SCHEMA))


def test_operator_llm_missing_required_key_raises():
    import pytest
    from src.agent.verify import make_operator_llm

    research = _research()
    del research["direction"]
    llm = make_operator_llm({"research": research, "skeptic": _skeptic()})
    with pytest.raises(ValueError, match="direction"):
        asyncio.run(llm("any prompt", RESEARCH_SCHEMA))


def test_operator_llm_unknown_schema_raises():
    import pytest
    from src.agent.verify import make_operator_llm

    llm = make_operator_llm({"research": _research(), "skeptic": _skeptic()})
    with pytest.raises(ValueError, match="unknown schema"):
        asyncio.run(llm("any prompt", {"required": ["other"]}))
