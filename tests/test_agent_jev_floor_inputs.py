"""Under --jev a malformed skeptic number refuses; the floor can't launder it into a clean one."""
import asyncio

import pytest

from src.agent.jev import with_jev_floor
from src.agent.verify import RESEARCH_SCHEMA, SKEPTIC_SCHEMA

RESEARCH = {
    "resolution_criteria": "x", "true_yes_pct": 2.0, "catalyst": "none", "direction": "none",
    "is_frontrunner": False, "verdict": "fade",
}
SKEPTIC = {
    "strongest_yes_path": "none", "adjusted_true_yes_pct": 2.0, "survives": True,
    "recommend": "BUY_NO", "size_hint": "full",
}


def _run(skeptic, jev_pct=1.0):
    calls = []

    async def llm(prompt, schema):
        return RESEARCH if schema is RESEARCH_SCHEMA else skeptic

    async def forecast(req):
        calls.append(req)
        return jev_pct

    wrapped = with_jev_floor(llm, "q", forecast=forecast)

    async def go():
        await wrapped("p", RESEARCH_SCHEMA)
        return await wrapped("p", SKEPTIC_SCHEMA)

    return asyncio.run(go()), calls


@pytest.mark.parametrize("value", [-50, -0.1, 100.1, True, False, "2", None, float("nan"), float("inf"), []])
def test_malformed_skeptic_number_refuses_before_any_jev_call(value):
    calls = []

    async def llm(prompt, schema):
        return RESEARCH if schema is RESEARCH_SCHEMA else {**SKEPTIC, "adjusted_true_yes_pct": value}

    async def forecast(req):
        calls.append(req)
        return 1.0

    wrapped = with_jev_floor(llm, "q", forecast=forecast)
    with pytest.raises(ValueError, match="adjusted_true_yes_pct"):
        asyncio.run(wrapped("p", SKEPTIC_SCHEMA))
    assert calls == []


def test_missing_skeptic_number_refuses():
    skeptic = {k: v for k, v in SKEPTIC.items() if k != "adjusted_true_yes_pct"}
    with pytest.raises(ValueError, match="adjusted_true_yes_pct"):
        _run(skeptic)


@pytest.mark.parametrize("skeptic_pct,jev_pct,expected", [(2.0, 1.0, 2.0), (2.0, 9.0, 9.0), (0, 0.0, 0), (100, 3.0, 100)])
def test_valid_numbers_are_still_floored_at_jev(skeptic_pct, jev_pct, expected):
    out, calls = _run({**SKEPTIC, "adjusted_true_yes_pct": skeptic_pct}, jev_pct)
    assert out["adjusted_true_yes_pct"] == expected
    assert len(calls) == 1
