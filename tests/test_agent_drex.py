"""Drex forecaster: the Jev request with Drex's model, the floor, and the fallback to Jev."""
import asyncio
import json
from functools import partial

import httpx
import pytest

from src.agent import drex
from src.agent.jev import build_jev_request, with_jev_floor
from src.agent.verify import make_operator_llm, run_verify
from tests.test_agent_jev import RESEARCH, SKEPTIC

REQ = build_jev_request("Will X?", RESEARCH, SKEPTIC, today="2026-09-25")


def _noul(p):
    return lambda request: httpx.Response(200, json={"answers": {"resolves_yes": {"noul": p}}})


def _drex_answers(monkeypatch, handler):
    """Route the httpx client in drex_yes_pct to ``handler``; returns the requests seen."""
    seen = []
    real = httpx.AsyncClient

    def record(request):
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(record), **kw)
    )
    return seen


@pytest.fixture
def jev_calls(monkeypatch):
    calls = []

    async def fake_jev(request):
        calls.append(request)
        return 30.0

    monkeypatch.setattr(drex, "jev_yes_pct", fake_jev)
    return calls


def test_drex_gets_the_jev_request_with_its_own_model(monkeypatch, jev_calls):
    monkeypatch.setenv("NACE_API_KEY", "test-key")
    seen = _drex_answers(monkeypatch, _noul(0.04))
    record = {}
    assert asyncio.run(drex.drex_yes_pct(REQ, record)) == pytest.approx(4.0)
    assert str(seen[0].url) == drex.DREX_URL
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(seen[0].content) == {**REQ, "model": drex.DREX_MODEL}
    assert record == {"drex_model": drex.DREX_MODEL} and jev_calls == []


def _refuse(request):
    raise httpx.ConnectError("down", request=request)


@pytest.mark.parametrize(
    "handler, logged",
    [
        (lambda r: httpx.Response(503, json={"error": {"type": "overloaded"}}), "Drex HTTP 503"),
        (lambda r: httpx.Response(200, json={"answers": {}}), "malformed Drex response"),
        (_noul(1.5), "Drex probability out of range"),
        (_refuse, "down"),
    ],
    ids=["http-error", "malformed", "out-of-range", "unreachable"],
)
def test_drex_failure_falls_back_to_jev(monkeypatch, caplog, jev_calls, handler, logged):
    monkeypatch.setenv("NACE_API_KEY", "test-key")
    _drex_answers(monkeypatch, handler)
    record = {}
    assert asyncio.run(drex.drex_yes_pct(REQ, record)) == 30.0
    assert jev_calls == [REQ] and record == {}
    assert logged in caplog.text and "falling back to Jev" in caplog.text


def test_no_key_goes_straight_to_jev(monkeypatch, jev_calls):
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    seen = _drex_answers(monkeypatch, _noul(0.04))
    assert asyncio.run(drex.drex_yes_pct(REQ)) == 30.0
    assert seen == [] and jev_calls == [REQ]


def test_drex_floor_tightens_the_gate_and_is_labelled(monkeypatch, jev_calls):
    monkeypatch.setenv("NACE_API_KEY", "test-key")
    _drex_answers(monkeypatch, _noul(0.08))
    record = {}
    llm = with_jev_floor(
        make_operator_llm({"research": RESEARCH, "skeptic": SKEPTIC}),
        "Will X?",
        forecast=partial(drex.drex_yes_pct, record=record),
        record=record,
    )
    verdict = asyncio.run(run_verify({"ticker": "T", "question": "Will X?", "no_ask": 0.90}, llm))
    assert verdict["true_yes"] == pytest.approx(8.0) and verdict["recommend"] == "PASS"
    assert record["drex_model"] == drex.DREX_MODEL
    assert record["jev_true_yes_pct"] == pytest.approx(8.0)
