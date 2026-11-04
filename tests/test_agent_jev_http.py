"""Jev HTTP paths: request shape, error mapping, batching and retries.

All HTTP goes through an httpx.MockTransport — no real network, no real key.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from src.agent import jev
from src.agent.jev import classify_texts, jev_yes_pct, with_jev_floor


def _patch_client(monkeypatch, handler):
    """Force every httpx.AsyncClient created in the code under test onto MockTransport."""
    seen = []
    real = httpx.AsyncClient

    def record(request):
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **kw: real(*a, transport=httpx.MockTransport(record), **kw),
    )
    return seen


def _req(**kw):
    return {"model": jev.JEV_MODEL, "state": {"market_question": "Will X?"}, "questions": {}, **kw}


def _body(request):
    return json.loads(request.content)


def test_jev_yes_pct_missing_key_raises_without_calling_http(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = _patch_client(monkeypatch, lambda r: pytest.fail("HTTP must not be called"))

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        asyncio.run(jev_yes_pct(_req()))

    assert seen == []


def test_jev_yes_pct_posts_request_and_returns_percent(monkeypatch):
    request = _req()
    seen = _patch_client(monkeypatch, lambda r: httpx.Response(200, json={"answers": {"resolves_yes": {"noul": 0.42}}}))

    result = asyncio.run(jev_yes_pct(request, api_key="test-key"))

    assert result == pytest.approx(42.0)
    assert len(seen) == 1
    sent = seen[0]
    assert str(sent.url) == jev.DECISIONS_URL
    assert sent.method == "POST"
    assert sent.headers["authorization"] == "Bearer test-key"
    assert _body(sent) == request


def test_jev_yes_pct_non_200_raises_with_status_but_no_key(monkeypatch):
    _patch_client(monkeypatch, lambda r: httpx.Response(503, json={"error": "busy"}))

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(jev_yes_pct(_req(), api_key="test-key"))

    message = str(excinfo.value)
    assert "503" in message
    assert "test-key" not in message


def test_jev_yes_pct_non_json_body_raises_with_status(monkeypatch):
    _patch_client(monkeypatch, lambda r: httpx.Response(502, text="<html>bad gateway</html>"))

    with pytest.raises(RuntimeError, match="502"):
        asyncio.run(jev_yes_pct(_req(), api_key="test-key"))


def test_jev_yes_pct_malformed_body_raises_value_error(monkeypatch):
    _patch_client(monkeypatch, lambda r: httpx.Response(200, json={"answers": {}}))

    with pytest.raises(ValueError):
        asyncio.run(jev_yes_pct(_req(), api_key="test-key"))


def test_jev_yes_pct_out_of_range_raises_value_error(monkeypatch):
    _patch_client(monkeypatch, lambda r: httpx.Response(200, json={"answers": {"resolves_yes": {"noul": 1.5}}}))

    with pytest.raises(ValueError):
        asyncio.run(jev_yes_pct(_req(), api_key="test-key"))


def test_classify_texts_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = _patch_client(monkeypatch, lambda r: pytest.fail("HTTP must not be called"))

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        asyncio.run(classify_texts("rule", ["a"], api_key=None))

    assert seen == []


def test_classify_texts_splits_into_batches_and_preserves_order(monkeypatch):
    values = {"alpha": 0.9, "beta": 0.5, "gamma": 0.1}

    def handler(request):
        questions = _body(request)["questions"]
        answers = {}
        for i in range(len(questions)):
            text = next(t for t in values if t in questions[f"q{i}"]["instructions"])
            answers[f"q{i}"] = {"noul": values[text]}
        return httpx.Response(200, json={"answers": answers})

    seen = _patch_client(monkeypatch, handler)

    result = asyncio.run(classify_texts("rule", ["alpha", "beta", "gamma"], batch=2, api_key="test-key"))

    assert len(seen) == 2
    assert [len(_body(r)["questions"]) for r in seen] == [2, 1]
    assert result == pytest.approx([0.9, 0.5, 0.1])


def test_classify_texts_retries_malformed_responses_three_times(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"answers": {"q0": {"noul": 0.7}}})

    _patch_client(monkeypatch, handler)

    result = asyncio.run(classify_texts("rule", ["only"], api_key="test-key"))

    assert result == pytest.approx([0.7])
    assert len(calls) == 3


def test_classify_texts_failed_batch_is_none_while_other_batch_still_fills(monkeypatch):
    def handler(request):
        instructions = _body(request)["questions"]["q0"]["instructions"]
        if "bad" in instructions:
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"answers": {"q0": {"noul": 0.8}}})

    _patch_client(monkeypatch, handler)

    result = asyncio.run(classify_texts("rule", ["good", "bad"], batch=1, api_key="test-key"))

    assert result[0] == pytest.approx(0.8)
    assert result[1] is None


def test_classify_texts_empty_list_makes_no_request(monkeypatch):
    seen = _patch_client(monkeypatch, lambda r: pytest.fail("HTTP must not be called"))

    result = asyncio.run(classify_texts("rule", [], api_key="test-key"))

    assert result == []
    assert seen == []


def test_with_jev_floor_passes_unknown_schema_through_untouched():
    forecast = AsyncMock()

    async def llm_call(_prompt, _schema):
        return {"x": 1}

    wrapped = with_jev_floor(llm_call, "Will X?", forecast=forecast)
    out = asyncio.run(wrapped("prompt", {"type": "object"}))

    assert out == {"x": 1}
    forecast.assert_not_awaited()
