"""Shared pytest configuration.

Safety contract: a plain ``pytest`` run must ALWAYS be safe to execute —
on contributor machines and in CI — with no credentials configured.

Tests marked ``live`` talk to the real Kalshi API (some place and cancel
real orders) and/or spend real LLM credits. They are skipped unless you
explicitly opt in with BOTH the env var and the marker filter:

    RUN_LIVE_TESTS=1 pytest -m live

Never run live tests against an account whose losses you are not
prepared to accept.
"""

import os

import pytest

RUN_LIVE_TESTS = os.getenv("RUN_LIVE_TESTS", "").lower() in {"1", "true", "yes"}


@pytest.fixture(autouse=True)
def _governor_defaults_in_tmp(tmp_path, monkeypatch):
    """The governor's default state file and halt flag live in the real repo's
    data/runtime/. Point them at this test's tmp dir so no test can read or
    clear the operator's live baseline or kill switch."""
    from src.risk import risk_governor

    runtime = tmp_path / "data" / "runtime"
    monkeypatch.setattr(risk_governor, "DEFAULT_STATE_PATH", str(runtime / "risk_governor_state.json"))
    monkeypatch.setattr(risk_governor, "DEFAULT_HALT_FLAG_PATH", str(runtime / "TRADING_HALTED"))
    monkeypatch.setattr(risk_governor, "DEFAULT_AUDIT_PATH", str(runtime / "governor_audit.jsonl"))
    monkeypatch.delenv("KALSHI_RUNTIME_DIR", raising=False)


def pytest_collection_modifyitems(config, items):
    if RUN_LIVE_TESTS:
        return
    skip_live = pytest.mark.skip(
        reason="live test (real Kalshi API / real money) — set RUN_LIVE_TESTS=1 and pass -m live to run"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
