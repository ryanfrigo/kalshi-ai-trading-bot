"""Every AI-client construction site hands its DatabaseManager to the client.

The client mirrors spend into ``db_manager.upsert_daily_cost``. A site that
builds ``XAIClient()`` / ``OpenRouterClient()`` without it leaves the DB budget
check blind. Fully offline: clients, DB and Kalshi are patched; no LLM calls.
"""
import ast
import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import cli
from src.clients.xai_client import XAIClient
from src.clients.openrouter_client import OpenRouterClient
from src.jobs import automated_performance_analyzer as apa

ROOT = Path(__file__).resolve().parent.parent


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _db_instance():
    db = MagicMock(name="DatabaseManager-instance")
    db.initialize = AsyncMock()
    db.upsert_daily_cost = AsyncMock()
    return db


# --- 1. AutomatedPerformanceAnalyzer.initialize -----------------------------

async def test_performance_analyzer_passes_db_to_xai_client():
    db = _db_instance()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(apa, "DatabaseManager", MagicMock(return_value=db))
        mp.setattr(apa, "KalshiClient", MagicMock())
        xai = MagicMock(name="XAIClient")
        mp.setattr(apa, "XAIClient", xai)

        analyzer = apa.AutomatedPerformanceAnalyzer()
        await analyzer.initialize()

    db.initialize.assert_awaited_once()
    xai.assert_called_once_with(db_manager=db)
    assert analyzer.db is db


# --- 2. scripts/quick_performance_analysis.py -------------------------------

async def test_quick_performance_analysis_passes_db_to_xai_client(tmp_path, monkeypatch):
    mod = _load("scripts/quick_performance_analysis.py", "quick_perf_under_test")
    db = _db_instance()
    xai = MagicMock(name="XAIClient")
    monkeypatch.setattr(mod, "DatabaseManager", MagicMock(return_value=db))
    monkeypatch.setattr(mod, "XAIClient", xai)
    monkeypatch.setattr(mod, "KalshiClient", MagicMock())
    monkeypatch.chdir(tmp_path)

    # The script body keeps going past client construction (Kalshi/DB reads on
    # mocks). Only the construction is under test, so a later failure is ignored.
    try:
        await mod.run_quick_analysis()
    except Exception:
        pass

    db.initialize.assert_awaited_once()
    xai.assert_called_once_with(db_manager=db)


# --- 3. scripts/extract_grok_analysis.py -----------------------------------

async def test_extract_grok_analysis_passes_db_to_xai_client(tmp_path, monkeypatch):
    mod = _load("scripts/extract_grok_analysis.py", "extract_grok_under_test")
    db = _db_instance()
    xai_instance = MagicMock(name="XAIClient-instance")
    xai_instance._make_completion_request = AsyncMock(return_value=("analysis", 0.0))
    xai_instance.close = AsyncMock()
    xai = MagicMock(return_value=xai_instance)
    monkeypatch.setattr(mod, "DatabaseManager", MagicMock(return_value=db))
    monkeypatch.setattr(mod, "XAIClient", xai)
    monkeypatch.chdir(tmp_path)  # the script writes grok4_full_analysis.txt to cwd

    await mod.extract_analysis()

    db.initialize.assert_awaited_once()
    xai.assert_called_once_with(db_manager=db)


# --- 4. cli.py verify: llm_call's OpenRouterClient -------------------------

def test_verify_passes_db_to_openrouter_client(monkeypatch):
    db = _db_instance()
    or_cls = MagicMock(name="OpenRouterClient")
    or_cls.return_value.get_completion = AsyncMock(return_value=None)

    kalshi = MagicMock(name="KalshiClient-instance")
    kalshi.get_orderbook = AsyncMock(
        return_value={"orderbook_fp": {"yes_dollars": [["0.40", "10"]]}}
    )
    kalshi.get_market = AsyncMock(return_value={"market": {}})
    kalshi.close = AsyncMock()

    monkeypatch.setattr("src.clients.kalshi_client.KalshiClient", MagicMock(return_value=kalshi))
    monkeypatch.setattr("src.clients.openrouter_client.OpenRouterClient", or_cls)
    monkeypatch.setattr("src.utils.database.DatabaseManager", MagicMock(return_value=db))

    args = MagicMock(ticker="TEST-TICKER", research_file=None, json=True, jev=False)
    # With the LLM returning None, verify stops after the client is built. The
    # error surfaces however run_verify chooses to report it; only construction matters.
    try:
        cli.cmd_verify(args)
    except (Exception, SystemExit):
        pass

    db.initialize.assert_awaited_once()
    or_cls.assert_called_once_with(db_manager=db)


# --- End-to-end: real XAIClient forwards db_manager to the spend mirror -----

async def test_xai_client_spend_reaches_db_upsert(tmp_path):
    db = MagicMock(name="DatabaseManager")
    db.upsert_daily_cost = AsyncMock()

    client = XAIClient(api_key="test-key", db_manager=db)
    client.usage_file = str(tmp_path / "usage.pkl")
    assert client.db_manager is db

    # XAIClient delegates completions to a real OpenRouterClient that carries the
    # same db_manager. Its _update_daily_cost is where the DB mirror is scheduled.
    or_client = client._get_openrouter_client()
    assert isinstance(or_client, OpenRouterClient)
    assert or_client.db_manager is db

    or_client.usage_file = str(tmp_path / "usage.pkl")
    or_client._update_daily_cost(0.5)  # schedules the DB write on the running loop
    await asyncio.gather(*list(or_client._db_cost_tasks))

    db.upsert_daily_cost.assert_awaited_once_with(0.5)


# --- AST guard: every XAIClient/OpenRouterClient call in the changed files ---

@pytest.mark.parametrize(
    "relpath",
    [
        "cli.py",
        "scripts/extract_grok_analysis.py",
        "scripts/quick_performance_analysis.py",
        "src/jobs/automated_performance_analyzer.py",
    ],
)
def test_ai_client_calls_pass_db_manager_keyword(relpath):
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
    targets = {"XAIClient", "OpenRouterClient"}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in targets
    ]
    assert calls, f"{relpath}: expected at least one client construction"
    for call in calls:
        kws = {k.arg for k in call.keywords}
        assert "db_manager" in kws, f"{relpath}:{call.lineno} builds a client without db_manager"
