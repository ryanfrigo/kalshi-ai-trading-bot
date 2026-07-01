"""Tests for the Kalshi MCP server — the FastMCP distribution layer.

These tests are OFFLINE: KalshiClient is mocked, no network, no real orders. They
verify (1) the module imports and registers the expected tools with the right
read-only / destructive annotations, and (2) — the safety-critical property — that
the mutating tools (``trade``/``close``) do NOT touch the live order path when
``confirm=False`` (dry-run), and only place a live order when ``confirm=True``. The
mutating tools must always route through the guarded ``place_guarded_order`` /
``close_position`` functions (which enforce the governor + 10% cap); we assert the
MCP layer adds no bypass.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import src.mcp_server as server


# ---------------------------------------------------------------------------
# Registration / annotations
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "brief", "settle", "learnings", "edge", "policy", "status",
    "scores", "history", "hunt", "trade", "close",
}
READ_ONLY = {"brief", "settle", "learnings", "edge", "policy", "status", "scores", "history", "hunt"}
MUTATING = {"trade", "close"}


def _list_tools():
    return asyncio.run(server.mcp.list_tools())


def test_module_imports_and_exposes_mcp_and_main():
    assert hasattr(server, "mcp")
    assert callable(server.main)


def test_expected_tools_registered():
    names = {t.name for t in _list_tools()}
    assert names == EXPECTED_TOOLS


def test_read_only_tools_annotated_read_only():
    by_name = {t.name: t for t in _list_tools()}
    for name in READ_ONLY:
        assert by_name[name].annotations.readOnlyHint is True, name


def test_mutating_tools_annotated_destructive_not_read_only():
    by_name = {t.name: t for t in _list_tools()}
    for name in MUTATING:
        ann = by_name[name].annotations
        assert ann.destructiveHint is True, name
        # Mutating tools must NOT be advertised as read-only.
        assert ann.readOnlyHint is not True, name


# ---------------------------------------------------------------------------
# Safety: mutating tools are dry-run by default and governor-gated
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_client():
    """Patch KalshiClient so no network/credentials are needed; close() is async."""
    fake = AsyncMock()
    fake.close = AsyncMock()
    with patch("src.clients.kalshi_client.KalshiClient", return_value=fake):
        yield fake


async def test_trade_default_is_dry_run(patched_client):
    """trade() with no confirm must call place_guarded_order with dry=True."""
    with patch("src.agent.toolbelt.place_guarded_order", new=AsyncMock(
        return_value={"ok": True, "dry": True})) as pgo:
        await server.trade(ticker="KXTEST", side="yes", count=5)

    pgo.assert_awaited_once()
    assert pgo.call_args.kwargs["dry"] is True


async def test_trade_confirm_true_places_live(patched_client):
    """Only confirm=True flips the guarded call to a live (dry=False) order."""
    with patch("src.agent.toolbelt.place_guarded_order", new=AsyncMock(
        return_value={"ok": True, "dry": False})) as pgo:
        await server.trade(ticker="KXTEST", side="yes", count=5, confirm=True)

    pgo.assert_awaited_once()
    assert pgo.call_args.kwargs["dry"] is False


async def test_trade_routes_through_guarded_function_only(patched_client):
    """The MCP trade tool must NOT call client.place_order directly — it must go
    through place_guarded_order (the governor + cap gate). Even in dry mode the
    raw order path stays untouched."""
    with patch("src.agent.toolbelt.place_guarded_order", new=AsyncMock(
        return_value={"ok": True, "dry": True})) as pgo:
        await server.trade(ticker="KXTEST", side="no", count=3)

    pgo.assert_awaited_once()
    # No direct order placement bypassing the guard.
    patched_client.place_order.assert_not_called()


async def test_close_default_is_dry_run(patched_client):
    with patch("src.agent.toolbelt.close_position", new=AsyncMock(
        return_value={"ok": True, "dry": True})) as cp:
        await server.close(ticker="KXTEST")

    cp.assert_awaited_once()
    assert cp.call_args.kwargs["dry"] is True


async def test_close_confirm_true_places_live(patched_client):
    with patch("src.agent.toolbelt.close_position", new=AsyncMock(
        return_value={"ok": True, "dry": False})) as cp:
        await server.close(ticker="KXTEST", confirm=True)

    cp.assert_awaited_once()
    assert cp.call_args.kwargs["dry"] is False


async def test_brief_is_read_only_passthrough(patched_client):
    """A read-only tool forwards the underlying function's JSON unchanged and
    closes the client."""
    with patch("src.agent.toolbelt.account_brief", new=AsyncMock(
        return_value={"equity": 123.45, "positions": []})) as ab:
        out = await server.brief()

    ab.assert_awaited_once()
    assert out == {"equity": 123.45, "positions": []}
    patched_client.close.assert_awaited_once()
