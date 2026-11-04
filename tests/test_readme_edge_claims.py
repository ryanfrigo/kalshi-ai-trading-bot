"""README "Prove Your Edge" + "Self-Improvement Loop" claims vs the code.

Read-only and offline: every number is parsed out of the README text and
compared with the source constant or argparse tree, so the prose cannot drift
from the behaviour. No order function is called, no CLI command is run and no
credentials are read.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import cli
import src.agent.edge as edge
import src.agent.policy as policy

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _section(title: str, next_title: str) -> str:
    start = README.index(title)
    return README[start:README.index(next_title, start)]


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    top = cli.build_parser()
    sub = next(a for a in top._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices


def _has_flag(parser: argparse.ArgumentParser, name: str) -> bool:
    return any(name in a.option_strings for a in parser._actions)


def _number(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    assert match, f"README pattern not found: {pattern!r}"
    return int(match.group(1))


def _func_source(rel: str, name: str) -> str:
    text = (ROOT / rel).read_text()
    tree = ast.parse(text)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    return ast.get_source_segment(text, node) or ""


# ---------------------------------------------------------------------------
# (1) "fewer than 10 forward-settled trades ... refuses to claim edge"
# ---------------------------------------------------------------------------

def test_forward_settled_gate_matches_edge_min_n():
    section = _section("## Prove Your Edge (the headline)", "## The Self-Improvement Loop")
    readme_n = _number(r"fewer than (\d+) forward-settled trades", section)
    assert readme_n == edge._MIN_EDGE_N


# ---------------------------------------------------------------------------
# (2) thin groups: the prose rule and the Block-table row both use min_n
# ---------------------------------------------------------------------------

def test_thin_group_thresholds_match_policy_min_n():
    section = _section("## The Self-Improvement Loop", "## Quick Start")
    prose_n = _number(r"group with fewer than (\d+) settled trades earns no opinion", section)
    block_n = _number(r"with ≥\s*(\d+)\s*settled trades and negative realized P&L", section)
    assert prose_n == policy.DEFAULT_MIN_N
    assert block_n == policy.DEFAULT_MIN_N


# ---------------------------------------------------------------------------
# (3) "Haircut ... where you're >=10pp overconfident"
# ---------------------------------------------------------------------------

def test_overconfidence_haircut_matches_policy_gap():
    section = _section("## The Self-Improvement Loop", "## Quick Start")
    readme_pp = _number(r"≥\s*(\d+)\s*pp overconfident", section)
    assert readme_pp == round(policy._OVERCONFIDENCE_GAP * 100)


# ---------------------------------------------------------------------------
# (4) "The measurement code (src/agent/edge.py) is pure and deterministic"
# ---------------------------------------------------------------------------

def test_edge_measurement_module_has_no_io_imports():
    source = (ROOT / "src/agent/edge.py").read_text()
    imports = [
        line for line in source.splitlines()
        if re.match(r"\s*(from|import)\s", line)
    ]
    assert imports
    for banned in ("requests", "httpx", "aiohttp", "sqlite3", "aiosqlite", "src.clients"):
        assert not any(banned in line for line in imports), banned
    assert "open(" not in source


# ---------------------------------------------------------------------------
# (5) trade exposes the documented --override-policy escape hatch
# ---------------------------------------------------------------------------

def test_trade_has_override_policy_flag():
    assert _has_flag(_subcommands()["trade"], "--override-policy")


# ---------------------------------------------------------------------------
# (6) the headline/loop commands are real subcommands; policy has --demo
# ---------------------------------------------------------------------------

def test_headline_and_loop_subcommands_exist():
    subs = _subcommands()
    assert {"edge", "policy", "improve", "report", "backtest"} <= set(subs)
    assert _has_flag(subs["policy"], "--demo")


# ---------------------------------------------------------------------------
# (7) "`cli backtest` is a placeholder" — no order path in the handler
# ---------------------------------------------------------------------------

def test_backtest_handler_is_a_placeholder_without_order_code():
    source = _func_source("cli.py", "cmd_backtest")
    assert "KalshiClient" not in source
    assert "place_order" not in source


# ---------------------------------------------------------------------------
# (8) "the example strategies (run, daily) don't read it"
# ---------------------------------------------------------------------------

def test_example_strategies_do_not_read_the_policy():
    bot_source = (ROOT / "beast_mode_bot.py").read_text()
    sources = [
        _func_source("cli.py", "cmd_run"),
        _func_source("cli.py", "cmd_daily"),
        bot_source,
    ]
    for source in sources:
        assert "src.agent.policy" not in source
        assert "load_policy" not in source


# ---------------------------------------------------------------------------
# (9) "the policy is checked by cli trade and the MCP trade tool"
# ---------------------------------------------------------------------------

def test_policy_gates_cli_trade_and_mcp_trade():
    # The gate itself lives in toolbelt.place_guarded_order; both documented
    # entry points route through it rather than re-implementing the check.
    cli_trade = _func_source("cli.py", "cmd_trade")
    mcp_trade = _func_source("src/mcp_server.py", "trade")
    toolbelt = (ROOT / "src/agent/toolbelt.py").read_text()

    assert "place_guarded_order" in cli_trade
    assert "place_guarded_order" in mcp_trade
    assert "from src.agent.policy import" in toolbelt
    assert "load_policy" in toolbelt and "apply_policy" in toolbelt


# ---------------------------------------------------------------------------
# (10) the linked live track record is committed
# ---------------------------------------------------------------------------

def test_track_record_exists():
    assert (ROOT / "docs/TRACK_RECORD.md").is_file()
