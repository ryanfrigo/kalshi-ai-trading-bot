"""README claims about the agent surface must match the actual CLI / MCP / strategy code.

Read-only and offline: the README is parsed as text and compared against the
argparse tree, the FastMCP registry, function signatures and strategy constants.
No order function is ever called.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import re
from pathlib import Path

import cli
import src.mcp_server as mcp_server
import src.strategies.safe_compounder as compounder

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _section(title: str, next_title: str) -> str:
    start = README.index(title)
    return README[start:README.index(next_title, start)]


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    top = cli.build_parser()
    sub = next(a for a in top._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices


def _default(parser: argparse.ArgumentParser, dest: str):
    return next(a for a in parser._actions if a.dest == dest).default


# ---------------------------------------------------------------------------
# (1) CLI table under "## Agent-Native Surface"
# ---------------------------------------------------------------------------

def test_cli_table_commands_are_real_subcommands():
    section = _section("## Agent-Native Surface", "## What's Included")
    table_rows = [
        line for line in section.splitlines()
        if line.startswith("| `") and line.count("|") >= 3
    ]
    assert table_rows, "CLI tool table not found"
    named = set()
    for row in table_rows:
        named.update(re.findall(r"`(\w+)`", row.split("|")[1]))
    assert named, "no CLI commands parsed from README table"
    assert named <= set(_subcommands()), f"README names missing commands: {named - set(_subcommands())}"


def test_readme_run_daily_health_are_real_subcommands():
    section = _section("## Agent-Native Surface", "## What's Included")
    m = re.search(r"Plus `run` / `daily`.*?`health`", section)
    assert m, "README run/daily/health line not found"
    named = set(re.findall(r"`(\w+)`", m.group(0)))
    assert {"run", "daily", "health"} <= named
    assert named <= set(_subcommands())


# ---------------------------------------------------------------------------
# (2) MCP tool list in the README
# ---------------------------------------------------------------------------

def test_mcp_tool_list_matches_registered_tools():
    line = next(line for line in README.splitlines() if "governor-gated tools (`" in line)
    listed = set(re.findall(r"`(\w+)`", re.search(r"governor-gated tools \(([^)]*)\)", line).group(1)))
    registered = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert registered == listed, f"README lists {listed}, server registers {registered}"


# ---------------------------------------------------------------------------
# (3) MCP trade/close require an explicit confirm=true
# ---------------------------------------------------------------------------

def test_mcp_trade_and_close_confirm_defaults_false():
    for fn in (mcp_server.trade, mcp_server.close):
        param = inspect.signature(fn).parameters["confirm"]
        assert param.default is False, fn.__name__


# ---------------------------------------------------------------------------
# (4) CLI trade/close are dry-run unless --live
# ---------------------------------------------------------------------------

def test_cli_trade_defaults_to_dry_run():
    parser = _subcommands()["trade"]
    assert _default(parser, "live") is False


def test_cli_close_defaults_to_dry_run():
    parser = _subcommands()["close"]
    assert _default(parser, "live") is False


# ---------------------------------------------------------------------------
# (5) Safe Compounder claims (README "### 2. Safe Compounder")
# ---------------------------------------------------------------------------

def test_safe_compounder_rule_constants():
    assert compounder.MIN_NO_ASK == 0.80   # README: "NO ask > 80¢"
    assert compounder.MIN_EDGE == 0.03     # README: "beats the ask by more than 3¢"
    assert compounder.MAX_POSITION_PCT == 0.10  # README: "max 10%/position"


def test_safe_compounder_yes_cap_and_orderbook_limit():
    source = inspect.getsource(compounder.SafeCompounder._find_no_candidates)
    assert re.search(r"yes_last > 0\.20", source)  # README: "YES last price of 20¢ or less"
    assert re.search(r"MAX_ORDERBOOK_CHECKS = 200", source)  # README: "up to 200 of them"


# ---------------------------------------------------------------------------
# (5) Paper Trading claims (README "## Paper Trading")
# ---------------------------------------------------------------------------

def test_paper_trader_flags_and_interval_match_readme():
    source = (ROOT / "paper_trader.py").read_text()
    for flag in ("--settle", "--dashboard", "--stats", "--loop"):
        assert f'"{flag}"' in source, flag
    assert re.search(r'"--interval", type=int, default=900', source)  # README: --interval 900


def test_paper_dashboard_path_matches_readme():
    import paper_trader

    out = Path(paper_trader.DASHBOARD_OUT).resolve()
    assert out == (ROOT / "docs" / "paper_dashboard.html").resolve()
    assert out.exists(), "README says docs/paper_dashboard.html ships as a template"
