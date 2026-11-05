"""README "What's Included" / "Data-Fed Families" claims vs the repo.

Read-only and offline: README sections are parsed as text and compared against
paths on disk, module sources and the argparse tree. Nothing under ``scripts/``
is ever imported or run (those modules may fetch from the network) and no order
function is called.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

import cli
import src.risk.risk_governor as governor

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _section(title: str, next_title: str) -> str:
    start = README.index(title)
    return README[start:README.index(next_title, start)]


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    top = cli.build_parser()
    sub = next(a for a in top._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices


def _included_paths() -> list[str]:
    """Backticked paths in the third column of the "What's Included" table."""
    section = _section("## What's Included", "## Example Strategies")
    paths = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        for token in re.findall(r"`([^`]+)`", cells[2]):
            # The dashboard row also names its command; only repo paths matter.
            if re.match(r"^(src|scripts)/", token):
                paths.append(token)
    assert len(paths) >= 10, f"expected the table to name its components, parsed {paths}"
    return paths


def _data_fed_tools() -> list[tuple[str, list[str]]]:
    """(script path, subcommand names) for every ``scripts/*.py`` tool token."""
    section = _section("## Data-Fed Families", "## Agent-Native Surface")
    tools = []
    for token in re.findall(r"`([^`]+)`", section):
        m = re.match(r"^(scripts/[\w/]+\.py)(?:\s+([\w|]+))?$", token)
        if m:
            tools.append((m.group(1), m.group(2).split("|") if m.group(2) else []))
    assert tools, "expected script tools in the Data-Fed Families table"
    return tools


# ---------------------------------------------------------------------------
# (1) "## What's Included" — every component path exists
# ---------------------------------------------------------------------------

def test_whats_included_paths_exist():
    missing = [p for p in _included_paths() if not (ROOT / p).exists()]
    assert not missing, f"README names paths that do not exist: {missing}"


# ---------------------------------------------------------------------------
# (2) "## Data-Fed Families" — every script and docs link exists
# ---------------------------------------------------------------------------

def test_data_fed_scripts_exist():
    missing = [script for script, _subs in _data_fed_tools() if not (ROOT / script).exists()]
    assert not missing, f"README names scripts that do not exist: {missing}"


def test_data_fed_docs_links_exist():
    section = _section("## Data-Fed Families", "## Agent-Native Surface")
    links = re.findall(r"\[docs/[\w.-]+\.md\]\(([^)]+)\)", section)
    assert links, "expected docs links in the Data-Fed Families table"
    missing = [link for link in links if not (ROOT / link).exists()]
    assert not missing, f"README links docs that do not exist: {missing}"


# ---------------------------------------------------------------------------
# (3) subcommand names are real strings in each script's source
# ---------------------------------------------------------------------------

def test_data_fed_subcommands_appear_in_script_source():
    checked = 0
    for script, subs in _data_fed_tools():
        source = (ROOT / script).read_text()
        for sub in subs:
            assert f'"{sub}"' in source or f"'{sub}'" in source, (script, sub)
            checked += 1
    assert checked >= 9, f"expected the table's subcommands to be checked, saw {checked}"


# ---------------------------------------------------------------------------
# (4) `python cli.py dashboard` is a real subcommand
# ---------------------------------------------------------------------------

def test_dashboard_is_a_cli_subcommand():
    section = _section("## What's Included", "## Example Strategies")
    assert "python cli.py dashboard" in section
    assert "dashboard" in _subcommands()


# ---------------------------------------------------------------------------
# (5) runtime-dir override: kill-switch artifacts, relative rejected, ~ expanded
# ---------------------------------------------------------------------------

def test_risk_governor_source_defines_halt_artifacts():
    section = _section("## What's Included", "## Example Strategies")
    assert "TRADING_HALTED" in section and "governor_audit.jsonl" in section
    assert "KALSHI_RUNTIME_DIR" in section

    source = (ROOT / "src/risk/risk_governor.py").read_text()
    assert "TRADING_HALTED" in source
    assert "governor_audit.jsonl" in source
    assert "KALSHI_RUNTIME_DIR" in source


def test_runtime_dir_reads_env_and_rejects_relative(monkeypatch):
    monkeypatch.setenv(governor.RUNTIME_DIR_ENV, "data/runtime")
    with pytest.raises(ValueError, match="absolute"):
        governor._env_runtime_dir()


def test_runtime_dir_expands_tilde(monkeypatch):
    monkeypatch.setenv(governor.RUNTIME_DIR_ENV, "~/x")
    resolved = governor._env_runtime_dir()
    assert resolved == Path.home() / "x"
    assert resolved.is_absolute()


# ---------------------------------------------------------------------------
# (6) halt-exempt sell is reduce-only + IOC
# ---------------------------------------------------------------------------

def test_halt_exempt_sell_is_reduce_only_ioc():
    # The governor documents the halt exemption; the order payload builder in
    # kalshi_client enforces IOC + reduce_only. Call the pure builder (no network).
    from src.clients.kalshi_client import build_order_v2_payload

    governor = (ROOT / "src/risk/risk_governor.py").read_text()
    assert "reduce-only sell" in governor
    payload = build_order_v2_payload(
        ticker="KXTEST-1", client_order_id="t", side="yes", action="sell",
        count=1, type_="limit", yes_price=40, reduce_only=True,
    )
    assert payload["time_in_force"] == "immediate_or_cancel"
    assert payload["reduce_only"] is True
