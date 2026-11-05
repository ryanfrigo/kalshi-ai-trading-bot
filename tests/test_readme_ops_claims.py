"""README operational claims (stopping, category scoring, performance) vs the code.

Read-only and offline: README sections are parsed as text and compared against
the argparse tree, the category scorer constants and module sources. No order
function is ever called and no credentials are read.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cli
import src.strategies.category_scorer as scorer

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
# (1) "## Stopping Cleanly" — close-all is dry-run by default
# ---------------------------------------------------------------------------

def test_close_all_is_real_subcommand():
    assert "close-all" in _subcommands()


def test_close_all_live_defaults_false():
    parser = _subcommands()["close-all"]
    assert _default(parser, "live") is False


# ---------------------------------------------------------------------------
# (2) Ctrl-C graceful shutdown (README "## Stopping Cleanly")
# ---------------------------------------------------------------------------

def test_ctrl_c_is_handled_by_the_run_path():
    section = _section("## Stopping Cleanly", "## Installation")
    assert "Ctrl-C" in section and "SIGINT" in section
    assert "graceful shutdown" in section

    cli_source = (ROOT / "cli.py").read_text()
    assert "from beast_mode_bot import BeastModeBot" in cli_source  # what `run` launches
    assert "except KeyboardInterrupt:" in cli_source

    bot_source = (ROOT / "beast_mode_bot.py").read_text()
    assert "import signal" in bot_source
    assert re.search(r'sig_names = \["SIGINT"', bot_source)
    assert "shutdown_event.set()" in bot_source


# ---------------------------------------------------------------------------
# (3) "## Category Scoring (display only)" — bands, caps and labels
# ---------------------------------------------------------------------------

def _score_table() -> list[tuple[int, int, int, str]]:
    section = _section("## Category Scoring (display only)", "## Performance Tracking")
    rows = re.findall(
        r"^\|\s*(\d+)[–-](\d+)\s*\|\s*(\d+)%?\s*\|\s*(\w+)\s*\|", section, re.M
    )
    assert len(rows) == 5, f"expected 5 score-band rows, parsed {rows}"
    return [(int(lo), int(hi), int(cap), label) for lo, hi, cap, label in rows]


def test_category_score_bands_match_scorer_labels():
    labels = scorer.CategoryScorer().score_label
    for lo, _hi, _cap, label in _score_table():
        assert labels(lo).split()[0] == label, (lo, label)


def test_category_allocation_caps_match_scorer_tiers():
    readme_tiers = [(lo, cap / 100) for lo, _hi, cap, _label in _score_table()]
    assert readme_tiers == [(lo, pct) for lo, pct in scorer.ALLOCATION_TIERS]
    assert scorer.BLOCK_THRESHOLD == 30


# ---------------------------------------------------------------------------
# (4) "Nothing in the order path reads it"
# ---------------------------------------------------------------------------

def test_category_scorer_not_imported_by_order_path():
    for rel in ("src/clients/kalshi_client.py", "src/risk/risk_governor.py"):
        imports = [
            line for line in (ROOT / rel).read_text().splitlines()
            if re.match(r"\s*(from|import)\s", line)
        ]
        assert imports, rel
        assert not any("category_scorer" in line for line in imports), rel


# ---------------------------------------------------------------------------
# (5) "## Performance Tracking" — history/status and the SQLite file
# ---------------------------------------------------------------------------

def test_history_and_status_are_real_subcommands():
    assert {"history", "status"} <= set(_subcommands())


def test_history_limit_defaults_to_50():
    section = _section("## Performance Tracking", "## Development")
    assert "Last 50 trades" in section
    assert _default(_subcommands()["history"], "limit") == 50


def test_trading_system_db_is_the_default_database_file():
    section = _section("## Performance Tracking", "## Development")
    assert "trading_system.db" in section
    db_source = (ROOT / "src/utils/database.py").read_text()
    assert 'db_path: str = "trading_system.db"' in db_source
