"""README troubleshooting and live-trading-lesson claims vs the code.

Read-only and offline: README sections are parsed as text and compared against
module sources and settings. No health command, order function or credential
file is touched.
"""
from __future__ import annotations

import re
from pathlib import Path

import src.strategies.safe_compounder as compounder
from src.config.settings import settings

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()

TROUBLESHOOTING = "## Troubleshooting"
LESSONS = "## Lessons From Live Trading"


def _section(title: str, next_title: str) -> str:
    start = README.index(title)
    return README[start:README.index(next_title, start)]


def _run_source() -> str:
    source = (ROOT / "cli.py").read_text()
    return source[source.index("def cmd_run("):source.index("def _run_safe_compounder(")]


def _src_files() -> list[Path]:
    return [
        path for path in (ROOT / "src").rglob("*.py")
        if path != ROOT / "src/config/settings.py"
    ]


# ---------------------------------------------------------------------------
# "## Troubleshooting" — 401 hint, shutdown signal logging, database -m
# ---------------------------------------------------------------------------

def test_health_check_has_a_401_specific_hint():
    section = _section(TROUBLESHOOTING, LESSONS)
    assert "The health check will print this hint when it detects a 401" in section

    cli_source = (ROOT / "cli.py").read_text()
    assert 'if "401" in msg or "authentication" in msg.lower():' in cli_source
    assert "A 401 from Kalshi almost always means one of:" in cli_source


def test_bot_logs_each_of_the_three_shutdown_signals():
    section = _section(TROUBLESHOOTING, LESSONS)
    assert "(SIGINT, SIGTERM, or SIGHUP)" in section

    bot_source = (ROOT / "beast_mode_bot.py").read_text()
    assert 'sig_names = ["SIGINT", "SIGTERM", "SIGHUP"]' in bot_source
    assert 'self.logger.info(f"🛑 Shutdown signal received: {sig_name} ({signum})")' in bot_source


def test_database_module_has_a_main_block_for_python_m():
    assert "python -m src.utils.database" in _section(TROUBLESHOOTING, LESSONS)
    db_source = (ROOT / "src/utils/database.py").read_text()
    assert 'if __name__ == "__main__":' in db_source


# ---------------------------------------------------------------------------
# Conservative thresholds quoted in the "no positions" panel
# ---------------------------------------------------------------------------

def _thresholds_from_readme() -> tuple[int, int, int, int]:
    section = _section(TROUBLESHOOTING, LESSONS)
    confidence = re.search(
        r"confidence ≥ (\d+)% on the legacy path \(≥ (\d+)% on the unified path\)",
        section,
    )
    compounder_limits = re.search(r"NO ask > (\d+)¢ AND edge > (\d+)¢", section)
    assert confidence, "could not parse confidence thresholds from README"
    assert compounder_limits, "could not parse Safe Compounder thresholds from README"
    return (
        int(confidence.group(1)),
        int(confidence.group(2)),
        int(compounder_limits.group(1)),
        int(compounder_limits.group(2)),
    )


def test_ai_confidence_thresholds_match_readme():
    legacy, unified, _no_ask, _edge = _thresholds_from_readme()

    assert round(settings.trading.min_confidence_to_trade * 100) == legacy

    immediate_source = (ROOT / "src/strategies/portfolio/immediate.py").read_text()
    gate = re.search(r"opportunity\.confidence >= (\d+\.\d+)", immediate_source)
    assert gate, "no confidence gate found in immediate.py"
    assert round(float(gate.group(1)) * 100) == unified


def test_safe_compounder_thresholds_match_readme():
    _legacy, _unified, no_ask, edge = _thresholds_from_readme()
    assert round(compounder.MIN_NO_ASK * 100) == no_ask
    assert round(compounder.MIN_EDGE * 100) == edge


# ---------------------------------------------------------------------------
# "## Lessons From Live Trading"
# ---------------------------------------------------------------------------

def test_lesson_2_quarter_kelly_is_the_configured_fraction():
    section = _section(LESSONS, "## Contributing")
    assert "Quarter-Kelly is what the example strategies use" in section
    assert settings.trading.kelly_fraction == 0.25


def test_lesson_3_run_path_uses_15_percent_drawdown():
    section = _section(LESSONS, "## Contributing")
    assert "15% under `cli.py run`" in section
    assert "cfg.max_drawdown = 0.15" in _run_source()


def test_lesson_4_no_per_category_cap_in_src():
    section = _section(LESSONS, "## Contributing")
    assert "There is no per-category cap in the code" in section
    readers = [
        path.relative_to(ROOT).as_posix()
        for path in _src_files()
        if "max_sector_exposure" in path.read_text()
    ]
    assert readers == [], f"max_sector_exposure is read in src: {readers}"


def test_lesson_5_trading_cycle_waits_60_seconds():
    # The 60s is hard-coded in the run loop; settings.scan_interval_seconds (also 60)
    # is read nowhere outside settings.py, so pin the loop, not the setting.
    section = _section(LESSONS, "## Contributing")
    assert "The default scan interval is 60 seconds" in section
    bot_source = (ROOT / "beast_mode_bot.py").read_text()
    loop = bot_source[bot_source.index("async def _run_trading_cycles("):bot_source.index("async def _check_daily_ai_limits(")]
    assert "# Wait for next cycle (60 seconds)\n                await asyncio.sleep(60)" in loop
