"""Every plain `settings.trading.<name>` read must name a real TradingConfig field.

A name that exists only at module level in src/config/settings.py is not an
attribute of settings.trading, so the read raises AttributeError at runtime.
"""
import ast
import dataclasses
from pathlib import Path

import src.config.settings as cfg

REPO = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [REPO / "src", REPO / "scripts", REPO / "cli.py"]


def _python_files():
    for root in SCAN_ROOTS:
        if root.is_file():
            yield root
        else:
            yield from root.rglob("*.py")


def _trading_reads():
    """Yield (file, line, name) for each `settings.trading.<name>` attribute access."""
    for path in _python_files():
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "trading"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "settings"
            ):
                yield path.relative_to(REPO), node.lineno, node.attr


TRADING_FIELDS = {f.name for f in dataclasses.fields(cfg.TradingConfig)}


def test_plain_settings_trading_reads_are_trading_fields():
    reads = list(_trading_reads())
    assert reads, "scan found no settings.trading reads; the scanner is broken"
    missing = sorted({(str(f), line, name) for f, line, name in reads if name not in TRADING_FIELDS})
    assert not missing, (
        "settings.trading.<name> reads with no TradingConfig field "
        f"(AttributeError at runtime): {missing}"
    )


def test_skip_news_fields_live_on_trading_config():
    assert "skip_news_for_low_volume" in TRADING_FIELDS
    assert "news_search_volume_threshold" in TRADING_FIELDS
    assert cfg.settings.trading.skip_news_for_low_volume is True
    assert cfg.settings.trading.news_search_volume_threshold == 1000.0
