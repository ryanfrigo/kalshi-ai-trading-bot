"""README risk defaults must match src/config/settings.py."""
import re
from pathlib import Path

import src.config.settings as cfg

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()
DRAWDOWN_PCT = round(cfg.max_drawdown * 100)
POSITION_PCT = cfg.settings.trading.max_position_size_pct


def test_defaults_line_matches_settings():
    m = re.search(r"Defaults: (\d+)% max drawdown, .*?(\d+)% max position size", README)
    assert m, "README 'Defaults:' line not found"
    assert int(m.group(1)) == DRAWDOWN_PCT
    assert int(m.group(2)) == POSITION_PCT


def test_drawdown_lesson_matches_settings():
    m = re.search(r"The default is (\d+)% with the circuit breaker", README)
    assert m, "README drawdown lesson not found"
    assert int(m.group(1)) == DRAWDOWN_PCT


def test_example_block_matches_settings():
    m = re.search(r"max_position_size_pct\s*=\s*([\d.]+)", README)
    assert m, "README example block not found"
    assert float(m.group(1)) == POSITION_PCT


def test_example_block_drawdown_matches_settings():
    m = re.search(r"^max_drawdown\s*=\s*([\d.]+)", README, re.M)
    assert m, "README example max_drawdown not found"
    assert float(m.group(1)) == cfg.max_drawdown
