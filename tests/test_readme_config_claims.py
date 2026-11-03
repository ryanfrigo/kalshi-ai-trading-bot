"""README "Configuration" block must match src/config/settings.py."""
import dataclasses
import re
from pathlib import Path

import src.config.settings as cfg

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
BLOCK = re.search(r"## Configuration.*?```python\n(.*?)```", README, re.S)
assert BLOCK, "README '## Configuration' python block not found"
CONFIG = BLOCK.group(1)


def _value(pattern):
    m = re.search(pattern, CONFIG, re.M)
    assert m, f"README Configuration line not found: {pattern}"
    return m.group(1)


def test_kelly_fraction():
    assert float(_value(r"^kelly_fraction\s*=\s*([\d.]+)")) == cfg.settings.trading.kelly_fraction


def test_min_volume():
    assert float(_value(r"^min_volume\s*=\s*([\d.]+)")) == cfg.settings.trading.min_volume


def test_max_time_to_expiry_days():
    assert int(_value(r"^max_time_to_expiry_days\s*=\s*(\d+)")) == cfg.settings.trading.max_time_to_expiry_days


def test_min_confidence_to_trade():
    assert float(_value(r"^min_confidence_to_trade\s*=\s*([\d.]+)")) == cfg.settings.trading.min_confidence_to_trade


def test_unified_path_hardcodes_point_six():
    assert "unified path hard-codes 0.60" in CONFIG
    source = (ROOT / "src/strategies/portfolio/immediate.py").read_text()
    m = re.search(r"confidence\s*>=\s*(0\.\d+)", source)
    assert m, "unified-path confidence gate not found"
    assert float(m.group(1)) == 0.60


def test_primary_model():
    assert _value(r'^primary_model\s*=\s*"([^"]+)"') == cfg.settings.trading.primary_model


def test_ai_temperature():
    assert float(_value(r"^ai_temperature\s*=\s*([\d.]+)")) == cfg.settings.trading.ai_temperature


def test_ai_max_tokens():
    assert int(_value(r"^ai_max_tokens\s*=\s*(\d+)")) == cfg.settings.trading.ai_max_tokens


def test_max_daily_loss_pct():
    assert float(_value(r"^max_daily_loss_pct\s*=\s*([\d.]+)")) == cfg.settings.trading.max_daily_loss_pct


def test_daily_ai_cost_limit(monkeypatch):
    monkeypatch.delenv("DAILY_AI_COST_LIMIT", raising=False)
    field = next(
        f for f in dataclasses.fields(cfg.TradingConfig) if f.name == "daily_ai_cost_limit"
    )
    assert float(_value(r"^daily_ai_cost_limit\s*=\s*([\d.]+)")) == field.default_factory()


def test_daily_ai_cost_limit_env_var_is_read():
    source = "\n".join(p.read_text() for p in (ROOT / "src").rglob("*.py"))
    assert re.search(r"os\.getenv\(\s*['\"]DAILY_AI_COST_LIMIT['\"]", source), (
        "DAILY_AI_COST_LIMIT is never read in src"
    )
