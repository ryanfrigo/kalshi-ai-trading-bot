"""src/data/news_aggregator.py imports feedparser; the project must declare it."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.xfail(strict=True, reason=(
    "src/data/news_aggregator.py:14 imports feedparser, but pyproject.toml declares it "
    "in neither dependencies nor any extra; on a clean install the sentiment pipeline "
    "raises ModuleNotFoundError, which src/jobs/decide.py:269 swallows and falls back"))
def test_feedparser_is_a_declared_dependency():
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r"(?im)^\s*[\"']feedparser\b", text)
