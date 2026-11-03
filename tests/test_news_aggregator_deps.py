"""src/data/news_aggregator.py imports feedparser; the project must declare it."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_feedparser_is_a_pinned_runtime_dependency():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(d.startswith("feedparser==") for d in deps)
