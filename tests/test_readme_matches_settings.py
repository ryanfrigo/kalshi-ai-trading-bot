"""The README quotes real settings values. Keep them honest.

Four stale claims were found in the README on 2026-10-23 by reading it against the code:
the drawdown default (said 15%, is 20%), the category-score allocation boundaries, a feature
table entry describing exits that cannot fire, and a scan interval nothing reads. None of that
is visible to any other test - 446 of them passed while the public face of the repo was wrong.

This checks the mechanically checkable part: every `name = value` in the README's settings
block must match the value the code actually uses. Prose claims still need a human.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


def readme_settings_block() -> dict[str, str]:
    """Parse the python block under "All trading parameters live in src/config/settings.py"."""
    text = README.read_text()
    marker = "All trading parameters live in"
    start = text.find(marker)
    assert start >= 0, "the README's settings section moved; update this test"
    block = re.search(r"```python\n(.*?)```", text[start:], re.S)
    assert block, "no python block after the settings section"
    out = {}
    for line in block.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([a-z_][a-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def real_value(name: str):
    from src.config import settings as mod

    if hasattr(mod, name):
        return getattr(mod, name)
    cfg = getattr(mod, "settings", None)
    for holder in (getattr(cfg, "trading", None), cfg):
        if holder is not None and hasattr(holder, name):
            return getattr(holder, name)
    return None


def doc_settings_claims() -> list[tuple[str, str, str]]:
    """Every `name = value` for a real setting, anywhere under docs/ or in the README.

    Prose like "Position <= 10%" needs interpretation and is deliberately NOT parsed here -
    a test keyed to magic phrases is worse than no test. Those still need a human; the values
    were checked by hand on 2026-10-23 and matched (confidence 45%, position 10%, sector 30%,
    drawdown 20%).
    """
    import src.config.settings as mod

    known = set(dir(mod))
    cfg = getattr(mod, "settings", None)
    for holder in (getattr(cfg, "trading", None), cfg):
        if holder is not None:
            known |= set(dir(holder))
    out = []
    for path in sorted(REPO.glob("docs/*.md")) + [README]:
        for m in re.finditer(r"^\s*([a-z_][a-z0-9_]{4,})\s*=\s*([\d.]+)\s*$", path.read_text(), re.M):
            if m.group(1) in known:
                out.append((path.name, m.group(1), m.group(2)))
    return out


@pytest.mark.parametrize("source,name,quoted", doc_settings_claims(),
                         ids=lambda v: str(v)[:40])
def test_doc_setting_matches_code(source, name, quoted):
    actual = real_value(name)
    if actual is None or isinstance(actual, bool):
        pytest.skip("not a numeric setting")
    assert float(actual) == float(quoted), (
        f"{source} says {name} = {quoted}, the code uses {actual}"
    )


@pytest.mark.parametrize("name,quoted", sorted(readme_settings_block().items()))
def test_readme_setting_matches_code(name, quoted):
    actual = real_value(name)
    if actual is None:
        pytest.skip(f"{name} is not a real setting (README may be illustrating a change)")
    try:
        expected = float(quoted)
    except ValueError:
        pytest.skip(f"{name}: non-numeric value {quoted!r}")
    if isinstance(actual, bool):
        pytest.skip("boolean")
    assert float(actual) == expected, (
        f"README says {name} = {quoted}, the code uses {actual}"
    )


def test_readme_allocation_table_matches_code():
    """The Category Scoring table must agree with ALLOCATION_TIERS / BLOCK_THRESHOLD.

    It did not: the README claimed 20-39 gets 2% and 0-19 is blocked, while the code blocks
    below 30 (ALLOCATION_TIERS has (30, 0.02) and (0, 0.00), commented "align with
    BLOCK_THRESHOLD=30"). A reader would have expected allocation at a score of 25; the code
    refuses to trade there at all.
    """
    from src.strategies.category_scorer import ALLOCATION_TIERS, BLOCK_THRESHOLD

    text = README.read_text()
    i = text.find("Category Scoring")
    table = text[i : text.find("```bash", i)]
    rows = re.findall(r"^\|\s*(\d+)[-\u2013](\d+)\s*\|\s*(\d+)%?\s*\|", table, re.M)
    assert rows, "category-scoring table not found or changed shape"

    code = {int(m): int(p * 100) for m, p in ALLOCATION_TIERS}
    for lo, hi, pct in rows:
        lo, hi, pct = int(lo), int(hi), int(pct)
        # the tier covering this band is the highest min_score <= lo
        applicable = max((m for m in code if m <= lo), default=None)
        assert applicable is not None, f"no tier covers score {lo}"
        assert code[applicable] == pct, (
            f"README table says {lo}-{hi} gets {pct}%, the code gives {code[applicable]}%"
        )
        # the band must not straddle the block threshold
        assert not (lo < BLOCK_THRESHOLD <= hi), (
            f"README band {lo}-{hi} straddles BLOCK_THRESHOLD={BLOCK_THRESHOLD}"
        )
