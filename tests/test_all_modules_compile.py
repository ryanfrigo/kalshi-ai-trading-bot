"""Every module in the repo must at least compile.

`src/jobs/performance_analyzer.py` shipped through the public release with an IndentationError
that made it unimportable, and no test touched it, so the suite stayed green. A syntax check
over every module is the cheapest possible guard against that class.

Compiles in memory (`compile`), never writing a `.pyc` beside the source.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", "experimental", "data"}


def _modules():
    out = []
    for root in ("src", "scripts"):
        for p in (REPO / root).rglob("*.py"):
            if SKIP_DIRS & set(p.parts):
                continue
            out.append(p)
    return sorted(out)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(REPO)))
def test_module_compiles(path):
    try:
        compile(path.read_text(), str(path), "exec")
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(REPO)} does not compile: {e}")
