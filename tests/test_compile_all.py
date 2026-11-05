"""Every Python file under src/ and scripts/ must byte-compile.

Catches syntax and indentation errors that no import-based test reaches.
"""

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Files excluded from compilation, each with a reason. Empty: nothing is skipped.
EXCLUDED: dict[str, str] = {}

PY_FILES = sorted(
    p
    for top in ("src", "scripts")
    for p in (ROOT / top).rglob("*.py")
    if str(p.relative_to(ROOT)) not in EXCLUDED
)


def test_found_python_files():
    assert PY_FILES, "expected .py files under src/ and scripts/"


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_compiles(path: Path):
    py_compile.compile(str(path), doraise=True, cfile=None)
