"""Test suite package.

Marking ``tests`` as a package lets sibling modules import shared
helpers via the ``tests.`` prefix (e.g. ``from tests.test_database import
TEST_DB``). Without this file, a bare ``pytest`` run — as used in CI —
fails at collection with ``ModuleNotFoundError: No module named 'tests'``,
because pytest's default ``prepend`` import mode only puts the project
root on ``sys.path`` when ``tests`` is an importable package.
"""
