"""
Issue #101: v1 position readers must understand Kalshi v2 position rows.

v2 ``get_positions`` rows carry a signed string ``position_fp`` ("-40.00" = 40 NO,
"+" = YES) and no ``position`` key. v2 ``get_market`` returns
``{"market": {...}}`` with ``*_dollars`` price keys. Each test below feeds that
shape through the helpers a script uses and guards the script source against
the legacy ``position`` read.
"""
import re
from pathlib import Path

import pytest

from src.utils.market_prices import held_side_mid, signed_position

REPO = Path(__file__).resolve().parents[1]

V2_NO_ROW = {"ticker": "X", "position_fp": "-40.00"}
V2_YES_ROW = {"ticker": "Y", "position_fp": "12.00"}

# Wrapped as get_market returns it: {"market": {...}} with *_dollars keys.
V2_WRAPPED_MARKET = {
    "market": {
        "ticker": "X",
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.44",
        "no_bid_dollars": "0.55",
        "no_ask_dollars": "0.58",
    }
}

LEGACY_READ = re.compile(r"\.get\(\s*['\"]position['\"]")


def _assert_v2_read(relpath: str) -> None:
    """The file must not read the legacy ``position`` key anywhere."""
    source = (REPO / relpath).read_text()
    assert not LEGACY_READ.search(source), f"{relpath} still reads .get('position')"
    assert "signed_position" in source, f"{relpath} does not use signed_position"


def _assert_fixture_semantics() -> None:
    assert signed_position(V2_NO_ROW) == -40.0
    assert signed_position(V2_YES_ROW) == 12.0
    assert signed_position({"ticker": "Z"}) == 0.0
    market = V2_WRAPPED_MARKET["market"]
    no_mid = held_side_mid(market, signed_position(V2_NO_ROW))
    yes_mid = held_side_mid(market, signed_position(V2_YES_ROW))
    assert no_mid == pytest.approx(0.565)  # NO mid = (0.55 + 0.58) / 2
    assert yes_mid == pytest.approx(0.42)  # YES mid = (0.40 + 0.44) / 2
    assert no_mid != 0.5 and yes_mid != 0.5  # not the legacy default


def test_trading_dashboard_v2_reads():
    _assert_fixture_semantics()
    _assert_v2_read("scripts/trading_dashboard.py")


def test_quick_performance_analysis_v2_reads():
    _assert_fixture_semantics()
    _assert_v2_read("scripts/quick_performance_analysis.py")


def test_portfolio_health_check_v2_reads():
    _assert_fixture_semantics()
    _assert_v2_read("scripts/portfolio_health_check.py")


def test_automated_performance_analyzer_v2_reads():
    _assert_fixture_semantics()
    _assert_v2_read("src/jobs/automated_performance_analyzer.py")


def test_performance_analysis_v2_reads():
    _assert_fixture_semantics()
    _assert_v2_read("scripts/performance_analysis.py")


def test_sync_positions_v2_reads():
    _assert_fixture_semantics()
    _assert_v2_read("scripts/sync_positions.py")


def test_performance_analyzer_v2_reads():
    _assert_fixture_semantics()
    _assert_v2_read("src/jobs/performance_analyzer.py")


def test_get_positions_script_v2_reads():
    _assert_fixture_semantics()
    _assert_v2_read("scripts/get_positions.py")
