"""Contract tests for PortfolioEnforcer (issue #99).

`max_position_pct` is a money-safety cap. It used to default to 0.03
(3%) while settings.trading.max_position_size_pct moved to 10.0 (percent)
— silently "fixing" the default would LOOSEN a safety limit, so the
parameter is now REQUIRED. Callers must pass an explicit fraction, e.g.
`settings.trading.max_position_size_pct / 100.0` (settings value is a
percent, the enforcer stores a fraction).

All tests are local (sqlite on tmp_path): no network, no credentials,
CI-safe.
"""

import inspect
from datetime import datetime, timezone

import aiosqlite
import pytest

from src.strategies.portfolio_enforcer import BlockedTradeError, PortfolioEnforcer

TEST_CATEGORY = "ncaab"


async def _seed_category(db_path, category=TEST_CATEGORY, score=80.0):
    """Give a category a passing score so tests reach the position-cap rule."""
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """INSERT OR REPLACE INTO category_scores
               (category, score, win_count, total_count, total_pnl,
                recent_trend, last_updated, blocked)
               VALUES (?, ?, 10, 20, 5.0, 0.0, ?, 0)""",
            (category, score, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()


def _make(db_path, portfolio_value=1000.0, max_position_pct=0.10):
    """Instantiate with an explicit cap, mirroring production call sites
    (settings.trading.max_position_size_pct / 100.0)."""
    return PortfolioEnforcer(
        db_path=str(db_path),
        portfolio_value=portfolio_value,
        max_position_pct=max_position_pct,
    )


# ------------------------------------------------------------------
# Issue #99: max_position_pct must be explicit (no default)
# ------------------------------------------------------------------

def test_max_position_pct_has_no_default():
    params = inspect.signature(PortfolioEnforcer.__init__).parameters
    assert params["max_position_pct"].default is inspect.Parameter.empty, (
        "max_position_pct must be required — a default on a money cap can go stale"
    )


def test_constructing_without_max_position_pct_raises():
    with pytest.raises(TypeError):
        PortfolioEnforcer(db_path=":memory:", portfolio_value=1000.0)


# ------------------------------------------------------------------
# Behavior with explicit values (regression contract)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_allows_trade_within_cap(tmp_path):
    enforcer = _make(tmp_path / "t.db", max_position_pct=0.10)
    await enforcer.initialize()
    await _seed_category(tmp_path / "t.db")
    allowed, reason = await enforcer.check_trade(
        ticker="KXNCAAB-TEST", side="no", amount=50.0, category=TEST_CATEGORY
    )
    assert allowed, reason


@pytest.mark.asyncio
async def test_blocks_trade_above_position_cap(tmp_path):
    enforcer = _make(tmp_path / "t.db", portfolio_value=1000.0, max_position_pct=0.10)
    await enforcer.initialize()
    await _seed_category(tmp_path / "t.db")
    allowed, reason = await enforcer.check_trade(
        ticker="KXNCAAB-TEST", side="no", amount=150.0, category=TEST_CATEGORY
    )
    assert not allowed
    assert "10%" in reason
    blocked = await enforcer.get_blocked_trades()
    assert len(blocked) == 1
    assert "10%" in blocked[0]["reason"]


@pytest.mark.asyncio
async def test_enforce_raises_blocked_trade_error(tmp_path):
    enforcer = _make(tmp_path / "t.db", portfolio_value=1000.0, max_position_pct=0.05)
    await enforcer.initialize()
    await _seed_category(tmp_path / "t.db")
    with pytest.raises(BlockedTradeError):
        await enforcer.enforce(
            ticker="KXNCAAB-TEST", side="no", amount=60.0, category=TEST_CATEGORY
        )
