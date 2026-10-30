"""
Portfolio Enforcer — Runs before every trade scan.

Hard-blocks:
  - Categories scoring < 30
  - Positions that would exceed category allocation limits
  - Positions that would exceed overall drawdown limits

Tracks and logs all blocked trades for analysis.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiosqlite

from src.strategies.category_scorer import CategoryScorer, infer_category, BLOCK_THRESHOLD, get_allocation_pct

logger = logging.getLogger(__name__)


class BlockedTradeError(Exception):
    """Raised when a trade is hard-blocked by the enforcer."""
    pass


class PortfolioEnforcer:
    """
    Enforces portfolio discipline before every trade.

    Call `check_trade()` before executing any order.
    It raises `BlockedTradeError` if the trade violates rules.

    Usage:
        enforcer = PortfolioEnforcer(max_position_pct=0.10,
                                     db_path=db_path, portfolio_value=1000.0)
        await enforcer.initialize()
        try:
            await enforcer.check_trade(ticker="KXNCAAB-...", side="no", amount=50.0)
        except BlockedTradeError as e:
            logger.warning(f"Trade blocked: {e}")

    max_position_pct is REQUIRED (no default): it is a money-safety cap, so
    callers must choose it explicitly, e.g. from
    settings.trading.max_position_size_pct / 100.0 (settings is a percent,
    this class stores a fraction).
    """

    def __init__(
        self,
        max_position_pct: float,
        db_path: str = "trading_system.db",
        portfolio_value: float = 0.0,
        max_drawdown_pct: float = 0.15,
        max_sector_pct: float = 0.30,
    ):
        self.db_path = db_path
        self.portfolio_value = portfolio_value
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.scorer = CategoryScorer(db_path)
        self._blocked_count = 0
        self._allowed_count = 0

    async def initialize(self) -> None:
        """Initialize scorer and create blocked trades log table."""
        await self.scorer.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS blocked_trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    side        TEXT NOT NULL,
                    amount      REAL NOT NULL,
                    reason      TEXT NOT NULL,
                    score       REAL,
                    blocked_at  TEXT NOT NULL
                )
            """)
            await db.commit()

    # ------------------------------------------------------------------
    # Main gate
    # ------------------------------------------------------------------

    async def check_trade(
        self,
        ticker: str,
        side: str,
        amount: float,
        title: str = "",
        category: Optional[str] = None,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a trade is allowed.

        Returns (allowed: bool, reason: str).
        Does NOT raise — callers decide whether to use BlockedTradeError.
        """
        cat = category or infer_category(ticker, title)
        score = await self.scorer.get_score(cat)
        max_alloc = get_allocation_pct(score)

        # --- Rule 1: Category score below block threshold ---
        if score < BLOCK_THRESHOLD:
            reason = (
                f"Category '{cat}' score {score:.1f} < {BLOCK_THRESHOLD} (blocked). "
                f"NCAAB NO-side is the only proven edge. "
                f"Economic categories have -70% ROI historically."
            )
            await self._log_blocked(ticker, cat, side, amount, reason, score)
            self._blocked_count += 1
            return False, reason

        # --- Rule 2: Category max allocation check ---
        if max_alloc == 0.0:
            reason = f"Category '{cat}' score {score:.1f} → 0% allocation (hard blocked)"
            await self._log_blocked(ticker, cat, side, amount, reason, score)
            self._blocked_count += 1
            return False, reason

        if self.portfolio_value > 0:
            max_allowed = self.portfolio_value * max_alloc
            if amount > max_allowed:
                reason = (
                    f"Trade amount ${amount:.2f} exceeds category '{cat}' "
                    f"max allocation ${max_allowed:.2f} "
                    f"({max_alloc*100:.0f}% of ${self.portfolio_value:.2f}, score={score:.1f})"
                )
                await self._log_blocked(ticker, cat, side, amount, reason, score)
                self._blocked_count += 1
                return False, reason

        # --- Rule 3: Overall position size limit ---
        if self.portfolio_value > 0:
            max_single = self.portfolio_value * self.max_position_pct
            if amount > max_single:
                reason = (
                    f"Trade amount ${amount:.2f} exceeds max position size "
                    f"${max_single:.2f} ({self.max_position_pct*100:.0f}% of portfolio)"
                )
                await self._log_blocked(ticker, cat, side, amount, reason, score)
                self._blocked_count += 1
                return False, reason

        # --- Rule 4: Sector concentration check ---
        if current_positions and self.portfolio_value > 0:
            sector_exposure = sum(
                v for k, v in current_positions.items()
                if infer_category(k) == cat
            )
            if (sector_exposure + amount) / self.portfolio_value > self.max_sector_pct:
                reason = (
                    f"Adding ${amount:.2f} to '{cat}' would exceed sector limit "
                    f"{self.max_sector_pct*100:.0f}% (current: ${sector_exposure:.2f})"
                )
                await self._log_blocked(ticker, cat, side, amount, reason, score)
                self._blocked_count += 1
                return False, reason

        self._allowed_count += 1
        return True, f"Trade allowed (category='{cat}', score={score:.1f}, max_alloc={max_alloc*100:.0f}%)"

    async def enforce(
        self,
        ticker: str,
        side: str,
        amount: float,
        title: str = "",
        category: Optional[str] = None,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Check and raise BlockedTradeError if not allowed.
        Use this when you want exceptions rather than booleans.
        """
        allowed, reason = await self.check_trade(
            ticker=ticker,
            side=side,
            amount=amount,
            title=title,
            category=category,
            current_positions=current_positions,
        )
        if not allowed:
            raise BlockedTradeError(reason)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    async def get_blocked_trades(self, limit: int = 50) -> List[Dict]:
        """Return the most recently blocked trades."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM blocked_trades
                ORDER BY blocked_at DESC
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_blocked_summary(self) -> Dict:
        """Summarize blocked trades by category and reason."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT category, COUNT(*) as count, SUM(amount) as total_amount
                FROM blocked_trades
                GROUP BY category
                ORDER BY count DESC
            """)
            rows = await cursor.fetchall()

        return {
            "by_category": [dict(r) for r in rows],
            "session_blocked": self._blocked_count,
            "session_allowed": self._allowed_count,
            "session_block_rate": (
                self._blocked_count / max(1, self._blocked_count + self._allowed_count)
            ),
        }

    def reset_session_counts(self) -> None:
        """Reset session-level counters."""
        self._blocked_count = 0
        self._allowed_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _log_blocked(
        self,
        ticker: str,
        category: str,
        side: str,
        amount: float,
        reason: str,
        score: Optional[float],
    ) -> None:
        """Log a blocked trade to the database."""
        now_iso = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "TRADE BLOCKED | ticker=%s category=%s score=%.1f reason=%s",
            ticker, category, score or 0.0, reason
        )
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO blocked_trades
                    (ticker, category, side, amount, reason, score, blocked_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (ticker, category, side, amount, reason, score, now_iso))
                await db.commit()
        except Exception as e:
            logger.error("Failed to log blocked trade: %s", e)

    def format_blocked_report(self, summary: Dict) -> str:
        """Format blocked trades summary as readable string."""
        lines = [
            "=" * 60,
            "  BLOCKED TRADES SUMMARY",
            f"  Session: {summary['session_blocked']} blocked / "
            f"{summary['session_blocked'] + summary['session_allowed']} checked "
            f"({summary['session_block_rate']*100:.0f}% block rate)",
            "",
            f"  {'Category':<20} {'Blocked':>8} {'$ Blocked':>12}",
            f"  {'-'*20} {'-'*8} {'-'*12}",
        ]
        for row in summary.get("by_category", []):
            lines.append(
                f"  {row['category']:<20} {row['count']:>8} ${row['total_amount']:>10.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
