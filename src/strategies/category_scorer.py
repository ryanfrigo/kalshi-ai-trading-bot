"""
Category Scorer — Score trading categories 0-100 based on historical performance.

Scoring formula:
  - ROI (40%): positive ROI = good, negative = bad
  - Sample size (20%): more data = more confidence
  - Recent trend (25%): last 10 trades trending up/down
  - Win rate (15%): percentage of winning trades

Allocation tiers:
  80-100 → 20% max position
  60-79  → 10% max position
  40-59  → 5%  max position
  30-39  → 2%  max position
  0-29   → 0%  (flagged BLOCKED in `cli scores`; display only, nothing in the
                order path reads it)
"""

import asyncio
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import aiosqlite

# Weights
W_ROI = 0.40
W_SAMPLE = 0.20
W_TREND = 0.25
W_WINRATE = 0.15

# Minimum score to allow trading
BLOCK_THRESHOLD = 30

# Allocation tiers: (min_score, max_pct_of_portfolio)
# Tiers align with BLOCK_THRESHOLD=30 so allocation 0% ↔ blocked
ALLOCATION_TIERS = [
    (80, 0.20),
    (60, 0.10),
    (40, 0.05),
    (30, 0.02),   # marginal — barely above block threshold
    (0,  0.00),   # blocked (< 30)
]

# Minimum samples before we score (below this → default score of 0 → blocked)
MIN_SAMPLES_FOR_SCORING = 5

# Real-world data snapshot (seeded from Ryan's trading history)
# NCAAB NO-side: 74% WR, +10% ROI — the ONLY profitable edge
# Economic (CPI etc): -70% ROI avg per trade, 78% of all losses
#
# total_pnl = SUM of per-trade ROI fractions (avg_roi = total_pnl / total_count)
# NCAAB: +10% avg ROI × 50 trades = +5.0
# CPI:   -70% avg ROI × 50 trades = -35.0  (more data = more confidence in the loss)
# FED:   -40% avg ROI × 50 trades = -20.0
# ECON:  -70% avg ROI × 100 trades = -70.0 (largest loss category — 78% of all losses)
# ECON_MACRO: -55% avg ROI × 60 trades = -33.0
KNOWN_DATA = {
    "NCAAB":      {"wins": 37,  "total": 50,  "total_pnl":  5.0,  "recent_trend":  0.15},
    "ECON":       {"wins": 22,  "total": 100, "total_pnl": -70.0, "recent_trend": -0.80},
    "CPI":        {"wins": 12,  "total": 50,  "total_pnl": -35.0, "recent_trend": -0.75},
    "FED":        {"wins": 16,  "total": 50,  "total_pnl": -20.0, "recent_trend": -0.50},
    "ECON_MACRO": {"wins": 18,  "total": 60,  "total_pnl": -33.0, "recent_trend": -0.65},
}


def is_seeded_row(row: Dict) -> bool:
    """True if a category_scores row still equals its KNOWN_DATA seed exactly
    (same wins, total and total_pnl). Display-only; once real trades change the
    counts the row is no longer marked."""
    seed = KNOWN_DATA.get(row["category"])
    return bool(seed) and (
        row["win_count"] == seed["wins"]
        and row["total_count"] == seed["total"]
        and row["total_pnl"] == seed["total_pnl"]
    )


def _compute_score(
    win_rate: float,
    avg_roi: float,
    sample_size: int,
    recent_trend: float,
) -> float:
    """Compute 0-100 score from component metrics."""
    # ROI score: map [-100%, +50%] → [0, 100]
    # -100% ROI → 0, 0% ROI → 40, +50% → 100
    roi_normalized = max(0.0, min(1.0, (avg_roi + 1.0) / 1.50))
    roi_score = roi_normalized * 100

    # Sample size score: max 40pts (represents data confidence, not performance).
    # Caps at 40 so poor-ROI categories can't float above block threshold on sample alone.
    # 5 samples → ~13pts, 50+ → 40pts, 200+ → 40pts
    if sample_size <= 0:
        sample_score = 0.0
    else:
        sample_score = min(40.0, math.log(sample_size + 1, 200) * 100 * 0.8)

    # Trend score: recent_trend is -1 to +1 (direction of last N trades)
    # +1 = strong up, -1 = strong down
    trend_score = max(0.0, min(100.0, (recent_trend + 1.0) / 2.0 * 100))

    # Win rate score: 50% → 50pts, 70% → 85pts, 100% → 100pts
    winrate_score = max(0.0, min(100.0, win_rate * 100))

    total = (
        roi_score * W_ROI
        + sample_score * W_SAMPLE
        + trend_score * W_TREND
        + winrate_score * W_WINRATE
    )
    return round(total, 1)


def get_allocation_pct(score: float) -> float:
    """Return max allocation % for a given score."""
    for min_score, pct in ALLOCATION_TIERS:
        if score >= min_score:
            return pct
    return 0.0


def is_blocked(score: float) -> bool:
    """Return True if category should be blocked from trading."""
    return score < BLOCK_THRESHOLD or get_allocation_pct(score) == 0.0


class CategoryScorer:
    """
    Scores trading categories and provides allocation limits.

    Usage:
        scorer = CategoryScorer()
        await scorer.initialize()
        score = await scorer.get_score("NCAAB")
        blocked = await scorer.is_blocked("ECON")
        max_pct = await scorer.get_max_allocation_pct("NCAAB")
        await scorer.update_score("NCAAB", trade_won=True, roi=0.12)
    """

    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path
        self._cache: Dict[str, dict] = {}
        self._cache_ts: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=15)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create tables and seed known data if empty."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS category_scores (
                    category        TEXT PRIMARY KEY,
                    score           REAL NOT NULL DEFAULT 50.0,
                    win_count       INTEGER NOT NULL DEFAULT 0,
                    total_count     INTEGER NOT NULL DEFAULT 0,
                    total_pnl       REAL NOT NULL DEFAULT 0.0,
                    recent_trend    REAL NOT NULL DEFAULT 0.0,
                    last_updated    TEXT NOT NULL,
                    blocked         INTEGER NOT NULL DEFAULT 0
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS category_trade_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    category        TEXT NOT NULL,
                    won             INTEGER NOT NULL,
                    roi             REAL NOT NULL,
                    trade_time      TEXT NOT NULL
                )
            """)
            await db.commit()

        # Seed known historical data
        await self._seed_known_data()

    async def _seed_known_data(self) -> None:
        """Seed initial scores from real trading history if tables are empty."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM category_scores")
            row = await cursor.fetchone()
            if row and row[0] > 0:
                return  # already seeded

            now = datetime.now(timezone.utc).isoformat()
            for category, data in KNOWN_DATA.items():
                wins = data["wins"]
                total = data["total"]
                pnl = data["total_pnl"]
                trend = data["recent_trend"]
                win_rate = wins / total if total > 0 else 0.5
                avg_roi = pnl / total if total > 0 else 0.0
                score = _compute_score(win_rate, avg_roi, total, trend)
                blocked = 1 if is_blocked(score) else 0
                await db.execute("""
                    INSERT OR REPLACE INTO category_scores
                    (category, score, win_count, total_count, total_pnl, recent_trend, last_updated, blocked)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (category, score, wins, total, pnl, trend, now, blocked))
            await db.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_score(self, category: str) -> float:
        """Return the current score for a category (0-100). Unknown = 0."""
        data = await self._load(category)
        if data is None:
            return 0.0
        return data["score"]

    async def get_all_scores(self) -> List[Dict]:
        """Return all category scores sorted by score descending."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM category_scores ORDER BY score DESC"
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def is_blocked(self, category: str) -> bool:
        """Return True if category is blocked."""
        score = await self.get_score(category)
        return is_blocked(score)

    async def get_max_allocation_pct(self, category: str) -> float:
        """Return max allocation % for a category."""
        score = await self.get_score(category)
        return get_allocation_pct(score)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def update_score(
        self,
        category: str,
        trade_won: bool,
        roi: float,
        recalculate: bool = True,
    ) -> float:
        """
        Record a trade outcome and recompute the score.
        Returns the new score.
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Log the trade
            await db.execute("""
                INSERT INTO category_trade_log (category, won, roi, trade_time)
                VALUES (?, ?, ?, ?)
            """, (category, 1 if trade_won else 0, roi, now_iso))

            # Load or init the category row
            cursor = await db.execute(
                "SELECT * FROM category_scores WHERE category = ?", (category,)
            )
            row = await cursor.fetchone()
            if row:
                win_count = row[2] + (1 if trade_won else 0)
                total_count = row[3] + 1
                total_pnl = row[4] + roi
            else:
                win_count = 1 if trade_won else 0
                total_count = 1
                total_pnl = roi

            # Compute recent trend from last 10 trades
            trend = await self._compute_recent_trend(db, category)

            win_rate = win_count / total_count if total_count > 0 else 0.5
            avg_roi = total_pnl / total_count if total_count > 0 else 0.0

            if total_count < MIN_SAMPLES_FOR_SCORING:
                score = 0.0  # Not enough data — block
            else:
                score = _compute_score(win_rate, avg_roi, total_count, trend)

            blocked = 1 if is_blocked(score) else 0

            await db.execute("""
                INSERT OR REPLACE INTO category_scores
                (category, score, win_count, total_count, total_pnl, recent_trend, last_updated, blocked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (category, score, win_count, total_count, total_pnl, trend, now_iso, blocked))

            await db.commit()

        # Invalidate cache
        self._cache.pop(category, None)
        self._cache_ts.pop(category, None)

        return score

    async def force_block(self, category: str, reason: str = "") -> None:
        """Manually block a category regardless of score."""
        now_iso = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO category_scores
                (category, score, win_count, total_count, total_pnl, recent_trend, last_updated, blocked)
                VALUES (?, 0, 0, 0, 0, 0, ?, 1)
            """, (category, now_iso))
            await db.execute(
                "UPDATE category_scores SET blocked = 1, last_updated = ? WHERE category = ?",
                (now_iso, category)
            )
            await db.commit()
        self._cache.pop(category, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load(self, category: str) -> Optional[Dict]:
        """Load from cache or DB."""
        now = datetime.now(timezone.utc)
        if category in self._cache:
            if now - self._cache_ts[category] < self._cache_ttl:
                return self._cache[category]

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM category_scores WHERE category = ?", (category,)
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        data = dict(row)
        self._cache[category] = data
        self._cache_ts[category] = now
        return data

    async def _compute_recent_trend(
        self, db: aiosqlite.Connection, category: str, n: int = 10
    ) -> float:
        """
        Compute recent trend from last N trades.
        Returns value in [-1, +1]:
          +1 = all wins, -1 = all losses
        """
        cursor = await db.execute("""
            SELECT won, roi FROM category_trade_log
            WHERE category = ?
            ORDER BY id DESC
            LIMIT ?
        """, (category, n))
        rows = await cursor.fetchall()

        if not rows:
            return 0.0

        # Weighted recency: more recent trades matter more
        total_weight = 0.0
        weighted_sum = 0.0
        for i, (won, roi) in enumerate(rows):
            weight = 1.0 / (i + 1)  # more recent = higher weight
            signal = 1.0 if won else -1.0
            weighted_sum += signal * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def score_label(self, score: float) -> str:
        """Human-readable label for a score."""
        if score >= 80:
            return "STRONG ✅"
        elif score >= 60:
            return "GOOD 🟢"
        elif score >= 40:
            return "WEAK 🟡"
        elif score >= BLOCK_THRESHOLD:
            return "POOR 🟠"
        else:
            return "BLOCKED 🚫"

    def format_scores_table(self, scores: List[Dict]) -> str:
        """Format scores as a readable table string."""
        any_seeded = False
        lines = [
            "=" * 70,
            "  CATEGORY SCORES",
            f"  {'Category':<18} {'Score':>6} {'WR':>6} {'ROI':>8} {'Trades':>7} {'Alloc':>6} {'Status'}",
            f"  {'-'*18} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*10}",
        ]
        for row in scores:
            cat = row["category"]
            score = row["score"]
            total = row["total_count"]
            wins = row["win_count"]
            pnl = row["total_pnl"]
            if is_seeded_row(row):
                cat += "*"
                any_seeded = True
            wr = f"{wins/total*100:.0f}%" if total > 0 else "n/a"
            avg_roi = f"{pnl/total*100:.1f}%" if total > 0 else "n/a"
            alloc = f"{get_allocation_pct(score)*100:.0f}%"
            label = self.score_label(score)
            lines.append(
                f"  {cat:<18} {score:>6.1f} {wr:>6} {avg_roi:>8} {total:>7} {alloc:>6}  {label}"
            )
        lines.append("=" * 70)
        if any_seeded:
            lines.append(
                "  Rows marked * are starting estimates hard-coded in "
                "category_scorer.py (KNOWN_DATA), not your trades."
            )
        return "\n".join(lines)


def infer_category(ticker: str, title: str = "") -> str:
    """
    Infer trading category from a Kalshi ticker or market title.
    Returns a normalized category string.
    """
    ticker_upper = ticker.upper()
    title_lower = title.lower()

    # Sports
    if any(ticker_upper.startswith(p) for p in ["KXNCAAB", "KXNCAAM", "NCAAB", "NCAAM", "KXBIG10", "KXBIG12", "KXACC", "KXSEC", "KXAAC", "KXBIGEAST"]):
        return "NCAAB"
    if any(ticker_upper.startswith(p) for p in ["KXNBA", "NBA"]):
        return "NBA"
    if any(ticker_upper.startswith(p) for p in ["KXNFL", "NFL"]):
        return "NFL"
    if any(ticker_upper.startswith(p) for p in ["KXNHL", "NHL"]):
        return "NHL"
    if any(ticker_upper.startswith(p) for p in ["KXMLB", "MLB"]):
        return "MLB"
    if any(ticker_upper.startswith(p) for p in ["KXUFC", "UFC"]):
        return "UFC"
    if any(ticker_upper.startswith(p) for p in ["KXPGA", "PGA"]):
        return "GOLF"

    # Economics
    if any(x in ticker_upper for x in ["CPI", "INFLATION"]):
        return "CPI"
    if any(x in ticker_upper for x in ["FED", "FOMC", "RATE"]):
        return "FED"
    if any(x in ticker_upper for x in ["GDP", "JOBS", "NFP", "UNEMPLOYMENT", "PCE"]):
        return "ECON_MACRO"
    if any(x in title_lower for x in ["federal reserve", "interest rate", "fomc"]):
        return "FED"
    if any(x in title_lower for x in ["cpi", "inflation", "consumer price"]):
        return "CPI"
    if any(x in title_lower for x in ["gdp", "nonfarm", "unemployment", "jobs report"]):
        return "ECON_MACRO"

    # Politics / Elections
    if any(x in ticker_upper for x in ["PRES", "SENATE", "HOUSE", "ELECT", "TRUMP", "BIDEN"]):
        return "POLITICS"
    if any(x in title_lower for x in ["election", "president", "senate", "congress"]):
        return "POLITICS"

    # Crypto
    if any(x in ticker_upper for x in ["BTC", "ETH", "CRYPTO", "SOL"]):
        return "CRYPTO"

    # Market / Stocks
    if any(x in ticker_upper for x in ["SPX", "SP500", "NASDAQ", "DOW"]):
        return "MARKETS"

    # Weather / Climate
    if any(x in ticker_upper for x in ["TEMP", "SNOW", "RAIN", "WEATHER"]):
        return "WEATHER"

    # Entertainment
    if any(x in ticker_upper for x in ["OSCAR", "GRAMMY", "AWARD", "MOVIE", "ALBUM", "SONG"]):
        return "ENTERTAINMENT"

    # Default
    return "OTHER"
