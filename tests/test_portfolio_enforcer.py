"""src/strategies/portfolio_enforcer.py: each block rule, its boundary, the blocked log and reports.

Nothing instantiates PortfolioEnforcer outside tests (tests/test_category_scorer.py pins that), so
these tests describe a reference implementation, not a live gate. Every DB lives in tmp_path. No
network, no Kalshi client, no credentials.

Seeded scores used below: NCAAB 62.8 (10% tier), FED 35.0 (2% tier), ECON 21.8 (blocked), and an
unknown category scores 0 (blocked).

Strict xfails pin real defects (file:line + example). Fixed bug -> XPASS(strict) -> delete marker.
"""
import sqlite3

import pytest

from src.strategies import portfolio_enforcer as pe
from src.strategies.portfolio_enforcer import BlockedTradeError, PortfolioEnforcer

NCAAB = "KXNCAAB-A"
FED = "KXFED-A"
ECON_CATEGORY = "ECON"


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "enforcer.db")


@pytest.fixture
async def enforcer(db_path):
    e = PortfolioEnforcer(db_path, portfolio_value=1000.0)
    await e.initialize()
    return e


def _blocked_rows(db_path):
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT ticker, category, side, amount, reason, score FROM blocked_trades ORDER BY id"
        ).fetchall()


# --- Rule 1: category score ---


async def test_blocked_category_is_refused_and_logged(enforcer, db_path):
    allowed, reason = await enforcer.check_trade(
        "ANY-TICKER", "no", 5.0, category=ECON_CATEGORY
    )
    assert allowed is False
    assert "score 21.8 < 30" in reason
    rows = _blocked_rows(db_path)
    assert len(rows) == 1
    assert rows[0][:4] == ("ANY-TICKER", "ECON", "no", 5.0)
    assert rows[0][4] == reason
    assert rows[0][5] == pytest.approx(21.8, abs=0.05)


async def test_unknown_category_scores_zero_and_is_refused(enforcer):
    allowed, reason = await enforcer.check_trade("ANY", "yes", 1.0, category="NEVER_SEEN")
    assert allowed is False
    assert "score 0.0" in reason


async def test_block_reason_states_no_returns_figure(enforcer):
    """KNOWN_DATA is hand-entered seed data with no committed backtest behind it, so a block
    reason, which lands in logs and the blocked_trades table, must not quote ROI or an edge."""
    _, reason = await enforcer.check_trade("ANY", "no", 5.0, category=ECON_CATEGORY)
    for needle in ("ROI", "-70%", "proven edge"):
        assert needle not in reason


async def test_score_at_the_threshold_passes_and_just_below_is_refused(enforcer, monkeypatch):
    def score(value):
        async def get_score(category):
            return value
        monkeypatch.setattr(enforcer.scorer, "get_score", get_score)

    score(30.0)
    allowed, _ = await enforcer.check_trade(NCAAB, "no", 1.0)
    assert allowed is True

    score(29.99)
    allowed, reason = await enforcer.check_trade(NCAAB, "no", 1.0)
    assert allowed is False
    assert "score 30.0 < 30" in reason  # 29.99 prints rounded; the comparison is on the raw score


async def test_title_feeds_category_inference(enforcer, monkeypatch):
    seen = []
    real = pe.infer_category

    def spy(ticker, title=""):
        seen.append((ticker, title))
        return real(ticker, title)

    monkeypatch.setattr(pe, "infer_category", spy)
    await enforcer.check_trade("ZZZ", "no", 1.0, title="some title")
    assert seen == [("ZZZ", "some title")]


# --- Rule 2: category allocation ---


async def test_amount_at_category_allocation_passes_and_one_cent_over_is_refused(enforcer, db_path):
    # FED is in the 2% tier: $20 of $1000. The 3% position limit ($30) stays out of the way.
    allowed, reason = await enforcer.check_trade(FED, "no", 20.0)
    assert allowed is True
    assert "category='FED'" in reason and "max_alloc=2%" in reason
    assert _blocked_rows(db_path) == []

    allowed, reason = await enforcer.check_trade(FED, "no", 20.01)
    assert allowed is False
    assert "max allocation $20.00" in reason
    assert len(_blocked_rows(db_path)) == 1


async def test_zero_allocation_above_the_block_threshold_is_refused(enforcer, monkeypatch, db_path):
    """Unreachable with today's tiers (0% is exactly the blocked range), so force it."""
    monkeypatch.setattr(pe, "get_allocation_pct", lambda score: 0.0)
    allowed, reason = await enforcer.check_trade(NCAAB, "no", 1.0)
    assert allowed is False
    assert "0% allocation" in reason
    assert len(_blocked_rows(db_path)) == 1


# --- Rule 3: single position limit ---


async def test_amount_at_position_limit_passes_and_one_cent_over_is_refused(enforcer, db_path):
    # NCAAB allocation is 10% ($100), so the 3% limit ($30) is the one that bites.
    allowed, _ = await enforcer.check_trade(NCAAB, "no", 30.0)
    assert allowed is True

    allowed, reason = await enforcer.check_trade(NCAAB, "no", 30.01)
    assert allowed is False
    assert "max position size $30.00" in reason
    assert len(_blocked_rows(db_path)) == 1


# --- Rule 4: sector concentration ---


async def test_sector_limit_counts_only_the_same_category(enforcer, db_path):
    # 30% of $1000 is $300. $280 already in NCAAB.
    positions = {"KXNCAAB-B": 280.0, "KXNBA-C": 500.0}

    allowed, _ = await enforcer.check_trade(NCAAB, "no", 20.0, current_positions=positions)
    assert allowed is True  # 300 / 1000 is at the limit, not over; the NBA position is ignored

    allowed, reason = await enforcer.check_trade(NCAAB, "no", 20.01, current_positions=positions)
    assert allowed is False
    assert "sector limit 30%" in reason
    assert "current: $280.00" in reason
    assert len(_blocked_rows(db_path)) == 1


async def test_constructor_limits_are_used(db_path):
    e = PortfolioEnforcer(db_path, portfolio_value=1000.0, max_position_pct=0.01, max_sector_pct=0.05)
    await e.initialize()

    allowed, reason = await e.check_trade(NCAAB, "no", 10.01)
    assert allowed is False
    assert "max position size $10.00" in reason

    allowed, reason = await e.check_trade(
        NCAAB, "no", 10.0, current_positions={"KXNCAAB-B": 40.01}
    )
    assert allowed is False
    assert "sector limit 5%" in reason


# --- Documented limits of the class ---


async def test_zero_portfolio_value_runs_only_the_score_check(db_path):
    """portfolio_value defaults to 0, and every size rule is guarded on it being positive."""
    e = PortfolioEnforcer(db_path)
    await e.initialize()
    allowed, _ = await e.check_trade(
        NCAAB, "no", 1_000_000.0, current_positions={"KXNCAAB-B": 1_000_000.0}
    )
    assert allowed is True
    allowed, _ = await e.check_trade("ANY", "no", 1.0, category=ECON_CATEGORY)
    assert allowed is False


async def test_max_drawdown_pct_is_stored_but_changes_no_decision(db_path):
    """The module docstring says so; a drawdown check added later should fail this and update it."""
    results = []
    for pct in (0.0, 0.15):
        e = PortfolioEnforcer(db_path, portfolio_value=1000.0, max_drawdown_pct=pct)
        await e.initialize()
        assert e.max_drawdown_pct == pct
        results.append(await e.check_trade(NCAAB, "no", 5.0))
    assert results[0] == results[1]
    assert results[0][0] is True


def test_module_docstring_does_not_claim_a_drawdown_check():
    doc = pe.__doc__
    assert "exceed overall drawdown limits" not in doc
    assert "max_drawdown_pct is stored but no check reads it" in doc


@pytest.mark.xfail(strict=True, reason=(
    "portfolio_enforcer.py:124,137,152 compare with `amount >`: NaN compares False everywhere and "
    "a negative amount is below every limit, so check_trade allows both."
))
@pytest.mark.parametrize("amount", [float("nan"), -5.0])
async def test_invalid_amount_is_refused(enforcer, amount):
    allowed, _ = await enforcer.check_trade(NCAAB, "no", amount)
    assert allowed is False


# --- Every rule counts its block once ---


@pytest.mark.parametrize("kwargs", [
    pytest.param(dict(ticker="ANY", category=ECON_CATEGORY, amount=1.0), id="score"),
    pytest.param(dict(ticker=FED, amount=20.01), id="allocation"),
    pytest.param(dict(ticker=NCAAB, amount=30.01), id="position"),
    pytest.param(
        dict(ticker=NCAAB, amount=20.01, current_positions={"KXNCAAB-B": 280.0}), id="sector"
    ),
])
async def test_each_block_rule_counts_one_block_and_no_allow(enforcer, kwargs):
    allowed, _ = await enforcer.check_trade(side="no", **kwargs)
    assert allowed is False
    assert (enforcer._blocked_count, enforcer._allowed_count) == (1, 0)


async def test_zero_allocation_counts_one_block(enforcer, monkeypatch):
    monkeypatch.setattr(pe, "get_allocation_pct", lambda score: 0.0)
    await enforcer.check_trade(NCAAB, "no", 1.0)
    assert (enforcer._blocked_count, enforcer._allowed_count) == (1, 0)


# --- enforce() ---


async def test_enforce_raises_with_the_reason_when_blocked(enforcer):
    with pytest.raises(BlockedTradeError, match="score 21.8"):
        await enforcer.enforce("ANY", "no", 5.0, category=ECON_CATEGORY)


async def test_enforce_returns_none_when_allowed_and_passes_every_argument(enforcer, monkeypatch):
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return True, "ok"

    monkeypatch.setattr(enforcer, "check_trade", fake)
    positions = {"KXNCAAB-B": 1.0}
    result = await enforcer.enforce(
        "T", "yes", 2.0, title="t", category="C", current_positions=positions
    )
    assert result is None
    assert calls == [dict(
        ticker="T", side="yes", amount=2.0, title="t", category="C", current_positions=positions
    )]


# --- Blocked log and reports ---


async def test_get_blocked_trades_is_newest_first_and_honours_the_limit(enforcer):
    for i in range(3):
        await enforcer.check_trade(f"T{i}", "no", 1.0, category=ECON_CATEGORY)
    rows = await enforcer.get_blocked_trades()
    assert [r["ticker"] for r in rows] == ["T2", "T1", "T0"]
    assert set(rows[0]) == {
        "id", "ticker", "category", "side", "amount", "reason", "score", "blocked_at"
    }
    assert [r["ticker"] for r in await enforcer.get_blocked_trades(limit=2)] == ["T2", "T1"]


async def test_summary_groups_by_category_and_counts_the_session(enforcer):
    await enforcer.check_trade("A", "no", 1.0, category=ECON_CATEGORY)
    await enforcer.check_trade("B", "no", 2.5, category=ECON_CATEGORY)
    await enforcer.check_trade("C", "no", 4.0, category="NEVER_SEEN")
    await enforcer.check_trade(NCAAB, "no", 1.0)

    summary = await enforcer.get_blocked_summary()
    assert summary["by_category"] == [
        {"category": "ECON", "count": 2, "total_amount": 3.5},
        {"category": "NEVER_SEEN", "count": 1, "total_amount": 4.0},
    ]
    assert summary["session_blocked"] == 3
    assert summary["session_allowed"] == 1
    assert summary["session_block_rate"] == 0.75


async def test_summary_with_no_checks_has_a_zero_rate(enforcer):
    summary = await enforcer.get_blocked_summary()
    assert summary["by_category"] == []
    assert summary["session_block_rate"] == 0.0


async def test_reset_session_counts_keeps_the_blocked_table(enforcer):
    await enforcer.check_trade("A", "no", 1.0, category=ECON_CATEGORY)
    await enforcer.check_trade(NCAAB, "no", 1.0)
    enforcer.reset_session_counts()
    summary = await enforcer.get_blocked_summary()
    assert (summary["session_blocked"], summary["session_allowed"]) == (0, 0)
    assert len(summary["by_category"]) == 1


async def test_a_failed_log_write_still_refuses_the_trade(enforcer, db_path, caplog):
    with sqlite3.connect(db_path) as db:
        db.execute("DROP TABLE blocked_trades")
    allowed, _ = await enforcer.check_trade("A", "no", 1.0, category=ECON_CATEGORY)
    assert allowed is False
    assert enforcer._blocked_count == 1
    assert "Failed to log blocked trade" in caplog.text


def test_format_blocked_report_renders_session_and_rows(db_path):
    report = PortfolioEnforcer(db_path).format_blocked_report({
        "session_blocked": 1,
        "session_allowed": 3,
        "session_block_rate": 0.25,
        "by_category": [{"category": "ECON", "count": 2, "total_amount": 3.5}],
    })
    assert "Session: 1 blocked / 4 checked (25% block rate)" in report
    line = next(ln for ln in report.splitlines() if ln.strip().startswith("ECON"))
    assert line.split() == ["ECON", "2", "$", "3.50"]


def test_format_blocked_report_with_no_rows(db_path):
    report = PortfolioEnforcer(db_path).format_blocked_report({
        "session_blocked": 0,
        "session_allowed": 0,
        "session_block_rate": 0.0,
    })
    assert "0 blocked / 0 checked (0% block rate)" in report
