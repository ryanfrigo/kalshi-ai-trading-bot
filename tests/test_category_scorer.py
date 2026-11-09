"""src/strategies/category_scorer.py: score math, tiers, unknown categories, persistence, corruption.

Every DB lives in tmp_path (the class default is the relative path "trading_system.db", so an
autouse fixture chdirs into tmp_path as well). No network, no Kalshi client, no credentials;
the module never touches an order method and a source-scan test pins that.

Who uses it (grep):
  * cli.py:1133 `cli.py scores`        -> display only
  * src/mcp_server.py:215              -> read-only MCP tool
  * src/strategies/portfolio_enforcer.py:59 builds a CategoryScorer, but nothing instantiates
    PortfolioEnforcer (cli.py:81 only imports it).
  * NOT on `cli.py daily` (SafeCompounder) and NOT in jobs/decide.py. No gate reads it.
  * update_score() has no caller outside this module, so the stored scores are the hard-coded
    KNOWN_DATA seed until someone wires trade outcomes in.

Strict xfails pin real defects (file:line + example). Fixed bug -> XPASS(strict) -> delete marker.
"""
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.strategies import category_scorer as cs
from src.strategies.category_scorer import (
    ALLOCATION_TIERS,
    BLOCK_THRESHOLD,
    KNOWN_DATA,
    CategoryScorer,
    _compute_score,
    get_allocation_pct,
    infer_category,
    is_blocked,
    is_seeded_row,
)

REPO = Path(__file__).resolve().parent.parent
NAN = float("nan")


@pytest.fixture(autouse=True)
def _cwd_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "scores.db")


@pytest.fixture
async def scorer(db_path):
    s = CategoryScorer(db_path)
    await s.initialize()
    return s


def sql(db_path, stmt, *args):
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(stmt, args)
        con.commit()
        return cur.fetchall()
    finally:
        con.close()


# ------------------------------------------------------------ score math (hand computed)
def test_weights_sum_to_one():
    assert cs.W_ROI + cs.W_SAMPLE + cs.W_TREND + cs.W_WINRATE == pytest.approx(1.0)


def test_neutral_inputs_score_46_7():
    # roi (0+1)/1.5 = .6667 -> 66.67*.4 = 26.667 ; sample 0 ; trend 50*.25 = 12.5 ; wr 50*.15 = 7.5
    assert _compute_score(0.5, 0.0, 0, 0.0) == 46.7


def test_ncaab_seed_score_62_8():
    # roi 1.1/1.5=.7333 -> 73.33*.4=29.333 ; sample capped 40*.2=8 ; trend 57.5*.25=14.375 ; wr 74*.15=11.1
    assert _compute_score(0.74, 0.10, 50, 0.15) == 62.8
    assert 29.333 + 8 + 14.375 + 11.1 == pytest.approx(62.808)


def test_best_case_five_trade_category_scores_85_4():
    # roi 1.0*100*.4=40 ; sample log200(6)*80=27.054*.2=5.411 ; trend 100*.25=25 ; wr 100*.15=15
    assert _compute_score(1.0, 0.5, 5, 1.0) == 85.4


@pytest.mark.parametrize("cat,hand", [
    ("NCAAB", 62.808), ("ECON", 21.8), ("CPI", 22.725), ("FED", 35.05), ("ECON_MACRO", 28.875),
])
def test_every_seed_scores_to_its_hand_value(cat, hand):
    d = KNOWN_DATA[cat]
    got = _compute_score(d["wins"] / d["total"], d["total_pnl"] / d["total"], d["total"], d["recent_trend"])
    assert got == pytest.approx(hand, abs=0.06)


def test_sample_score_ramps_then_caps_at_40_points():
    # 40 pts is reached once log200(n+1)*80 >= 40, i.e. n >= 14 (the docstring's "~13pts at 5" is 27).
    s = lambda n: _compute_score(0.5, 0.0, n, 0.0)
    assert s(0) < s(1) < s(5) < s(13) < s(14)
    assert s(14) == s(15) == s(1000)
    assert s(5) - s(0) == pytest.approx(27.0538 * cs.W_SAMPLE, abs=0.06)


def test_negative_sample_size_scores_like_zero():
    assert _compute_score(0.5, 0.0, -3, 0.0) == _compute_score(0.5, 0.0, 0, 0.0)


@pytest.mark.parametrize("roi,expected_roi_pts", [(-5.0, 0.0), (-1.0, 0.0), (0.5, 40.0), (9.0, 40.0)])
def test_roi_component_clamps_to_0_100(roi, expected_roi_pts):
    base = _compute_score(0.0, -1.0, 0, -1.0)           # roi, sample, trend, winrate all at floor
    assert _compute_score(0.0, roi, 0, -1.0) - base == pytest.approx(expected_roi_pts, abs=0.06)


def test_trend_and_winrate_clamp():
    assert _compute_score(0.5, 0.0, 0, 5.0) == _compute_score(0.5, 0.0, 0, 1.0)
    assert _compute_score(0.5, 0.0, 0, -5.0) == _compute_score(0.5, 0.0, 0, -1.0)
    assert _compute_score(7.0, 0.0, 0, 0.0) == _compute_score(1.0, 0.0, 0, 0.0)
    assert _compute_score(-2.0, 0.0, 0, 0.0) == _compute_score(0.0, 0.0, 0, 0.0)


def test_score_is_monotone_in_each_input():
    base = dict(win_rate=0.5, avg_roi=0.0, sample_size=10, recent_trend=0.0)
    ref = _compute_score(**base)
    for k, up in (("win_rate", 0.6), ("avg_roi", 0.1), ("sample_size", 12), ("recent_trend", 0.2)):
        assert _compute_score(**{**base, k: up}) > ref, k


def test_score_is_always_within_0_100_over_a_grid():
    for wr in (0, 0.5, 1):
        for roi in (-3, -1, 0, 0.5, 3):
            for n in (0, 5, 50, 10**6):
                for tr in (-1, 0, 1):
                    assert 0.0 <= _compute_score(wr, roi, n, tr) <= 100.0


@pytest.mark.xfail(strict=True, reason=(
    "src/strategies/category_scorer.py:88,101,104 `max(0.0, min(1.0, nan))` returns 1.0: NaN roi/win-rate/"
    "trend each clamp to the TOP of their range, so _compute_score(nan, nan, 10, nan) = 87.2 "
    "(STRONG, 20% allocation) instead of failing closed."))
def test_nan_inputs_do_not_score_high():
    assert _compute_score(NAN, NAN, 10, NAN) < BLOCK_THRESHOLD


# ------------------------------------------------------------ tiers and blocking
@pytest.mark.parametrize("score,pct", [
    (100, 0.20), (80, 0.20), (79.9, 0.10), (60, 0.10), (59.9, 0.05),
    (40, 0.05), (39.9, 0.02), (30, 0.02), (29.9, 0.0), (0, 0.0), (-1, 0.0),
])
def test_allocation_tier_boundaries(score, pct):
    assert get_allocation_pct(score) == pct


def test_block_threshold_is_the_lowest_nonzero_tier():
    nonzero = [m for m, p in ALLOCATION_TIERS if p > 0]
    assert min(nonzero) == BLOCK_THRESHOLD == 30
    assert [m for m, _ in ALLOCATION_TIERS] == sorted((m for m, _ in ALLOCATION_TIERS), reverse=True)


@pytest.mark.parametrize("score,blocked", [(29.99, True), (30, False), (0, True), (-5, True), (100, False)])
def test_is_blocked_boundaries(score, blocked):
    assert is_blocked(score) is blocked


def test_nan_score_fails_closed_in_both_helpers():
    assert get_allocation_pct(NAN) == 0.0
    assert is_blocked(NAN) is True


@pytest.mark.parametrize("score,label", [
    (80, "STRONG"), (79.9, "GOOD"), (60, "GOOD"), (59.9, "WEAK"), (40, "WEAK"),
    (39.9, "POOR"), (30, "POOR"), (29.9, "BLOCKED"),
])
def test_score_label_boundaries(score, label):
    assert label in CategoryScorer("unused.db").score_label(score)


# ------------------------------------------------------------ infer_category
@pytest.mark.parametrize("ticker,title,cat", [
    ("KXNCAAB-26MAR-DUKE", "", "NCAAB"),
    ("kxnba-1", "", "NBA"),
    ("KXNFLGAME-1", "", "NFL"),
    ("KXNHL-1", "", "NHL"),
    ("KXMLB-1", "", "MLB"),
    ("KXUFC-1", "", "UFC"),
    ("KXPGA-1", "", "GOLF"),
    ("KXCPI-26OCT", "", "CPI"),
    ("KXFEDDECISION-26", "", "FED"),
    ("KXRATECUT-26", "", "FED"),
    ("KXGDP-26Q3", "", "ECON_MACRO"),
    ("KXPRES-28", "", "POLITICS"),
    ("KXBTC-1", "", "CRYPTO"),
    ("KXSPX-1", "", "MARKETS"),
    ("KXTEMP-NYC", "", "WEATHER"),
    ("KXOSCAR-1", "", "ENTERTAINMENT"),
    ("X1", "Will the Federal Reserve cut?", "FED"),
    ("X1", "Will inflation top 3%?", "CPI"),
    ("X1", "Will unemployment fall?", "ECON_MACRO"),
    ("X1", "Who wins the election?", "POLITICS"),
])
def test_infer_category_table(ticker, title, cat):
    assert infer_category(ticker, title) == cat


@pytest.mark.parametrize("ticker,title", [("", ""), ("ZZZZ", ""), ("KXMYSTERY-1", "Will it happen?")])
def test_unknown_ticker_and_title_fall_to_other(ticker, title):
    assert infer_category(ticker, title) == "OTHER"


def test_ticker_prefix_wins_over_title():
    assert infer_category("KXNBA-1", "Federal Reserve rate decision") == "NBA"


@pytest.mark.xfail(strict=True, reason=(
    "src/strategies/category_scorer.py:430 bare startswith('KXSEC') (no delimiter) files any ticker "
    "that merely starts KXSEC... as NCAAB, the one category the seed scores 62.8 / 10% allocation: "
    "infer_category('KXSECRETARY-28') == 'NCAAB'. A mislabel into the best-scored bucket loosens "
    "the gate (same for KXACC*, KXAAC*)."))
def test_prefix_match_does_not_pull_unrelated_tickers_into_ncaab():
    assert infer_category("KXSECRETARY-28") != "NCAAB"


@pytest.mark.xfail(strict=True, reason=(
    "src/strategies/category_scorer.py:474 substring 'RAIN' matches inside KXUKRAINE-CEASEFIRE, so a "
    "geopolitics market is scored as WEATHER (also KXSOLAR -> CRYPTO via 'SOL', KXHOUSEPRICE -> "
    "POLITICS via 'HOUSE')."))
def test_substring_match_does_not_misfile_ukraine_as_weather():
    assert infer_category("KXUKRAINE-CEASEFIRE") != "WEATHER"


# ------------------------------------------------------------ persistence (temp DB)
async def test_initialize_creates_tables_and_seeds_five_rows(scorer, db_path):
    rows = await scorer.get_all_scores()
    assert {r["category"] for r in rows} == set(KNOWN_DATA)
    assert all(is_seeded_row(r) for r in rows)
    tables = {t for (t,) in sql(db_path, "select name from sqlite_master where type='table'")}
    assert {"category_scores", "category_trade_log"} <= tables


async def test_seeded_rows_store_the_computed_score_and_blocked_flag(scorer):
    by = {r["category"]: r for r in await scorer.get_all_scores()}
    assert by["NCAAB"]["score"] == 62.8 and by["NCAAB"]["blocked"] == 0
    assert by["FED"]["score"] == 35.0 and by["FED"]["blocked"] == 0
    for cat in ("ECON", "CPI", "ECON_MACRO"):
        assert by[cat]["blocked"] == 1 and by[cat]["score"] < BLOCK_THRESHOLD


async def test_seeded_tier_outcomes(scorer):
    assert await scorer.get_max_allocation_pct("NCAAB") == 0.10
    assert await scorer.get_max_allocation_pct("FED") == 0.02
    for cat in ("ECON", "CPI", "ECON_MACRO"):
        assert await scorer.is_blocked(cat) is True
        assert await scorer.get_max_allocation_pct(cat) == 0.0


async def test_initialize_twice_does_not_reseed_or_duplicate(scorer, db_path):
    await scorer.update_score("NCAAB", True, 0.1)
    before = sql(db_path, "select win_count,total_count from category_scores where category='NCAAB'")
    await scorer.initialize()
    assert sql(db_path, "select win_count,total_count from category_scores where category='NCAAB'") == before
    assert sql(db_path, "select count(*) from category_scores") == [(5,)]


async def test_get_all_scores_sorted_descending(scorer):
    scores = [r["score"] for r in await scorer.get_all_scores()]
    assert scores == sorted(scores, reverse=True)


async def test_unknown_category_is_score_zero_blocked_zero_allocation(scorer):
    assert await scorer.get_score("NO_SUCH") == 0.0
    assert await scorer.is_blocked("NO_SUCH") is True
    assert await scorer.get_max_allocation_pct("NO_SUCH") == 0.0


async def test_empty_history_category_stays_blocked_until_five_trades(scorer):
    for i in range(cs.MIN_SAMPLES_FOR_SCORING - 1):
        assert await scorer.update_score("NEWCAT", True, 0.5) == 0.0
        assert await scorer.is_blocked("NEWCAT") is True
    fifth = await scorer.update_score("NEWCAT", True, 0.5)
    assert fifth == 85.4                       # see test_best_case_five_trade_category_scores_85_4
    assert await scorer.get_score("NEWCAT") == 85.4
    assert await scorer.is_blocked("NEWCAT") is False
    assert await scorer.get_max_allocation_pct("NEWCAT") == 0.20


async def test_five_losses_stay_blocked(scorer):
    for _ in range(5):
        s = await scorer.update_score("LOSER", False, -0.5)
    # roi (.5/1.5)=.3333*100*.4=13.33 ; sample 5.41 ; trend 0 ; wr 0  -> 18.7
    assert s == 18.7 and await scorer.is_blocked("LOSER") is True


async def test_update_score_accumulates_counts_pnl_and_logs_each_trade(scorer, db_path):
    await scorer.update_score("X", True, 0.30)
    await scorer.update_score("X", False, -0.10)
    row = sql(db_path, "select win_count,total_count,total_pnl from category_scores where category='X'")[0]
    assert row[0] == 1 and row[1] == 2 and row[2] == pytest.approx(0.20)
    assert sql(db_path, "select won,roi from category_trade_log where category='X' order by id") == [(1, 0.3), (0, -0.1)]


async def test_update_score_on_seeded_category_adds_to_the_seed(scorer):
    d = KNOWN_DATA["NCAAB"]
    await scorer.update_score("NCAAB", True, 0.2)
    row = next(r for r in await scorer.get_all_scores() if r["category"] == "NCAAB")
    assert (row["win_count"], row["total_count"]) == (d["wins"] + 1, d["total"] + 1)
    assert row["total_pnl"] == pytest.approx(d["total_pnl"] + 0.2)
    assert not is_seeded_row(row)


async def test_recent_trend_is_recency_weighted(scorer, db_path):
    # newest first: loss weight 1, win weight 1/2 -> (-1 + .5) / 1.5 = -1/3
    await scorer.update_score("T", True, 0.1)
    await scorer.update_score("T", False, -0.1)
    row = sql(db_path, "select recent_trend from category_scores where category='T'")[0]
    assert row[0] == pytest.approx(-1 / 3)


async def test_trend_looks_back_only_ten_trades(scorer, db_path):
    for _ in range(10):
        await scorer.update_score("W", False, -0.1)
    for _ in range(10):
        await scorer.update_score("W", True, 0.1)
    assert sql(db_path, "select recent_trend from category_scores where category='W'")[0][0] == pytest.approx(1.0)


async def test_scores_persist_across_instances(scorer, db_path):
    await scorer.update_score("P", True, 0.5)
    other = CategoryScorer(db_path)
    rows = {r["category"]: r for r in await other.get_all_scores()}
    assert rows["P"]["total_count"] == 1
    assert await other.get_score("NCAAB") == 62.8


async def test_update_score_invalidates_own_cache(scorer):
    assert await scorer.get_score("NCAAB") == 62.8
    await scorer.update_score("NCAAB", False, -1.0)
    assert await scorer.get_score("NCAAB") == 47.6


async def test_nan_roi_is_rejected_and_nothing_is_written(scorer, db_path):
    with pytest.raises(sqlite3.IntegrityError):
        await scorer.update_score("N", True, NAN)
    assert sql(db_path, "select count(*) from category_trade_log where category='N'") == [(0,)]
    assert sql(db_path, "select count(*) from category_scores where category='N'") == [(0,)]


# ------------------------------------------------------------ stale and corrupt scores
async def test_cache_serves_a_stale_score_within_the_ttl_then_refreshes(scorer, db_path):
    assert await scorer.get_score("NCAAB") == 62.8
    sql(db_path, "update category_scores set score=0 where category='NCAAB'")   # e.g. another process
    assert await scorer.get_score("NCAAB") == 62.8                              # stale, within 15 min
    scorer._cache_ts["NCAAB"] = datetime.now(timezone.utc) - timedelta(minutes=16)
    assert await scorer.get_score("NCAAB") == 0.0
    assert await scorer.is_blocked("NCAAB") is True


async def test_negative_stored_score_is_blocked(scorer, db_path):
    sql(db_path, "update category_scores set score=-5 where category='FED'")
    assert await CategoryScorer(db_path).is_blocked("FED") is True


async def test_non_numeric_stored_score_raises_rather_than_allowing(scorer, db_path):
    sql(db_path, "update category_scores set score='abc' where category='CPI'")
    with pytest.raises(TypeError):
        await CategoryScorer(db_path).is_blocked("CPI")


@pytest.mark.xfail(strict=True, reason=(
    "src/strategies/category_scorer.py:209 get_score() returns the stored value unvalidated: a "
    "corrupt score of 1e9 on ECON (seeded 21.8, blocked) reads as not-blocked with a 20% "
    "allocation. Scores are 0-100; anything outside should read as blocked."))
async def test_out_of_range_stored_score_cannot_loosen_a_blocked_category(scorer, db_path):
    sql(db_path, "update category_scores set score=1e9 where category='ECON'")
    fresh = CategoryScorer(db_path)
    assert await fresh.is_blocked("ECON") is True


@pytest.mark.xfail(strict=True, reason=(
    "src/strategies/category_scorer.py:302 force_block() sets the `blocked` column, but "
    "is_blocked()/get_score()/get_max_allocation_pct() only read `score`: force_block('NCAAB') on a "
    "62.8-score category leaves is_blocked False and a 10% allocation."))
async def test_force_block_actually_blocks(scorer):
    await scorer.force_block("NCAAB", "manual")
    assert await scorer.is_blocked("NCAAB") is True
    assert await scorer.get_max_allocation_pct("NCAAB") == 0.0


@pytest.mark.xfail(strict=True, reason=(
    "src/strategies/category_scorer.py:286 update_score() recomputes `blocked` from the score, so "
    "the next recorded trade silently clears a manual force_block flag."))
async def test_manual_block_survives_the_next_trade(scorer, db_path):
    await scorer.force_block("NCAAB", "manual")
    await scorer.update_score("NCAAB", False, -0.1)
    assert sql(db_path, "select blocked from category_scores where category='NCAAB'") == [(1,)]


async def test_force_block_creates_a_zero_score_row_for_unknown_category(scorer):
    await scorer.force_block("FRESH")
    assert await scorer.get_score("FRESH") == 0.0
    assert await scorer.is_blocked("FRESH") is True


# ------------------------------------------------------------ table formatting
def _row(cat, wins, total, pnl, score):
    return {"category": cat, "score": score, "win_count": wins, "total_count": total, "total_pnl": pnl}


def test_format_table_zero_trades_shows_na_and_tier(db_path):
    out = CategoryScorer(db_path).format_scores_table([_row("EMPTY", 0, 0, 0.0, 0.0)])
    assert "n/a" in out and "BLOCKED" in out and "0%" in out and "*" not in out


def test_format_table_wr_roi_and_alloc_columns(db_path):
    out = CategoryScorer(db_path).format_scores_table([_row("AB", 3, 4, 1.0, 62.8)])
    line = next(l for l in out.splitlines() if l.strip().startswith("AB"))
    assert "75%" in line and "25.0%" in line and "10%" in line and "GOOD" in line


# ------------------------------------------------------------ safety guards (source scans)
def test_module_has_no_order_or_client_surface():
    text = (REPO / "src/strategies/category_scorer.py").read_text()
    for needle in ("kalshi_client", "place_order", "cancel_order", "KalshiClient", ".env", "private_key"):
        assert needle not in text


def test_nothing_outside_the_module_records_trade_outcomes_or_builds_the_enforcer():
    """Pins that scores are seed-only and unenforced; wiring either in should fail this test and
    force a review of the gaps above."""
    callers, enforcers = [], []
    for root in (REPO / "src", REPO / "cli.py", REPO / "beast_mode_bot.py"):
        files = [root] if root.is_file() else list(root.rglob("*.py"))
        for p in files:
            if p.name in ("category_scorer.py", "portfolio_enforcer.py"):
                continue
            t = p.read_text()
            if ".update_score(" in t:
                callers.append(str(p.relative_to(REPO)))
            if "PortfolioEnforcer(" in t:
                enforcers.append(str(p.relative_to(REPO)))
    assert callers == []
    assert enforcers == []
