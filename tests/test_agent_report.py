"""Tests for the track-record renderer — the honest public face of the loop.

Pure-function tests with in-memory fixtures (no network, no live Kalshi, no IO).
They cover:
  - render_track_record: full render with every section populated; the offline
    mode (no equity snapshot); the empty-history / no-policy fresh-clone case.
  - determinism: same inputs -> identical output (the function never reads the
    clock or the filesystem).
  - honesty invariants: losses render with their sign, the verdict line is
    quoted verbatim, and a missing policy renders a pointer, not a claim.

The inputs mirror the real shapes emitted by ``settle.summarize_settlements``,
``edge.edge_report`` and the persisted Edge Policy so the renderer stays a
thin, honest consumer.
"""
from src.agent.report import render_track_record


# ---------------------------------------------------------------------------
# Fixtures — shaped exactly like the producers' outputs
# ---------------------------------------------------------------------------

def _settle_summary():
    return {
        "n": 146,
        "wins": 90,
        "win_rate": 90 / 146,
        "pnl": -726.18,
        "by_side": {
            "yes": {"n": 63, "wins": 20, "pnl": -684.96},
            "no": {"n": 83, "wins": 70, "pnl": -41.22},
        },
    }


def _edge(verdict="NO EDGE PROVEN YET — keep the track record honest."):
    return {
        "date": "2026-07-02",
        "n_settled": 24,
        "n_forward": 1,
        "n_suspect": 0,
        "n_unknown": 23,
        "brier": {"value": 0.0404, "n": 1},
        "log_loss": {"value": 0.2231, "n": 1},
        "calibration_table": [
            {"bucket": "0.95-1.00", "lo": 0.95, "hi": 1.0,
             "n": 1, "predicted": 0.98, "actual": 1.0},
        ],
        "edge_vs_book": {"overall": None, "by_category": {}, "by_side": {}},
        "verdict": verdict,
    }


def _policy():
    return {
        "meta": {"version": 1, "generated_date": "2026-07-02", "settled_n": 146},
        "blocks": [
            {"dimension": "category", "label": "KXCPI", "n": 6, "pnl": -66.57},
        ],
        "warnings": [],
        "haircuts": [
            {"band": "0.80-0.90", "lo": 0.8, "hi": 0.9,
             "shrink_to": 0.62, "n": 8, "gap": 0.23},
        ],
    }


def _equity():
    return {
        "equity": 1822.77,
        "cash": 1196.87,
        "portfolio": 625.90,
        "positions": [{"ticker": "KXALIENS-27"}, {"ticker": "OAIAGI-26"}],
        "governor": {
            "halted": False,
            "drawdown_pct": 0.0,
            "daily_pnl_cents": 937,
        },
    }


def _render(**overrides):
    kwargs = dict(
        date="2026-07-02",
        equity=_equity(),
        settle_summary=_settle_summary(),
        edge=_edge(),
        policy=_policy(),
    )
    kwargs.update(overrides)
    return render_track_record(**kwargs)


# ---------------------------------------------------------------------------
# Full render
# ---------------------------------------------------------------------------

def test_full_render_contains_every_section():
    md = _render()
    for heading in (
        "# Live Track Record",
        "## Account",
        "## The honest edge verdict (forward-only)",
        "## Full account settlement history",
        "## Calibration (predicted vs. actual)",
        "## Edge Policy — what the record currently blocks",
    ):
        assert heading in md


def test_losses_render_with_their_sign():
    md = _render()
    assert "-$726.18" in md          # net account P&L, not hidden
    assert "-$684.96" in md          # the YES-side disaster, published
    assert "-$66.57" in md           # the blocked category's loss


def test_verdict_is_quoted_verbatim():
    verdict = "NO EDGE PROVEN YET — keep the track record honest."
    assert verdict in _render(edge=_edge(verdict))


def test_account_snapshot_and_governor_render():
    md = _render()
    assert "$1,822.77" in md
    assert "$1,196.87" in md
    assert "| Governor halted | no |" in md


def test_rebased_governor_still_publishes_all_time_drawdown():
    # After the governor peak is re-baselined, its drawdown restarts near 0 —
    # the public page must still show the drawdown from the all-time peak.
    eq = _equity()
    eq["rebase"] = {"date": "2026-09-25", "prior_peak_cents": 297525}
    md = _render(equity=eq)
    assert "| Drawdown from all-time peak ($2,975.25) | 38.7% |" in md
    assert "| Drawdown since governor re-baseline (2026-09-25) | 0.0% |" in md
    assert "| Drawdown from peak |" not in md


def test_account_section_flags_blended_operator_plus_strategy():
    # The account equity blends the operator's manual trades with the strategy;
    # the page must say so, so a manual-driven drawdown is never misread as the
    # strategy failing. The strategy's real edge lives in the verdict section.
    md = _render()
    assert "blended" in md.lower()
    assert "manually" in md.lower()


def test_policy_block_and_haircut_render():
    md = _render()
    assert "category=KXCPI" in md
    assert "shrink to 62%" in md


# ---------------------------------------------------------------------------
# Degraded modes — offline render and fresh clone
# ---------------------------------------------------------------------------

def test_offline_render_notes_missing_snapshot():
    md = _render(equity=None)
    assert "Live account snapshot unavailable" in md
    assert "## The honest edge verdict (forward-only)" in md  # rest still renders


def test_missing_policy_points_at_improve():
    md = _render(policy=None)
    assert "python cli.py improve" in md


def test_empty_calibration_renders_placeholder():
    edge = _edge()
    edge["calibration_table"] = []
    md = _render(edge=edge)
    assert "No forward-settled journaled trades" in md


def test_empty_history_renders_without_error():
    md = _render(
        equity=None,
        settle_summary={"n": 0, "wins": 0, "win_rate": None,
                        "pnl": 0.0, "by_side": {}},
        edge={"date": "2026-07-02", "n_settled": 0, "n_forward": 0,
              "n_suspect": 0, "n_unknown": 0, "brier": None,
              "log_loss": None, "calibration_table": [],
              "edge_vs_book": {}, "verdict": "no data"},
        policy=None,
    )
    assert "| Settled markets | 0 |" in md
    assert "n/a" in md  # win rate and metrics degrade to n/a, never crash


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_header_keeps_reproducibility_note():
    md = _render()
    assert (
        "These figures are the owner's live account, computed from local "
        "settlements. Other people cannot reproduce them without the account."
    ) in md


def test_render_is_deterministic():
    assert _render() == _render()


def _account_md(**snapshot):
    equity = {"equity": 12.5, "cash": 10.0, "positions": [{"ticker": "A"}, {"ticker": "B"}], **snapshot}
    return render_track_record(
        date="2026-07-02", equity=equity, settle_summary=_settle_summary(), edge=_edge(), policy=None)


def test_a_clean_snapshot_gives_exact_figures():
    for snapshot in ({}, {"read_errors": []}, {"read_errors": None}):
        md = _account_md(**snapshot)
        assert "| Equity | $12.50 |" in md and "| Cash | $10.00 |" in md
        assert "| Open positions | 2 |" in md


def test_a_partial_position_read_never_publishes_a_plain_count():
    for error in ("position SECRET-TICKER", "positions: page cap", "positions: no list", "positions: bad row"):
        md = _account_md(read_errors=[error])
        assert "| Open positions | at least 2 (read incomplete) |" in md
        assert "| Open positions | 2 |" not in md


def test_read_errors_never_reach_the_public_page():
    md = _account_md(read_errors=["position SECRET-TICKER", "balance", "resting_orders"])
    assert "SECRET-TICKER" not in md and "resting_orders" not in md


def test_a_resting_order_error_leaves_the_position_count_alone():
    md = _account_md(read_errors=["resting_orders", "resting_orders: first page only"])
    assert "| Open positions | 2 |" in md and "| Equity | $12.50 |" in md


def test_a_failed_balance_read_publishes_no_figure():
    md = _account_md(equity=0.0, cash=0.0, positions=[], read_errors=["balance"])
    assert "| Equity | unavailable (read failed) |" in md
    assert "| Cash | unavailable (read failed) |" in md
    assert "$0.00" not in md.split("## The honest edge verdict")[0]
    assert "| Open positions | 0 |" in md


def test_a_failed_balance_read_publishes_no_all_time_drawdown():
    rebase = {"date": "2026-09-25", "prior_peak_cents": 100000}
    bad = _account_md(equity=0.0, cash=0.0, read_errors=["balance"], rebase=rebase)
    assert "Drawdown from all-time peak" not in bad
    good = _account_md(equity=500.0, rebase=rebase)
    assert "| Drawdown from all-time peak ($1,000.00) | 50.0% |" in good


def test_a_failed_balance_read_publishes_no_governor_figures_but_still_shows_the_halt():
    gov = {"halted": True, "drawdown_pct": 100.0, "daily_pnl_cents": -50000}
    rebase = {"date": "2026-09-25", "prior_peak_cents": 100000}
    bad = _account_md(equity=0.0, cash=0.0, governor=gov, rebase=rebase, read_errors=["balance"])
    assert "| Governor halted | YES |" in bad
    assert "100.0%" not in bad and "Day P&L" not in bad and "Drawdown since" not in bad
    assert "all-time drawdown includes" not in bad
    good = _account_md(equity=500.0, governor=gov, rebase=rebase)
    assert "| Drawdown since governor re-baseline (2026-09-25) | 100.0% |" in good
    assert "| Day P&L | -$500.00 |" in good
    assert "all-time drawdown includes" in good
