"""The live-book fade backtest must be arithmetically honest and count events.

Pins: the fade filter, the fee-charged P&L for buying NO at the captured ask, the
event grouping that stops correlated strikes inflating n, and the refusal to score
an unsettled corpus as if it were zero.
"""
import scripts.backtest_books as bb


def _row(ticker, yes_ask=0.08, no_ask=0.90):
    return {"ticker": ticker, "yes_ask": yes_ask, "no_ask": no_ask, "cat": "x"}


def test_fade_filter_requires_both_thresholds():
    rows = [_row("A"), _row("B", yes_ask=0.40), _row("C", no_ask=0.50)]
    assert [r["ticker"] for r in bb.fade_rows(rows)] == ["A"]


def test_event_key_groups_strikes_of_one_event():
    assert bb.event_key("KXDIESELW-26OCT26-T6.32") == "KXDIESELW-26OCT26"
    assert bb.event_key("CONTROLH-2026-R") == "CONTROLH-2026"
    assert bb.event_key("NOSUFFIX") == "NOSUFFIX"


def test_score_charges_the_fee_and_pays_one_on_a_no_win():
    # buy NO at 0.90, fee = ceil(7*0.9*0.1)/100 = 0.01 -> a win nets 1 - 0.90 - 0.01
    s = bb.score([_row("KXTEST-26JAN-A")], {"KXTEST-26JAN-A": "no"})
    assert s["settled"] == 1 and s["wins"] == 1
    assert abs(s["pnl_per_contract"] - 0.09) < 1e-9
    assert abs(s["edge_vs_paid"] - 0.10) < 1e-9


def test_score_treats_a_yes_resolution_as_a_full_loss():
    s = bb.score([_row("KXTEST-26JAN-A")], {"KXTEST-26JAN-A": "yes"})
    assert s["wins"] == 0
    assert abs(s["pnl_per_contract"] + 0.91) < 1e-9


def test_unsettled_candidates_are_reported_not_scored():
    s = bb.score([_row("A"), _row("B")], {})
    assert s["settled"] == 0 and s["unsettled"] == 2
    assert s["pnl_per_contract"] is None
    assert "too young to score" in bb.render(s)


def test_render_flags_a_thin_event_sample():
    s = bb.score([_row(f"KXTEST-26JAN-{i}") for i in range(5)],
                 {f"KXTEST-26JAN-{i}": "no" for i in range(5)})
    assert s["events"] == 1                     # five strikes, one event
    assert "not enough to claim edge" in bb.render(s)
