"""Pins the headline of the offline AAA alert-filter fee backtest.

Reads only committed files (data/aaa/alerts, data/aaa/backtest_outcomes.csv).
If an alert or outcome file changes, these numbers move on purpose: re-run
`python scripts/fee_backtest_aaa.py` and re-pin.
"""
import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fee_backtest_aaa as fb  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return fb.run()


@pytest.fixture(scope="module")
def trades():
    return fb.build_trades()[0]


def test_counts_add_up(res):
    assert res["n_raw_alert_rows"] == res["n_unique_tickers"] + res["n_duplicate_rows_dropped"]
    assert res["n_unique_tickers"] == res["n_scored"] + len(res["missing_outcome_tickers"])
    assert res["n_raw_alert_rows"] == 453
    assert res["n_duplicate_rows_dropped"] == 15
    assert res["n_scored"] == 420
    assert len(res["missing_outcome_tickers"]) == 18
    # NC and VA have no archived 10-03 print; nothing else is missing
    assert {t.split("-")[0] for t in res["missing_outcome_tickers"]} == {
        "KXAAAGASDNC", "KXAAAGASDVA"}


def test_headline_counts(res):
    h = res["headline"]
    assert (h["n_rows"], h["n_events"], h["n_print_days"]) == (420, 52, 4)


def test_headline_means(res):
    h = res["headline"]
    assert round(h["mean_gross"], 4) == -0.2237
    assert round(h["mean_maker"], 4) == -0.2237
    assert round(h["win_rate"], 4) == 0.0738
    assert round(h["mean_claimed_edge"], 4) == 0.4916
    assert {c: round(v, 4) for c, v in h["mean_net"]["per_contract"].items()} == {
        1: -0.2386, 10: -0.2386, 100: -0.2386}
    assert {c: round(v, 4) for c, v in h["mean_net"]["per_order"].items()} == {
        1: -0.2386, 10: -0.2329, 100: -0.2325}


def test_event_bootstrap_ci(res):
    ci = res["headline"]["ci_event"]
    got = {k: (round(lo, 4), round(hi, 4)) for k, (lo, hi) in ci.items()}
    assert got == {
        "gross": (-0.2919, -0.1608),
        "net_per_contract_1": (-0.3073, -0.1752),
        "net_per_contract_10": (-0.3073, -0.1752),
        "net_per_contract_100": (-0.3073, -0.1752),
        "net_per_order_1": (-0.3073, -0.1752),
        "net_per_order_10": (-0.3018, -0.1695),
        "net_per_order_100": (-0.3013, -0.1690),
    }


def test_naive_row_ci_is_narrower_than_cluster_cis(res):
    h = res["headline"]
    naive = h["ci_row_naive_gross"]
    for ci in (h["ci_event"]["gross"], h["ci_day"]["gross"]):
        assert naive[1] - naive[0] < ci[1] - ci[0]


def test_near_strike_count(res):
    assert res["n_near_strike"] == 12
    assert len(res["near_strike_tickers"]) == 12
    assert res["headline"]["n_rows"] - res["excluding_near_strike"]["n_rows"] == 12


def test_verdict_is_computed_from_the_numbers(res):
    assert res["headline"]["verdict"].startswith("LOSES BEFORE FEES")
    assert "over 4 print days" in res["headline"]["verdict"]
    assert fb.verdict((-0.1, 0.1), 0.0, (-0.1, 0.1), 3).startswith("TOO SMALL TO CONCLUDE")
    assert fb.verdict((0.01, 0.1), 0.05, (-0.01, 0.1), 3).startswith("WINS BEFORE FEES")
    assert fb.verdict((0.01, 0.1), 0.05, (0.01, 0.1), 3).startswith("WINS AFTER FEES")


def test_count_one_fee_forms_agree(trades):
    for t in trades:
        assert t["net_per_contract_1"] == pytest.approx(t["net_per_order_1"], abs=1e-12)


def test_count_hundred_per_order_not_worse_than_per_contract(trades):
    for t in trades:
        assert t["net_per_order_100"] >= t["net_per_contract_100"] - 1e-12
    n = len(trades)
    assert (sum(t["net_per_order_100"] for t in trades) / n
            >= sum(t["net_per_contract_100"] for t in trades) / n)


def test_trade_construction():
    row = {"ticker": "X-26SEP27-4.00", "series": "KXAAAGASD", "target_date": "2026-09-27",
           "strike": 4.0, "fair_yes": 0.7, "yes_bid": 0.1, "yes_ask": 0.08999999,
           "buy_yes_edge": 0.5, "buy_no_edge": -0.6}
    yes = fb.make_trade(row, 4.2)
    assert (yes["side"], yes["price"], yes["payoff"]) == ("YES", 0.09, 1.0)
    assert yes["gross"] == pytest.approx(0.91)
    assert yes["claimed"] == pytest.approx(0.61)
    row.update(buy_yes_edge=-0.6, buy_no_edge=0.5)
    no = fb.make_trade(row, 4.2)  # NO pays 1 - yes_bid = 0.90, loses when print > strike
    assert (no["side"], no["price"], no["payoff"]) == ("NO", 0.9, 0.0)
    assert fb.ticker_date("KXDIESELD-26SEP29-T6.440") == "2026-09-29"


def test_outcomes_fixture_shape():
    with open(fb.OUTCOMES_CSV) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["series", "target_date", "print", "source",
                             "snapshot_url", "snapshot_ts"]
    keys = [(r["series"], r["target_date"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(r["source"] in {"committed_csv", "wayback", "wayback_noxcheck"} for r in rows)
    assert all(r["snapshot_url"] for r in rows)
