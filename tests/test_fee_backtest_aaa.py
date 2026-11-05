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
    assert res["n_unique_tickers"] == (res["n_scored"] + len(res["missing_outcome_tickers"])
                                       + res["n_excluded_lookahead"])
    assert res["n_excluded_lookahead"] == 0  # every alert preceded its print by >2h
    assert res["n_label_mismatch_tickers"] == 95
    assert res["n_raw_alert_rows"] == 453
    assert res["n_duplicate_rows_dropped"] == 15
    assert res["n_scored"] == 420
    assert len(res["missing_outcome_tickers"]) == 18
    # NC and VA have no archived 10-03 print; nothing else is missing
    assert {t.split("-")[0] for t in res["missing_outcome_tickers"]} == {
        "KXAAAGASDNC", "KXAAAGASDVA"}


def test_headline_counts(res):
    h = res["headline"]
    assert (h["n_rows"], h["n_events"], h["n_print_days"]) == (420, 62, 5)


def test_headline_means(res):
    h = res["headline"]
    assert round(h["mean_gross"], 4) == -0.1952
    assert round(h["mean_maker"], 4) == -0.1952
    assert round(h["win_rate"], 4) == 0.1024
    assert round(h["mean_claimed_edge"], 4) == 0.4916
    assert {c: round(v, 4) for c, v in h["mean_net"]["per_contract"].items()} == {
        1: -0.2100, 10: -0.2100, 100: -0.2100}
    assert {c: round(v, 4) for c, v in h["mean_net"]["per_order"].items()} == {
        1: -0.2100, 10: -0.2044, 100: -0.2039}


def test_event_bootstrap_ci(res):
    ci = res["headline"]["ci_event"]
    got = {k: (round(lo, 4), round(hi, 4)) for k, (lo, hi) in ci.items()}
    assert got == {
        "gross": (-0.2657, -0.1324),
        "net_per_contract_1": (-0.2809, -0.1469),
        "net_per_contract_10": (-0.2809, -0.1469),
        "net_per_contract_100": (-0.2809, -0.1469),
        "net_per_order_1": (-0.2809, -0.1469),
        "net_per_order_10": (-0.2754, -0.1410),
        "net_per_order_100": (-0.2749, -0.1406),
    }


def test_naive_row_ci_is_narrower_than_cluster_cis(res):
    h = res["headline"]
    naive = h["ci_row_naive_gross"]
    for ci in (h["ci_event"]["gross"], h["ci_day"]["gross"]):
        assert naive[1] - naive[0] < ci[1] - ci[0]


def test_near_strike_count(res):
    assert res["n_near_strike"] == 17
    assert len(res["near_strike_tickers"]) == 17
    assert res["headline"]["n_rows"] - res["excluding_near_strike"]["n_rows"] == 17


def test_verdict_is_computed_from_the_numbers(res):
    assert res["headline"]["verdict"].startswith("LOSES BEFORE FEES")
    assert "over 5 print days" in res["headline"]["verdict"]
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
           "settle_date": "2026-09-27", "strike": 4.0, "fair_yes": 0.7, "yes_bid": 0.1, "yes_ask": 0.08999999,
           "buy_yes_edge": 0.5, "buy_no_edge": -0.6}
    yes = fb.make_trade(row, 4.2)
    assert (yes["side"], yes["price"], yes["payoff"]) == ("YES", 0.09, 1.0)
    assert yes["gross"] == pytest.approx(0.91)
    assert yes["claimed"] == pytest.approx(0.61)
    row.update(buy_yes_edge=-0.6, buy_no_edge=0.5)
    no = fb.make_trade(row, 4.2)  # NO pays 1 - yes_bid = 0.90, loses when print > strike
    assert (no["side"], no["price"], no["payoff"]) == ("NO", 0.9, 0.0)
    assert fb.ticker_date("KXDIESELD-26SEP29-T6.440") == "2026-09-29"


def _synthetic(ticker, alert_ts, target_date="2026-09-28"):
    return {"ticker": ticker, "series": "KXAAAGASD", "target_date": target_date,
            "settle_date": fb.ticker_date(ticker), "alert_ts": alert_ts, "strike": 4.0}


def test_lookahead_guard_excludes_rows_alerted_after_publication():
    # 26SEP27 print assumed public 2026-09-27 00:00 New York = 04:00 UTC (EDT)
    before = _synthetic("KXAAAGASD-26SEP27-4.00", "2026-09-27T03:59:59+00:00")
    at = _synthetic("KXAAAGASD-26SEP27-4.05", "2026-09-27T04:00:00+00:00")
    after = _synthetic("KXAAAGASD-26SEP27-4.10", "2026-09-27T15:00:00+00:00")
    assert not fb.is_lookahead(before)
    assert fb.is_lookahead(at)
    assert fb.is_lookahead(after)


def test_lookahead_guard_in_build_trades(tmp_path):
    import json
    def snap(ts, ticker):
        return {"ts": ts, "series": "KXAAAGASD", "target_date": "2026-09-28", "threshold": 0.1,
                "rows": [{"ticker": ticker, "strike": 4.0, "fair_yes": 0.7, "yes_bid": 0.1,
                          "yes_ask": 0.2, "buy_yes_edge": 0.4, "buy_no_edge": -0.5}]}
    (tmp_path / "a.json").write_text(json.dumps(snap("2026-09-27T01:00:00+00:00", "KXAAAGASD-26SEP27-4.00")))
    (tmp_path / "b.json").write_text(json.dumps(snap("2026-09-27T09:00:00+00:00", "KXAAAGASD-26SEP27-4.05")))
    outcomes = {("KXAAAGASD", "2026-09-27"): 4.2}  # keyed by ticker date, not target_date
    trades, missing, look, raw, uniq, mismatch = fb.build_trades(outcomes, tmp_path)
    assert [t["ticker"] for t in trades] == ["KXAAAGASD-26SEP27-4.00"]
    assert look == ["KXAAAGASD-26SEP27-4.05"]
    assert missing == [] and (raw, uniq, mismatch) == (2, 2, 2)
    assert trades[0]["settle_date"] == "2026-09-27"


def test_scoring_ignores_alert_target_date():
    row = _synthetic("KXAAAGASD-26SEP27-4.00", "2026-09-27T01:00:00+00:00", "2026-09-28")
    row.update(fair_yes=0.7, yes_bid=0.1, yes_ask=0.2, buy_yes_edge=0.4, buy_no_edge=-0.5)
    t = fb.make_trade(row, 4.2)
    assert t["day"] == "2026-09-27" and t["event"] == "KXAAAGASD|2026-09-27"


def test_outcomes_fixture_shape():
    with open(fb.OUTCOMES_CSV) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["series", "target_date", "print", "source",
                             "snapshot_url", "snapshot_ts"]
    keys = [(r["series"], r["target_date"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(r["source"] in {"committed_csv", "wayback", "wayback_noxcheck"} for r in rows)
    assert all(r["snapshot_url"] for r in rows)
