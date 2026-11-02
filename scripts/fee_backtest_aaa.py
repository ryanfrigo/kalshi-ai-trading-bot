"""Offline, fee-aware backtest of the AAA pricer's alert filter.

Reads only committed files: data/aaa/alerts/*.json (what the pricer flagged)
and data/aaa/backtest_outcomes.csv (realized prints; built by
scripts/build_backtest_outcomes.py). No network, no orders, no credentials.

Method
------
* Dedup by ticker: the earliest alert (by `ts`) for a ticker wins.
* Settlement is by the date in the ticker (26SEP27 -> 2026-09-27) and nothing
  else. The alert's own target_date is ignored for scoring: the pricer labels a
  whole open ladder with one target_date, so it can differ from the ticker. The
  number of tickers where they differ is printed for information only. YES iff
  print > strike; the outcomes CSV column named target_date holds this
  settlement date.
* Look-ahead guard. Assumption: AAA publishes the print for day D at D 00:00
  America/New_York. A row whose first-alert ts is at or after that moment for
  its ticker date was decided when the print was known or the market was
  closed, so it is excluded from every number and listed as EXCLUDED_LOOKAHEAD.
* Every remaining row is a trade; nothing is cherry-picked. Side is the larger
  of buy_yes_edge / buy_no_edge (a tie goes to YES).
* Price paid, rounded to the cent: YES = yes_ask, NO = 1 - yes_bid. Payoff is 1
  if the side wins, else 0. Gross P&L per contract = payoff - price.
* Net P&L per contract = gross - fee/count under two fee forms, at order sizes
  of 1, 10 and 100 contracts: `per_contract` (count * taker_fee_per_contract,
  so the fee per contract does not depend on size) and `per_order`
  (taker_fee_order(count, price), rounded up once per order). `maker` is the
  zero-fee reference and equals gross. Both fee functions are imported from
  aaa_pricer.
* Model edge claimed at alert time = the model's probability for the side
  minus the price paid; the alert's own fee-aware edge field is shown beside it.
* 95% bootstrap CIs on each mean, 10000 resamples, seed 0, resampling whole
  events (series, settlement date) and, separately, whole print days (settlement
  date; every series on one day shares the same market move). The per-row CI
  treats trades as independent and overstates precision; it is shown only to be
  contrasted with the cluster CIs.
* Rows with |print - strike| < 0.001 sit within rounding of the strike; they
  are flagged and the headline is repeated without them.
* The verdict line is computed from the print-day gross CI, the widest and
  therefore most honest of the three. Whatever sign comes out is printed.
* Sanity asserts: raw rows = unique tickers + duplicates, and unique tickers =
  scored + excluded-lookahead + missing-outcome.

Known issue elsewhere: aaa_data.backfill_state fetches the national Wayback
URL for state timestamps, so the `wayback` rows in data/aaa/states/*.csv are
national values. Every state outcome used here comes from a `live` row or from
build_backtest_outcomes.py, which builds the state URL itself.

Usage:  python scripts/fee_backtest_aaa.py [--json]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import datetime as dt
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aaa_pricer import MONTHS, taker_fee_order, taker_fee_per_contract  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ALERTS_DIR = REPO / "data" / "aaa" / "alerts"
OUTCOMES_CSV = REPO / "data" / "aaa" / "backtest_outcomes.csv"
COUNTS = (1, 10, 100)
FEE_FORMS = ("per_contract", "per_order")
N_BOOT = 10000
SEED = 0
NEAR_STRIKE = 0.001
PUBLISH_TZ = "America/New_York"  # AAA's day-D print is assumed public at D 00:00 here


# ---------------------------------------------------------------- loading

def load_outcomes(path: Path = OUTCOMES_CSV) -> dict[tuple[str, str], float]:
    with open(path) as f:
        return {(r["series"], r["target_date"]): float(r["print"]) for r in csv.DictReader(f)}


def load_alert_rows(alerts_dir: Path = ALERTS_DIR):
    """-> (raw_row_count, unique rows). Earliest alert (by ts) wins per ticker."""
    snaps = [json.loads(Path(p).read_text()) for p in glob.glob(str(alerts_dir / "*.json"))]
    snaps.sort(key=lambda s: s["ts"])
    seen: dict[str, dict] = {}
    raw = 0
    for s in snaps:
        for r in s["rows"]:
            raw += 1
            if r["ticker"] not in seen:
                seen[r["ticker"]] = {**r, "series": s["series"],
                                     "target_date": s["target_date"], "alert_ts": s["ts"],
                                     "settle_date": ticker_date(r["ticker"])}
    return raw, list(seen.values())


def ticker_date(ticker: str) -> str | None:
    """'KXAAAGASD-26SEP27-4.38' -> '2026-09-27' (the event date in the ticker)."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(?:-|$)", ticker)
    if not m or m.group(2) not in MONTHS:
        return None
    return f"20{m.group(1)}-{MONTHS[m.group(2)]:02d}-{m.group(3)}"


def make_trade(row: dict, print_value: float) -> dict:
    yes = row["buy_yes_edge"] >= row["buy_no_edge"]
    if yes:
        side, price = "YES", round(row["yes_ask"], 2)
        model_p = row["fair_yes"]
        wins = print_value > row["strike"]
    else:
        side, price = "NO", round(1 - row["yes_bid"], 2)
        model_p = 1 - row["fair_yes"]
        wins = not (print_value > row["strike"])
    payoff = 1.0 if wins else 0.0
    t = {
        "ticker": row["ticker"], "series": row["series"], "target_date": row["target_date"],
        "settle_date": row["settle_date"],
        "event": f'{row["series"]}|{row["settle_date"]}', "day": row["settle_date"],
        "strike": row["strike"], "print": print_value, "side": side, "price": price,
        "payoff": payoff, "gross": payoff - price,
        "claimed": model_p - price,
        "claimed_fee_aware": max(row["buy_yes_edge"], row["buy_no_edge"]),
        "near_strike": round(abs(print_value - row["strike"]), 6) < NEAR_STRIKE,
    }
    fee1 = taker_fee_per_contract(price)
    for c in COUNTS:
        t[f"net_per_contract_{c}"] = t["gross"] - fee1
        t[f"net_per_order_{c}"] = t["gross"] - taker_fee_order(c, price) / c
    return t


def published_at(day: str) -> dt.datetime:
    """Assumed moment AAA's print for `day` is public: day 00:00 New York."""
    return dt.datetime.combine(dt.date.fromisoformat(day), dt.time(0), tzinfo=ZoneInfo(PUBLISH_TZ))


def is_lookahead(row: dict) -> bool:
    return dt.datetime.fromisoformat(row["alert_ts"]) >= published_at(row["settle_date"])


def build_trades(outcomes: dict | None = None, alerts_dir: Path = ALERTS_DIR):
    """-> (trades, missing_tickers, excluded_lookahead_tickers, raw_count, unique_count,
    n_label_mismatch)."""
    outcomes = load_outcomes() if outcomes is None else outcomes
    raw, rows = load_alert_rows(alerts_dir)
    trades, missing, lookahead = [], [], []
    for r in rows:
        if r["settle_date"] is None:
            raise ValueError(f"no date in ticker {r['ticker']}")
        if is_lookahead(r):
            lookahead.append(r["ticker"])
            continue
        p = outcomes.get((r["series"], r["settle_date"]))
        if p is None:
            missing.append(r["ticker"])
        else:
            trades.append(make_trade(r, p))
    assert len(trades) + len(missing) + len(lookahead) == len(rows), "rows unaccounted for"
    assert raw >= len(rows)
    mismatch = sum(1 for r in rows if r["settle_date"] != r["target_date"])
    return trades, missing, lookahead, raw, len(rows), mismatch


# ------------------------------------------------------------- statistics

def metric_names() -> list[str]:
    return ["gross"] + [f"net_{f}_{c}" for f in FEE_FORMS for c in COUNTS]


def _mean(xs):
    return sum(xs) / len(xs)


def bootstrap_ci(trades: list[dict], cluster_key: str | None, metrics: list[str],
                 n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, tuple[float, float]]:
    """95% percentile CI on the mean of each metric, resampling whole clusters
    (cluster_key=None resamples single rows)."""
    groups: dict = defaultdict(list)
    for i, t in enumerate(trades):
        groups[t[cluster_key] if cluster_key else i].append(t)
    keys = list(groups)
    sums = {m: [sum(t[m] for t in groups[k]) for k in keys] for m in metrics}
    ns = [len(groups[k]) for k in keys]
    rng = random.Random(seed)
    k = len(keys)
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.choices(range(k), k=k)
        n = sum(ns[i] for i in idx)
        for m in metrics:
            s = sums[m]
            draws[m].append(sum(s[i] for i in idx) / n)
    out = {}
    for m in metrics:
        d = sorted(draws[m])
        out[m] = (d[round(0.025 * (n_boot - 1))], d[round(0.975 * (n_boot - 1))])
    return out


def verdict(ci_gross, mean_gross, ci_net_worst, n_days) -> str:
    lo, hi = ci_gross
    if hi < 0:
        v = "LOSES BEFORE FEES"
    elif lo > 0:
        v = ("WINS AFTER FEES" if ci_net_worst[0] > 0
             else "WINS BEFORE FEES, NOT PROVEN AFTER FEES")
    else:
        v = "TOO SMALL TO CONCLUDE"
    return (f"{v} (gross mean {mean_gross:+.4f}/contract, 95% CI [{lo:+.4f}, {hi:+.4f}] "
            f"over {n_days} print days, day-cluster bootstrap)")


def analyze(trades: list[dict]) -> dict:
    ms = metric_names()
    out = {
        "n_rows": len(trades),
        "n_events": len({t["event"] for t in trades}),
        "n_print_days": len({t["day"] for t in trades}),
        "mean_price_paid": _mean([t["price"] for t in trades]),
        "win_rate": _mean([t["payoff"] for t in trades]),
        "mean_claimed_edge": _mean([t["claimed"] for t in trades]),
        "mean_claimed_edge_fee_aware": _mean([t["claimed_fee_aware"] for t in trades]),
        "mean_gross": _mean([t["gross"] for t in trades]),
        "mean_net": {f: {c: _mean([t[f"net_{f}_{c}"] for t in trades]) for c in COUNTS}
                     for f in FEE_FORMS},
        "mean_maker": _mean([t["gross"] for t in trades]),
    }
    ev = bootstrap_ci(trades, "event", ms)
    dy = bootstrap_ci(trades, "day", ms)
    rw = bootstrap_ci(trades, None, ["gross"])
    out["ci_event"] = ev
    out["ci_day"] = dy
    out["ci_row_naive_gross"] = rw["gross"]
    worst_net = dy["net_per_contract_1"]
    out["verdict"] = verdict(dy["gross"], out["mean_gross"], worst_net, out["n_print_days"])
    return out


def run(outcomes: dict | None = None, alerts_dir: Path = ALERTS_DIR) -> dict:
    trades, missing, lookahead, raw, uniq, mismatch = build_trades(outcomes, alerts_dir)
    near = [t for t in trades if t["near_strike"]]
    far = [t for t in trades if not t["near_strike"]]
    return {
        "bootstrap": {"resamples": N_BOOT, "seed": SEED, "ci": 0.95},
        "publish_assumption": f"day-D print public at D 00:00 {PUBLISH_TZ}",
        "n_raw_alert_rows": raw,
        "n_duplicate_rows_dropped": raw - uniq,
        "n_unique_tickers": uniq,
        "n_label_mismatch_tickers": mismatch,
        "n_excluded_lookahead": len(lookahead),
        "excluded_lookahead_tickers": lookahead,
        "n_scored": len(trades),
        "missing_outcome_tickers": missing,
        "near_strike_threshold": NEAR_STRIKE,
        "n_near_strike": len(near),
        "near_strike_tickers": [t["ticker"] for t in near],
        "headline": analyze(trades),
        "excluding_near_strike": analyze(far),
    }


# ----------------------------------------------------------------- output

def _fmt_ci(ci) -> str:
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def _print_block(title: str, a: dict) -> None:
    print(f"== {title} ==")
    print(f"rows {a['n_rows']}  events {a['n_events']}  print days {a['n_print_days']}")
    print(f"mean price paid {a['mean_price_paid']:.4f}   win rate {a['win_rate']:.4f}")
    print(f"mean model edge claimed (model p - price) {a['mean_claimed_edge']:+.4f}   "
          f"(alert fee-aware edge field {a['mean_claimed_edge_fee_aware']:+.4f})   "
          f"realized gross {a['mean_gross']:+.4f}")
    print("mean P&L per contract, with 95% CI (event | print-day clusters):")
    rows = [("gross (= maker, zero fee)", "gross", a["mean_gross"])]
    for f in FEE_FORMS:
        for c in COUNTS:
            rows.append((f"net {f} count={c}", f"net_{f}_{c}", a["mean_net"][f][c]))
    for label, key, mean in rows:
        print(f"  {label:<28} {mean:+.4f}   event {_fmt_ci(a['ci_event'][key])}   "
              f"day {_fmt_ci(a['ci_day'][key])}")
    print(f"  per-row naive CI on gross {_fmt_ci(a['ci_row_naive_gross'])}  "
          "(treats rows as independent; OVERSTATES precision)")
    print(f"VERDICT: {a['verdict']}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)
    res = run()
    if args.json:
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    b = res["bootstrap"]
    print(f"AAA alert-filter backtest (offline). bootstrap: {b['resamples']} resamples, "
          f"seed {b['seed']}, {int(b['ci'] * 100)}% percentile CI")
    print(f"settlement: by the date in the ticker only. look-ahead guard: {res['publish_assumption']}")
    print(f"raw alert rows {res['n_raw_alert_rows']}, duplicates dropped "
          f"{res['n_duplicate_rows_dropped']}, unique tickers {res['n_unique_tickers']}, "
          f"excluded lookahead {res['n_excluded_lookahead']}, "
          f"missing outcome {len(res['missing_outcome_tickers'])}, scored {res['n_scored']}")
    print(f"unique tickers whose ticker date != alert target_date (info only): "
          f"{res['n_label_mismatch_tickers']}")
    print(f"EXCLUDED_LOOKAHEAD: {res['n_excluded_lookahead']}")
    for t in res["excluded_lookahead_tickers"]:
        print(f"  EXCLUDED_LOOKAHEAD: {t}")
    for t in res["missing_outcome_tickers"]:
        print(f"  MISSING OUTCOME: {t}")
    print(f"near-strike rows (|print - strike| < {res['near_strike_threshold']}): "
          f"{res['n_near_strike']}")
    for t in res["near_strike_tickers"]:
        print(f"  near strike: {t}")
    print()
    _print_block("HEADLINE (all scored rows)", res["headline"])
    _print_block("EXCLUDING near-strike rows", res["excluding_near_strike"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
