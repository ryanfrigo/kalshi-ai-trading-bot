#!/usr/bin/env python3
"""Paper-trade log for model-gated maker close-outs (the only scalp variant the
study didn't kill). See docs/SCALP_STUDY.md.

record: scan the latest pricing snapshot per ladder series; a strike qualifies as
        a paper fill when model fair >= 0.995 AND yes_ask <= 0.99 (fillable).
settle: when the target print lands, mark each paper entry won/lost at
        (1 - entry - fee) / -(entry + fee) with the per-order fee model.

State: data/runtime/scalp_paper.jsonl
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRICINGS = REPO / "data" / "aaa" / "pricings"
STATE = REPO / "data" / "runtime" / "scalp_paper.jsonl"


def fee_order(c: int, p: float) -> float:
    return math.ceil(7.0 * c * p * (1 - p)) / 100.0


def latest_by_series() -> dict:
    snaps = {}
    for p in sorted(PRICINGS.glob("*.json")):
        try:
            s = json.loads(p.read_text())
        except Exception:
            continue
        if s.get("rows"):
            snaps[s["series"]] = s
    return snaps


def record() -> int:
    done = set()
    if STATE.exists():
        for line in STATE.open():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:
                continue
    n = 0
    now = datetime.now(timezone.utc).isoformat()
    for series, snap in latest_by_series().items():
        for r in snap["rows"]:
            if r["ticker"] in done:
                continue
            fair, ask = r.get("fair_yes"), r.get("yes_ask")
            bid = r.get("yes_bid")
            if fair is None or ask is None or bid is None:
                continue
            # 9/30 lesson (n=110 paper): wide-book / deep-discount rows were
            # snap-blind phantom gaps (28% win rate); tight close-outs went 18/18.
            spread = ask - bid
            if fair >= 0.995 and 0.90 <= ask <= 0.99 and spread <= 0.25:
                e = {"ts": now, "ticker": r["ticker"], "series": series,
                     "target": snap["target_date"], "entry": ask, "fair": fair,
                     "qty": 100, "status": "open"}
                with open(STATE, "a") as f:
                    f.write(json.dumps(e) + "\n")
                n += 1
    print(f"paper fills recorded: {n}")
    return n


def settle(known_actuals: dict) -> int:
    if not STATE.exists():
        return 0
    rows = [json.loads(l) for l in STATE.open()]
    n = 0
    for e in rows:
        if e["status"] != "open":
            continue
        actual = known_actuals.get((e["series"], e["target"]))
        if actual is None:
            continue
        # strike tail: 'T6.435' (diesel) or '5.5050' (state gas) - dollars in both
        strike = float(e["ticker"].rsplit("-", 1)[-1].lstrip("T"))
        y = 1 if actual > strike else 0
        fee = fee_order(e["qty"], e["entry"]) / e["qty"]
        pnl = (1 - e["entry"] - fee) if y else -(e["entry"] + fee)
        e.update({"status": "settled", "y": y, "actual": actual,
                  "pnl_per_contract": round(pnl, 4)})
        n += 1
    with open(STATE, "w") as f:
        for e in rows:
            f.write(json.dumps(e) + "\n")
    won = [e for e in rows if e.get("status") == "settled"]
    if won:
        wins = sum(1 for e in won if e["y"])
        print(f"settled {n}; cumulative: {wins}/{len(won)} wins, "
              f"PnL/ct {sum(e['pnl_per_contract'] for e in won):+.3f}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "settle"])
    args = ap.parse_args()
    if args.cmd == "record":
        record()
    else:
        import csv
        actuals = {}
        rows = {r["date"]: r for r in csv.DictReader(open(REPO / "data" / "aaa" / "aaa_daily.csv"))}
        for date, r in rows.items():
            actuals[("KXDIESELD", date)] = float(r["diesel"]) if r["diesel"] else None
            actuals[("KXAAAGASD", date)] = float(r["regular"]) if r["regular"] else None
        actuals = {k: v for k, v in actuals.items() if v is not None}
        sys.path.insert(0, str(REPO / "scripts"))
        import aaa_data
        for st in ["NV", "WA", "OR", "MA", "NJ", "CA", "AZ", "CO", "CT", "FL", "GA",
                   "IL", "MI", "MN", "NC", "NY", "OH", "PA", "TX", "VA", "WI"]:
            seq = aaa_data.build_print_series(aaa_data.load_live_state(st), value_col="regular")  # no src=wayback state rows (national prints)
            for s in seq:
                actuals[(f"KXAAAGASD{st}", s["date"])] = s["value"]
        settle(actuals)


if __name__ == "__main__":
    import sys
    main()
