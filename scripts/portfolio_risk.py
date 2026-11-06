#!/usr/bin/env python3
"""Portfolio risk report: exposure by family, max-loss, time-to-resolution.

Reads live positions and marks each at the conservative (bid) price. Empty
bid sides make mark-to-bid overstate losses on illiquid books (e.g. a book
with no bids is marked 0) - the report flags those rows instead of crying wolf.

The point is the CLUSTERS: positions that resolve on the same event share fate
(e.g. the OpenRouter-share family all settles on the Monday 10am chart), so
the family max-loss - not any single position's - is the real risk number the
governor caps should be read against.

Usage:  python scripts/portfolio_risk.py [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FAMILY = {
    "KXDIESEL": "aaa_diesel", "KXDIESELMON": "aaa_diesel", "KXDIESELW": "aaa_diesel",
    "KXAAAGAS": "aaa_gas", "KXRAIN": "weather", "KXHIGH": "weather",
    "KXTRUMP": "trump_say", "KXTRUTHSOCIAL": "trump_say", "KXNOBEL": "nobel",
    "KXSTATELEG": "elections", "KXANTHSHARE": "orshare", "KXOPENSHARE": "orshare",
    "KXDEEPSHARE": "orshare", "KXXIAOMISHARE": "orshare", "KXTOKENUSE": "orshare",
    "KXGOOGSHARE": "orshare", "KXMISTRALSHARE": "orshare", "KXZAISHARE": "orshare",
    "KXBABASHARE": "orshare", "KXTENCENTSHARE": "orshare",
    "KXA100MS": "ornn", "KXALIENS": "paranormal", "KXSCOTUS": "legal",
    "OAIAGI": "ai_milestones", "KXLLM": "ai_milestones", "KXPS": "ai_milestones",
}


def family_of(ticker: str) -> str:
    for k, v in FAMILY.items():
        if ticker.startswith(k):
            return v
    return "other"


async def collect() -> list[dict]:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from src.clients.kalshi_client import KalshiClient
    from src.strategies.safe_compounder import _fetch_all_pages
    c = KalshiClient()
    try:
        rows = await _fetch_all_pages(c.get_positions, "market_positions")
        mps = [m for m in rows if abs(float(m.get("position_fp") or 0)) > 0.5]
        out = []
        for m in mps:
            t = m["ticker"]
            fp = float(m["position_fp"])
            cnt = abs(fp)
            side = "yes" if fp > 0 else "no"
            cost = float(m["market_exposure_dollars"] or 0)
            try:
                r = await c.get_market(t)
                mk = r.get("market", r)
            except Exception:
                mk = {}
            yb = float(mk.get("yes_bid_dollars") or 0)
            na = float(mk.get("no_bid_dollars") or 0)
            mark = yb if side == "yes" else na
            out.append({"ticker": t, "family": family_of(t), "side": side, "count": cnt,
                        "cost": round(cost, 2), "value": round(cnt * mark, 2),
                        "mtm": round(cnt * mark - cost, 2), "max_loss": round(cost, 2),
                        "mark_is_zero_bid": mark == 0,
                        "close": (mk.get("close_time") or "")[:10]})
        return out
    finally:
        await c.close()


def summarize(positions: list[dict], equity: float) -> dict:
    byf = defaultdict(lambda: {"n": 0, "cost": 0.0, "value": 0.0, "max_loss": 0.0,
                               "zero_bid": 0})
    for p in positions:
        d = byf[p["family"]]
        d["n"] += 1
        d["cost"] += p["cost"]
        d["value"] += p["value"]
        d["max_loss"] += p["max_loss"]
        d["zero_bid"] += 1 if p["mark_is_zero_bid"] else 0
    rows = sorted(({"family": f, **{k: round(v, 2) if isinstance(v, float) else v
                                     for k, v in d.items()},
                    "max_loss_pct_equity": round(100 * d["max_loss"] / max(equity, 1), 1)}
                   for f, d in byf.items()),
                  key=lambda r: -r["cost"])
    return {"positions": len(positions),
            "total_cost": round(sum(p["cost"] for p in positions), 2),
            "total_value_bid_marked": round(sum(p["value"] for p in positions), 2),
            "by_family": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--equity", type=float, default=875.0, help="equity for pct-of-equity")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    import asyncio
    positions = asyncio.run(collect())
    s = summarize(positions, args.equity)
    if args.json:
        print(json.dumps(s, indent=1))
        return
    print(f"{'family':14s} {'n':>3s} {'cost':>8s} {'bidval':>8s} {'mtm':>8s} {'maxloss%eq':>10s} {'0bid':>5s}")
    for r in s["by_family"]:
        mtm = r["value"] - r["cost"]
        print(f"{r['family']:14s} {r['n']:3d} {r['cost']:8.2f} {r['value']:8.2f} "
              f"{mtm:+8.2f} {r['max_loss_pct_equity']:9.1f}% {r['zero_bid']:5d}")
    print(f"{'TOTAL':14s} {s['positions']:3d} {s['total_cost']:8.2f} "
          f"{s['total_value_bid_marked']:8.2f}")
    print("note: bid marks overstate losses on zero-bid books (flagged per family); "
          "family max-loss rows that resolve on one event are one correlated bet.")


if __name__ == "__main__":
    main()
