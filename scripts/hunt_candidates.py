#!/usr/bin/env python3
"""
hunt_candidates.py — broad edge-candidate scanner for the agent-at-the-helm process.

`cli.py daily` only surfaces the mechanical "near-certain NO, YES<=0.20, model-edge>=3c"
slice — which on an efficient day coughs up only un-tradeable 96c temperature buckets.
This tool casts a wider net across the FULL open universe (via the events API, the only
endpoint that returns real individual markets — /markets returns KXMVE parlay tickers),
buckets candidates by the two edge structures worth researching, and — critically —
enriches the shortlist with LIVE orderbook prices.

WHY live books matter: the events snapshot's *_dollars fields are STALE (last-trade/mid).
Real executable prices live in the orderbook under `orderbook_fp.{yes_dollars,no_dollars}`
(prices in $). best yes_ask = 1 - best_no_bid ; best no_ask = 1 - best_yes_bid.
Always price edge off the live book, never the snapshot. (Learned 2026-06-19: a market
showed snapshot YES 0.68 while the live book was 0.85.)

Output is RAW MATERIAL for the RESEARCH step, not a buy list. A deep, tight, liquid book
is already a sharp price — large apparent "edge" there is a red flag (stale / live-vs-
pre-match data), not a gift. Real edge lives in genuine <5% structural longshots or
thin/obscure mispriced markets with enough liquidity to fill and exit.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/hunt_candidates.py            # scan + enrich top shortlist
    PYTHONPATH=. .venv/bin/python scripts/hunt_candidates.py --no-books # faster, snapshot prices only
    PYTHONPATH=. .venv/bin/python scripts/hunt_candidates.py --top 40   # enrich more candidates
"""
import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone

sys.path.append(".")
from src.clients.kalshi_client import KalshiClient

# numeric-distribution buckets = the loser category (real continuous distribution; a
# "narrow bucket" can quietly carry 10-20%). Exclude from the longshot-NO hunt.
NUMERIC = re.compile(
    r"(KXCPI|KXPCE|KXGDP|KXNGDP|KXFED|KXRATE|KXUNRATE|KXJOBS|KXPAYROLL|"
    r"KXHIGH|KXLOW|KXTEMP|KXRAIN|KXSNOW|KXBTC|KXETH|KXSOL|KXXRP|KXDOGE|KXINX|KXSPX|"
    r"KXNASDAQ|KXNDX|KXDJIA|KXVIX|KXGAS|KXOIL|KXEGG|KXMORT|KXWTI|KXAISPIKE)", re.I)
BUCKET_SUFFIX = re.compile(r"-[TB]\d", re.I)  # -T64499.99 / -B80.5 price/temp buckets


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _days_to(close, now):
    try:
        dt = datetime.fromisoformat(close.replace("Z", "+00:00"))
        return round((dt - now).total_seconds() / 86400, 1)
    except Exception:
        return None


def best_book(ob):
    """Parse orderbook_fp into best executable yes/no prices ($)."""
    o = ob.get("orderbook_fp", {}) or {}
    yes = [(_f(p), _f(s)) for p, s in (o.get("yes_dollars") or [])]
    no = [(_f(p), _f(s)) for p, s in (o.get("no_dollars") or [])]
    yb = max((p for p, _s in yes), default=None)  # best yes bid
    nb = max((p for p, _s in no), default=None)    # best no bid
    return {
        "yes_ask": round(1 - nb, 2) if nb is not None else None,
        "yes_bid": yb,
        "no_ask": round(1 - yb, 2) if yb is not None else None,
        "no_bid": nb,
        "depth_yes": len(yes),
        "depth_no": len(no),
    }


async def fetch_events(client, max_pages=160):
    """All open events with nested markets (the real tradeable universe)."""
    out, cursor, page = [], None, 0
    while page < max_pages:
        params = {"status": "open", "limit": 100, "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        resp = await client._make_authenticated_request(
            "GET", "/trade-api/v2/events", params=params
        )
        evs = resp.get("events", [])
        if not evs:
            break
        out.extend(evs)
        cursor = resp.get("cursor")
        if not cursor:
            break
        page += 1
        await asyncio.sleep(0.05)
    return out


def scan(events, now):
    longshot_no, directional = [], []
    for e in events:
        cat = e.get("category", "")
        etitle = e.get("title", "")
        for m in e.get("markets", []):
            t = m.get("ticker", "")
            if t.startswith("KXMVE"):  # parlay combos — skip
                continue
            ya = _f(m.get("yes_ask_dollars"))
            if ya <= 0:
                continue
            v24 = _f(m.get("volume_24h_fp"))
            vol = _f(m.get("volume_fp"))
            oi = _f(m.get("open_interest_fp"))
            dtc = _days_to(m.get("close_time", ""), now)
            sub = m.get("yes_sub_title") or m.get("no_sub_title") or ""
            liquid = (v24 >= 50) or (oi >= 300) or (vol >= 1500)
            is_num = bool(NUMERIC.search(t) or BUCKET_SUFFIX.search(t))
            rec = {
                "ticker": t, "cat": cat, "q": (etitle + " :: " + sub)[:95],
                "snap_yes_ask": round(ya, 3), "snap_no_ask": round(_f(m.get("no_ask_dollars")), 3),
                "vol24": int(v24), "oi": int(oi), "dtc": dtc,
            }
            # genuine longshot-NO zone: YES 2-15c, NO has room for a >=5c edge, liquid, not numeric
            if 0.02 <= ya <= 0.15 and _f(m.get("no_ask_dollars")) <= 0.96 and liquid and not is_num \
                    and (dtc is None or dtc <= 220):
                longshot_no.append(rec)
            # contested near-close liquid markets (research-driven directional edge)
            if 0.25 <= ya <= 0.78 and (v24 >= 800 or vol >= 6000) and dtc is not None \
                    and 0 < dtc <= 20 and not is_num:
                directional.append(rec)
    longshot_no.sort(key=lambda r: (r["vol24"], r["oi"]), reverse=True)
    directional.sort(key=lambda r: r["vol24"], reverse=True)
    return longshot_no, directional


async def enrich(client, rows):
    """Attach live executable book prices to each candidate row."""
    for r in rows:
        try:
            ob = await client.get_orderbook(r["ticker"], depth=10)
            b = best_book(ob)
            r["yes_ask"], r["no_ask"] = b["yes_ask"], b["no_ask"]
            r["yes_bid"], r["no_bid"] = b["yes_bid"], b["no_bid"]
        except Exception as ex:  # noqa: BLE001
            r["book_err"] = str(ex)[:60]
        await asyncio.sleep(0.04)
    return rows


def _print(title, rows, books):
    print(f"\n=== {title} (n={len(rows)}) ===")
    for r in rows:
        if books and r.get("no_ask") is not None:
            px = f"live YES_ask={r['yes_ask']:.2f} NO_ask={r['no_ask']:.2f}"
        else:
            px = f"snap YES_ask={r['snap_yes_ask']:.2f} NO_ask={r['snap_no_ask']:.2f}"
        print(f"  {r['ticker']:<40} {px}  vol24={r['vol24']:>9} dtc={r['dtc']}  [{r['cat']}]")
        print(f"      {r['q']}")


async def main():
    ap = argparse.ArgumentParser(description="Broad Kalshi edge-candidate scanner")
    ap.add_argument("--top", type=int, default=25, help="how many of each bucket to live-price")
    ap.add_argument("--no-books", action="store_true", help="skip live orderbook enrichment")
    ap.add_argument("--out", default="data/runtime/hunt_candidates.json")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    client = KalshiClient()
    try:
        events = await fetch_events(client)
        nmkts = sum(len(e.get("markets", [])) for e in events)
        print(f"scanned {len(events)} events / {nmkts} markets")
        longshot_no, directional = scan(events, now)
        ls_top, dir_top = longshot_no[:args.top], directional[:args.top]
        if not args.no_books:
            print("enriching shortlist with LIVE orderbooks...")
            await enrich(client, ls_top)
            await enrich(client, dir_top)
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass

    _print("LONGSHOT-NO CANDIDATES (fade overpriced lottery-ticket YES)", ls_top, not args.no_books)
    _print("DIRECTIONAL near-close liquid (research vs sharp consensus)", dir_top, not args.no_books)
    try:
        with open(args.out, "w") as f:
            json.dump({"longshot_no": ls_top, "directional": dir_top}, f, indent=2)
        print(f"\nwrote {args.out}")
    except Exception as ex:  # noqa: BLE001
        print(f"\n(could not write {args.out}: {ex})")


if __name__ == "__main__":
    asyncio.run(main())
