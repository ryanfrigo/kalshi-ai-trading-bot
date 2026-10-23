#!/usr/bin/env python
"""Backtest the longshot-NO fade on the LIVE-BOOK corpus.

Joins ``data/corpus/books_*.jsonl.gz`` (captured live orderbooks for each day's
candidate slice) to realized market outcomes, and reports what buying the NO side
at the *captured executable ask* would actually have returned — the honest test of
the fade, priced off the live book rather than the stale snapshot fields.

Honesty rules baked in:
  * a trade counts only where the market has actually settled (no interpolation);
  * fees are charged on entry (``taker_fee_per_contract``, the repo's formula);
  * the report shows the number of distinct **events**, not just rows, because one
    event contributes several strikes and correlated rows inflate n.

Outcomes come from ``--outcomes`` (a JSONL of {"ticker":..., "result":"yes"|"no"})
if given, else they are fetched once from Kalshi's settled markets and cached.
"""
import argparse
import asyncio
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aaa_pricer import taker_fee_per_contract  # noqa: E402

CORPUS_DIR = Path("data/corpus")
OUTCOMES = CORPUS_DIR / "outcomes.jsonl"


def load_books(corpus_dir: Path = CORPUS_DIR):
    """All captured live-book rows, oldest file first."""
    rows = []
    for p in sorted(corpus_dir.glob("books_*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_outcomes(path: Path = OUTCOMES):
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["ticker"]] = r["result"]
    return out


def event_key(ticker: str) -> str:
    """Group correlated strikes: the part of the ticker before the final '-'.

    One realized print/event yields several strikes, so an honest edge claim counts
    EVENTS, not strike rows.
    """
    return ticker.rsplit("-", 1)[0] if "-" in ticker else ticker


def fade_rows(rows, max_yes_ask: float = 0.15, min_no_ask: float = 0.80):
    """Pure: rows the longshot-NO fade would have entered, at the captured ask."""
    out = []
    for r in rows:
        ya, na = r.get("yes_ask"), r.get("no_ask")
        if ya is None or na is None:
            continue
        if ya <= max_yes_ask and na >= min_no_ask:
            out.append(r)
    return out


def score(rows, outcomes, max_yes_ask: float = 0.15, min_no_ask: float = 0.80):
    """Pure: buy NO at the captured ``no_ask``; settle against ``outcomes``.

    Returns a stats dict. Unsettled rows are counted but never scored.
    """
    picks = fade_rows(rows, max_yes_ask, min_no_ask)
    settled = [r for r in picks if outcomes.get(r["ticker"]) in ("yes", "no")]
    wins = 0
    pnl = 0.0
    cost = 0.0
    for r in settled:
        price = r["no_ask"]
        won = outcomes[r["ticker"]] == "no"      # buying NO wins when the market resolves NO
        fee = taker_fee_per_contract(price)
        wins += int(won)
        pnl += (1.0 if won else 0.0) - price - fee
        cost += price
    n = len(settled)
    return {
        "rows": len(rows),
        "candidates": len(picks),
        "events": len({event_key(r["ticker"]) for r in picks}),
        "settled": n,
        "unsettled": len(picks) - n,
        "wins": wins,
        "win_rate": (wins / n) if n else None,
        "mean_no_ask": (sum(r["no_ask"] for r in settled) / n) if n else None,
        "edge_vs_paid": ((wins / n) - (cost / n)) if n else None,
        "pnl_per_contract": (pnl / n) if n else None,
        "pnl_total": pnl,
    }


def render(s: dict) -> str:
    lines = [
        f"  candidate rows:     {s['candidates']}   (events: {s['events']})",
        f"  settled / unscored: {s['settled']} / {s['unsettled']}",
    ]
    if not s["settled"]:
        lines.append("  no settled candidates yet — the live-book series is too young to score."
                     " This is the honest answer, not a zero.")
        return "\n".join(lines)
    lines += [
        f"  NO wins:            {s['wins']}  ({s['win_rate']:.0%})",
        f"  mean NO paid:       {s['mean_no_ask']:.3f}",
        f"  edge vs paid:       {s['edge_vs_paid']:+.3f}",
        f"  P&L / contract:     {s['pnl_per_contract']:+.4f}   (total {s['pnl_total']:+.2f}, fees on entry)",
    ]
    if s["events"] < 30:
        lines.append(f"  ⚠️ only {s['events']} distinct events — not enough to claim edge (want >=30).")
    return "\n".join(lines)


async def fetch_outcomes(max_pages: int = 20) -> int:
    """Pull settled market results once and cache them (tickers we captured).

    Paginates the settled-markets feed newest-first and stops as soon as every
    captured ticker is accounted for — our candidates settle in the future, so they
    surface near the front once they resolve. The page cap keeps a cold cache from
    walking the entire settled history: hitting it just means "run again later".
    """
    from src.clients.kalshi_client import KalshiClient

    want = {r["ticker"] for r in load_books()}
    have = set(load_outcomes())
    todo = want - have
    if not todo:
        print(f"outcomes cache already covers all {len(want)} captured tickers")
        return 0
    client = KalshiClient()
    written = 0
    pages = 0
    try:
        cursor = None
        while pages < max_pages:
            pages += 1
            params = {"status": "settled", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            resp = await client._make_authenticated_request("GET", "/trade-api/v2/markets", params=params)
            markets = resp.get("markets", [])
            for m in markets:
                t = m.get("ticker")
                res = m.get("result")
                if t in todo and res in ("yes", "no"):
                    with open(OUTCOMES, "a") as f:
                        f.write(json.dumps({"ticker": t, "result": res}) + "\n")
                    written += 1
            if len(have) + written >= len(want):
                break                      # everything captured is accounted for
            cursor = resp.get("cursor")
            if not markets or not cursor:
                break
        else:
            print(f"stopped after {max_pages} pages ({written} cached so far) — "
                  f"settled history is deeper; run again later")
    finally:
        await client.close()
    print(f"cached {written} outcomes -> {OUTCOMES}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-yes-ask", type=float, default=0.15)
    ap.add_argument("--min-no-ask", type=float, default=0.80)
    ap.add_argument("--fetch-outcomes", action="store_true",
                    help="fetch settled results for captured tickers first")
    args = ap.parse_args()

    if args.fetch_outcomes:
        asyncio.run(fetch_outcomes())

    rows = load_books()
    if not rows:
        print("no captured books yet — run: scripts/capture_books.py")
        return 0
    s = score(rows, load_outcomes(), args.max_yes_ask, args.min_no_ask)
    print("=" * 64)
    print(f"  LONGSHOT-NO FADE BACKTEST — buy NO when yes_ask <= {args.max_yes_ask:.2f}"
          f" and no_ask >= {args.min_no_ask:.2f}")
    print("=" * 64)
    print(f"  captured rows:      {s['rows']}")
    print(render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
