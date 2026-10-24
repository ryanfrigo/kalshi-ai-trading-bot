#!/usr/bin/env python
"""Capture LIVE orderbook prices for today's candidate slice.

``capture_corpus.py`` records the whole universe from the events API. Its
``*_dollars`` prices *can* lag the live book — the `kalshi-trade` skill documents
a measured 0.68-vs-0.85 miss (thin market) — so this script captures what a
price/edge backtest actually needs: the **live book** (``orderbook_fp``) for the
candidates the strategies would consider — the genuine longshot-NO zone
(YES 2-15c, liquid, non-numeric) plus the contested near-close directional set.

Calibration note (2026-10-23, first run, top ~120 liquid candidates): snapshot and
live agreed within 1c (mean |Δyes_ask| = 0.0007, max 0.01). So the stale-field risk
concentrates in thin/fast markets — which is exactly where a longshot fade would be
sized, and exactly why the captured series should be the live book, not the snapshot.

One gzipped JSONL file per UTC day under ``data/corpus/books_<date>.jsonl.gz``,
one row per candidate, so the fade strategy can be forward-scored against real
executable prices later. The slice is capped (default 250) because capturing
live books for the whole ~120k-market universe is not feasible.

Idempotent per day (skip if today's file exists; --force to redo). Gaps are
honest: a day with no run is simply missing, and the backtest must tolerate
that — never interpolate through it.
"""
import argparse
import asyncio
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clients.kalshi_client import KalshiClient  # noqa: E402
from scripts.hunt_candidates import fetch_events, scan, enrich  # noqa: E402

CORPUS_DIR = Path("data/corpus")


def book_rows(rows, captured_at: str):
    """Pure: flatten enriched candidate rows into live-book corpus rows.

    Rows that failed enrichment (no live book) are dropped rather than written
    with the stale snapshot price — a corpus of mirages is worse than a gap.
    """
    out = []
    for r in rows:
        if r.get("yes_ask") is None or r.get("no_ask") is None:
            continue
        out.append({
            "ts": captured_at,
            "ticker": r["ticker"],
            "cat": r.get("cat", ""),
            "q": r.get("q", ""),
            "yes_bid": r.get("yes_bid"),
            "yes_ask": r.get("yes_ask"),
            "no_bid": r.get("no_bid"),
            "no_ask": r.get("no_ask"),
            "snap_yes_ask": r.get("snap_yes_ask"),
            "dtc": r.get("dtc"),
            "vol24": r.get("vol24"),
            "oi": r.get("oi"),
        })
    return out


def append_day(rows, path: Path) -> int:
    """Append rows to a gzipped JSONL day file. Returns the number written."""
    with gzip.open(path, "at", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=250, help="max candidates to price")
    ap.add_argument("--force", action="store_true", help="re-capture even if today's file exists")
    args = ap.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out = CORPUS_DIR / f"books_{today}.jsonl.gz"
    if out.exists() and not args.force:
        print(f"already captured today: {out} — skipping (use --force to redo)")
        return 0

    captured_at = datetime.now(timezone.utc).isoformat()
    client = KalshiClient()
    try:
        events = await fetch_events(client)
        longshot, directional = scan(events, datetime.now(timezone.utc))
        picks = (longshot + directional)[: args.limit]
        print(f"pricing {len(picks)} candidates "
              f"({len(longshot)} longshot-NO, {len(directional)} directional)")
        await enrich(client, picks)
    finally:
        await client.close()

    rows = book_rows(picks, captured_at)
    n = append_day(rows, out)
    missing = len(picks) - n
    print(f"wrote {n} live-book rows -> {out}"
          + (f"  ({missing} dropped: no live book)" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
