#!/usr/bin/env python
"""Capture a daily price snapshot of the full open-market universe.

The one thing a real backtest needs that this repo never had: entry prices
over time. Each run appends one gzipped JSONL file per UTC day under
``data/corpus/`` — one row per open market with its snapshot prices, volume,
open interest and close time. Joined later against Kalshi's authoritative
settlements, this corpus makes an honest, out-of-sample backtest of the
longshot-fade strategy possible (see docs: the backtest is data-blocked
until this corpus exists).

Idempotent per day: if today's file already exists it exits without touching
anything (pass --force to re-capture). Run it from any loop tick — a day with
two runs stays one file, a day with zero runs is simply a gap, and gaps are
honest (the backtest must tolerate them, never interpolate through them).
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
from scripts.hunt_candidates import fetch_events, _f  # noqa: E402

CORPUS_DIR = Path("data/corpus")


def snapshot_rows(events, captured_at: str):
    """Flatten the events universe into corpus rows. Pure."""
    rows = []
    for e in events:
        cat = e.get("category", "")
        for m in e.get("markets", []):
            t = m.get("ticker", "")
            if not t:
                continue
            rows.append({
                "ts": captured_at,
                "ticker": t,
                "cat": cat,
                "yes_ask": _f(m.get("yes_ask_dollars")),
                "yes_bid": _f(m.get("yes_bid_dollars")),
                "no_ask": _f(m.get("no_ask_dollars")),
                "no_bid": _f(m.get("no_bid_dollars")),
                "last": _f(m.get("last_price_dollars")),
                "vol24": _f(m.get("volume_24h_fp")),
                "vol": _f(m.get("volume_fp")),
                "oi": _f(m.get("open_interest_fp")),
                "close": m.get("close_time", ""),
            })
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser(description="Capture daily market-price corpus")
    ap.add_argument("--force", action="store_true", help="re-capture even if today's file exists")
    ap.add_argument("--out-dir", default=str(CORPUS_DIR))
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    out_dir = Path(args.out_dir)
    out = out_dir / f"markets_{now:%Y%m%d}.jsonl.gz"
    if out.exists() and not args.force:
        print(f"already captured today: {out} — skipping (use --force to redo)")
        return 0

    client = KalshiClient()
    try:
        events = await fetch_events(client)
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass

    rows = snapshot_rows(events, now.isoformat())
    if not rows:
        print("no open markets returned — refusing to write an empty snapshot")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(out)
    print(f"captured {len(rows)} markets -> {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
