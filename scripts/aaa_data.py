#!/usr/bin/env python3
"""AAA Fuel Gauge data layer: durable daily national gas/diesel print series.

The Kalshi AAA gas/diesel markets (KXDIESELD, KXAAAGASD, KXDIESELMON, KXDIESELW,
KXAAAGASMAXM, ...) resolve on the AAA national daily average ("Fuel Gauge") for a
given calendar date. The market for date D CLOSES before the D print is published
(~1:59am ET close vs a ~3-4am ET AAA update), so pricing these markets means
forecasting the next print from the prior print and the distribution of
consecutive-print changes.

This module builds and maintains the historical print series that forecast needs.

Site semantics (verified 2026-09-26 against settled Kalshi prints):
  * A page load shows ``Current Avg.`` and ``Yesterday Avg.`` — the two most
    recent published daily prints (V_t, V_{t-1}).
  * The page occasionally serves a stale pair (CDN cache / pre-publish capture),
    so the same (V, V_prev) pair can appear on several days, and the calendar day
    a page was captured is NOT reliably the calendar day of its Current value.
  * Therefore the curated store keeps one row per captured page (deduplicated by
    calendar day: the LAST snapshot of that day), and ``build_print_series``
    stitches (cur, yes) pairs into an ordered sequence of consecutive prints.
    The sequence — not calendar dates — is the model's sample of daily changes.

Commands:
  * ``backfill``: reconstruct history from the Wayback Machine's daily snapshots
    of https://gasprices.aaa.com/. Raw pages cached under ``data/aaa/raw/``
    (gitignored); parsed rows written to ``data/aaa/aaa_daily.csv`` (committed).
  * ``today``: fetch the live page and merge today's row (idempotent per day).

Provenance is preserved per row (src, snapshot_ts).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import httpx
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AAA_DIR = REPO / "data" / "aaa"
RAW_DIR = AAA_DIR / "raw"
CSV_PATH = AAA_DIR / "aaa_daily.csv"

CDX_URL = "http://web.archive.org/cdx/search/cdx"
SNAP_URL = "http://web.archive.org/web/{ts}id_/https://gasprices.aaa.com/"
LIVE_URL = "https://gasprices.aaa.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

ROW_RE = re.compile(
    r"Current Avg\.</td>\s*<td>\$([\d.]+)</td>\s*<td>\$([\d.]+)</td>\s*"
    r"<td>\$([\d.]+)</td>\s*<td>\$([\d.]+)</td>(?:\s*<td>\$([\d.]+)</td>)?",
    re.S,
)
YDAY_RE = re.compile(
    r"Yesterday Avg\.</td>\s*<td>\$([\d.]+)</td>\s*<td>\$([\d.]+)</td>\s*"
    r"<td>\$([\d.]+)</td>\s*<td>\$([\d.]+)</td>",
    re.S,
)
COLUMNS = [
    "date", "diesel", "diesel_yes", "regular", "regular_yes",
    "src", "snapshot_ts",
]


def parse_aaa_page(html: str) -> dict | None:
    """Parse the Current/Yesterday averages table from an AAA fuel gauge page.

    Returns dict with cur_*/yes_* floats, or None if the table is absent.
    Pure function; covered by tests.
    """
    m = ROW_RE.search(html)
    if not m:
        return None
    out = {
        "cur_reg": float(m.group(1)),
        "cur_mid": float(m.group(2)),
        "cur_prem": float(m.group(3)),
        "cur_die": float(m.group(4)),
    }
    y = YDAY_RE.search(html)
    if y:
        out["yes_reg"] = float(y.group(1))
        out["yes_mid"] = float(y.group(2))
        out["yes_prem"] = float(y.group(3))
        out["yes_die"] = float(y.group(4))
    return out


def load_csv() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for r in csv.DictReader(f):
                rows[r["date"]] = r
    return rows


def write_csv(rows: dict[str, dict]) -> None:
    AAA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for date in sorted(rows):
            w.writerow(rows[date])


def _row(date: str, p: dict, src: str, snap_ts: str) -> dict:
    return {
        "date": date,
        "diesel": p["cur_die"],
        "diesel_yes": p.get("yes_die", ""),
        "regular": p["cur_reg"],
        "regular_yes": p.get("yes_reg", ""),
        "src": src,
        "snapshot_ts": snap_ts,
    }


def backfill_wayback(from_date: str = "20250101", to_date: str = "", max_workers: int = 4,
                     sleep_s: float = 0.2) -> int:
    """Fetch + parse Wayback snapshots of the AAA page; merge into the CSV."""
    params = {
        "url": "gasprices.aaa.com/",
        "from": from_date,
        "to": to_date or dt.datetime.now().strftime("%Y%m%d"),
        "output": "json",
        "fl": "timestamp,statuscode",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",
        "limit": "2000",
    }
    r = httpx.get(CDX_URL, params=params, timeout=60)
    snaps = r.json()[1:]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"CDX returned {len(snaps)} snapshots")

    def fetch(ts: str):
        path = RAW_DIR / f"{ts}.html"
        if path.exists() and path.stat().st_size > 20000:
            return ts, True
        try:
            resp = httpx.get(SNAP_URL.format(ts=ts), timeout=45, headers=HEADERS,
                             follow_redirects=True)
            if resp.status_code == 200 and "Current Avg" in resp.text:
                path.write_text(resp.text)
                return ts, True
        except Exception:
            pass
        time.sleep(sleep_s)
        return ts, False

    from concurrent.futures import ThreadPoolExecutor
    got = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ts, ok in ex.map(fetch, [s[0] for s in snaps]):
            got += ok
    print(f"cached {got}/{len(snaps)} pages")

    rows = load_csv()
    for ts, _ in snaps:
        path = RAW_DIR / f"{ts}.html"
        if not path.exists():
            continue
        p = parse_aaa_page(path.read_text())
        if not p:
            continue
        day = dt.datetime.strptime(ts[:8], "%Y%m%d").date().isoformat()
        # keep the LAST snapshot of each calendar day (most likely to be updated)
        if day in rows and rows[day]["snapshot_ts"] >= ts:
            continue
        rows[day] = _row(day, p, "wayback", ts)
    write_csv(rows)
    print(f"series now has {len(rows)} dated rows -> {CSV_PATH}")
    return len(rows)


def fetch_via_browser(url: str = LIVE_URL) -> dict | None:
    """Browser fallback when AAA challenges httpx (Cloudflare).

    Uses agent-browser to render the page and extracts the Current/Yesterday
    averages straight from the DOM table cells.
    """
    import subprocess

    from src.utils.agent_browser import find as find_agent_browser

    def ab(*args: str) -> str:
        # resolve the binary: a bare "agent-browser" fails under launchd's minimal PATH
        # (the same bug that silently broke the orshare capture on 51 scheduled runs)
        return subprocess.run([find_agent_browser(), *args], capture_output=True,
                              text=True, timeout=120).stdout
    ab("open", url)
    ab("wait", "4000")
    raw = ab("eval", "(() => { const cells = [...document.querySelectorAll('td')];"
                     " const i = cells.findIndex(c => c.innerText.includes('Current'));"
                     " return i >= 0 ? JSON.stringify(cells.slice(i, i+10).map(c => c.innerText)) : '[]'; })()")
    s = raw.strip()
    if s.startswith('"') and s.endswith('"'):
        try:
            s = json.loads(s)
        except json.JSONDecodeError:
            pass
    try:
        cells = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        print(f"browser fallback failed: {raw[:120]}", file=sys.stderr)
        return None
    vals = [float(str(c).lstrip("$")) for c in cells[1:] if str(c).startswith("$")]
    if len(vals) < 4:
        print(f"browser fallback: unexpected cells {cells[:8]}", file=sys.stderr)
        return None
    # national table: cur regular, cur mid, cur premium, cur diesel
    raw2 = ab("eval", "(() => { const cells = [...document.querySelectorAll('td')];"
                      " const i = cells.findIndex(c => c.innerText.includes('Yesterday'));"
                      " return i >= 0 ? JSON.stringify(cells.slice(i, i+6).map(c => c.innerText)) : '[]'; })()")
    s2 = raw2.strip()
    if s2.startswith('"') and s2.endswith('"'):
        try:
            s2 = json.loads(s2)
        except json.JSONDecodeError:
            pass
    try:
        ycells = [float(str(c).lstrip("$")) for c in json.loads(s2)[1:] if str(c).startswith("$")]
    except Exception:
        ycells = []
    out = {"cur_reg": vals[0], "cur_mid": vals[1], "cur_prem": vals[2], "cur_die": vals[3]}
    if len(ycells) >= 4:
        out.update({"yes_reg": ycells[0], "yes_mid": ycells[1], "yes_prem": ycells[2], "yes_die": ycells[3]})
    return out


def fetch_today(base_url: str = LIVE_URL) -> dict | None:
    """Fetch the live page and merge today's row (idempotent per calendar day)."""
    resp = httpx.get(base_url, timeout=30, headers=HEADERS, follow_redirects=True)
    if resp.status_code != 200 or "Current Avg" not in resp.text:
        print(f"direct fetch failed (status={resp.status_code}); trying browser fallback", file=sys.stderr)
        b = fetch_via_browser(base_url)
        if not b:
            return None
        now = dt.datetime.now()
        day = now.date().isoformat()
        rows = load_csv()
        row = {"date": day, "diesel": b["cur_die"], "diesel_yes": b.get("yes_die", ""),
               "regular": b["cur_reg"], "regular_yes": b.get("yes_reg", ""),
               "src": "live-browser", "snapshot_ts": now.strftime("%Y%m%d%H%M%S")}
        rows[day] = row
        write_csv(rows)
        print(f"merged (browser) {day}: cur diesel {row['diesel']} regular {row['regular']}")
        return row
    p = parse_aaa_page(resp.text)
    if not p:
        print("live page parsed but no table found", file=sys.stderr)
        return None
    now = dt.datetime.now()
    day = now.date().isoformat()
    rows = load_csv()
    # staleness guard: the page updates ~3-4am ET; a pre-update capture repeats
    # the previous day's (cur, yes) pair. Merging it would fabricate a duplicate
    # print and corrupt the streak/delta series (2026-09-28 lesson).
    prior = rows.get(max(rows) if rows else "", None)
    if prior and prior.get("date") != day and _same_pair(row := _row(day, p, "live", ""), prior):
        print(f"stale pair for {day} (identical to {prior['date']}); NOT merging",
              file=sys.stderr)
        return None
    row = _row(day, p, "live", now.strftime("%Y%m%d%H%M%S"))
    rows[day] = row
    write_csv(rows)
    print(f"merged {day}: cur diesel {row['diesel']} (yes {row['diesel_yes']})")
    return row


def states_csv_path(state: str) -> Path:
    return AAA_DIR / "states" / f"{state.lower()}.csv"


def _same_pair(a: dict, b: dict) -> bool:
    """True if two rows carry the same (cur, yes) values for both fuels."""
    for col in ("diesel", "regular"):
        try:
            if abs(float(a.get(col, 0)) - float(b.get(col, 0))) > 1e-9:
                return False
            if abs(float(a.get(col + "_yes", 0)) - float(b.get(col + "_yes", 0))) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
    return True


def fetch_state(state: str, base_url: str = "https://gasprices.aaa.com/?state={state}") -> dict | None:
    """Fetch one state's page and merge its row into data/aaa/states/<state>.csv.

    Same page structure and CSV schema as the national series. Idempotent per
    calendar day (the day's latest fetch wins).
    """
    resp = httpx.get(base_url.format(state=state.upper()), timeout=30, headers=HEADERS,
                     follow_redirects=True)
    if resp.status_code != 200 or "Current Avg" not in resp.text:
        print(f"{state}: direct fetch failed (status={resp.status_code}); browser fallback", file=sys.stderr)
        b = fetch_via_browser(base_url.format(state=state.upper()))
        if not b:
            return None
        now = dt.datetime.now()
        path = states_csv_path(state)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = load_csv_from(path)
        rows[now.date().isoformat()] = _row(now.date().isoformat(), b, "live-browser",
                                            now.strftime("%Y%m%d%H%M%S"))
        write_csv_to(path, rows)
        return rows[now.date().isoformat()]
    p = parse_aaa_page(resp.text)
    if not p:
        print(f"{state}: no table", file=sys.stderr)
        return None
    now = dt.datetime.now()
    day = now.date().isoformat()
    path = states_csv_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_csv_from(path)
    prior = rows.get(max(rows) if rows else "", None)
    if prior and prior.get("date") != day and _same_pair(_row(day, p, "", ""), prior):
        print(f"{state}: stale pair for {day}; NOT merging", file=sys.stderr)
        return None
    rows[day] = _row(day, p, "live", now.strftime("%Y%m%d%H%M%S"))
    write_csv_to(path, rows)
    return rows[day]


def load_csv_from(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                rows[r["date"]] = r
    return rows


def write_csv_to(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for date in sorted(rows):
            w.writerow(rows[date])


def load_state(state: str) -> dict[str, dict]:
    return load_csv_from(states_csv_path(state))


def build_print_series(rows: dict[str, dict], value_col: str = "diesel") -> list[dict]:
    """Stitch page pairs (cur, yes) into an ordered sequence of consecutive prints.

    Returns a list of {"value": float, "prev": float, "delta_cents": float,
    "date": str} sorted by capture time. Pure function; covered by tests.
    """
    seq = []
    for date in sorted(rows):
        r = rows[date]
        try:
            cur = float(r[value_col])
            yes = float(r[value_col + "_yes"]) if r.get(value_col + "_yes") else None
        except (TypeError, ValueError):
            continue
        if yes is None:
            continue
        seq.append({"date": date, "value": cur, "prev": yes,
                    "delta_cents": round((cur - yes) * 100, 4)})
    return seq


def backfill_state(state: str, from_date: str = "20250101", to_date: str = "",
                   max_workers: int = 4, sleep_s: float = 0.3) -> int:
    """Fetch + parse Wayback snapshots of one state's AAA gauge page.

    Same mechanics as backfill_wayback but for gasprices.aaa.com/?state=<ST>.
    Merges into data/aaa/states/<st>.csv; never overwrites a live-captured row.
    Resumable: raw pages cached under data/aaa/raw/states/<ST>/.
    """
    params = {
        "url": f"gasprices.aaa.com/?state={state}",
        "from": from_date,
        "to": to_date or dt.datetime.now().strftime("%Y%m%d"),
        "output": "json",
        "fl": "timestamp,statuscode",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",
        "limit": "3000",
    }
    snaps = None
    for attempt in range(6):
        try:
            r = httpx.get(CDX_URL, params=params, timeout=60)
            snaps = r.json()[1:]
            break
        except Exception as e:
            print(f"{state}: CDX attempt {attempt+1} failed ({e}); backing off",
                  file=sys.stderr)
            time.sleep(20 * (attempt + 1))
    if snaps is None:
        print(f"{state}: CDX unavailable after retries; run again later", file=sys.stderr)
        return 0
    raw_dir = RAW_DIR / "states" / state.upper()
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"{state}: CDX returned {len(snaps)} snapshots")

    def fetch(ts: str):
        path = raw_dir / f"{ts}.html"
        if path.exists() and path.stat().st_size > 20000:
            return ts, True
        try:
            resp = httpx.get(SNAP_URL.format(ts=ts), timeout=45, headers=HEADERS,
                             follow_redirects=True)
            if resp.status_code == 200 and "Current Avg" in resp.text:
                path.write_text(resp.text)
                return ts, True
        except Exception:
            pass
        time.sleep(sleep_s)
        return ts, False

    from concurrent.futures import ThreadPoolExecutor
    got = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ts, ok in ex.map(fetch, [s[0] for s in snaps]):
            got += ok
    print(f"{state}: cached {got}/{len(snaps)} pages")

    path = states_csv_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_csv_from(path)
    added = 0
    for ts, _ in snaps:
        f = raw_dir / f"{ts}.html"
        if not f.exists():
            continue
        p = parse_aaa_page(f.read_text())
        if not p:
            continue
        day = dt.datetime.strptime(ts[:8], "%Y%m%d").date().isoformat()
        old = rows.get(day)
        # keep the last snapshot of each day; never displace a live capture
        if old and (old.get("src") != "wayback" or old.get("snapshot_ts", "") >= ts):
            continue
        if not old:
            added += 1
        rows[day] = _row(day, p, "wayback", ts)
    write_csv_to(path, rows)
    seq = build_print_series(rows, value_col="regular")
    print(f"{state}: {added} new wayback rows, series n={len(seq)} -> {path}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["backfill", "today", "states", "backfill_state"],
                    help="subcommand")
    ap.add_argument("--from", dest="from_date", default="20250101")
    ap.add_argument("--to", dest="to_date", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--state", action="append", default=[],
                    help="state code for backfill_state (repeatable)")
    ap.add_argument("--sleep", type=float, default=None,
                    help="per-fetch sleep for backfill_state (rate-limit control)")
    args = ap.parse_args()
    if args.command == "backfill":
        backfill_wayback(args.from_date, args.to_date, args.workers)
    elif args.command == "states":
        for st in ["NV", "WA", "OR", "MA", "NJ", "CA", "AZ", "CO", "CT", "FL", "GA",
                   "IL", "MI", "MN", "NC", "NY", "OH", "PA", "TX", "VA", "WI"]:
            fetch_state(st)
            time.sleep(2)
    elif args.command == "backfill_state":
        states = args.state or ["NV", "WA", "OR", "MA", "NJ", "CA", "AZ", "CO", "CT",
                                "FL", "GA", "IL", "MI", "MN", "NC", "NY", "OH", "PA",
                                "TX", "VA", "WI"]
        for st in states:
            kw = {"sleep_s": args.sleep} if args.sleep else {}
            backfill_state(st, args.from_date, args.to_date, args.workers, **kw)
    else:
        fetch_today()


if __name__ == "__main__":
    main()
