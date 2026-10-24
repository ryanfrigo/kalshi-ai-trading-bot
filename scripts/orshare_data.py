#!/usr/bin/env python3
"""OpenRouter "text request share by model author" snapshot (Kalshi KX*SHARE source).

The KXDEEPSHARE / KXOPENSHARE / KXANTHSHARE / KXXIAOMISHARE markets resolve on
the **10:00 AM ET Monday updated "Market Share" values from
openrouter.ai/rankings** — share of TEXT requests by model AUTHOR for the week
(Mon-Sun), rounded to one decimal place, with a trap: an author not separately
shown (folded into "Others") resolves ALL its strikes NO.

The public JSON APIs (rankings/models, model-rankings-chart) do NOT reproduce
the chart's numbers (they differ on modality filtering and aggregation), so the
source of truth is the page itself. This tool drives agent-browser to the
rankings page, opens the Market Share tab, and parses the accessibility text:
  * the in-progress week's author shares (the chart's live bucket), and
  * the "last complete week" text summary (ranked list with share %).
Snapshots land in data/orshare/snapshots/<ts>.json (gitignored); the resolved
history is reconstructable from settled Kalshi strike brackets.

Usage:
  python scripts/orshare_data.py snapshot
  python scripts/orshare_data.py read FILE      # parse a saved raw dump
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SNAP_DIR = REPO / "data" / "orshare" / "snapshots"


def _ab(*args: str) -> str:
    from src.utils.agent_browser import find as find_agent_browser

    out = subprocess.run([find_agent_browser(), *args], capture_output=True, text=True,
                         timeout=120)
    return out.stdout


def _unescape_cli_output(out: str) -> str:
    """agent-browser eval prints the value as a JSON string literal.

    Some CLI versions emit it already JSON-escaped (literal backslash-n);
    normalize both forms to real newlines/tabs.
    """
    s = out.strip()
    if s.startswith('"') and s.endswith('"'):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    if "\n" in s and "\n" not in s.replace("\\n", ""):
        s = s.replace("\\n", "\n").replace("\\t", "\t")
    return s


def grab_raw() -> str:
    """Drive agent-browser to the Market Share chart and return its text."""
    _ab("open", "https://openrouter.ai/rankings")
    _ab("wait", "--load", "networkidle")
    _ab("wait", "2500")
    # click the Market Share tab link (nav or section anchor)
    snap = _ab("snapshot", "-i")
    ref = None
    for line in snap.splitlines():
        if 'link "Market Share"' in line and "[" in line:
            ref = line.split("[ref=", 1)[1].split("]", 1)[0]
            break
    if ref:
        _ab("click", f"@{ref}")
        _ab("wait", "3000")
    txt = _ab("eval", "document.body.innerText")
    return _unescape_cli_output(txt)


CHART_ROW_RE = re.compile(
    r"\d+\.\n([a-zA-Z0-9\-]+)\n([\d.]+[BM]?)\n([\d.]+)%")
SUMMARY_RE = re.compile(
    r"Share of text requests made on OpenRouter in the week beginning ([\w ,\d]+?)"
    r"(?:, with the change|\.\s|[\n\r]).*?leads at ([\d.]+)%", re.S)
WEEKLIST_RE = re.compile(
    r"Share of requests\s+Change in requests\n((?:\d+\t[^\n]+\n|\t[^\n]+\n)+)")


def parse_raw(raw: str) -> dict:
    """Extract author shares from the Market Share tab text. Pure; tested.

    The page has two author-share datasets: the chart's live bucket
    (in-progress week: "1.\ndeepseek\n1.14B\n24.2%") and the ranked table
    for the most recent COMPLETE week (tab-separated rows).
    """
    out: dict = {"in_progress_week": {}, "last_complete_week": {}}
    # only trust chart rows that appear before the "as text" summary
    chart_zone = raw.split("Market share by model author, as text")[0]
    for m in CHART_ROW_RE.finditer(chart_zone):
        author, pct = m.group(1), float(m.group(3))
        out["in_progress_week"].setdefault(author, pct)
    m = WEEKLIST_RE.search(raw)
    if m:
        pct_re = re.compile(r"^([\d.]+)%$")
        for row in m.group(1).strip().splitlines():
            cells = row.split("\t")
            names = [c.strip() for c in cells
                     if c.strip() and not pct_re.match(c.strip()) and not c.strip().isdigit()]
            pcts = [c.strip() for c in cells if pct_re.match(c.strip())]
            if not names or not pcts:
                continue
            author = names[0]
            if author == "All other authors":
                author = "Others"
            try:
                out["last_complete_week"][author] = float(pcts[0].rstrip("%"))
            except ValueError:
                continue
    m = SUMMARY_RE.search(raw)
    if m:
        out["summary"] = {"week_label": m.group(1).strip(), "leader": m.group(2)}
    return {k: v for k, v in out.items() if v}


def snapshot() -> dict:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    raw = grab_raw()
    parsed = parse_raw(raw)
    ts = dt.datetime.now(dt.UTC).isoformat()
    rec = {"ts": ts, "parsed": parsed, "raw_head": raw[:20000]}
    path = SNAP_DIR / f"{ts.replace(':', '').replace('-', '')}.json"
    path.write_text(json.dumps(rec, indent=1))
    print(f"snapshot -> {path}")
    print(json.dumps(parsed, indent=1)[:1500])
    return parsed


DAY_URL = "https://openrouter.ai/api/frontend/v1/rankings/models?view=day"


def day_shares() -> dict:
    """Author shares of ALL requests for the most recent complete UTC day.

    The resolution chart is TEXT-filtered, so treat these as the raw signal and
    convert with the per-author text/all gap measured from the chart snapshot.
    """
    import httpx
    from collections import defaultdict
    r = httpx.get(DAY_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    rows = r.json()["data"]
    tot = sum(x["count"] or 0 for x in rows)
    by_author = defaultdict(int)
    for x in rows:
        by_author[x["model_permaslug"].split("/")[0]] += x["count"] or 0
    shares = {a: round(v / tot * 100, 2) for a, v in
              sorted(by_author.items(), key=lambda kv: -kv[1]) if v > 0}
    return {"date": rows[0]["date"][:10], "total_requests": tot, "shares": shares}


def latest_snapshot() -> dict | None:
    files = sorted(SNAP_DIR.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())["parsed"]


AUTHORS_DIR = REPO / "data" / "orshare" / "authors"


def author_shares(view: str = "day") -> dict:
    """Per-author ALL-request shares + totals from the public rankings API.

    `view=day` = the most recent complete UTC day; `view=week` = weekly totals
    per model (rows are keyed to the last date of the window). The resolution
    chart is TEXT-filtered; these ALL-request shares are the raw signal - the
    per-author text/all ratio is fit over time as daily captures accumulate
    (started 2026-09-28). Two weeks of daily captures enable the weekend
    dilution model for the KX*SHARE strikes.
    """
    import httpx
    from collections import defaultdict
    r = httpx.get(
        f"https://openrouter.ai/api/frontend/v1/rankings/models?view={view}",
        timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    rows = r.json()["data"]
    day = rows[0]["date"][:10]
    by_author_req = defaultdict(int)
    by_author_tok = defaultdict(int)
    for x in rows:
        a = x["model_permaslug"].split("/")[0]
        by_author_req[a] += x["count"] or 0
        by_author_tok[a] += (x.get("total_prompt_tokens") or 0) + \
            (x.get("total_completion_tokens") or 0)
    tot_req = sum(by_author_req.values())
    tot_tok = sum(by_author_tok.values())
    return {
        "fetched": dt.datetime.now(dt.UTC).isoformat(),
        "view": view,
        "label_date": day,
        "total_requests": tot_req,
        "total_tokens": tot_tok,
        "shares_by_author_req": {a: round(v / tot_req * 100, 3)
                                 for a, v in by_author_req.items() if v > 0},
        "tokens_by_author": dict(sorted(by_author_tok.items(), key=lambda kv: -kv[1])),
    }


def capture_authors() -> int:
    """Idempotent daily capture of day+week author shares -> data/orshare/authors/."""
    AUTHORS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for view in ("day", "week"):
        d = author_shares(view)
        path = AUTHORS_DIR / f"{d['label_date']}_{view}.json"
        is_new = not path.exists()
        if is_new:
            path.write_text(json.dumps(d, indent=1))
            n += 1
        print(f"{view}: label {d['label_date']} req {d['total_requests']/1e6:.0f}M "
              f"tok {d['total_tokens']/1e12:.2f}T -> {path.name}"
              f"{' (new)' if is_new else ' (exists)'}")
    return n


def strike_note(current_pct: float | None, strike: float, shown: bool) -> str:
    """Resolution logic per the rules: rounded 1dp must be > strike; not shown => NO."""
    if not shown:
        return "NOT SHOWN in chart -> trap clause makes NO the default"
    if current_pct is None:
        return "author absent from chart"
    gap = current_pct - strike
    return f"current share {'+' if gap >= 0 else ''}{gap:.1f}pts vs strike (1dp rounding; weekend mix still pending)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    sub.add_parser("day")
    sub.add_parser("authors")
    p = sub.add_parser("read")
    p.add_argument("file")
    p2 = sub.add_parser("week")
    p2.add_argument("--strikes", nargs="+", type=float, required=True,
                    help="e.g. 24.1 17.8 2.8 1.9")
    p2.add_argument("--author", required=True, help="author to check, e.g. deepseek")
    args = ap.parse_args()
    if args.cmd == "snapshot":
        snapshot()
    elif args.cmd == "day":
        print(json.dumps(day_shares(), indent=1))
    elif args.cmd == "authors":
        capture_authors()
    elif args.cmd == "read":
        print(json.dumps(parse_raw(_unescape_cli_output(Path(args.file).read_text())), indent=1))
    else:
        parsed = latest_snapshot()
        if not parsed:
            print("no snapshots; run snapshot first", file=sys.stderr)
            return
        cur = parsed.get("in_progress_week", {})
        shown = args.author in cur
        val = cur.get(args.author)
        print(f"author {args.author}: in-progress share {val} (shown={shown})")
        for s in args.strikes:
            print(f"  strike >{s}: {strike_note(val, s, shown)}")


if __name__ == "__main__":
    main()
