"""Build data/aaa/backtest_outcomes.csv: the realized print for every
(series, target_date) that appears in data/aaa/alerts/*.json.

Dates already in the committed CSVs are copied (source=committed_csv). Dates
that are not (the 10-01 / 10-03 targets) are read from Wayback Machine
snapshots of gasprices.aaa.com, the ONLY network access this script makes, and
only when run with --fetch. Parsing reuses aaa_data.parse_aaa_page.

Snapshot rule: the last snapshot whose UTC date is the target date D and whose
time is >= 11:00 UTC (after AAA's daily update). Cross-check: the first
snapshot of D+1 taken >= 11:00 UTC must show "Yesterday" equal to D's
"Current"; source is `wayback` when it matches and `wayback_noxcheck` when no
D+1 snapshot exists. A mismatch or a missing D snapshot leaves the row out
(listed as missing on stdout). Nothing is guessed.

NOTE: aaa_data.backfill_state fetches the NATIONAL url for state timestamps
(SNAP_URL has no ?state=), so its `wayback` rows in data/aaa/states/*.csv are
national values. This script builds the state URL itself. Every alert outcome
currently used comes from `live` rows, so the committed backtest is unaffected.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aaa_data  # noqa: E402

REPO = aaa_data.REPO
ALERTS = REPO / "data" / "aaa" / "alerts"
OUT = REPO / "data" / "aaa" / "backtest_outcomes.csv"
COLS = ["series", "target_date", "print", "source", "snapshot_url", "snapshot_ts"]
MIN_HHMMSS = "110000"  # UTC; AAA's daily update has landed by then


def alert_keys() -> list[tuple[str, str]]:
    keys = set()
    for f in glob.glob(str(ALERTS / "*.json")):
        d = json.loads(Path(f).read_text())
        keys.add((d["series"], d["target_date"]))
    return sorted(keys)


def spec(series: str):
    """-> (state or None, column, csv path) per aaa_pricer.cmd_score.realized_for."""
    if series.startswith("KXAAAGASD") and len(series) > len("KXAAAGASD"):
        st = series[len("KXAAAGASD"):len("KXAAAGASD") + 2]
        return st, "regular", aaa_data.states_csv_path(st)
    if "GAS" in series.upper():
        return None, "regular", aaa_data.CSV_PATH
    return None, "diesel", aaa_data.CSV_PATH


def page_url(ts: str, state: str | None) -> str:
    base = f"http://web.archive.org/web/{ts}id_/https://gasprices.aaa.com/"
    return base + (f"?state={state}" if state else "")


def _get(url: str, params=None):
    """GET with backoff. None on 404; raises if Wayback keeps failing, so a
    flaky network can never be mistaken for a missing snapshot."""
    last = None
    for i in range(8):
        time.sleep(1.0)  # be polite to archive.org
        try:
            r = httpx.get(url, params=params, timeout=60, headers=aaa_data.HEADERS,
                          follow_redirects=True)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            last = f"status {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"GET {url} failed after retries: {last}")


def snapshots(state: str | None, day: str) -> list[str]:
    """All 200-status snapshot timestamps for the page from `day` to day+2."""
    import datetime as dt
    d0 = dt.date.fromisoformat(day)
    r = _get(aaa_data.CDX_URL, {
        "url": "gasprices.aaa.com/" + (f"?state={state}" if state else ""),
        "from": d0.strftime("%Y%m%d"),
        "to": (d0 + dt.timedelta(days=2)).strftime("%Y%m%d"),
        "output": "json", "fl": "timestamp,statuscode", "filter": "statuscode:200",
    })
    return [row[0] for row in r.json()[1:]] if r else []


def fetch_row(series: str, day: str):
    """-> (row dict | None, note)."""
    import datetime as dt
    state, col, _ = spec(series)
    pcol, ycol = ("cur_reg", "yes_reg") if col == "regular" else ("cur_die", "yes_die")
    d = day.replace("-", "")
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).strftime("%Y%m%d")
    snaps = snapshots(state, day)
    cands = [t for t in snaps if t[:8] == d and t[8:] >= MIN_HHMMSS]
    if not cands:
        return None, "no snapshot on target date after 11:00 UTC"
    ts = cands[-1]
    r = _get(page_url(ts, state))
    p = aaa_data.parse_aaa_page(r.text) if r else None
    if not p:
        return None, f"snapshot {ts} unparseable"
    nx = [t for t in snaps if t[:8] == nxt and t[8:] >= MIN_HHMMSS]
    source, note = "wayback_noxcheck", "no D+1 snapshot; unverified"
    if nx:
        r2 = _get(page_url(nx[0], state))
        p2 = aaa_data.parse_aaa_page(r2.text) if r2 else None
        if p2 and abs(p2.get(ycol, -1) - p[pcol]) < 1e-9:
            source, note = "wayback", f"xcheck ok vs {nx[0]}"
        else:
            return None, f"xcheck MISMATCH: D current {p[pcol]} vs D+1 {nx[0]} yesterday {p2 and p2.get(ycol)}"
    return {"series": series, "target_date": day, "print": p[pcol], "source": source,
            "snapshot_url": page_url(ts, state), "snapshot_ts": ts}, note


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="allow Wayback network access for dates not in the committed CSVs")
    args = ap.parse_args()
    out, missing = [], []
    for series, day in alert_keys():
        _, col, path = spec(series)
        row = aaa_data.load_csv_from(path).get(day)
        if row and row.get(col) not in (None, ""):
            out.append({"series": series, "target_date": day, "print": float(row[col]),
                        "source": "committed_csv",
                        "snapshot_url": str(path.relative_to(REPO)),
                        "snapshot_ts": row.get("snapshot_ts", "")})
            continue
        if not args.fetch:
            missing.append((series, day, "not in committed CSV; rerun with --fetch"))
            continue
        got, note = fetch_row(series, day)
        print(f"{series} {day}: {note}", file=sys.stderr)
        if got:
            out.append(got)
        else:
            missing.append((series, day, note))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows -> {OUT}")
    for m in missing:
        print("MISSING", *m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
