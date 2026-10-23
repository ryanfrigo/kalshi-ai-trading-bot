#!/usr/bin/env bash
#
# Daily bounded trading run, intended for launchd/cron.
#
# Runs `cli.py daily` once and exits. The risk governor (daily-loss + drawdown
# kill switch) runs first inside the command; new buys are skipped if halted.
#
# Mode is controlled by the KALSHI_DAILY_MODE env var:
#   dry  (default) -> no real orders, safe to schedule immediately
#   live           -> places real orders with real money
#
# This script derives the repo path from its own location, so it works for any
# checkout without editing.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${KALSHI_DAILY_MODE:-dry}"
FLAG=""
if [ "$MODE" = "live" ]; then
  FLAG="--live"
fi

PYBIN="$REPO/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="python3"

mkdir -p logs/daily_runs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/daily_runs/daily_${TS}.log"

echo "[run_daily] $(date) mode=$MODE -> $LOG"

# Data capture first — no orders involved, safe in both modes. These keep the
# evidence base growing even on days the trading loop does nothing:
#   * full-universe price snapshot (UNIVERSE corpus: what was open, and what
#     resolved; its *_dollars prices are snapshots, not reliable entry prices)
#   * live-book slice for today's candidates (the ENTRY-PRICE corpus a fade
#     can honestly be backtested against)
#   * AAA fuel-gauge print series (data/aaa/aaa_daily.csv, idempotent per day)
#   * AAA ladder pricing snapshot vs the live book (forward-scored by
#     scripts/aaa_pricer.py score once the print lands)
PYTHONPATH="$REPO" "$PYBIN" scripts/capture_corpus.py >>"$LOG" 2>&1 || true
# live-book slice (the corpus prices a fade can actually be backtested against)
PYTHONPATH="$REPO" "$PYBIN" scripts/capture_books.py >>"$LOG" 2>&1 || true
PYTHONPATH="$REPO" "$PYBIN" scripts/aaa_data.py today >>"$LOG" 2>&1 || true
# AAA state dailies for the KXAAAGASD<ST> ladders (21 states, idempotent per day)
PYTHONPATH="$REPO" "$PYBIN" scripts/aaa_data.py states >>"$LOG" 2>&1 || true
PYTHONPATH="$REPO" "$PYBIN" scripts/aaa_futures.py fetch >>"$LOG" 2>&1 || true
# Ornn GPU compute price index (KXA100MS-family resolution source)
PYTHONPATH="$REPO" "$PYBIN" scripts/ornn_data.py fetch >>"$LOG" 2>&1 || true
PYTHONPATH="$REPO" "$PYBIN" scripts/ornn_data.py ladder >>"$LOG" 2>&1 || true
# OpenRouter text-share-by-author chart (KX*SHARE resolution source; browser dump)
PYTHONPATH="$REPO" "$PYBIN" scripts/orshare_data.py snapshot >>"$LOG" 2>&1 || true
# score any pricing snapshots whose print has landed (model vs book, Brier)
PYTHONPATH="$REPO" "$PYBIN" scripts/aaa_pricer.py score >>"$LOG" 2>&1 || true
# score the Jev forward-pricing log against settled markets (model vs Jev vs book)
PYTHONPATH="$REPO" "$PYBIN" scripts/jev_score.py >>"$LOG" 2>&1 || true

echo "[run_daily] $(date) mode=$MODE -> $LOG (trading loop next)"
PYTHONPATH="$REPO" "$PYBIN" cli.py daily $FLAG >>"$LOG" 2>&1
STATUS=$?
echo "[run_daily] $(date) exit=$STATUS mode=$MODE log=$LOG"
exit $STATUS
