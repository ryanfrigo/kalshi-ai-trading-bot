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
#
# Each step is allowed to fail without stopping the run (a data source being down
# must not block the trading loop), but failures are no longer silent: they are
# recorded and summarised at the end. This is how orshare_data.py stayed broken
# for 51 runs — the step failed, `|| true` swallowed it, and only a traceback
# buried in the log said so.
FAILED=()
step() {  # run a data step; never abort the run, but never hide the failure either
  local name="$1"; shift
  if ! PYTHONPATH="$REPO" "$PYBIN" "$@" >>"$LOG" 2>&1; then
    FAILED+=("$name")
    echo "[run_daily] STEP FAILED: $name" >>"$LOG"
  fi
}

# compress logs older than a week (the disk was at 100%; nothing is ever deleted)
step rotate_logs scripts/rotate_logs.py

step capture_corpus scripts/capture_corpus.py
# live-book slice (the corpus prices a fade can actually be backtested against)
step capture_books scripts/capture_books.py
step aaa_data_today scripts/aaa_data.py today
# AAA state dailies for the KXAAAGASD<ST> ladders (21 states, idempotent per day)
step aaa_data_states scripts/aaa_data.py states
step aaa_futures scripts/aaa_futures.py fetch
# Ornn GPU compute price index (KXA100MS-family resolution source)
step ornn_fetch scripts/ornn_data.py fetch
step ornn_ladder scripts/ornn_data.py ladder
# OpenRouter text-share-by-author chart (KX*SHARE resolution source; browser dump)
step orshare_snapshot scripts/orshare_data.py snapshot
# score any pricing snapshots whose print has landed (model vs book, Brier)
step aaa_pricer_score scripts/aaa_pricer.py score
# score the Jev forward-pricing log against settled markets (model vs Jev vs book)
step jev_score scripts/jev_score.py

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[run_daily] WARNING: ${#FAILED[@]} data step(s) failed: ${FAILED[*]}" >>"$LOG"
  printf '%s\n' "${FAILED[@]}" > logs/daily_runs/LAST_FAILURES.txt
else
  rm -f logs/daily_runs/LAST_FAILURES.txt
fi

echo "[run_daily] $(date) mode=$MODE -> $LOG (trading loop next)"
PYTHONPATH="$REPO" "$PYBIN" cli.py daily $FLAG >>"$LOG" 2>&1
STATUS=$?
echo "[run_daily] $(date) exit=$STATUS mode=$MODE log=$LOG"
exit $STATUS
