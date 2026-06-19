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
PYTHONPATH="$REPO" "$PYBIN" cli.py daily $FLAG >>"$LOG" 2>&1
STATUS=$?
echo "[run_daily] $(date) exit=$STATUS mode=$MODE log=$LOG"
exit $STATUS
