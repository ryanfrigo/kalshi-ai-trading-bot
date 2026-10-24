#!/usr/bin/env bash
#
# Morning data + repricing run (no orders, ever).
#
# Runs shortly after the ~3-4am ET AAA update (4:05am PT): captures the fresh
# print, snapshots all the data families, prices the ladders vs the live books,
# scores what has settled, and writes gap alerts for the agent loop. The
# trading loop stays the agent's decision - this only feeds it evidence while
# the post-print repricing window is live.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYBIN="$REPO/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="python3"

mkdir -p logs/daily_runs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/daily_runs/morning_${TS}.log"
echo "[run_morning] $(date) -> $LOG"

FAILED=()
run() {  # never abort the run on one failure, but never hide it either
  if ! PYTHONPATH="$REPO" "$PYBIN" "$@" >>"$LOG" 2>&1; then
    FAILED+=("$*")
    echo "[run_morning] STEP FAILED: $*" >>"$LOG"
  fi
}

TARGET="$(date -u -d '+1 day' +%F 2>/dev/null || date -u -v+1d +%F)"          # next print (daily ladders)
TARGET_WEEK="$(date -u -d '+2 days' +%F 2>/dev/null || date -u -v+2d +%F)"     # weekly resolves on the print after next
TARGET_MONTH="$(date -u -d '+4 days' +%F 2>/dev/null || date -u -v+4d +%F)"    # monthly, 30 days ahead of the week

run scripts/capture_corpus.py
run scripts/aaa_data.py today
run scripts/aaa_data.py states
run scripts/aaa_futures.py fetch
run scripts/ornn_data.py fetch
run scripts/ornn_data.py ladder
run scripts/orshare_data.py snapshot
run scripts/orshare_data.py authors
run scripts/weather_data.py capture
run scripts/aaa_pricer.py path --series KXDIESELMON --target "$TARGET_MONTH"
run scripts/aaa_pricer.py path --series KXDIESELW --target "$TARGET_WEEK"
run scripts/aaa_pricer.py price --series KXDIESELD --target "$TARGET" --alert-min 0.10
run scripts/aaa_pricer.py price --series KXAAAGASD --target "$TARGET" --retail regular --alert-min 0.10
for ST in NV WA OR MA NJ CA AZ CO CT FL GA IL MI MN NC NY OH PA TX VA WI; do
  run scripts/aaa_pricer.py price --series "KXAAAGASD$ST" --target "$TARGET" --retail regular --alert-min 0.15
done
run scripts/aaa_pricer.py score
run scripts/scalp_paper.py record || true
run scripts/scalp_paper.py settle || true
run scripts/jev_score.py

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[run_morning] WARNING: ${#FAILED[@]} step(s) failed: ${FAILED[*]}" >>"$LOG"
fi
echo "[run_morning] $(date) done -> $LOG"
grep -c ALERT "$LOG" || true
echo "[run_morning] standing verdicts: data/aaa/alerts/ADJUDICATIONS.md"
