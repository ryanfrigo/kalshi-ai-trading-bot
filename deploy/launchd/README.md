# Daily run on macOS launchd

Runs `cli.py daily` once per day. The command runs the risk-governor kill switch
first, then a single governed trading cycle, then writes account snapshots — and
exits (unlike `cli.py run`, which loops forever).

## Why launchd (not cron)

launchd survives reboots and runs even when you are not logged in interactively
(cron on macOS is deprecated and flaky). The trade run must be **local** — your
Kalshi key and position DB are not in the cloud.

## Install (starts in DRY mode — no real orders)

```bash
REPO="$(pwd)"                       # run from the repo root
PLIST="$HOME/Library/LaunchAgents/com.kalshi-daily.plist"
sed "s#__REPO__#$REPO#g" deploy/launchd/com.kalshi-daily.plist.template > "$PLIST"
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
launchctl list | grep kalshi-daily   # confirm it is registered
```

Test it immediately without waiting for 10:00:

```bash
launchctl start com.kalshi-daily
tail -f logs/daily_runs/daily_*.log
```

## Go live

Only after you have reviewed a successful **live** run (`KALSHI_DAILY_MODE=live
bash scripts/run_daily.sh`), edit the plist's `KALSHI_DAILY_MODE` value from
`dry` to `live`, then reload:

```bash
launchctl unload "$PLIST" && launchctl load "$PLIST"
```

## Kill switch

To halt all new buys immediately (the next run still settles/exits, just no new
positions), drop the manual halt file:

```bash
mkdir -p data/runtime && echo "manual stop $(date)" > data/runtime/TRADING_HALTED
```

Remove it to resume. The governor also auto-halts at -10% day / -15% drawdown.

## Uninstall

```bash
launchctl unload "$HOME/Library/LaunchAgents/com.kalshi-daily.plist"
rm "$HOME/Library/LaunchAgents/com.kalshi-daily.plist"
```
