# Autonomous Agentic Daily Trading — Design

**Date:** 2026-06-18
**Branch:** `feat/autonomous-daily-trading`
**Status:** Approved direction; building in phases via a daily `/loop`.

## Mission

Use this repo to autonomously and *profitably* trade the owner's **live** Kalshi account,
every day. **Claude is the decision engine** — not a metered LLM API. The deterministic,
math-only **Safe Compounder** carries reliable live trading even when Claude isn't running;
Claude's researched directional trades layer on top when it runs in the loop. Pivot the repo
toward this agentic model. Collect data for backtesting **locally only** (never committed).
Goal: **build real account value.**

## Why "live now" still means "safe first"

The fastest way to *lose* money — the opposite of the goal — is an unbounded agent firing
live orders. The account is small ($798.88 total on 2026-06-18); one bad day or one order bug
can do real damage. So every live action runs under hard guards, and new order code is
dry-run-verified before it is scheduled.

## Current-state findings (verified 2026-06-18)

- **No realized daily-loss kill switch exists.** `max_daily_loss_pct` (10%) and `max_drawdown`
  (15%) are defined in `src/config/settings.py` but enforced **nowhere** in the order path.
  This is the #1 safety gap.
- `cli.py run --live` → `BeastModeBot.run()` is an **infinite loop** (4 background tasks), not
  a run-once process. Unsuitable for a daily cron → need a bounded entrypoint.
- Per-order guards already exist and are good: price sanity (issue #42), fail-closed balance
  check (`src/jobs/execute.py`).
- **Safe Compounder** (`src/strategies/safe_compounder.py`) is bounded, math-only (no API),
  has a clean `dry_run` switch, and trades the repo's only evidence-backed profitable category
  (NO-side sports). Best base for live capital.
- Existing risk guards: `PositionLimitsManager`, `PortfolioEnforcer`, `CategoryScorer`,
  edge filter (>8%), Kelly (quarter-Kelly). Logging to SQLite is thorough.
- Test baseline: 85 passed, 8 skipped (live) — green.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
   DAILY (launchd)  │  cli.py daily  (NEW, bounded, runs once)     │
   ───────────────► │  1. risk_governor.check()  ── kill switch    │
                    │  2. trade ONE bounded cycle (Safe Compounder)│
                    │  3. exits/stop-loss/take-profit (all posns)  │
                    │  4. learning.update()  (auto, bounded)       │
                    │  5. data_collector snapshot (LOCAL only)     │
                    │  6. write daily report  → exit               │
                    └─────────────────────────────────────────────┘
   IN-LOOP (Claude) │  Agentic layer: research markets, decide     │
   ───────────────► │  directional + cleanup trades, execute under │
                    │  risk_governor, log as strategy="claude_*"   │
                    └─────────────────────────────────────────────┘
```

- **Deterministic Python layer** = reliable execution, risk, logging, data. No LLM cost.
- **Agentic Claude layer** = the intelligence. Runs in the session `/loop` now; a documented
  local headless-Claude schedule can run it unattended later. Never required for the account
  to stay safe — Safe Compounder + governor carry it.

## Components

1. **`src/risk/risk_governor.py` (NEW)** — the kill switch. Persists a start-of-day equity
   baseline + running peak; computes realized+unrealized P&L; returns HALT (block new buys,
   allow exits) when daily loss ≥ 10% or drawdown from peak ≥ 15%. Unit-tested, adversarially
   verified. Called first in `cli.py daily` and before every Claude-initiated buy.
2. **`cli.py daily` (NEW)** — bounded run-once entrypoint wiring the steps above.
3. **`src/data/collector.py` (NEW)** — append-only local snapshots (markets, signals, fills,
   settlements) to a git-ignored store for backtesting. Never committed.
4. **`src/learning/` (NEW)** — reads settled outcomes, recalibrates CategoryScorer + confidence
   + Kelly within bounded reversible ranges; logs every change to `learning_log`.
5. **launchd plist + wrapper script** — durable daily local schedule.
6. **Daily report** (`docs/reports/` or local) — human-readable run summary.
7. **Existing-book management** — analyze the 6 positions, sell broken theses, hold live ones,
   apply exits going forward. Owner delegated the call ("hold or sell, your judgment").

## Risk model (owner-selected: repo % defaults)

3%/position · 10% daily-loss kill · 15% drawdown halt · quarter-Kelly · fail-closed balance ·
price sanity. No extra absolute caps for now; revisit as the account grows.

## Data & repo hygiene

- Trading/backtest data: **local, git-ignored** (`data/`, `*.db`, snapshots). Already covered
  by `.gitignore`; extend as needed.
- Public repo stays clean: branches → PRs, no secrets, no trading data, no graph artifacts.

## Phasing (executed across daily loop iterations)

1. Mission saved + design committed. ← done
2. `risk_governor` + tests (safety gate).
3. Live portfolio cleanup + first new trades (supervised, this session).
4. `cli.py daily` bounded entrypoint + dry-run pre-flight.
5. Local data collector for backtesting.
6. Auto-applying learning loop.
7. launchd daily schedule + keep `/loop` on mission.
8. Agentic repo pivot + docs.
