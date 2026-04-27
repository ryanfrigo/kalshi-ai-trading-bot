# Kalshi AI Trading Bot

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ryanfrigo/kalshi-ai-trading-bot?style=flat&color=yellow)](https://github.com/ryanfrigo/kalshi-ai-trading-bot/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ryanfrigo/kalshi-ai-trading-bot?style=flat&color=blue)](https://github.com/ryanfrigo/kalshi-ai-trading-bot/network)
[![GitHub Issues](https://img.shields.io/github/issues/ryanfrigo/kalshi-ai-trading-bot)](https://github.com/ryanfrigo/kalshi-ai-trading-bot/issues)

**A toolkit for building automated trading strategies on [Kalshi](https://kalshi.com) prediction markets.**

Signed Kalshi API client, market-data ingestion, position tracking, SQLite telemetry, a Streamlit dashboard, and a pluggable LLM client (any model on OpenRouter). Three example strategies ship with the repo as starting points — fork them, replace them, or write your own from scratch.

[Quick Start](#quick-start) · [What's Included](#whats-included) · [Example Strategies](#example-strategies) · [Configuration](#configuration) · [Contributing](CONTRIBUTING.md) · [Kalshi API Docs](https://trading-api.readme.io/reference/getting-started)

</div>

---

> **Read this before running with real money.** No strategy in this repo is guaranteed to make money. The examples lose money on certain markets. Trading prediction markets is hard, the edges are small, and what worked last quarter may not work this quarter. This is a toolkit, not a turnkey bot. Read the code, understand what it does, and tune it for the markets you care about. The authors are not responsible for losses you incur using this software.

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/ryanfrigo/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot
python setup.py        # creates .venv, installs deps

# 2. Add your API keys
cp env.template .env
# then open .env and fill in KALSHI_API_KEY and OPENROUTER_API_KEY

# 3. Verify connectivity
python cli.py health

# 4. Run an example strategy in paper mode
python cli.py run --paper                  # AI directional (LLM-driven)
python cli.py run --safe-compounder        # Edge-based NO-side, no LLM
```

Open the dashboard in another terminal:

```bash
python cli.py dashboard
```

> **Need API keys?**
> - Kalshi key + private key → [kalshi.com/account/settings](https://kalshi.com/account/settings)
> - OpenRouter key → [openrouter.ai](https://openrouter.ai/)

---

## What's Included

This repo gives you the building blocks. The example strategies use them — your own strategies can too.

| Component | What it does | Where it lives |
|---|---|---|
| **Kalshi client** | Authenticated REST + WebSocket client (RSA signing, retries, rate-limit handling) | `src/clients/kalshi_client.py` |
| **Market ingestion** | Pulls the full tradeable universe via the Events API, persists to SQLite | `src/jobs/ingest.py` |
| **Position tracking** | Stop-loss, take-profit, time-based, and resolution-based exits with real Kalshi sell orders | `src/jobs/track.py` |
| **LLM client** | Single OpenRouter API key, swap models with one config line, fallback chain on errors, persistent daily-cost tracker | `src/clients/openrouter_client.py`, `src/clients/xai_client.py` |
| **SQLite telemetry** | Every trade, AI decision, and cost metric logged locally | `src/utils/database.py` |
| **Streamlit dashboard** | Real-time portfolio, positions, P&L, decision logs | `beast_mode_dashboard.py` |
| **Paper trading** | Log signals against settled markets without sending orders | `paper_trader.py` |
| **CLI** | `run`, `dashboard`, `status`, `health`, `scores`, `history`, `close-all` | `cli.py` |
| **Risk helpers** | Kelly sizing, stop-loss math, drawdown circuit breaker | `src/utils/`, `src/strategies/` |

The repo also ships scaffolding for things that aren't fully wired — multi-agent debate runners in `src/agents/`, sentiment analyzer in `src/data/`, etc. Treat them as starting points if you want to extend them.

---

## Example Strategies

Three strategies ship with the repo. **None of them is "the right answer."** They exist so you can run something end-to-end and see how the pieces connect, then fork the one closest to what you want to build.

### 1. AI Directional — `python cli.py run`

The default. For each candidate market, it calls a single LLM via OpenRouter (with a fallback chain on errors) to score directional confidence, then sizes positions with fractional Kelly and applies category/sector guardrails.

> **It is not a "5-model ensemble"** despite earlier README claims. One model is called per decision. The fallback chain only triggers on errors. The agents/ directory contains scaffolding for real parallel multi-model voting, but it's not wired into the live trading path. If you want a real ensemble, fork `src/jobs/decide.py` and build it.

```bash
python cli.py run --paper          # paper trading
python cli.py run --live           # live trading (real money)
```

Defaults: 15% max drawdown, 45% min confidence, 3% max position size, 30% max sector concentration, quarter-Kelly. All configurable in `src/config/settings.py`.

### 2. Safe Compounder — `python cli.py run --safe-compounder`

Pure edge-based math, no LLM required. Scans every active Kalshi market for NO-side asks above a price threshold with a positive expected-value edge, then places resting maker orders one cent below the ask.

```bash
python cli.py run --safe-compounder              # dry-run preview
python cli.py run --safe-compounder --live       # live execution

# Run continuously instead of one cycle and exit:
python cli.py run --safe-compounder --live --loop --interval 300
```

Rules: NO side only, YES last ≤ 20¢, NO ask > 80¢, edge > 5¢, max 10%/position, skips sports/entertainment/"mention" markets.

### 3. Beast Mode — `python cli.py run --beast`

Aggressive settings with no category guardrails. Available for comparison and experimentation — **not recommended for live trading**. Running this with real money historically led to significant losses on this repo.

---

## Stopping Cleanly

Ctrl-C sends SIGINT and triggers graceful shutdown — the bot finishes the in-flight cycle, logs, and exits. Open positions remain on Kalshi until they resolve.

If you want to **liquidate everything** before stepping away:

```bash
# 1. Stop the bot
Ctrl-C

# 2. Place limit sells at the current best bid for every open position
python cli.py close-all              # dry-run preview
python cli.py close-all --live       # actually send orders

# 3. Verify
python cli.py status
```

`close-all` queries Kalshi directly (not the local DB), so it works even when local state is stale. Sells are limit-priced, so they may rest unfilled on thin books — check Kalshi or `cli.py status` after a minute.

---

## Installation

### Prerequisites

- Python 3.12 or later
- A [Kalshi](https://kalshi.com) account with API access ([API docs](https://trading-api.readme.io/reference/getting-started))
- An [OpenRouter](https://openrouter.ai/) API key (only needed for the AI directional strategy)

### Automated Setup

```bash
git clone https://github.com/ryanfrigo/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot
python setup.py
```

Creates a virtual env, installs dependencies, and prints next steps.

### Manual Setup

```bash
git clone https://github.com/ryanfrigo/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### Configuration

```bash
cp env.template .env
# then edit .env with your keys
```

| Variable | Description |
|---|---|
| `KALSHI_API_KEY` | Your Kalshi API key ID |
| `OPENROUTER_API_KEY` | OpenRouter key (only for AI directional strategy) |

Place your Kalshi private key as `kalshi_private_key` (no extension) in the project root. Download it from [Kalshi Settings → API](https://kalshi.com/account/settings). It's git-ignored.

Verify everything is wired:

```bash
python cli.py health
```

---

## Configuration

All trading parameters live in `src/config/settings.py`. The most useful knobs:

```python
# Position sizing
max_position_size_pct  = 3.0     # Max 3% of balance per position
max_positions          = 10      # Max concurrent positions
kelly_fraction         = 0.25    # Quarter-Kelly (conservative)

# Market filtering
min_volume             = 500     # Minimum contract volume
max_time_to_expiry_days = 14     # How far out to trade
min_confidence_to_trade = 0.45   # Minimum AI confidence to enter

# LLM (OpenRouter)
primary_model          = "anthropic/claude-sonnet-4.5"
ai_temperature         = 0       # Deterministic
ai_max_tokens          = 8000

# Risk management
max_daily_loss_pct     = 10.0    # Daily loss circuit breaker
max_drawdown           = 0.15    # Portfolio drawdown halt
daily_ai_cost_limit    = 10.0    # Max daily LLM spend in USD
```

**Swapping models:** change `primary_model` to any slug from [openrouter.ai/models](https://openrouter.ai/models). The fallback chain in `src/clients/openrouter_client.py` controls what happens when the primary errors.

**Controlling LLM spend:** the bot checks the daily limit before every API call and skips trading until the next calendar day once exhausted. Set `DAILY_AI_COST_LIMIT` in `.env` to override.

---

## Project Structure

```
kalshi-ai-trading-bot/
├── beast_mode_bot.py          # Example AI directional bot — main loop orchestration
├── cli.py                     # Unified CLI: run, dashboard, status, health, close-all, scores, history
├── paper_trader.py            # Paper-trading signal logger + static dashboard
├── setup.py                   # Bootstrap script
├── env.template               # Environment variable template
│
├── src/
│   ├── agents/                # UNWIRED scaffolding for multi-agent debate (fork to use)
│   ├── clients/               # Kalshi, OpenRouter, WebSocket clients
│   ├── config/                # Settings and trading parameters
│   ├── data/                  # News + sentiment helpers (optional)
│   ├── events/                # Async event bus
│   ├── jobs/                  # ingest, decide, execute, track, evaluate
│   ├── strategies/            # Safe compounder, category scorer, portfolio enforcer
│   └── utils/                 # Database, logging, prompts, risk helpers
│
├── scripts/                   # Diagnostic and utility scripts
├── docs/                      # Additional docs + paper-trading dashboard HTML
└── tests/                     # Pytest suite
```

---

## Paper Trading

Simulate trades without sending real orders. Every signal is logged to SQLite and a static HTML dashboard renders cumulative P&L after markets settle.

```bash
python paper_trader.py                            # one scan
python paper_trader.py --loop --interval 900      # continuous, every 15m
python paper_trader.py --settle                   # update outcomes for resolved markets
python paper_trader.py --dashboard                # regenerate HTML
python paper_trader.py --stats                    # print stats
```

Output goes to `docs/paper_dashboard.html`.

---

## Category Scoring (used by AI Directional)

The category scorer evaluates each Kalshi market category on a 0-100 scale based on historical ROI, win rate, recent trend, and sample size. Allocation per category is gated by score.

| Score | Max Position | Status |
|---|---|---|
| 80–100 | 20% | STRONG |
| 60–79 | 10% | GOOD |
| 40–59 | 5% | WEAK |
| 20–39 | 2% | POOR |
| 0–19 | 0 | BLOCKED |

```bash
python cli.py scores
```

This is one heuristic for category-level risk control. If it doesn't fit your strategy, ignore it — it's only used by the AI directional path.

---

## Performance Tracking

Every trade, AI decision, and cost metric is recorded to `trading_system.db` (local SQLite). Inspect via the dashboard or:

```bash
python cli.py history                # Last 50 trades
python cli.py history --limit 100    # Last 100
python cli.py status                 # Live balance + open positions from Kalshi
```

---

## Development

### Running Tests

```bash
pytest tests/          # full suite
pytest tests/ -v       # verbose
pytest --cov=src       # with coverage
```

### Code Quality

```bash
black src/ tests/ cli.py beast_mode_bot.py
isort src/ tests/ cli.py beast_mode_bot.py
mypy src/
```

### Adding a New Strategy

1. Create a module under `src/strategies/`
2. Wire it into a CLI flag in `cli.py` (or invoke it directly)
3. Use the `KalshiClient` for orders/positions and `DatabaseManager` for state
4. Add tests under `tests/`

---

## Troubleshooting

<details>
<summary><strong>Health check fails with HTTP 401</strong></summary>

A 401 from Kalshi almost always means one of three things:

1. `KALSHI_API_KEY` in `.env` doesn't match the API key ID shown in Kalshi
2. The private key file (`kalshi_private_key`) is the wrong key for that API key, or its path is wrong
3. The key was created on Kalshi's demo environment but you're pointing at production (or vice versa)

Re-download the key pair from Kalshi and verify both values point to the matching pair. The health check will print this hint when it detects a 401.

</details>

<details>
<summary><strong>"Shutdown signal received" without pressing Ctrl-C</strong></summary>

The bot now logs which signal arrived (SIGINT, SIGTERM, or SIGHUP). If you see SIGTERM or SIGHUP without sending it yourself, common causes:

- Parent shell closed (run inside `tmux`, `screen`, or with `nohup`)
- A cloud platform / systemd / launchd timeout
- Another shell sent `kill <pid>`
- An OOM killer warning before SIGKILL

The bot did not kill itself — something external ended the process.

</details>

<details>
<summary><strong>Bot ran for weeks but placed no positions</strong></summary>

This is expected behavior, not a bug. The example strategies are conservative by design:

- **AI Directional** requires confidence ≥ 45%, category score ≥ 30, and is gated by drawdown / sector caps. On many days, no markets clear all four filters.
- **Safe Compounder** requires NO ask > 80¢ AND edge > 5¢. Most NO-side markets don't meet both.

If you want more activity, lower the thresholds in `src/config/settings.py` (or the relevant strategy file) — but that means taking lower-edge bets. Or write your own strategy that targets the markets you actually have an edge on. This repo is a toolkit; the example thresholds are starting points.

</details>

<details>
<summary><strong>"no such table: positions" error on fresh install</strong></summary>

The DB file isn't committed; it's created at runtime. The bot auto-initializes on startup, but you can do it manually:

```bash
python -m src.utils.database
```

Use `-m` — running `python src/utils/database.py` directly fails with an import error.

</details>

<details>
<summary><strong>AdGuard (macOS) blocks dependency downloads</strong></summary>

If AdGuard is running as a system-level proxy, `pip install` may time out during setup. Disable AdGuard at the system level for the install, then re-enable it. AdGuard as a browser extension is fine.

</details>

<details>
<summary><strong>Bot not placing live trades despite --live</strong></summary>

```bash
grep -i "live trading\|paper trading\|LIVE ORDER" logs/trading_system.log | tail -20
```

If you see "Paper trading mode" the flag isn't taking effect. Verify the API key has trading permissions in [Kalshi Settings](https://kalshi.com/account/settings).

</details>

<details>
<summary><strong>Model not found / OpenRouter API errors</strong></summary>

Model names on OpenRouter change. Update `primary_model` in `src/config/settings.py` with a current slug from [openrouter.ai/models](https://openrouter.ai/models), or set `PRIMARY_MODEL` in `.env`.

</details>

<details>
<summary><strong>Bot only seeing "KXMVE" tickers</strong></summary>

The Kalshi `/markets` endpoint returns parlay tickers; real markets live under the Events API. The ingestion pipeline already uses Events with nested markets. If you see only `KXMVE*`, check API permissions and run `python cli.py health`.

</details>

<details>
<summary><strong>Python 3.14 PyO3 compatibility error</strong></summary>

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
pip install -r requirements.txt
```

Or use Python 3.13:

```bash
pyenv install 3.13.1 && pyenv local 3.13.1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

</details>

---

## Lessons From Live Trading

These are observations from running the example strategies with real money on Kalshi. They informed the defaults shipped here. They are **not** universal trading wisdom — they're notes from one set of experiments.

**1. Category discipline mattered more than AI confidence.** The LLM could be 80% confident on a CPI trade and still be wrong. Market-implied probabilities on highly-watched economic releases are already efficient.

**2. Kelly fraction matters enormously.** Three-quarter Kelly compounds losses catastrophically on a 45% win-rate strategy. Quarter-Kelly is what the example strategies use.

**3. A 50% drawdown limit isn't a limit.** The default is 15% with the circuit breaker actually halting trades, not just logging.

**4. Sector concentration creates correlated losses.** When 90% of capital is in economic categories on a Fed day, everything moves together. The default cap is 30% per category.

**5. More trades without edge is faster path to zero.** The default scan interval is 60 seconds, not 30, and trades are gated by both confidence and category score.

Your edge is probably somewhere else. Use the toolkit to find it.

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

```bash
# 1. Fork
# 2. Create a feature branch
git checkout -b feature/your-feature
# 3. Make changes, add tests, run pytest and black
# 4. Commit using conventional commits (feat:, fix:, refactor:)
# 5. Open a PR
```

---

## Resources

- [Kalshi Trading API](https://trading-api.readme.io/reference/getting-started)
- [Kalshi API Authentication](https://trading-api.readme.io/reference/authentication)
- [Kalshi Markets](https://kalshi.com/markets)
- [OpenRouter Model Catalog](https://openrouter.ai/models)

---

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

If this is useful to you, a star helps others find it.

</div>
