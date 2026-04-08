# 🤖 Kalshi AI Trading Bot

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ryanfrigo/kalshi-ai-trading-bot?style=flat&color=yellow)](https://github.com/ryanfrigo/kalshi-ai-trading-bot/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ryanfrigo/kalshi-ai-trading-bot?style=flat&color=blue)](https://github.com/ryanfrigo/kalshi-ai-trading-bot/network)
[![GitHub Issues](https://img.shields.io/github/issues/ryanfrigo/kalshi-ai-trading-bot)](https://github.com/ryanfrigo/kalshi-ai-trading-bot/issues)
[![Last Commit](https://img.shields.io/github/last-commit/ryanfrigo/kalshi-ai-trading-bot)](https://github.com/ryanfrigo/kalshi-ai-trading-bot/commits/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An autonomous trading bot for [Kalshi](https://kalshi.com) prediction markets powered by a five-model AI ensemble.**

Five frontier LLMs debate every trade. The system only enters when they agree.

[Quick Start](#-quick-start) · [Features](#-features) · [How It Works](#-how-it-works) · [Configuration](#configuration-reference) · [Contributing](CONTRIBUTING.md) · [Runbook](docs/RUNBOOK.md) · [Testing](docs/TESTING.md) · [AI Collaboration](docs/AI_COLLABORATION.md) · [Kalshi API Docs](https://trading-api.readme.io/reference/getting-started)

</div>

---

> ⚠️ **Disclaimer — This is experimental software for educational and research purposes only.** Trading involves substantial risk of loss. Only trade with capital you can afford to lose. Past performance does not guarantee future results. This software is not financial advice. The authors are not responsible for any financial losses incurred through the use of this software.

> 📊 **Why Discipline Mode Exists** — Through extensive live trading on Kalshi across multiple strategies, we learned that trading without category enforcement and risk guardrails leads to significant losses. The most common mistakes: over-allocating to economic events (CPI, Fed decisions) with no real edge, and using aggressive position sizing. The consistently profitable edge we found was **NCAAB NO-side** trading (74% win rate, +10% ROI). This repo now ships with discipline systems enabled by default — category scoring, portfolio enforcement, and sane risk parameters.

---

## 🚀 Quick Start

**Three steps to get running in paper-trading mode (no real money):**

```bash
# 1. Clone and set up
git clone https://github.com/ryanfrigo/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot
python setup.py        # creates .venv, installs deps, checks config

# 2. Add your API keys
cp env.template .env   # then open .env and fill in KALSHI_API_KEY,
                       # XAI_API_KEY, and OPENROUTER_API_KEY

# 3. Run in disciplined mode (default — category scoring + guardrails)
python cli.py run --paper

# Or run the safe compounder (NO-side edge-based, most conservative)
python cli.py run --safe-compounder
```

Then open the live dashboard in another terminal:

```bash
python cli.py dashboard
```

> **Need API keys?**
> - Kalshi key + private key → [kalshi.com/account/settings](https://kalshi.com/account/settings) ([API docs](https://trading-api.readme.io/reference/getting-started))
> - xAI key → [console.x.ai](https://console.x.ai/)
> - OpenRouter key → [openrouter.ai](https://openrouter.ai/)

---

## ✅ Features

### Multi-Model AI Ensemble
- ✅ **Five frontier LLMs** collaborate on every decision — Grok-3, Claude 3.5 Sonnet, GPT-4o, Gemini Flash 1.5, DeepSeek R1
- ✅ **Role-based specialization** — each model plays a distinct analytical role (forecaster, bull, bear, risk manager, news analyst)
- ✅ **Consensus gating** — positions are skipped when models diverge beyond a configurable confidence threshold
- ✅ **Deterministic outputs** — temperature=0 for reproducible AI reasoning

### Trading Strategies
- ✅ **Directional trading** (50% of capital) — AI-predicted probability edge with Kelly Criterion sizing
- ✅ **Market making** (40%) — automated limit orders capturing bid-ask spread
- ✅ **Arbitrage detection** (10%) — cross-market opportunity scanning

### Risk Management
- ✅ **Fractional Kelly** position sizing (0.75x Kelly for volatility control)
- ✅ **Hard daily loss limit** — stops trading at 15% drawdown
- ✅ **Max drawdown circuit breaker** — halts at 50% portfolio drawdown
- ✅ **Sector concentration cap** — no more than 90% in any single category
- ✅ **Daily AI cost budget** — stops spending when API costs hit the configurable daily limit (default: $10/day)

### Dynamic Exit Strategies
- ✅ Trailing take-profit at 20% gain
- ✅ Stop-loss at 15% per position
- ✅ Confidence-decay exits when AI conviction drops
- ✅ Time-based exits (10-day max hold)
- ✅ Volatility-adjusted thresholds

### Observability
- ✅ **Real-time Streamlit dashboard** — portfolio value, positions, P&L, AI decision logs
- ✅ **Paper trading mode** — simulate trades without real orders; track outcomes on settled markets
- ✅ **SQLite telemetry** — every trade, AI decision, and cost metric logged locally
- ✅ **Unified CLI** — `run`, `dashboard`, `status`, `health`, `backtest` commands

---

## 🧠 How It Works

The bot runs a four-stage pipeline on a continuous loop:

```
  INGEST               DECIDE (5-Model Ensemble)    EXECUTE       TRACK
 --------             ─────────────────────────    ---------    --------
                      ┌─────────────────────────┐
  Kalshi    ────────► │  Grok-β  (Forecaster 30%)│
  REST API            ├─────────────────────────┤
                      │  Claude  (News Analyst 20%)│
  WebSocket ────────► ├─────────────────────────┤
  Stream              │  GPT-4o  (Bull Case   20%)│  ──► Kalshi  ──► P&L
                      ├─────────────────────────┤      Order       Win Rate
  RSS / News ───────► │  Gemini  (Bear Case   15%)│      Router     Sharpe
  Feeds               ├─────────────────────────┤               Drawdown
                      │  DeepSeek(Risk Mgr    15%)│      Kelly    Cost
  Volume &  ────────► └─────────────────────────┘      Sizing   Budget
  Price Data             Debate → Consensus
                         Confidence Calibration
```

### Stage 1 — Ingest
Market data, order book snapshots, and news feeds are pulled via the Kalshi REST API and WebSocket stream. RSS feeds from financial news sources supplement the signal.

### Stage 2 — Decide (Multi-Model Ensemble)
Each of the five models analyzes the incoming data from its assigned perspective and returns a probability estimate + confidence score. The ensemble combines weighted votes:

| Model | Role | Weight |
|---|---|---|
| Grok-3 (xAI) | Lead Forecaster | 30% |
| Claude 3.5 Sonnet (OpenRouter) | News Analyst | 20% |
| GPT-4o (OpenRouter) | Bull Researcher | 20% |
| Gemini Flash 1.5 (OpenRouter) | Bear Researcher | 15% |
| DeepSeek R1 (OpenRouter) | Risk Manager | 15% |

If the weighted confidence falls below `min_confidence_to_trade` (default: 0.50), the opportunity is skipped. If models disagree significantly, position size is automatically reduced.

### Stage 3 — Execute
Qualifying trades are sized using the **Kelly Criterion** (fractional 0.75x) and routed through Kalshi's order API. Market-making orders are placed symmetrically around the mid-price.

### Stage 4 — Track
Every decision is written to a local SQLite database. The dashboard and `--stats` commands surface cumulative P&L, win rate, Sharpe ratio, and per-strategy breakdowns in real time.

---

## 📦 Installation

### Prerequisites

- Python 3.12 or later
- A [Kalshi](https://kalshi.com) account with API access ([API docs](https://trading-api.readme.io/reference/getting-started))
- An [xAI](https://console.x.ai/) API key (Grok-4)
- An [OpenRouter](https://openrouter.ai/) API key (Claude, GPT-4o, Gemini, DeepSeek)

### Automated Setup (Recommended)

```bash
git clone https://github.com/ryanfrigo/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot
python setup.py
```

The setup script will:
- ✅ Check Python version compatibility
- ✅ Create virtual environment
- ✅ Install all dependencies (with Python 3.14 compatibility handling)
- ✅ Test that the dashboard can run
- ✅ Print troubleshooting guidance

### Manual Installation

```bash
git clone https://github.com/ryanfrigo/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate          # Windows

# Python 3.14 users only:
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

pip install -r requirements.txt
```

### Configuration

```bash
cp env.template .env   # fill in your keys
```

| Variable | Description |
|---|---|
| `KALSHI_API_KEY` | Your Kalshi API key ID |
| `XAI_API_KEY` | xAI key for Grok-4 |
| `OPENROUTER_API_KEY` | OpenRouter key (Claude, GPT-4o, Gemini, DeepSeek) |
| `OPENAI_API_KEY` | Optional fallback |

Place your Kalshi private key as `kalshi_private_key` (no extension) in the project root. Download from [Kalshi Settings → API](https://kalshi.com/account/settings). This file is git-ignored.

### Initialize the Database

```bash
python -m src.utils.database
```

> ⚠️ Use `-m` flag — running `python src/utils/database.py` directly will fail with a module import error.

---

## 🖥️ Running

```bash
# Paper trading (no real orders — safe to test)
python cli.py run --paper

# Live trading (real money)
python cli.py run --live

# Launch monitoring dashboard
python cli.py dashboard

# Check portfolio balance and open positions
python cli.py status

# Verify all API connections
python cli.py health
```

Or invoke the bot script directly:

```bash
python beast_mode_bot.py              # Paper trading
python beast_mode_bot.py --live       # Live trading
python beast_mode_bot.py --dashboard  # Dashboard mode
```

---

## 📊 Paper Trading Dashboard

Simulate trades without risking real money. Every signal is logged to SQLite and a static HTML dashboard renders cumulative P&L, win rate, and per-signal details after markets settle.

```bash
# Scan markets and log signals
python paper_trader.py

# Continuous scanning every 15 minutes
python paper_trader.py --loop --interval 900

# Settle markets and update outcomes
python paper_trader.py --settle

# Regenerate HTML dashboard
python paper_trader.py --dashboard

# Print stats to terminal
python paper_trader.py --stats
```

The dashboard writes to `docs/paper_dashboard.html` — open locally or host via GitHub Pages.

---

## 🗂️ Project Structure

```
kalshi-ai-trading-bot/
├── beast_mode_bot.py          # Main bot entry point
├── cli.py                     # Unified CLI: run, dashboard, status, health, backtest
├── paper_trader.py            # Paper trading signal tracker
├── pyproject.toml             # PEP 621 project metadata
├── requirements.txt           # Pinned dependencies
├── env.template               # Environment variable template
│
├── src/
│   ├── agents/                # Multi-model ensemble (forecaster, bull/bear, risk, trader)
│   ├── clients/               # API clients (Kalshi, xAI, OpenRouter, WebSocket)
│   ├── config/                # Settings and trading parameters
│   ├── data/                  # News aggregation and sentiment analysis
│   ├── events/                # Async event bus for real-time streaming
│   ├── jobs/                  # Core pipeline: ingest, decide, execute, track, evaluate
│   ├── strategies/            # Market making, portfolio optimization, quick flip
│   └── utils/                 # Database, logging, prompts, risk helpers
│
├── scripts/                   # Utility and diagnostic scripts
├── docs/                      # Additional documentation + paper dashboard HTML
└── tests/                     # Pytest test suite
```

---

## ⚙️ Configuration Reference

All trading parameters live in `src/config/settings.py`:

```python
# Position sizing
max_position_size_pct  = 5.0     # Max 5% of balance per position
max_positions          = 15      # Up to 15 concurrent positions
kelly_fraction         = 0.75    # Fractional Kelly multiplier

# Market filtering
min_volume             = 200     # Minimum contract volume
max_time_to_expiry_days = 30     # Trade contracts up to 30 days out
min_confidence_to_trade = 0.50   # Minimum ensemble confidence to enter

# AI settings
primary_model          = "grok-4"
ai_temperature         = 0       # Deterministic outputs
ai_max_tokens          = 8000

# Risk management
max_daily_loss_pct     = 15.0    # Hard daily loss limit
daily_ai_cost_limit    = 10.0    # Max daily AI API spend (USD) — default $10/day
```

> **💸 Controlling AI spend (important for `movement_prediction` / xAI costs)**
>
> The bot checks daily spend limits **before every xAI API call** — including `search()` and all market analysis calls. Once the limit is reached, all AI calls are skipped until the next calendar day.
>
> Key knobs:
> - `DAILY_AI_COST_LIMIT` env var (or `daily_ai_cost_limit` in `TradingConfig`) — max USD per day. **Default: $10.** Raise it only when you're comfortable with the spend. Example: `export DAILY_AI_COST_LIMIT=25`
> - `scan_interval_seconds` — how often the bot scans markets. Lower = more AI calls per hour. Default: 60 seconds.
> - `max_analyses_per_market_per_day` — cap on AI analyses per individual market per day. Default: 4.
>
> The `movement_prediction` strategy runs AI analysis on **every scan cycle** for all candidate markets. If you have many active markets and a short scan interval, costs add up fast. Reduce `scan_interval_seconds` (e.g. `120`) or lower `max_analyses_per_market_per_day` (e.g. `2`) to cut frequency.

The ensemble configuration (model roster, weights, debate settings) lives in `EnsembleConfig` in the same file.

> **⚠️ AI Model Names** — xAI periodically renames Grok models. The default is currently set to `grok-3`. If you see a `model not found` error, update `primary_model` in `TradingConfig` and the `"grok-3"` key in `EnsembleConfig.models` to match the latest model name from [console.x.ai](https://console.x.ai/). You can also override via environment variable: set `PRIMARY_MODEL=grok-3-mini` (or any valid model ID) in your `.env` file.

---

## 📈 Performance Tracking

Every trade, AI decision, and cost metric is recorded to `trading_system.db` (local SQLite). Use the dashboard or scripts in `scripts/` to review:

- Cumulative P&L and win rate
- Sharpe ratio and maximum drawdown
- AI confidence calibration
- Cost per trade and daily API budget utilization
- Per-strategy breakdowns (directional vs. market making)

---

## 🛠️ Development

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

1. Create a module in `src/strategies/`
2. Wire it into `src/strategies/unified_trading_system.py`
3. Set allocation percentage in `src/config/settings.py`
4. Add tests in `tests/`

---

## 🔧 Troubleshooting

<details>
<summary><strong>Bot not placing live trades despite --live flag</strong></summary>

Check logs for the mode confirmation string:

```bash
grep -i "live trading\|paper trading\|LIVE ORDER\|PAPER TRADE" logs/trading_system.log | tail -20
```

- `"LIVE TRADING MODE ENABLED"` → correct
- `"Paper trading mode"` → still in paper mode; verify API key has TRADING permissions in [Kalshi Settings](https://kalshi.com/account/settings)

</details>

<details>
<summary><strong>Dashboard won't launch / import errors</strong></summary>

Import errors in VS Code are IDE linter warnings, not runtime errors.

```bash
# Fix: activate venv, then run from project root
source .venv/bin/activate
python beast_mode_dashboard.py
```

Set VS Code Python interpreter to `.venv/bin/python` via `Cmd+Shift+P → Python: Select Interpreter`.

</details>

<details>
<summary><strong>Python 3.14 PyO3 compatibility error</strong></summary>

```bash
# Quick fix
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
pip install -r requirements.txt

# Recommended: use Python 3.13
pyenv install 3.13.1 && pyenv local 3.13.1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

</details>

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

**Quick steps:**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes, add tests, run `pytest` and `black`
4. Commit with [conventional commit](https://www.conventionalcommits.org/) format: `feat: add new model weight config`
5. Open a Pull Request

**Good first issues:** look for the [`good first issue`](https://github.com/ryanfrigo/kalshi-ai-trading-bot/issues?q=label%3A%22good+first+issue%22) label.

---

---

## 🛡️ Trading Modes

The bot now supports three distinct trading modes. **Disciplined is the default.**

### 1. Disciplined Mode (DEFAULT) — `python cli.py run`

The safe, category-aware mode. Runs the AI ensemble but with guardrails:

```bash
python cli.py run --paper        # Paper trading (safe, no real money)
python cli.py run --live         # Live trading with discipline enforced
python cli.py run --disciplined  # Explicit flag (same as default)
```

Settings enforced:
- Max drawdown: **15%** (vs 50% in beast mode)
- Min confidence: **65%** (vs 50% in beast mode)
- Max position size: **3%** of portfolio
- Max sector concentration: **30%**
- Kelly fraction: **0.25** (quarter-Kelly)
- Category scoring active — blocks categories with score < 30

### 2. Safe Compounder — `python cli.py run --safe-compounder`

The most conservative and historically validated strategy:

```bash
python cli.py run --safe-compounder           # Dry run (shows opportunities)
python cli.py run --safe-compounder --live    # Live execution
```

Strategy rules:
- **NO side ONLY** — never buys YES
- YES last price must be ≤ 20¢ (near-certain NO outcome)
- NO ask must be > 80¢
- Edge (EV - price) must be > 5¢
- Places resting maker orders at `lowest_ask - 1¢` (near-zero fees)
- Max 10% of portfolio per position (half-Kelly sizing)
- Skips all sports, entertainment, and "mention" markets

This strategy is the closest thing to a pure edge play on Kalshi.

### 3. Beast Mode — `python cli.py run --beast`

> ⚠️ **Not recommended.** Aggressive settings with no category guardrails have historically led to significant losses in live prediction market trading.

The original aggressive mode with minimal guardrails. Available for comparison/research:

```bash
python cli.py run --beast --paper  # Only run beast mode in paper trading
```

Aggressive settings:
- Max drawdown: 50%
- Min confidence: 50%
- Max position: 5%
- Sector cap: 90%
- Kelly fraction: 0.75

---

## 📊 Category Scoring System

The category scorer evaluates each Kalshi market category on a 0-100 scale and enforces allocation limits.

### Scoring Formula

| Factor | Weight | Description |
|--------|--------|-------------|
| ROI | 40% | Average return on investment across all trades |
| Recent Trend | 25% | Direction of last 10 trades (recency-weighted) |
| Sample Size | 20% | More data = more confidence in the score |
| Win Rate | 15% | Percentage of winning trades |

### Allocation Tiers

| Score Range | Max Position Size | Status |
|-------------|-------------------|--------|
| 80-100 | 20% of portfolio | STRONG ✅ |
| 60-79 | 10% of portfolio | GOOD 🟢 |
| 40-59 | 5% of portfolio | WEAK 🟡 |
| 20-39 | 2% of portfolio | POOR 🟠 |
| 0-19 | 0% (blocked) | BLOCKED 🚫 |

**Categories scoring below 30 are hard-blocked** — the bot will not enter any trade in those categories regardless of AI confidence.

### Check Current Scores

```bash
python cli.py scores
```

Example output:
```
======================================================================
  CATEGORY SCORES
  Category           Score     WR      ROI  Trades   Alloc  Status
  ------------------ ------ ------ -------- ------- ------ ----------
  NCAAB               72.3   74%   +10.0%      50    10%   GOOD 🟢
  NBA                 41.2   52%    +1.5%      28     5%   WEAK 🟡
  POLITICS            31.0   48%    -8.0%      15     2%   MARGINAL 🔴
  CPI                  8.4   25%   -65.0%      20     0%   BLOCKED 🚫
  FED                 12.1   32%   -40.0%      25     0%   BLOCKED 🚫
  ECON_MACRO          10.5   30%   -55.0%      40     0%   BLOCKED 🚫
======================================================================
```

### Real Trading Data (Seeded)

The scorer is pre-seeded with real historical data:
- **NCAAB**: 74% win rate, +10% ROI → score ~72 → allowed at 10% allocation
- **ECON/CPI**: 25% win rate, -65% ROI → score ~8 → **blocked**
- **FED**: 32% win rate, -40% ROI → score ~12 → **blocked**

---

## 📈 Trade History & Analysis

```bash
python cli.py history           # Last 50 trades with category breakdown
python cli.py history --limit 100  # Last 100 trades
```

---

## 🧠 Lessons Learned

After extensive live trading across multiple strategies, here's what the data taught:

### 1. Category discipline > AI confidence

The AI ensemble can be 80% confident on a CPI trade and still be wrong. Market-implied probabilities on economic releases are already efficient — there's no structural edge for a retail bot. The bot was trading these with the same aggression as sports markets where it had actual edge.

**Fix:** Category scoring now hard-blocks economic markets until they prove a positive edge over ≥5 trades.

### 2. Kelly fraction matters enormously

A Kelly fraction of 0.75 sounds reasonable. It's not — it compounds losses catastrophically. At 0.75x Kelly with a 45% win rate, you can lose 80% of capital in a standard drawdown scenario.

**Fix:** Default is now 0.25x Kelly (quarter-Kelly), which is more conservative than most professional traders use.

### 3. Max drawdown must have teeth

A 50% drawdown limit means you can lose half your money before the bot stops. That's not a limit — it's a suggestion. A 15% limit forces the bot to stop while you still have capital to analyze and adjust.

**Fix:** 15% max drawdown, with the circuit breaker actually stopping trades (not just logging a warning).

### 4. Sector concentration = correlated losses

When 90% of capital is in economic categories and there's a Fed meeting, everything moves together. Correlated losses compound faster than diversified losses.

**Fix:** 30% sector cap means no single category can dominate the portfolio.

### 5. Consistency > frequency

The bot was scanning every 30 seconds and trading everything it found. More trades with no edge = faster path to zero.

**Fix:** 60-second scan interval. Trades only when confidence ≥ 65% AND category score ≥ 30.

---

## 📚 Resources

- [Kalshi Trading API Docs](https://trading-api.readme.io/reference/getting-started)
- [Kalshi API Authentication](https://trading-api.readme.io/reference/authentication)
- [Kalshi Markets Overview](https://kalshi.com/markets)
- [OpenRouter Model Catalog](https://openrouter.ai/models)
- [xAI API (Grok)](https://console.x.ai/)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**If this project is useful to you, consider giving it a ⭐**

Made with ❤️ for the Kalshi trading community

</div>
