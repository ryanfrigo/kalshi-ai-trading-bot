# Kalshi AI Trading Bot

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Multi-model AI trading bot for Kalshi prediction markets.**

An autonomous trading system that combines a five-model AI ensemble, portfolio
optimization, market making, and dynamic exit strategies to find and exploit
edges on [Kalshi](https://kalshi.com) event contracts.

> **Disclaimer -- This is experimental software for educational and research
> purposes only.** Trading involves substantial risk of loss. Only trade with
> capital you can afford to lose. Past performance does not guarantee future
> results. This software is not financial advice. Use at your own risk. The
> authors are not responsible for any financial losses incurred through the use
> of this software.

---

## Architecture

```
                           Kalshi AI Trading Bot

  INGEST               DECIDE (Multi-Agent)         EXECUTE          TRACK
 --------             ----------------------        ---------       --------
                      +--------------------+
  Kalshi    --------> |  Grok-4            |
  REST API            |  (Forecaster, 30%) |
                      +--------------------+
  WebSocket --------> +--------------------+
  Stream              |  Claude Sonnet 4   |
                      |  (News Analyst,20%)|        Kalshi        Portfolio
  RSS/News  --------> +--------------------+ -----> Order  -----> P&L, Win
  Feeds               |  GPT-4o            |        Router        Rate,
                      |  (Bull Case, 20%)  |                      Sharpe,
  Volume &  --------> +--------------------+        Kelly         Drawdown
  Price Data          |  Gemini 2.5 Flash  |        Criterion     Tracking
                      |  (Bear Case, 15%)  |        Sizing
                      +--------------------+                      Cost &
                      |  DeepSeek R1       |        Risk          Budget
                      |  (Risk Mgr, 15%)   |        Parity        Monitor
                      +--------------------+        Alloc

                        Debate & Consensus
                        Confidence Calibration
```

---

## Features

### Multi-Model Ensemble

Five frontier LLMs collaborate on every trading decision. Each model is assigned
a distinct analytical role and weighted vote. When model opinions diverge beyond
a configurable threshold the system reduces position size or skips the trade.

| Model | Provider | Role | Weight |
|---|---|---|---|
| Grok-4 | xAI | Lead Forecaster | 30% |
| Claude Sonnet 4 | OpenRouter | News Analyst | 20% |
| GPT-4o | OpenRouter | Bull Researcher | 20% |
| Gemini 2.5 Flash | OpenRouter | Bear Researcher | 15% |
| DeepSeek R1 | OpenRouter | Risk Manager | 15% |

### Trading Strategies

| Strategy | Allocation | Description |
|---|---|---|
| Directional Trading | 50% | AI-predicted probability edge with Kelly sizing |
| Market Making | 40% | Automated limit orders capturing bid-ask spreads |
| Arbitrage Detection | 10% | Cross-market opportunity scanning |

### Portfolio Optimization

- **Kelly Criterion** position sizing with fractional Kelly (0.75x) for volatility control
- **Risk parity** allocation across concurrent positions
- **Dynamic rebalancing** every six hours
- Hard limits on daily loss (15%), max drawdown (50%), and sector concentration (90%)

### Dynamic Exit Strategies

- Trailing take-profit at 20% gain
- Stop-loss at 15% drawdown per position
- Confidence-decay exits when AI conviction drops
- Time-based exits with a 10-day maximum hold
- Volatility-adjusted thresholds

### Real-Time Dashboard

A Streamlit web dashboard provides live views of:

- Portfolio value and balance
- Open positions with entry prices and P&L
- AI decision logs and confidence scores
- Cost monitoring and daily budget utilization
- Strategy-level performance breakdown

---

## Quick Start

### Prerequisites

- Python 3.12 or later
- A [Kalshi](https://kalshi.com) account with API access
- An [xAI](https://console.x.ai/) API key (Grok-4)
- An [OpenRouter](https://openrouter.ai/) API key (for the remaining four models)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as an editable package (includes the kalshi-bot CLI)
pip install -e ".[dev,dashboard]"
```

### Configuration

```bash
# Copy the template and fill in your keys
cp env.template .env
```

Required variables in `.env`:

| Variable | Description |
|---|---|
| `KALSHI_API_KEY` | Your Kalshi API key ID |
| `XAI_API_KEY` | xAI API key for Grok-4 |
| `OPENROUTER_API_KEY` | OpenRouter key for ensemble models |
| `OPENAI_API_KEY` | OpenAI key (optional fallback) |

You must also place your Kalshi private key file at `kalshi_private_key` in the
project root (this file is git-ignored).

### Running

```bash
# Paper trading (default -- no real orders)
python cli.py run --paper

# Live trading (real money)
python cli.py run --live

# Launch the monitoring dashboard
python cli.py dashboard

# Check portfolio balance and positions
python cli.py status

# Verify all connections and configuration
python cli.py health
```

Or run the bot directly:

```bash
python beast_mode_bot.py              # Paper trading
python beast_mode_bot.py --live       # Live trading
python beast_mode_bot.py --dashboard  # Dashboard mode
```

---

## Project Structure

```
kalshi-ai-trading-bot/
|-- beast_mode_bot.py          # Main bot entry point (BeastModeBot class)
|-- cli.py                     # Unified CLI: run, dashboard, status, health, backtest
|-- pyproject.toml             # PEP 621 project metadata and build config
|-- requirements.txt           # Pinned dependencies
|-- env.template               # Environment variable template
|
|-- src/
|   |-- agents/                # Multi-agent ensemble (forecaster, bull/bear, risk, trader)
|   |-- clients/               # API clients (Kalshi, xAI, OpenRouter, WebSocket)
|   |-- config/                # Settings and trading parameters
|   |-- data/                  # News aggregation and sentiment analysis
|   |-- events/                # Async event bus for real-time streaming
|   |-- jobs/                  # Core pipeline: ingest, decide, execute, track, evaluate
|   |-- strategies/            # Market making, portfolio optimization, quick flip
|   +-- utils/                 # Database, logging, prompts, risk helpers
|
|-- scripts/                   # Utility and diagnostic scripts (20 scripts)
|-- docs/                      # Additional documentation
+-- tests/                     # Pytest test suite
```

---

## Configuration Reference

All trading parameters live in `src/config/settings.py`. Key defaults:

```python
# Position sizing
max_position_size_pct  = 5.0     # Max 5% of balance per position
max_positions          = 15      # Up to 15 concurrent positions
kelly_fraction         = 0.75    # Fractional Kelly multiplier

# Market filtering
min_volume             = 200     # Minimum contract volume
max_time_to_expiry_days = 30     # Trade contracts up to 30 days out
min_confidence_to_trade = 0.50   # Minimum ensemble confidence

# AI settings
primary_model          = "grok-4"
ai_temperature         = 0       # Deterministic outputs
ai_max_tokens          = 8000

# Risk management
max_daily_loss_pct     = 15.0    # Hard daily loss limit
daily_ai_cost_limit    = 50.0    # Max daily AI API spend (USD)
```

The ensemble configuration (model roster, weights, debate settings) is in
`EnsembleConfig` within the same file.

---

## Performance

Performance tracking is built in. The bot records every trade, AI decision, and
cost metric to a local SQLite database (`trading_system.db`). Use the dashboard
or the scripts in `scripts/` to review:

- Cumulative P&L and win rate
- Sharpe ratio and maximum drawdown
- AI confidence calibration curves
- Cost per trade and daily API budget utilization
- Per-strategy breakdowns (directional vs. market making)

---

## Development

### Running Tests

```bash
# Run the full test suite
python run_tests.py

# Or use pytest directly
pytest tests/
```

### Code Quality

```bash
# Format
black src/ tests/ cli.py beast_mode_bot.py
isort src/ tests/ cli.py beast_mode_bot.py

# Type check
mypy src/
```

### Adding a New Strategy

1. Create a module in `src/strategies/`.
2. Implement the strategy logic and wire it into `src/strategies/unified_trading_system.py`.
3. Add allocation percentage in `src/config/settings.py`.
4. Write tests in `tests/`.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes with a descriptive message.
4. Push and open a Pull Request.

Please follow the existing code style (Black, isort) and add tests for new
functionality.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.

---

**Disclaimer**: This software is for educational and research purposes. Trading
involves risk, and you should only trade with capital you can afford to lose.
The authors are not responsible for any financial losses incurred through the use
of this software.
