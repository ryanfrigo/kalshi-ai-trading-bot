# Changelog

All notable changes to the Kalshi AI Trading Bot project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`cli verify --research-file`** — run the adversarial-verify gate with **no LLM
  API key**: supply the research + skeptic judgments as JSON (a human or agent does
  them out-of-band), and the deterministic gate still recomputes the edge off the
  live orderbook, so the fade can't be fudged by optimistic pricing.
- **`cli report`** — renders a public **[Live Track Record](docs/TRACK_RECORD.md)**
  from the persisted settlements / journal / policy, losses included. The account
  section is explicitly framed as *blended* (operator manual trades + autonomous
  strategy) so a manual-driven drawdown is never misread as the strategy failing;
  the strategy's real edge lives in the journal-based metrics. Offline-capable.
- **`cli fills`** — reconciles the decision journal against actual order fills:
  voids records whose maker orders were cancelled unfilled and shrinks partial
  fills, so calibration and the Edge Policy only ever learn from trades that
  actually executed (no phantom predictions).
- **`scripts/capture_corpus.py`** — captures a daily price snapshot of the full
  open-market universe to `data/corpus/` (idempotent per UTC day). Joined against
  settlements, this is the entry-price corpus a real out-of-sample backtest needs.
- **Non-sports longshot bucket** in `scripts/hunt_candidates.py` — surfaces the
  pond where researched fades actually pay (liquid sports books are already sharp).
- **The self-improvement loop is closed.** A new data-driven **Edge Policy**
  (`src/agent/policy.py`) turns your settled track record into a pre-trade gate:
  it **blocks** category/method groups your record proves lose money (≥5 settled
  trades, negative realized P&L), **warns** on a net-negative side (advisory — a
  losing side never disables the whole strategy), and **haircuts** `est_prob`
  bands where you're ≥10pp overconfident. Honest gating throughout: a group with
  fewer than 5 settled trades earns no rule.
- **`cli policy`** — read-only view of the gate your settled record earns
  (`--demo` runs on a shipped fixture, no keys needed; `--json` for machines).
- **`cli improve`** — the loop end to end: settle → reconcile → re-derive the
  policy → diff what the newest settlements changed → persist the active gate.
  `--dry` previews; falls back to the local settlements log when offline.
- **Edge Policy gate in `place_guarded_order`.** A blocked category/method is a
  hard refusal (`blocked_by_policy`); the agent keeps full authority via
  `cli trade --override-policy` (the override is recorded). Backward-compatible:
  no policy file means no opinion, so nothing changes until you run `improve`.
- **MCP `policy` tool** — the gate is now drivable from Claude Desktop/Code
  (read-only, derives fresh in memory).
- `settle.settlement_to_record` / `series_category` — adapt Kalshi's
  authoritative settlements into journal-shaped records so the policy learns
  from real outcomes, with the Kalshi series prefix as the category.

### Changed
- `cli backtest` no longer claims a fake "coming soon" engine. It honestly
  explains that a strategy backtest needs a captured price/outcome corpus this
  repo doesn't ship yet, and points to the feedback loop (`edge`/`policy`/
  `improve`) that works today without one.

## [2.0.1] - 2026-06-12

### Fixed
- **`pytest` is now safe to run by default.** Tests that hit the real Kalshi
  API — several of which place real orders — are marked `live` and skipped
  unless `RUN_LIVE_TESTS=1` is set. Previously, running the test suite with a
  configured `.env` could place real-money orders.
- **`pip install -e .` works again.** The declared build backend
  (`setuptools.backends._legacy:_Backend`) does not exist, and `setup.py` was
  an interactive wizard that setuptools executed (and hung on) during every
  install. The backend is now `setuptools.build_meta` and the wizard lives at
  `setup_env.py`.
- Balance guard in order execution: live orders are skipped (fail-closed) if
  the account balance cannot be verified or is insufficient for the order.
- `MODEL_PRICING` was missing entries for `anthropic/claude-sonnet-4.5` (the
  default model), `google/gemini-3-pro-preview`, and `deepseek/deepseek-v3.2`,
  so the daily AI cost limit was not tracking their spend.
- Replaced 6 bare `except:` clauses (which also swallow
  `asyncio.CancelledError` and can block shutdown) with narrow handlers.
- Fixed placeholder `yourusername` URLs in `pyproject.toml` and the broken
  mocked execution test.

### Added
- GitHub Actions CI: test suite (no secrets required) + gitleaks secret scan.
- `SECURITY.md` with vulnerability reporting and credential-handling guidance.
- Regression test: orders exceeding the available balance are refused.

### Removed
- `requirements.txt` (and stale `requirements-dev.txt` /
  `dashboard_requirements.txt` references) — `pyproject.toml` is the single
  source of truth: `pip install -e ".[dev]"` / `".[dashboard]"`.
- One-off debug scripts `verify_fix.py` and `test_live_mode.py` from the
  repo root.

## [Unreleased]

### Changed
- Split the 1,300-line `portfolio_optimization.py` into the
  `src/strategies/portfolio/` package (`models`, `optimizer`, `immediate`,
  `runner`). `portfolio_optimization.py` remains as a compatibility shim, so
  all existing imports — including in forks — keep working unchanged.
- Consolidated the Kelly criterion kernel into `src/utils/position_sizing.py`.
  It was previously implemented three times (twice in
  `portfolio_optimization`, once in `safe_compounder`) with subtle
  differences. Strategy-level policy (fractional multipliers, regime/time
  adjustments, caps) is unchanged; behavior is pinned by 28 characterization
  tests in `tests/test_position_sizing.py`.
- `market_making._calculate_optimal_sizes` docstring now states that its
  formula is a linear edge scaler, not Kelly (the math itself is untouched).

### Added
- Initial public release of Kalshi AI Trading Bot
- Multi-agent AI decision engine with Forecaster, Critic, and Trader agents
- Real-time market scanning and analysis
- Portfolio optimization using Kelly Criterion and risk parity
- Live trading integration with Kalshi API
- Web-based dashboard for monitoring and control
- Performance analytics and reporting
- Market making strategy implementation
- Dynamic exit strategies
- Cost optimization for AI usage
- Comprehensive test suite
- Database management with SQLite support
- Configuration management system
- Logging and monitoring capabilities

### Features
- **Beast Mode Trading**: Aggressive multi-strategy trading system
- **Grok-4 Integration**: Primary AI model for market analysis
- **Real-time Dashboard**: Web interface for monitoring and control
- **Portfolio Management**: Advanced position sizing and risk management
- **Market Making**: Automated spread trading and liquidity provision
- **Performance Tracking**: Comprehensive analytics and reporting

### Technical
- Python 3.12+ compatibility
- Async/await architecture for high performance
- Type hints throughout the codebase
- Comprehensive error handling
- Rate limiting and API management
- Modular design for easy extension

## [1.0.0] - 2024-01-XX

### Added
- Initial release
- Core trading system with AI integration
- Multi-agent decision making
- Portfolio optimization
- Real-time market analysis
- Web dashboard
- Performance monitoring
- Database management
- Configuration system
- Testing framework

---

## Version History

### Version 1.0.0
- **Release Date**: January 2024
- **Status**: Initial public release
- **Key Features**: 
  - Multi-agent AI trading system
  - Real-time market analysis
  - Portfolio optimization
  - Web dashboard
  - Performance tracking

---

## Migration Guide

### From Development to Production
1. Set up environment variables in `.env` file
2. Initialize database with `python init_database.py`
3. Configure trading parameters in `src/config/settings.py`
4. Test with paper trading before live trading
5. Monitor performance and adjust settings as needed

---

## Deprecation Notices

No deprecations in current version.

---

## Breaking Changes

No breaking changes in current version.

---

## Known Issues

- Limited to SQLite database (PostgreSQL support planned)
- Requires manual API key management
- Performance may vary based on market conditions

---

## Future Roadmap

### Planned Features
- PostgreSQL database support
- Additional AI models
- Advanced risk management
- Mobile dashboard
- API rate limit optimization
- Enhanced backtesting capabilities

### Version 1.1.0 (Planned)
- Database migration tools
- Enhanced error handling
- Performance optimizations
- Additional trading strategies

### Version 1.2.0 (Planned)
- PostgreSQL support
- Advanced analytics
- Mobile interface
- API improvements 