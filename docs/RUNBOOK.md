# Runbook

This repo ships a paper-trading bot plus a dashboard and a separate paper signal logger.
Use this guide for day-to-day operation and for helping new contributors or agents get oriented quickly.

## Prerequisites

- Python 3.12+
- Kalshi API credentials if you want live market access
- xAI / OpenRouter credentials if you want the LLM-backed decision flow
- A virtual environment with `requirements.txt` installed

## Common Commands

```bash
python cli.py run --paper
python cli.py dashboard
python cli.py status
python paper_trader.py
python paper_trader.py --loop
python paper_trader.py --dashboard
python paper_trader.py --stats
```

## What Each Mode Does

- `cli.py run --paper` runs the bot in paper mode.
- `cli.py run --live` runs the trading flow against live execution.
- `cli.py dashboard` launches the Streamlit monitoring UI.
- `paper_trader.py` logs paper signals to SQLite and builds the static dashboard.
- `paper_trader.py --loop` keeps scanning on a schedule.

## Recommended Operational Sequence

1. Pull the latest branch or check out the PR branch.
2. Create or activate the virtual environment.
3. Confirm environment variables are present.
4. Run the targeted test subset for the change.
5. Start the paper bot.
6. Open the dashboard in a browser.
7. Verify status, signals, and error panels after the first cycle.

## Restart and Recovery

- If a backend or scheduler process is restarted, confirm the bot version and loop state before resuming analysis.
- After a restart, verify the dashboard has reconnected and is showing fresh data.
- If the signal stack is stale, check the latest logs before changing code.

## Repo Hygiene

- Do not commit API keys or private credentials.
- Keep generated artifacts out of source control unless they are intentionally checked in.
- If you add a new operational workflow, document the exact command and expected outcome here.

