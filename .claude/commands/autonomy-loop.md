---
description: One full autonomy tick — trade the live Kalshi account, improve the project, ship to GitHub. Rerun it (or wrap in /loop) to keep the mission going.
---

# Autonomy loop — one tick

Run one complete cycle of the standing mission: trade Ryan's live Kalshi account
profitably with Claude as the decision engine, and keep compounding the project.
Do the steps in order; each tick must end with a report.

## 1. TRADE (always first)

Invoke the `kalshi-trade` skill and follow it end to end:
ASSESS → SURFACE EDGE → RESEARCH → DECIDE → EXECUTE → JOURNAL →
LEARN (`cli.py learnings`) → IMPROVE (`cli.py improve`).

Non-negotiables:
- Respect the governor and the kill switch (`data/runtime/TRADING_HALTED`).
- Respect the Edge Policy gate and the adversarial-verify gate (`cli.py verify`).
- An honest no-trade tick after a rigorous hunt is a valid, often-correct outcome.
- Near a drawdown limit, deploy conservatively; halted ⇒ hold the sound book, no new buys.
- If a position settles today, check its live status before assuming it's fine.

## 2. IMPROVE THE PROJECT (one finished slice)

Pick ONE small, high-value improvement and finish it — code with tests, or docs. Priority:
1. Whatever the trading tick just exposed (bug, missing data, friction in the loop).
2. The roadmap in memory (next up: capture a price-history corpus so a real backtest is possible).
3. Repo health: CI failures, open issues/PRs, doc drift. This is a 500★ public repo — README honesty is a feature.

Keep it surgical (Karpathy guidelines). Don't start what one tick can't finish;
if a slice is too big, land the first shippable piece and note the rest in memory.

## 3. SHIP

- Run the suite: `PYTHONPATH=. .venv/bin/python -m pytest -q`. Ship only on green.
- Commit with a conventional message; `git push` the current branch. Never force-push.
- Never commit secrets or runtime state (`kalshi_private_key*`, `data/runtime/`,
  `trading_system.db` are gitignored — verify with `git status` before committing).
- If on `main`, branch first.

## 4. REPORT + MEMORY

End the tick with: equity and day P&L, trades placed (or why none), what the
learn/improve loop changed in the Edge Policy, and what shipped to GitHub.
If a durable lesson emerged, write it to the memory directory (update the
existing file if one covers it).

---

**How to rerun (for Ryan):**
- One tick: `/autonomy-loop`
- Keep it running: `/loop /autonomy-loop` (self-paced) or `/loop 1h /autonomy-loop`
