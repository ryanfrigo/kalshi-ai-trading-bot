---
description: One full autonomy tick — trade the live Kalshi account, publish the honest track record, grow the repo, ship to GitHub. Rerun it (or wrap in /loop) to keep the mission going.
---

# Autonomy loop — one tick

Two standing goals, one loop: **trade the live account profitably** and **grow the
GitHub repo**. They're wired together — every trading tick produces the honest,
settlement-grounded track record that is this repo's unique growth asset. Do the
steps in order; each tick must end with a report.

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

## 2. PUBLISH THE RECORD (every tick)

Regenerate `docs/TRACK_RECORD.md` from real data — equity + drawdown, settled P&L
by category, the calibration table, current Edge Policy blocks, and the `cli.py edge`
verdict. Losses included; that honesty IS the brand ("prove your edge, don't claim it")
and the reason to star the repo. If the generator doesn't exist yet, building it is
this tick's GROW slice. Never publish keys, account IDs, or anything beyond what the
CLI's own reports print.

## 3. GROW (one finished slice, rotate by value)

Pick ONE lane and finish a slice — code with tests, or docs:
- **Product** — whatever the trading tick just exposed, else the roadmap in memory
  (next: price-history corpus → real backtester, the open-core future).
- **Community** — triage any open issues/PRs same-tick (respond, label, fix or close);
  maintain good-first-issues; tag a release with changelog when something meaningful landed.
- **Distribution (draft-only)** — draft release notes or a post (HN/Reddit/X) for Ryan
  to publish. Never post outward-facing content yourself; leave drafts in `docs/drafts/`.

Keep it surgical (Karpathy guidelines). Don't start what one tick can't finish;
if a slice is too big, land the first shippable piece and note the rest in memory.

## 4. SHIP

- Run the suite: `PYTHONPATH=. .venv/bin/python -m pytest -q`. Ship only on green.
- Commit with a conventional message; `git push` the current branch. Never force-push.
- Never commit secrets or runtime state (`kalshi_private_key*`, `data/runtime/`,
  `trading_system.db` are gitignored — verify with `git status` before committing).
- If on `main`, branch first.

## 5. REPORT + MEMORY

End the tick with: equity and day P&L, trades placed (or why none), what the
learn/improve loop changed in the Edge Policy, what shipped to GitHub, and which
GROW lane ran. If a durable lesson emerged, write it to the memory directory
(update the existing file if one covers it).

---

**How to rerun (for Ryan):**
- One tick: `/autonomy-loop`
- Keep it running: `/loop /autonomy-loop` (self-paced, ~1–2 ticks/day is plenty) or `/loop 4h /autonomy-loop`
