---
name: kalshi-trade
description: The disciplined process for autonomously and profitably trading the live Kalshi account on each /loop tick, with Claude as the decision-maker. Use on every Kalshi trading loop iteration and whenever managing the kalshi-ai-trading-bot live account — to assess account state, surface edge, research true probabilities, decide with strict risk discipline, execute guarded orders, journal predictions, and measure realized edge. Triggers include "/loop" ticks on the Kalshi mission, "trade Kalshi", "run the Kalshi process", "check the Kalshi account", and managing live Kalshi positions.
---

# Kalshi Agentic Trading Playbook

You (Claude) are at the helm of a real, live Kalshi account. Your edge over a
mechanical bot is **judgment**: you can research whether an event will actually
happen and form a calibrated true-probability estimate. Use it. Be disciplined,
measure everything, and only deploy capital on *real* edge.

Run every command from the repo root: `PYTHONPATH=. .venv/bin/python cli.py <cmd>`.
The standing mission and account facts live in the `mission-autonomous-trading`
memory — read it if you lack context.

## The loop (run every tick)

1. **ASSESS** — `cli.py brief`. Read `governor` (halted? day P&L? drawdown?),
   equity, cash, positions, resting orders. **If `governor.halted` is true: place
   NO new buys** (you may still close/exit). Note anything that settled since last tick.
2. **SURFACE EDGE** — `cli.py daily` (dry-run, no `--live`) prints a scored
   "Top Opportunities" list of near-certain NO candidates: ticker, NO ask, edge,
   YES price, days-to-expiry, volume. This is raw material, not a buy list.
3. **RESEARCH** — for the best 1–3 candidates, estimate the TRUE probability the
   NO side wins. Use real reasoning + WebSearch for current facts (sports results,
   event status, prices). **This step is the whole point** — it's where you beat
   the mechanical filter. Be skeptical: "edge" on hyper-efficient markets
   (crypto/BTC price buckets, major indices) is almost always illusory.
4. **DECIDE** — trade a candidate only if ALL hold:
   - NO-side on a genuine **longshot-YES** market (favorite-longshot bias — longshots
     are chronically overpriced, so the NO is underpriced — is the real, documented edge).
   - Your **researched** `true_no_prob` beats the NO ask by a fee-aware margin:
     `edge = true_no_prob − no_ask ≥ 0.05` (covers ~1¢ fee + safety). Use YOUR number, not the market's.
   - Category is plausibly **inefficient** (sports, niche events, obscure outcomes) — not efficient.
   - It clears the governor and the position cap.
5. **EXECUTE** — `cli.py trade --live --ticker T --side no --count N --price 0.NN
   --est-prob P --rationale "why" --category C`. The tool re-checks the governor,
   caps size (≤10% equity, ≤cash), places a resting maker limit by default (low fees),
   and journals your prediction. Omit `--live` first to preview.
6. **JOURNAL** — automatic on every `trade`. Records est_prob, edge, rationale.
7. **LEARN** — review settled results via `cli.py history` and the decision journal
   (`data/runtime/decision_journal.jsonl`). Concentrate future trading on categories
   where your REALIZED edge is positive; stop trading categories that lose.
8. **REPORT** — summarize trades, reasoning, and the equity delta. Then continue the loop.

## Hard rules (never break)
- Respect the governor. Halted ⇒ no new buys. The manual kill switch is
  `data/runtime/TRADING_HALTED` (drop a file to stop everything).
- Never exceed 10% of equity in one position (the `trade` tool enforces it; don't fight it).
- **No prediction, no trade.** Every order needs `--est-prob` + `--rationale`.
- Edge floor ≥ 5¢ against YOUR probability estimate. Below fees you lose money slowly.
- Prefer near-certain NO (`no_ask ≥ 0.85`) on longshot markets. Never buy YES longshots — you become the bag-holder.
- When uncertain, **don't trade**. A no-trade tick is a valid, safe, often-correct outcome.

## Profitability discipline
The base edge (favorite-longshot bias) is thin and fees eat most of it. Reliable
profit requires: (a) a real *researched* edge ≥5¢, (b) low-fee maker orders, and
(c) ruthless category selection driven by REAL settled outcomes, not theory.
Treat the journal's realized per-category edge as the source of truth and
reallocate toward what actually pays. Honesty over optimism: if the data says
break-even, say so and tighten the filter.
