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

## What the REAL settlement data says (measured 2026-06-19, 122 settled bets)

`cli.py settle` revealed the actual track record — read it before trading:
- **YES longshots: −$528 over 46 bets.** Buying a longshot YES is the bag-holder
  trade. **NEVER buy YES longshots.** (Holding/closing existing YES is fine.)
- **NO side: 79% win rate but −$59 net** over 76 bets — wins were small (+$81),
  the 16 losses were large (−$141). Picking up pennies, then run over.
- **The losses concentrated in FAKE longshots**, not genuine ones:
  - **Economic-data buckets** (`KXCPI`, inflation, GDP-point, Fed-rate): the
    outcome has a real distribution — a "narrow bucket" can carry 10–20%, not 3–5%.
    **AVOID NO bets on numeric/economic-data buckets.**
  - **Multi-outcome sports brackets/totals** (`KXMARMAD`, `KXNCAAMBTOTAL`):
    several outcomes stay live; the NO is not near-certain. **Avoid / size tiny.**
- **The winners were GENUINE longshots** (`KXGDP` overshoot, `KXGUINEAWORM`,
  `KXBTCMAX150` extreme price, `KXGOVTSHUTLENGTH`, alien-confirmation-type):
  true YES < 5%, NO won ~97%, real edge (~+11¢/contract on winners).

**Refined edge (the only version the data supports):** NO-only, on **genuine
<5% longshots** — extreme/binary events where YES is a real long shot — and
**avoid economic-data buckets and multi-outcome sports brackets**. Run
`cli.py settle` each tick and let the realized per-category P&L keep tightening
this list. If a category's realized edge is negative, stop trading it.

## Profitability discipline
The base edge (favorite-longshot bias) is thin and fees eat most of it, and the
real data shows undisciplined NO trading LOST money. Reliable profit requires:
(a) genuine <5% longshots only, (b) the category exclusions above, (c) low-fee
maker orders, (d) a real researched reason the YES is overpriced. Treat
`cli.py settle` realized P&L as the source of truth. Honesty over optimism: the
measured edge is currently negative — tighten the filter until it isn't, and
when in doubt, don't trade.
