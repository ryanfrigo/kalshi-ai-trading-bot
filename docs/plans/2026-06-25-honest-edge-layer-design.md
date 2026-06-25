# Design: "The Honest Edge Layer" — agent-native Kalshi toolkit repositioning

**Date:** 2026-06-25 · **Status:** direction approved; executing incrementally

## Why (research-backed)
The Kalshi/prediction-market AI niche is **saturated at the surface, thin underneath**: every framing
(single-LLM toolkit, multi-LLM ensemble, multi-agent+skills, pure-quant) already exists, and *every
honest repo disclaims profitability*. Independent out-of-sample studies (Profit Mirage, The Alpha
Illusion) show the famous multi-agent "hedge fund" repos' edge collapses 50–72% past the training
cutoff, below buy-and-hold net of fees — and the flagships don't even trade live. The one thing the
whole field lacks: **rigorous, settlement-grounded proof of edge.**

This repo (~486★, #1 Kalshi repo, "honest toolkit" brand) is one repositioning from owning that
white space — and it already seeds it (`settle`, `hunt_candidates`, the calibration/learnings system).

## What — north star: "Prove your edge against the sharp book."
Four layers:
1. **Lean agent-native foundation** — clean tools/skills/utils; ~30% cruft quarantined.
2. **Edge-measurement harness (headline)** — Brier/log-loss vs settlements, realized-edge-vs-orderbook
   attribution, calibration curve, out-of-sample/cutoff honesty. The differentiator.
3. **First-mover MCP distribution** — no official Kalshi MCP exists; thin FastMCP over the CLI,
   governor-encoded, "runs on your keys."
4. **Adversarial-verify engine** — LLM-as-skeptic research orchestration (proven this session), NOT
   investor personas; repurpose the unwired `agents/` scaffolding.

## Constraints
- Preserve trading value; restructure for composability.
- **Import paths for live code stay stable** — the make-money loop + forks keep working.
- Each phase: refactor → live-test (`brief`/`daily`) → `/code-review` → commit. Never break governor/kill switch.

## Phases
P1 lean foundation · P2 edge harness (flagship) · P3 MCP server · P4 adversarial-verify engine ·
P5 reposition README/docs. (Detail + live progress tracked in /tmp/refactor-kalshi-ai-trading-bot.md.)

## Non-goals
Generalizing beyond Kalshi (yet); investor-persona debate (rejected — saturated + hollow); big-bang
package rename (would break the live loop).
