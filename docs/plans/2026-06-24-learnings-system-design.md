# Learnings System — Design (proposed)

**Date:** 2026-06-24
**Status:** scope confirmed with user; awaiting go-ahead on one open question (auto-flagging)
**Goal:** Build a good system for tracking trading learnings over time, integrating measured
outcomes with derived insights so the strategy evolves on evidence, not vibes.

## Decisions locked with user
- **Purpose:** *both, integrated* — calibration/edge numbers feed a structured learnings log; one
  system that goes measure outcomes → derive lessons → update strategy.
- **Build weight:** *code in the repo* — a real, durable, testable command; not a manual ritual.

## The gap this closes
Infrastructure already present:
- `src/agent/journal.py` — `make_decision_record` (writes my prediction with `outcome: None`) and
  `summarize_journal` (aggregates win-rate / P&L **by category** — but only over records whose
  `outcome` is filled).
- `src/agent/settle.py` — `fetch_settlements` (`/portfolio/settlements`), `settlement_pnl`,
  and a `settlements.jsonl` cache.

**Missing link:** nothing joins settlements back into `decision_journal.jsonl` to fill each trade's
`outcome`. So per-trade calibration and per-category realized edge on *my own* predictions is blank;
`cli.py settle` only measures raw settlements (mostly legacy bot), not my journaled `est_prob` vs reality.

## Components

### 1. Reconciliation — the spine (`reconcile_outcomes`)
Match each settled ticker → its journal entry; compute `outcome = {won: bool, pnl: float}` from my
`side` + the settlement result; write it back into `decision_journal.jsonl`. Idempotent (skip
entries already reconciled). This is what makes everything downstream operate on *my* trades.

### 2. Calibration + edge engine (extends `summarize_journal`)
- **Calibration table:** bucket my settled trades by `est_prob` (0.90–0.95, 0.95–1.0, …); show
  *predicted vs actual* win-rate + n. Answers "when I say 97%, do ~97% win?"
- **Per-category & per-side realized edge:** n, win-rate, P&L, and *edge-at-entry vs realized edge*.
- **Per-method:** new journal field `method` (`manual` | `workflow`) to learn which research
  approach actually pays.

### 3. Structured learnings store (`data/runtime/learnings.jsonl`, append-only)
Record: `{date, kind, claim, evidence:{metric,n,value}, confidence, status, supersedes}`.
Confidence scales with sample size. `status`: candidate | confirmed | retired.

### 4. The integrated human loop
Each tick's LEARN step runs `cli.py learnings`: reconcile → print tables → surface *new* candidate
learnings. I review and promote confirmed ones into `SKILL.md` (the playbook) + memory. learnings.jsonl
is the audit trail; SKILL.md stays the distilled strategy. No rule changes silently — every change
traces to a numbered, evidenced learning.

### 5. Tests
`reconcile_outcomes`, calibration bucketing, and the flag-rules are pure functions → deterministic
unit tests with fixtures (matches the repo's pytest setup; CI = 3.12 + editable + bare pytest).

## Open question (my recommendation: YES)
**Should `cli.py learnings` auto-flag candidate learnings from the numbers** (e.g. "category=EconBucket,
n≥5, realized edge <0 → STOP"; "0.95+ bucket wins only 80% → overconfident, haircut est_prob"), which
I then confirm — **vs.** only printing raw tables and me writing every insight by hand?

Recommend **auto-flag**: the rule-based pass catches regressions I'd miss eyeballing tables, scales as
trades accumulate, and every flag is just a *candidate* (status=candidate) until I confirm it — so no
loss of judgment, only a safety net. Builds as a small, isolated, easily-toggled `flag_rules()` function.

## Alternatives rejected
- Standalone calibration script with no learnings log → the "hybrid" the user declined.
- Charting/dashboard analytics → YAGNI.

## Next step
On go-ahead: `writing-plans` → implementation plan → build (reconcile → engine → store → CLI → tests),
atomic commits, then wire into the kalshi-trade LEARN step.
