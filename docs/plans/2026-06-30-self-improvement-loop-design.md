# Design: Close the self-improvement loop — a data-driven Edge Policy gate

**Status:** approved direction (2026-06-30). First slice of the "keep building a
self-improving system" program. Branch: `feat/autonomous-daily-trading`.

## Why

The "honest edge layer" refactor gave the repo a real *measurement* half —
`journal → settle → learnings → edge`. `learnings.flag_rules()` already turns the
141 authoritative settlements into evidenced signals ("category X is losing money
over N settled trades"; "you're overconfident in the 90–95% band"). But **nothing
consumes those signals.** They are appended as text candidates and read by a
human. The next trade does not know what the settled record proved.

So today the system *measures* but does not *learn*. Closing that one feedback
edge is the highest-leverage, most brand-consistent, buildable-now move: it turns
the honest edge layer from a measurement tool into a **self-improving loop**, and
it is the exact thing the mission asks for — "a system that can self-improve using
AI agents." AI research (`verify.py`) proposes; the settled track record disposes.

A universe-wide *strategy backtest* was considered and rejected as the first
slice: the local corpus (`trading_system.db`) is one upserted row per market with
no price history and no settled outcomes, so a backtest engine would have nothing
real to score — a hollow shell this repo's brand refuses to ship. Building the
capture corpus that unblocks a real backtest is a later slice (see Non-goals).

## What — north star

**Your own settled outcomes automatically constrain your next trade.**

```bash
python cli.py policy          # the honest, gated view of what your record proved
python cli.py improve         # run the loop: settle → learn → re-derive → diff what changed & why
```

`cli policy` shows a machine-readable **Edge Policy** derived purely from settled
data: which category/side groups your record proves lose money (→ block), and
which `est_prob` bands you're overconfident in (→ haircut). `cli improve` runs the
full loop end to end and reports the *diff* — which new settlements moved the
policy and why. The policy becomes a **pre-trade gate**: `cli trade` consults it
and refuses a blocked group / haircuts an overconfident estimate before placing.

The honesty rule is absolute and mirrors `cli edge`: **the policy never blocks or
haircuts on thin data.** A group with n < 5 settled trades earns *no opinion* —
the gate stays silent, never inventing a constraint the data can't support. The
policy is only ever as confident as the settled sample behind it.

## Architecture

New pure module `src/agent/policy.py` (no IO, no `datetime.now` — `date` injected,
same contract as `edge.py`/`learnings.py`):

- `derive_policy(edge_breakdown, calibration, *, date, min_n=5) -> Policy`
  Consumes the existing `learnings.edge_breakdown(...)` and
  `learnings.calibration_table(...)` outputs. Emits:
  - `blocks`: `[{dimension, label, reason, n, pnl}]` — a **category or method**
    group with `n >= min_n` and realized `pnl < 0`. This granularity matches the
    real measured lesson (losses concentrate in *specific categories* — economic
    buckets, sports brackets — not in a whole side).
  - `warnings`: same shape, for the **side** dimension. Side is the strategy axis,
    not a selection axis: a net-negative side (e.g. NO is 79% win-rate but
    net-negative *because of* a few bad categories) must NOT hard-block the whole
    strategy. Side negativity is surfaced as a warning reason only, never a BLOCK.
  - `haircuts`: `[{band, lo, hi, shrink_to, n, gap}]` — an `est_prob` calibration
    bucket with `n >= min_n` where actual win-rate is ≥10pp below predicted;
    `shrink_to` = the bucket's realized `actual` win-rate.
  - `meta`: `{version, generated_date, settled_n}` for auditability.
  Deterministic; a superset of what `flag_rules` already computes, shaped for
  consumption rather than display.
- `apply_policy(policy, decision) -> Verdict`
  `decision = {ticker, side, category, est_prob, method?}`. Returns
  `{verdict: ALLOW|BLOCK|HAIRCUT, reasons: [...], adjusted_est_prob?}`.
  BLOCK when a block rule matches the decision's category or method; else HAIRCUT
  (with `adjusted_est_prob = shrink_to`) when `est_prob` lands in a haircut band;
  else ALLOW. A matching side `warning` is appended to `reasons` but never
  changes the verdict on its own. Pure.
- `diff_policy(old, new) -> {added_blocks, removed_blocks, changed_haircuts}`
  For the `improve` report — what the newest settlements changed.

IO / CLI layer (thin, mirrors existing commands):
- Persist/load `data/runtime/edge_policy.json` (git-ignored; regenerated from
  settled data, so it is derived state, never a source of truth).
- `cmd_policy` — load journal+settlements, reconcile, derive, print (honest gated
  view); `--json`; `--demo` runs against a shipped fixture so a fresh clone works.
- `cmd_improve` — `settle` (pull authoritative outcomes) → `reconcile_outcomes` →
  `edge_breakdown` + `calibration_table` → `derive_policy` → `diff_policy` vs the
  saved file → persist (unless `--dry`) → print the diff. The visible heartbeat of
  the self-improving loop.
- Gate wiring: `place_guarded_order` (`src/agent/toolbelt.py`) gains an optional
  policy consult. Default behavior is unchanged when no policy file exists (no
  opinion → allow), so it is backward-compatible; when a policy exists a BLOCK is
  a hard refusal (returns a `blocked_by_policy` result, places nothing) and a
  HAIRCUT annotates the journaled record with the shrunk estimate.

Replaces the dead `cmd_backtest` "coming soon" placeholder text with an honest
pointer: real backtest needs the capture corpus (later slice), and here is the
feedback mechanism that works *today*.

## Data flow (the closed loop)

```
decide (verify: AI research proposes) ─▶ trade (guarded, journaled)
        ▲                                        │
        │                                        ▼
   apply_policy  ◀── edge_policy.json  ◀── settle (authoritative outcomes)
   (gate: block/haircut)      ▲                  │
                              │                  ▼
                       derive_policy ◀── reconcile → edge_breakdown + calibration
```

Outcomes now feed forward into the next decision. That is the whole point.

## Constraints

- Pure calculation core, IO only in CLI — the repo's standing contract; keeps it
  unit-testable and deterministic.
- Honest gating everywhere: `n < min_n` ⇒ no rule. Never assert a constraint the
  settled sample can't support.
- Surgical: reuse `journal`, `settle`, `learnings`, `edge`, `verify`, `toolbelt`.
  One new pure module + two CLI verbs + one optional gate consult. No new
  subsystem, no schema migration.
- `edge_policy.json` is derived, git-ignored, per-user. No trading data or policy
  ships in the public repo. A tiny synthetic fixture ships for tests + `--demo`.
- TDD: pure functions land test-first (losing group → block; overconfident band →
  haircut; thin data → silence; diff correctness; gate ALLOW/BLOCK/HAIRCUT).

## Success criteria

1. `cli policy --demo` prints a gated policy from the shipped fixture on a fresh
   clone (no live keys).
2. `cli improve --dry` runs the full loop against the real runtime files and
   prints a diff without mutating anything.
3. A blocked category is refused by `cli trade` (dry-run shows `blocked_by_policy`
   and places nothing); an overconfident estimate is haircut in the journaled
   record.
4. Thin-data groups produce no rule (asserted by test).
5. Full suite green (existing 205 + new policy tests); README documents the loop.

## Non-goals (explicitly deferred to later slices)

- Universe-capture corpus (extend the collector to snapshot the whole market
  universe + settlements over time). The prerequisite for a real backtest.
- Real backtest / replay engine (needs the corpus above).
- An agentic optimizer that proposes strategy/param tweaks and scores them against
  the corpus.
- Auto-*loosening* the policy. This slice only ever *tightens* (block/haircut)
  from evidence; re-enabling a group stays a human/agent judgment call.
```
