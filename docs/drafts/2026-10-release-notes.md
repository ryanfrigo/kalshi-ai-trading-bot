# Release notes — DRAFT (for Ryan to publish after PR #95 merges)

_Status: draft. Nothing has been tagged or published._

**Why this exists:** the repo has **no tags and no releases at all**, despite using
Keep-a-Changelog + SemVer and declaring `version = "2.0.1"` in `pyproject.toml`.
So users cannot pin a version, and the CHANGELOG's `[Unreleased]` work has nowhere
to land.

**Version choice (your call):** the last changelogged release is `2.0.1`; the
unreleased section is additive (new playbook + edge-verdict work) plus a substantial
fix set, so
**`2.1.0`** is the natural next tag. If you'd rather start simple, tag `2.0.1` at the
merge point instead.

## Suggested tag
`v2.1.0` → target: the merge commit of PR #95 on `main`

## Body

```markdown
### Added
- **Operator's Playbook** (`docs/OPERATOR_PLAYBOOK.md`) — the free field guide for the
  setup traps: the private-key 401 wall, "it runs but never trades", staying online.
  Linked from the README nav.
- **First forward-only edge verdict on the live account** — `cli edge` scores
  forward-settled trades and publishes the verdict, losses included.
- **Live-book corpus + fade backtest** — `scripts/capture_books.py` captures the live
  orderbook for each day's candidate slice (run by the daily job);
  `scripts/backtest_books.py` joins it to settlements, charges fees on entry, and counts
  **events, not strike rows**.
- **`aaa_pricer.py summary`** — per-series forward Brier with the effective sample
  (distinct prints), so a ladder of correlated strikes can't masquerade as n=85.

### Fixed
- **`cli.py close-all` — the kill switch — did not work.** It filtered positions on the v1
  `position` field, so it printed *"No open positions on Kalshi"* on an account holding 13.
  Three more v1/v2 mismatches sat behind it: a `KeyError` on the row lookup, the wrong
  orderbook key, and bid levels parsed as integer cents where the API returns dollar
  strings. Verified fixed by running it dry: 12 sells at real bid prices.
- **The position cap priced NO orders off the YES leg** — a 95c fade counted as ~5c of
  exposure, understating it ~19x against a "10% of equity" rule.
- **The governor enforced 10% daily loss while the config said 5%** — three call sites
  built it with no arguments and took the defaults, so the configured limit was decoration.
- **`src/jobs/performance_analyzer.py` had not compiled since the first public release** —
  an `IndentationError` nothing caught because no test imported it. Repaired, and
  `tests/test_all_modules_compile.py` now compiles every module in `src/` and `scripts/`.
- **`scripts/get_positions.py` printed `None` for every position** (v1 `position` field).
- **51 consecutive scheduled runs failed silently** — two data tools could not find their
  browser binary under the scheduler's minimal `PATH`, and the scheduler swallowed every
  step failure. Both fixed; failures now land in `logs/daily_runs/LAST_FAILURES.txt`.
- **Unbounded logs** — `scripts/rotate_logs.py` caps them, so a full disk can't take the
  bot down.
- **Track record: the all-time peak now ratchets with deposits** — a funded account can
  no longer publish a nonsense `-393.2%` drawdown.
- **Edge verdict: a sub-point edge is noise** — it no longer prints "BEATING THE BOOK"
  for a +0.1 pt edge.

### Known, not fixed (tracked as issues)
- **The risk gate fails OPEN on infrastructure errors** (#98) — the approved design requires
  fail-closed. Changing it alters live trading behaviour, so it is your call.
- **`taker_fee_per_contract` over-states Kalshi's fee by 39%** (#102) — it rounds up per
  contract rather than per order. Real but small: ~0.25c/contract at realistic sizes, so it
  does not change any measured verdict. A correctness cleanup, not an edge unlock.
- **`src/jobs/track.py` is uninvoked** (#97) — and it is the sole caller of every take-profit
  and stop-loss function, which also no-op on a v1 price read. Net effect: no exit management.

### Docs
- Corrected stale defaults in the playbook (position cap 3%→10%, drawdown 15%→20%,
  Safe Compounder edge 5c→3c); refreshed `docs/DAILY_STATUS.md`; added the operator's
  playbook link + listing badge to the README.
```

## How to publish (after the merge)

```bash
cd ~/dev/kalshi-ai-trading-bot && git checkout main && git pull
gh release create v2.1.0 --target main --title "v2.1.0 — the operator's playbook + honest edge fixes" --notes-file docs/drafts/2026-10-release-notes.md
# gh release create v2.1.0 --target main --title "..." --notes-file ... --draft   # to review first
```
