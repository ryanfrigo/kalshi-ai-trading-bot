# Release notes — DRAFT (for Ryan to publish after PR #95 merges)

_Status: draft. Nothing has been tagged or published._

**Why this exists:** the repo has **no tags and no releases at all**, despite using
Keep-a-Changelog + SemVer and declaring `version = "2.0.1"` in `pyproject.toml`.
So users cannot pin a version, and the CHANGELOG's `[Unreleased]` work has nowhere
to land.

**Version choice (your call):** the last changelogged release is `2.0.1`; the
unreleased section is additive (new playbook + edge-verdict work) plus two fixes, so
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
- **Track record: the all-time peak now ratchets with deposits** — a funded account can
  no longer publish a nonsense `-393.2%` drawdown.
- **Edge verdict: a sub-point edge is noise** — it no longer prints "BEATING THE BOOK"
  for a +0.1 pt edge.

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
