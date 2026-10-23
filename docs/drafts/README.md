# Launch checklist — turning the drafts into a launch

_Drafts only. Nothing here has been posted or published by an agent._

## Prerequisites (each is a few minutes)
- [ ] **Merge PR #95.** Until then none of the repo work is public — the README playbook
      link, the badge, and the two honesty fixes all live on the branch.
- [ ] **Checkout link** (Gumroad / Lemon Squeezy / Stripe / Payhip) → then
      `~/kalshi-monetization/guide/wire-checkout.sh <url>`. Without this the guide cannot
      be bought, and the CTA stays "coming soon".
- [ ] **(Optional, fastest revenue) Activate GitHub Sponsors.** `.github/FUNDING.yml` is
      already wired; enrollment is the only missing step.
- [ ] **(Optional) Publish the release.** `docs/drafts/2026-10-release-notes.md` has the
      body and the exact command (tag choice flagged: `2.1.0` vs `2.0.1`).

## Sequence (do them in this order)
1. **Merge PR #95** → then **publish the release** (tag + notes).
2. **Publish the free playbook publicly** so the lead magnet has a URL outside the repo.
3. **Fire ONE launch post** — Show HN *or* Reddit, not both. HN suits "honest tooling";
   Reddit suits a practitioner audience.
4. **Answer every substantive comment with a real number** from `docs/TRACK_RECORD.md`.
   The credibility comes from quoting your own losses correctly.
5. **Only then** mention the paid guide, in a follow-up ("the playbook got N replies —
   here's the deeper guide"). Never lead with the paid thing.

## Drafts in this folder
| file | channel | notes |
|---|---|---|
| `2026-10-launch-playbook.md` | Show HN / Reddit / X | the posts; has a "fire-today" update block with current numbers |
| `2026-10-release-notes.md` | GitHub Release | repo has **no tags or releases** yet; body + command ready |
| `2026-10-blog-post.md` | Ryan's own site | evergreen SEO piece; the honesty story told through three concrete bug fixes |

## Rules for whoever fires these
- No fabricated numbers, testimonials, ratings, or user counts — ever.
- Every number must match `docs/TRACK_RECORD.md` **at post time**; if it has moved, use the
  new one (the drafts say this too).
- Never post outward-facing content from an agent tick. These are for Ryan to publish.
