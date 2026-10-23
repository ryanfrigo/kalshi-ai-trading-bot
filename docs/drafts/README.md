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

## Evidence check (2026-10-23, GitHub traffic API)

Before you choose a channel, the data says this audience arrives through **search**:
14 days — 729 repo views / 418 uniques, top referrers **Google 172 (139 uniques)**, github.com 61,
brave 26, Bing 23, DuckDuckGo 12, botforkalshi.com 9, t.co 9; and `docs/TRACK_RECORD.md` gets
direct visits. That *lowers* the expected value of a one-off HN post and *raises* the value of
anything that ranks: the free field map, the two calculators, the sitemap and the structured data.
A post is still worth firing — just not first.

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
