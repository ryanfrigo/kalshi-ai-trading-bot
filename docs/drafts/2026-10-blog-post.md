# Blog post draft — for Ryan's own site (ryanfrigo.com), not a third-party platform

_Status: draft. Nothing published. See `README.md` in this folder for the launch sequence._

---

**Title options**
1. We spent three weeks trying to prove our own trading bot had an edge. It doesn't.
2. The most useful thing we shipped was a verdict against ourselves.
3. What a prediction-market bot taught us about measuring anything honestly

**Suggested slug:** `we-measured-our-trading-bot-and-it-had-no-edge`

---

We built an open-source toolkit for trading Kalshi prediction markets. It has a risk
governor, a signed API client, an MCP server, and a dashboard. None of that is the
interesting part.

The interesting part is that the flagship feature is a command called `edge` whose job is to
tell us we are wrong.

## The problem with every other bot

Search "AI trading bot" and you will find a wall of repos that claim an edge and prove
nothing: a screenshot of a green P&L, a handful of settled markets, a confident README. The
claims are unfalsifiable by construction.

So we built the measurement first.

`cli edge` reconciles our own decision journal against Kalshi's authoritative settlements and
scores three things: Brier, log-loss, and **edge vs the book** — our realized win rate minus
the average price we paid. It counts only trades where the market resolved *after* we traded,
and with fewer than ten of those it refuses to give a verdict at all.

## The verdict

**No measured edge.** Not "slightly negative" — *indistinguishable from the book*, at a
sample large enough to say so.

We published it. It is on the repo's front page, losses included, and it has been the single
most useful thing we have shipped, because it changed what we built next.

## Four things we got wrong, and the fix for each

**1. We printed a win out of noise.** Our harness reported *"BEATING THE BOOK by 0.1 pts"* —
a margin of a tenth of a point over sixty trades. That is not an edge; that is arithmetic
noise wearing a medal. We added a floor: a positive edge under one point now reports as
**within noise**.

**2. We counted rows instead of events.** One realized price print generates a whole ladder
of strike contracts that all settle together. Our scoring briefly credited a model with
beating the book on "n = 85" observations that were really **four** independent prints. Now
the report prints the event count next to the sample size, and refuses to claim anything
without enough events.

**3. We traded a price we could not get.** A stored snapshot showed a market at 0.68; the
live book was 0.85. Separately, a ladder that looked like a comfortable 25-point fade at
midday had repriced from 0.62 to 0.99 by evening. If we had acted on the morning file, we
would have bought exactly the wrong side. Rule now: price off the live book, at the moment
you decide.

**4. Our kill switch reported success while doing nothing.** `close-all` is the command you
reach for when you need to flatten the book in a hurry. It printed *"No open positions on
Kalshi"* on an account holding thirteen — it filtered on a field the API had renamed, and
three more renames hid behind that first one. The same class of silent failure had killed
**51 consecutive** scheduled runs without anyone noticing: two data tools could not find their
browser under the scheduler's minimal `PATH`, and the scheduler swallowed every step failure.
A guard that never fires and a guard that is broken look identical from the outside. We now
compile every module in CI, and scheduled step failures are written to a file the daily report
reads.

## Why publish a losing record

Because the alternative is marketing.

A measurement harness that only ever prints flattering numbers is not a measurement harness.
The gate that blocks our own losing categories, the overconfidence haircuts that shrink our
own estimates, the refusal to claim edge on thin data — none of it means anything if the
front page hides the verdict.

And practically: the honest verdict is what made the thing worth using. People can tell the
difference between a tool that flatters them and a tool that constrains them.

## What we'd tell anyone building one of these

- Measure before you trade. Your first result will be uncomfortable; that is information.
- Write down your estimate *before* the market resolves. Everything else is hindsight.
- Count events, not rows.
- Price off the live book.
- Below a few points of edge, fees eat you slowly enough that you'll never notice.
- Expect the answer to be no. Most of the time it is — and now we can tell.

The toolkit is MIT-licensed, and the full track record is in the repo, losses and all.

---

**Notes for publishing:** link the repo and the free playbook at the end; do **not** add a
call to action for the paid guide inside the post itself (a footer link is fine). All
figures must match `docs/TRACK_RECORD.md` at publication time.
