# Launch drafts — the honest track record + the Operator's Playbook

_Status: DRAFTS for Ryan to publish. Nothing here has been posted. Do not post
outward-facing content from an agent tick (autonomy-loop rule 3)._

**What is being launched:** the repo (tools + the honest live track record), the free
**Operator's Playbook**, and the paid **Build & Run an AI Kalshi Bot** guide
(https://kalshi-community.vercel.app).

**Rules for whoever fires these:** no fabricated numbers, no "guaranteed", no
testimonials, no income claims. Link only what exists. If a number is used, it must
match `docs/TRACK_RECORD.md` at post time.

---

## 1. Show HN (primary)

**Title:** Show HN: An open-source Kalshi bot that measures whether you actually beat the book

**URL:** https://github.com/ryanfrigo/kalshi-ai-trading-bot

**Body:**

> Most prediction-market bots claim an edge and prove nothing. This one does the
> opposite: it ships the harness to test whether you beat the price the sharp book
> charged you — and it will refuse to claim edge on thin data.
>
> The headline is `cli.py edge`: it reconciles your decision journal against Kalshi's
> own settlements and scores Brier, log-loss, and *edge vs book* (your realized win
> rate minus the mean price you paid). It only counts **forward-settled** trades, and
> below 10 of them it refuses to give a verdict. On our real account the first honest
> verdict was "no measured edge" — we published it anyway, losses included
> (docs/TRACK_RECORD.md).
>
> There's a second loop: `cli.py improve` turns your settled record into a pre-trade
> gate that blocks the categories your own record proves you lose money in, and
> haircuts the estimate bands where you're overconfident. Measurement constrains the
> next trade, automatically.
>
> Also included: a signed Kalshi client, a risk governor (daily-loss + drawdown kill
> switch), an MCP server, and example strategies that are starting points, not
> turnkey money-makers — the README says so plainly.
>
> MIT, Python. Happy to answer questions about the edge math or why we ship the
> "no edge" verdict.

---

## 2. r/algotrading or r/PredictionMarkets

**Title:** I built a Kalshi bot that refuses to claim an edge until you prove one (Brier/log-loss/edge-vs-book, forward-only)

**Body:**

> I kept seeing Kalshi bots assert profitability from a handful of lucky settlements,
> so I built the opposite: a measurement harness first.
>
> - `cli edge` scores your settled trades (Brier, log-loss, and edge vs the book's
>   implied price) using **only forward-settled** trades — where the market resolved
>   strictly after you traded. Under 10 of them, no verdict.
> - `cli improve` derives a machine-readable gate from your record: block categories
>   with ≥5 settled trades and negative P&L; haircut the estimate bands where you're
>   overconfident. It only ever tightens from evidence.
> - The risk governor is authoritative for every live order: daily-loss and drawdown
>   kill switch, manual halt file, hard position cap.
>
> Our own account's first forward verdict was "no measured edge (-1.9 pts)". It's
> published in the repo, losses and all. That honesty is the point.
>
> Repo + the docs (there's a free operator's playbook for the setup traps — the
> private-key 401 wall, reading the silence, keeping it alive).

---

## 3. X / Twitter thread (short)

1/ Most Kalshi bots claim an edge. This one measures one — and publishes "no edge"
when that's the truth. Open source, MIT.
2/ `cli edge`: Brier, log-loss, and **edge vs book**, on forward-settled trades only.
Under 10 forward trades, it refuses to guess.
3/ `cli improve` closes the loop: your settled record auto-blocks the categories you
lose in and shrinks the estimates where you're overconfident.
4/ The first honest verdict on our own live account: **no measured edge (-1.9 pts).**
Publishing it is the feature.
5/ New: the **Operator's Playbook** (free) for the setup traps — auth walls, "why is
it silent", staying online. And a full guide for running it properly.
6/ Repo → {REPO_URL} · Playbook + guide → {SITE_URL}

---

## 4. Release-note style (for a GitHub Release)

**Tag:** v1.x — "the honest edge layer + operator's playbook"
- Adds `docs/OPERATOR_PLAYBOOK.md` (the free field guide).
- Fixes the track-record renderer: the all-time peak now ratchets with deposits, so a
  funded account can never publish a negative drawdown.
- Edge harness, policy gate, and risk governor unchanged in behaviour.

---

## Sequencing suggestion

1. Publish the repo README line linking the playbook + site (traffic target exists).
2. Fire Show HN or the Reddit post (pick one; HN skews better for "honest tooling").
3. Reply to every substantive comment with real numbers from `docs/TRACK_RECORD.md`.
4. Only then push the paid guide in a follow-up ("the playbook got X requests, here's
   the deeper guide") — do not lead with the paid thing.
