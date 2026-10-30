# Jev on Kalshi: what worked, what didn't

[Jev](https://openrouter.ai/~typesafe/jev-latest) is TypeSafe's "decisions" model. You don't chat with it. You send a `state` (facts) and typed questions, and it sends back calibrated probabilities in about 0.3 seconds for roughly $0.00002 a call. It isn't on OpenRouter's chat endpoint at all — it lives at `POST https://openrouter.ai/api/alpha/decisions`, and `~typesafe/jev-latest` resolved to `typesafe/jev-1.13-20260917` when I tested it.

A fast, cheap, calibrated probability engine sounds made for prediction markets. So on 2026-09-25 I pointed it at a live account and measured it. Most of what I expected was wrong.

## The short version

| Test | n | Result |
|---|---|---|
| Blind pricing (rules only, no price) vs the Kalshi book | 601 settled markets | Brier **0.209** vs book **0.157**; trading the disagreements lost at every threshold |
| Fed live settlement data (gas, weather, RT, OpenRouter share, TSA...) | 1,235 markets, 10 families | squashed into ~0.25-0.75 regardless of strike; mutually exclusive buckets summed to 1.67 and 2.55 |
| "Does this text satisfy this rule?" | 207 posts / 107 mention cases | 206/207 and 92/107 correct |
| Headline x market-rule matching | 281 | 167/281 — weak |
| Throughput as a classifier | — | ~50,000 text x rule pairs a minute |

It's a good text classifier. It's a bad pricer.

## Blind pricing loses to the book

`scripts/jev_backtest.py` reproduces this. I took every market that settled Sep 19-25 (after Jev's build date, so it couldn't have seen the outcomes), read the book mid from Kalshi candlesticks on Sep 18, and asked Jev for P(YES) as of Sep 18 from the rules alone. No price in the prompt, because the price anchors it.

```
ALL     n=601  Brier book 0.1568  jev 0.2090  blend 0.1707
  disagreement > 0.05: 526 trades, P&L $-17.39
  disagreement > 0.30:  91 trades, P&L $-4.66
```

The raw results file is not committed. `scripts/jev_backtest.py` needs live Kalshi and OpenRouter calls to run.

Jev compresses everything toward 50% — all 601 forecasts landed between 0.2 and 0.7, so it never once called a market nearly settled in either direction, and averaging it with the book made the book's own score worse. The sample was 98% sports (that's what settles in a random week on Kalshi), so non-sports is barely tested — but nothing here suggests a reason to keep going.

## Feeding it data didn't fix it

So I gave Jev the settlement source itself. I ran 10 families that settle on a public feed — AAA gas, NWS weather, Rotten Tomatoes, OpenRouter's rankings JSON, Truth Social counts, YouTube views — and asked for P(YES) with the fresh numbers in the state.

It can't do threshold arithmetic. It said 0.27 for a gas contract that needed a 49¢ crash in five days. It gave 0.62-0.92 on weather markets that had already settled. When I switched to the `choice` question type the buckets finally summed to 1, but Rotten Tomatoes at 86.54% came back with 29% on "84 or lower" and 6% on "exactly 85". Plain arithmetic beat it on every disagreement.

In every liquid data family the book was already ahead of the public source, and that matters more than anything about Jev. Gas traders price off GasBuddy before AAA prints. Weather books reprice within about a minute of NWS products. The edges that survived weren't latency — they were structural lags: the weekend usage mix on OpenRouter, and monthly rain books that ignore the forecast entirely.

## Reading text is the one job it does well

Mention markets ("will Trump say X") and count markets ("how many endorsements") resolve on text. That's Jev's job. `scripts/jev_classify.py` batches 20 questions per call, 8 calls in flight:

```bash
python scripts/jev_classify.py input.json > out.json
# input.json = {"rule": "<exact market rule>", "items": [{"id": "...", "text": "..."}]}
```

On one market's word-form rules it went 16/16 — "tariffs" counts, "tariffed" and "counter-tariff" don't. Long rule text hurts it; paste only the operative clause.

Speed still doesn't win the race. Bots price new Truth Social posts in 0-5 seconds and sweep live mention markets within about a second of him saying the words, and any pipeline that has to turn audio into text first shows up roughly ten seconds late to a book that's already at 99¢. Classification isn't the bottleneck.

What it did do: confirm *absence* across thousands of transcripts and posts fast enough to make base-rate fades cheap. Trump hadn't said "North Korea" since Aug 25 or "whack job" since Jul 1. Retail pays up for those YES longshots anyway. I filter with Jev, then read every hit myself.

## Commands and code

- `python cli.py verify --ticker T --research-file r.json --jev` floors the skeptic's true-YES at Jev's independent P(YES). It can only make the gate stricter. In practice it vetoes almost every sound trade (because of the compression above), so I don't run it by default.
- `src/agent/jev.py` holds the request builders and the batch classifier, and `tests/test_agent_jev.py` pins them.
- Never put the market price in the state.

## The part I got wrong

I started out treating Jev as the decision engine and the account as the test. That was backwards. The best trade of the weekend came from reading a rule, not from any model: OpenRouter share markets settle all-NO if an author isn't named in the top-9 chart, and a surging stealth model was about to push one out. Kalshi settled Tencent all-NO that exact way on Aug 24. The book priced Xiaomi's 1.9% strike at 98¢ YES anyway. That position is still open.

Read the rules.
