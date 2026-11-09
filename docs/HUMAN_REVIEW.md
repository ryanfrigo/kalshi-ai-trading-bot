> Operator log, dated, not documentation. Figures may be stale. The risk governor is wired into the order path; no correlated-family cap exists in code.

# Human review queue — for Ryan

_Updated 2026-09-28 ~19:00 UTC. Ryan granted full trading autonomy and confirmed the risk
constitution (see src/config/settings.py). New goal set: autonomous, risk-controlled profit
system; success = positive realized P&L on rolling 30-day windows inside the limits._

## Confirmed operating parameters (2026-09-28)

* **Bankroll: ~$925** (cash $315.60 + marks $609.84). Up from ~$900 baseline two days ago.
* **Limits (in settings.py, Ryan-approved):** 5% max daily loss, 20% peak-to-trough drawdown,
  10% max single position, 15% correlated-family cap. New models earn size only via forward
  performance (start 5 contracts, n>=50 beating the book to double).
* **No purchases**: no pro API (doesn't exist), no paid data for now. Polymarket creds exist on
  the machine but IP-ban risk -> cross-platform scanner is deprioritized.
## 1. KXA100MS-26SEP-1.000 (Sep A100 compute index, >$1.000 strike) — RESOLVED TO HOLD

_Ryan granted full trading autonomy 2026-09-28; the approval request is void. Correction: the
original "exit half at 0.38" was based on misreading the book — the 0.62-0.81 NO prices are
sellers of YES, not buyers. **The YES bid is empty (0.00-0.01, zero volume in 24h)**: there is
no exit at any acceptable price. Autonomous plan now automated (scripts/ornn_stop_check.py):
print ≥ 0.93 → rest GTC sell on half at model-fair−5c; print ≤ 0.88 → sell all into any bid
≥ 0.03; 0.89-0.92 → recompute + decide. Position: 115 YES, cost $61.35.

Live position: **115 YES, cost $61.35 (avg 0.533)**. The 9/27 print came in **0.91 — second
consecutive all-time low** (1.02 → 0.95 → 0.91). Sep mean 1.01037 with 3 prints left; the strike
loses if the last 3 prints average ≤ 0.9067 — i.e. any single print ≤ 0.90 flips it.

* Book now: YES ask 0.29 (130 deep) / 0.38 (1051 deep). Market P(YES) ≈ 0.29–0.38.
* Model: P(YES) 0.39 (recent-30 bootstrap) – 0.53 (full history). No edge either way.
* **Recommendation: exit ~half (57 contracts) into the 0.38 bid now** (recovers ~$21,
  cuts tail risk), hold the rest with a hard exit if the 9/28 print ≤ 0.90. Full exit at 0.38 is
  also defensible. No adds.
## 2. Settling today — informational, no action

OpenRouter share week (Sep 21–27) settled values (from the 05:40Z snapshot; official 10am ET
update will be re-captured at 14:05Z): deepseek 24.4, google 20.5, openai 17.9, anthropic 2.7,
xiaomi 2.5. Our NOs: DEEPSHARE-24.1 −$42.55, OPENSHARE-17.8 −$21.22, OPENSHARE-17.3 −$4.00,
XIAOMI −$4.00/−$4.50, ANTHSHARE-2.8 **+$5.60**. Net ≈ −$70.7. Weekend-dilution lesson recorded:
dilution was real but weaker than last weekend (openai Sunday ~15–16% vs 13.6–14.1% prior);
never size off a single weekend's dilution.

## 3. Your 99c scalping idea — tested, verdict: not mechanical, see below

Full report: `docs/SCALP_STUDY.md` (two independent samples, 2,683-market 1-min study).

* **Taker at the 0.99 ask: loses** — 93.8% win, EV −6.3%/trade on 354 fills. You
  only get filled at 0.99 when the outcome is about to flip (adverse selection).
  Also, when the bid is ≥0.99 the ask is 1.00 in 239/239 cases — the last cent
  usually isn't offered at all.
* **Naive resting maker bid: loses worse** — 77.5% win on 111 fills. Your bid is a
  free option for informed sellers.
* **The level is real**: markets whose bid sat ≥0.99 near close settled ~100%
  (242/242, 155/156, 126/126). Information yes; mechanical trade no.
* The only testable version: maker close-outs **gated by our own pricer**
  (fair ≥ 0.995, gas/diesel ladders, ~2.3 fillable/day, small clips, breakeven
  99.1%, one loss wipes ~460 compounding wins). I will paper-log these going
  forward before any real size.
* **Question for you**: when it "worked for you for a while", were those
  source-locked outcomes you judged manually? If yes, the edge was the judgment —
  share your fills (dates/tickers/entries) and I'll replicate the gate and test it.

## 4. Data sources I'd like (highest value first)

1. **Hourly / paid Ornn OCPI data** — the public feed is daily-settled values only; a large
   seller was right against us on KXA100MS. Hourly OCPI-A100 (paid feed, or any Bloomberg
   export) would show intraday direction before the daily settle. Even a screenshot export is
   useful.
2. **GasBuddy or OPIS access** (or an account you already have) — gas traders price off
   GasBuddy before AAA prints; a pre-print retail gauge would sharpen all KXAAAGAS* families.
3. **Any historical fills/records from your manual 99c scalping** (dates, tickers, entry price)
   — turns the scalp study from estimated to measured.
4. **Your risk budget in numbers**: current bankroll you want me to manage, max single-position
   %, max family %. The docs assume ~6-10% caps; confirm or correct.
5. Kalshi **fee schedule confirmation** — I'm using 0.07 × p × (1−p) per contract; if your
   account has different maker/taker terms, tell me and I'll re-derive every edge calc.

## 5. Note

The repo's old `docs/QUICK_FLIP_STRATEGY.md` (1–20c lottery scalps with AI prediction) is a
different, unshipped idea and I don't rate it; your 99c version is the one worth testing.

## 6. Portfolio mark-to-market (2026-09-28 ~08:00 UTC, equity assumed $1000 — confirm)

Cost $762.75 across 33 positions; conservative bid marks $523 (unrealized ≈ −$240, overstated
where books have no bids — the OpenRouter markets are settled-but-unpaid (≈ −$70 expected) and
the A100 book has no YES bids (real exit ≈ 0.38 → −$19.6 not −$61)). Real clusters:
paranormal/KXALIENS +$12 (biggest single exposure $150), nobel +$14.5, trump_say −$20,
diesel weekly/monthly ≈ −$11 to −$1, ORNN as above. Full report: `scripts/portfolio_risk.py`.
