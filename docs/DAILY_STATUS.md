> Operator log, dated, not documentation. Figures may be stale. The risk governor is wired into the order path; no correlated-family cap exists in code.

# Daily status — 2026-10-02 22:30 UTC

## Account
* Equity ≈ $929 (cash $484.49 + marks ~$444.42); 12 active positions, exposure $423.40.
* Governor: not halted (0.3% from re-baseline peak). Risk constitution absolute:
  5% daily / 20% drawdown / 10% position / 15% family cap.
* ⚠️ KXALIENS-27 grandfathered at 16.1% of equity vs the 10% cap — no adds permitted;
  no trim without a model signal (long-dated, bid-thin).

## First forward-only edge verdict (headline metric now LIVE)
* Fixed: the settle→journal reconcile dropped `settled_time`, so `cli edge` had never
  scored a single forward-settled trade. Add-only fix (won/pnl immutable); 310 tests pass.
* Verdict: **NO MEASURED EDGE** — n=55 forward trades, won 58% vs 60% implied, edge −1.9 pts.
  Brier 0.2356 (better than coin flip), log-loss 0.6817.
* Splits that matter: **NO side +7.6 pts edge (n=37)**; YES side −22.7 pts (n=17).
  Calibration overconfidence ([0.50,0.85]: pred 73% vs actual 44%) matches the Edge Policy
  haircuts — keep shrinking YES-side estimates, keep biasing toward NO.
* Verdict published in docs/TRACK_RECORD.md, losses included.

## Model scorecard
* Path model (diesel weekly): first live tranche filled 9/29 — 5 YES KXDIESELW-26OCT05-T6.30
  @ 0.77 (maker, max loss 0.4% eq). Settles on the Oct 5 print. Family size doubles on a win.
* Diesel daily ladder: model weak on evidence (both scored days lost to book) — alerts observed,
  not traded, per policy.
* ORNN Sep >1.000: settled LOSS — Sep mean 0.99967 (missed by 0.0003); 115 YES → $0.
  Economic −$61.35, pre-marked to ~0 since 9/28, equity unaffected. Redemption trade resting:
  5 NO KXA100MS-26OCT-1.000 @ 0.80 (max risk $1.00; model P(Oct mean>1.00) ≈ 8%).
* Oct index so far: 0.91, 0.90 — needs 1.0059 avg from 29 remaining (dead by drift).
* State family (21 states): alerts batched 18:32 UTC; forward calibration audited across all
  21 states (beta 0.80–0.98) — tradeability decided by accumulating forward scores.
* Weather v1: capture live (GEFS-31 + NWS obs, ET-day bucketed); first calibration datapoints banking.

## Infrastructure
* git history: 4.16 GB of data/external parquet backfill removed from branch history
  (gitignored); repo pushed clean through ba5ba91. `.git` 3.0 GB → 1.3 MB.
* ornn stop-check now month-aware (targets the active KXA100MS-26MMM-1.000 strike,
  exits gracefully on resolved months).
* Evening repricing 22:20 UTC: diesel path + daily ladders (25 strike alerts logged).

## Pending decisions
* Oct 5 diesel weekly print settles the first path-model tranche — score forward, act per rules.
* State family: keep accumulating forward scores before sizing.
* Daily ladders stay un-traded until the model beats the book on scored evidence.
