# AAA gas/diesel markets: mechanics, data, model

_Family: `KXDIESELD` (daily diesel), `KXAAAGASD` (daily gas), `KXDIESELMON`,
`KXDIESELW`, `KXAAAGASMAXM`, `KXAAAGASM`, `KXAAAGASW<NJ/...>`. Built 2026-09-26._

## Market mechanics (verified against settled prints)

* Each market resolves on the **AAA Fuel Gauge national daily average** for date D
  ("If the Diesel Price on September 27, 2026 is above $6.480..."). A 21-strike
  ladder at 0.5c increments is listed per day.
* The market for date D **closes before the D print is published** (close ~1:59am
  ET vs the AAA update ~3-4am ET). You can never trade the print after seeing it;
  pricing = forecasting the next print from the prior print.
* Ladder books must be read through the complement: a NO bid at `p` is a YES ask
  at `1-p`. With spot `S` (the last print), the ladder directly quotes the
  implied distribution of the next print around `S`.

## Data

* `scripts/aaa_data.py` — the print series. The AAA page shows
  `Current Avg.` / `Yesterday Avg.` (the two most recent prints). The page
  sometimes serves stale pairs (CDN cache, pre-publish captures), so calendar
  day ≠ print day; the model therefore uses the **page-pair delta**
  (Current − Yesterday) — the change between consecutive prints, calendar-free.
  * `backfill` rebuilds history from Wayback Machine snapshots (~20 months,
    ~550 pages, ~294 usable deltas as of 2026-09-26; archive.org is slow — the
    fetch is resumable and cached under `data/aaa/raw/`, gitignored).
  * `today` merges the live page (idempotent per day; `run_daily.sh` calls it).
* `scripts/aaa_futures.py` — wholesale futures (`HO=F` ULSD, `RB=F`, `CL=F`)
  from Yahoo Finance, cached under `data/aaa/futures/`.

## The change distribution (20-month sample, n=294)

mean +0.53c, std 2.53c; median 0; P(≤ −0.4c) = 36%; P(≤ 0) = 53%; p95 +4.4c.
Weekday effects are second-order; weekend prints move nearly as much as weekdays.

## The two forecasting lenses — and the 2026-09-26 lesson

1. **Retail streak model** (episode-based): consecutive-decline streaks
   (≤ −0.4c/day) historically extend ~55-60% of the time; **exact-4-day
   streaks broke 3/3 times** (day-5 deltas +0.2, 0.0, +2.2c). On 2026-09-26 the
   retail-only model priced `P(print(9/27) ≤ 6.480) ≈ 55%` vs the book's ~95% —
   an apparent 40pt edge on YES at 4-5c.
2. **Wholesale convergence model (the one that matters):** retail prints lag
   futures by ~4-7 days (lag-7 regression R² ≈ 0.42). The retail-minus-wholesale
   gap relative to its trailing baseline measures the pending retail decline.
   On 2026-09-26 the gap was **46.7c above baseline** (ULSD fell 5.25 → 4.68,
   −11%, while retail had only given back 4.4c of its spike) — historical
   gap-closing episodes (2026-04: 18c/11d; 2026-05: 49c/23d) run at ~2c/day.
   That fully justifies the book's continued-decline pricing.

**Verdict: the 2026-09-26 ladder was efficient.** The apparent streak-model edge
was an artifact of ignoring the wholesale driver. The books for 9/27 (median
−1.8c) and the weekly/monthly ladders (median 6.455/6.40 vs model 6.45/6.41)
were all consistent with convergence. No trade was warranted; the pricing
snapshot was recorded for forward scoring.

Rule of thumb for this family: **never fade the ladder on retail history alone.
Check the excess gap first (`scripts/aaa_futures.py gap`).** Genuine edge, if it
exists, is more likely in (a) the *rate* of convergence (book too slow/fast vs
the ~2c/day episodes) and (b) the multi-day path (weekly/monthly ladders), not
in tomorrow's single print.

## Forward scoring (how this becomes measured edge)

`scripts/aaa_pricer.py price` writes a snapshot (`data/aaa/pricings/`, gitignored)
with model fair, book quotes and fee-aware edges per strike.
`scripts/aaa_pricer.py score` joins those snapshots with realized prints and
appends per-strike Brier (model vs book mid) to `data/aaa/scores.jsonl`. Over
time the summary answers the only question that matters: **does the model beat
the book out-of-sample?** Until the answer is yes with enough n, this family
gets no size.

## State transfer calibration (unreliable)

`data/aaa/states/CALIBRATION.json` (fitted 2026-09-29, 21 states, 236-364 pairs
each) holds per-state fits of `state_delta = alpha + beta*nat_delta + resid`.
Treat it as unreliable. It was fit on `src=wayback` state rows, and about 91%
of those are national prints: `backfill_state` fetched the national AAA page
for every state snapshot instead of the `?state=` page. The fit is mostly the
national series regressed on itself, so its alpha, beta, corr and resid_sd mean
little. Nothing reads the file at runtime.

Until the rows are re-fetched, the pricer ignores `src=wayback` state rows
(also in `cmd_score` and `scalp_paper.py` settlement). Each state has 4 live
rows, so `fit_state_transfer` (needs 60) and the state own-series path (needs
30) are both skipped. States are priced from their latest live print plus the
national change distribution. That is the model the 2026-09-27 scoring found
biased.
