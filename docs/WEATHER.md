# Weather family (KXHIGH*) — build plan + calibration log

_Built 2026-09-28. Capture runs daily from `scripts/run_morning.sh`. No trades, no size, and no forward score yet: the pricing and scoring steps below are not built._

## Why this family

* Kalshi weather daily-high ladders (KXHIGHNY/CHI/MIA/LA/AUS/ATL/DEN/PHL), settled
  on the NWS Daily Climate Report — a public, fast settlement source (the actual
  source, like AAA for gas).
* Becker 2026: Weather category maker-vs-taker gap 2.57pp (vs Finance 0.17pp) —
  there is real flow to provide liquidity against.
* Free probabilistic inputs exist: Open-Meteo ensemble API (GEFS 31 members),
  ECMWF open-data ensembles (50 members), NBM (blend, best short-range per the
  polymarket-kalshi-weather-bot validated research).

## First probe (2026-09-28, markets for Sep 29)

GEFS 31-member daily-max vs live book mid (F, day+1):
* NYC: ensemble mean 73.3 (std 2.1) vs book mode ~69-70  -> ~3F apart
* CHI: ensemble mean 74.3 (std 0.9) vs book mode ~76-78  -> ~3F apart, opposite sign
* MIA: ensemble 83.6 vs book mode ~85-86

**Reading: a single-model GEFS pull is NOT market-grade.** The book prices a
multi-model blend (NBM/ECMWF/HRRR). Opposite-sign gaps across cities rule out a
units bug. Conclusion: build the multi-model pipeline BEFORE trusting any edge;
log ensemble-vs-book every day from now on so the ensemble's bias vs settlement
becomes measurable (the ensemble may still win out-of-sample at longer leads).

## Build plan

1. `scripts/weather_data.py`: daily capture (a) Open-Meteo GEFS + ECMWF-ensemble
   daily-max distributions per city, (b) NWS Climate Report actuals (settlement
   source) -> data/weather/. Runs from run_morning.sh.
2. Bias-correct each ensemble against the NWS-station history (gridpoint-vs-station
   offset), blend by recent out-of-sample performance.
3. Price ladders (above/below/bucket strikes) -> `data/weather/pricings/`,
   forward-score vs settled NWS actuals like the AAA family.
4. Alert-only until the model beats the book over n>=50 strike-observations.

## Probe snapshot (for future calibration)

NYC ens [mean 73.3 sd 2.1] book: <69: 0.26, 69-70: 0.40, 71-72: 0.29
CHI ens [mean 74.3 sd 0.9] book: <74: 0.01(?), 76-77: 0.45, 78-79: 0.36
MIA ens [mean 83.6 sd 1.2] book: <83: 0.08, 85-86: 0.36, 87-88: 0.29
