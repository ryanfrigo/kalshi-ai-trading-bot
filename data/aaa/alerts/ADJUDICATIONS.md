


## 2026-10-23 (later) — WHY THE DIESEL-WEEKLY "EDGE" WAS NOT TRADEABLE

Follow-up to the refresh above. The 12:50 UTC `KXDIESELW` path snapshot showed a fat
apparent edge (`T6.32`: model fair 0.368 vs book bid 0.64 / ask 0.78 → +25c "buy NO").
Checked it against the live book at ~18:07 UTC: **the book had repriced hard.**

| ticker | snapshot (12:50) | live (18:07) |
|---|---|---|
| KXDIESELW-26OCT26-T6.32 | yes_bid 0.64 / ask 0.78 | yes_bid **0.99** (prev close 0.62) |
| KXDIESELW-26OCT26-T6.34 | yes_bid 0.03 / ask 0.39 | yes_bid **0.79** (prev close 0.30) |

Verified this is a real repricing, not a stale-quote bug: `aaa_pricer.py` reads the live
orderbook (`orderbook_fp`), and the market's own `previous_price_dollars` moved
0.62 → 0.99. The model's input is also current (`data/aaa/aaa_daily.csv` has today's
print, 6.3172, matching the snapshot's `last_print`).

**The disagreement is drift, not data.** The model's implied median is ~6.31 (flat);
the market's is ~6.33-6.34 (a ~2c rise by Oct 26). Because these distributions are very
tight, a 2c median difference explodes into a 60-point probability difference (0.37 vs
0.99) — i.e. *the "edge" is a statement about the drift assumption, nothing else.*

**Lesson (operational, keep):** a diesel-weekly pricing snapshot goes stale within hours
on these markets. **Never act on a stored snapshot's edge — recompute off the LIVE book at
decision time**, and treat a large model-vs-book gap on a tight distribution as a claim
about drift, not a gift.
