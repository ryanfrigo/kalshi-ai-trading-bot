


## 2026-10-23 (CORRECTION) — the diesel "n=85 / n=80" is really 4 and 2 EVENTS

The refresh above reported the model beating the book on `KXDIESELW` (n=85) and
`KXDIESELMON` (n=80). **That n counts strike-observations, not independent events** — one
realized print generates the entire strike ladder. By distinct target print:

| series | obs | **events** | model Brier | book Brier | winner |
|---|---|---|---|---|---|
| KXDIESELW | 85 | **4** | 0.0306 | 0.0925 | model |
| KXDIESELMON | 80 | **2** | 0.0477 | 0.0576 | model |
| KXDIESELD | 101 | 9 | 0.1176 | 0.0578 | book |
| KXAAAGASD* states | 82–126 | 9–13 | 0.04–0.44 | 0.01–0.10 | book |

So the diesel-wins result rests on **four and two prints**. That is not a forward sample —
it is exactly the "assert skill from a handful of lucky settlements" failure this toolkit
exists to refuse. **Retract the line above that said diesel W/MON "earned a forward-scored
right to be considered": they are UNPROVEN, not proven.** The states stay proven bad (each
has 9–13 independent events and loses decisively).

New tool: `scripts/aaa_pricer.py summary` prints obs + **events** + Brier per series, so
the effective sample is always visible and this mistake cannot recur silently.
