# Daily status — 2026-10-23

_Snapshot, not a live feed. Current figures always come from `cli.py report` →
[docs/TRACK_RECORD.md](TRACK_RECORD.md)._

## Account
* Equity ≈ **$14,673** (cash $8,102.75 + marks ~$6,550); 13 active positions.
* Governor: **not halted**. Day P&L **+$423.79**; drawdown **0.0%** from the all-time peak.
  Risk constitution absolute: 5% daily / 20% drawdown / 10% position / 15% family cap.
* ⚠️ A resting **buy of 8,000 `CONTROLH-2026-D` @ 0.913** (~$7,304 if filled) sits far
  above the 10%-of-equity cap ($1,467). It was not placed by the `trade` tool. Flagged,
  not touched — it needs an explicit call from the operator.

## Edge verdict (forward-only, the headline)
* **NO MEASURED EDGE** — edge **+0.1 pts over 60 forward trades is within noise**
  (won 62% vs. 62% implied); Brier 0.2114 (n=48), log-loss 0.6150 (n=48).
* Harness change this cycle: a positive edge below the new `_MIN_EDGE_PTS = 1.0` floor no
  longer prints "BEATING THE BOOK" — a sub-point edge is noise, and saying otherwise was
  the overclaim this toolkit exists to refuse.
* Full settlement history: 232 settled markets, 65% win rate, **net realized +$465.58**
  (NO 110 @ 78% WR +$229.09; YES 122 @ 52% +$236.49).
* Calibration overconfidence persists: [0.50,0.85] predicted 73% vs actual 44%;
  [0.90,0.95] predicted 92% vs actual 75% → the Edge Policy haircuts encode this.

## Edge Policy (the gate the record earns)
12 categories blocked (Economics −$245, KXMAKEMARMAD −$310, Politics −$150, KXCPI −$67,
KXAISPIKE −$52, KXNBAPLAYOFF −$52, KXETH15M −$39, …). Derived from 251 settled trades.

## Model scorecard
* **AAA / state gas** — model proven bad forward vs a near-perfect book. No size.
* **Diesel path model** — the only live development thread; needs a forward sample before
  it earns size.
* **Ornn / orshare / TruthSocial** — watched, forward-scored, not yet beating the book.
* **No live trades this cycle** — every candidate failed the 5¢ fee-aware floor, or sat in
  a policy-blocked / efficient-liquid category. The sharpest near-miss was
  `KXIPOANTHROPIC-DATE-26NOV16` (NO 0.86): research confirmed the book's timing read and
  killed the fade.

## Repo / infrastructure
* Track-record renderer: the all-time peak now **ratchets with deposits**, so a funded
  account can never print a negative drawdown.
* Docs: added the free **Operator's Playbook** (`docs/OPERATOR_PLAYBOOK.md`, linked from
  the README nav) and launch-post drafts under `docs/drafts/`.
* **PR #95** (branch → main) carries the 10/3–10/13 data cycle plus the above; CI green.

## Pending decisions
* Merge **PR #95**.
* The 8,000-lot `CONTROLH-2026-D` resting order: intended accumulation, or a cap breach to
  cancel?
