"""Risk governor — the realized-P&L kill switch.

The repo defines ``max_daily_loss_pct`` and ``max_drawdown`` in settings but
enforces them *nowhere* in the order path. This module closes that gap. It is
deliberately small, pure where possible, and has no dependency on the trading
strategy so it can wrap any execution path.

Equity model: total equity = available cash + portfolio (position) value, both
read from Kalshi's ``/portfolio/balance`` response (``balance`` and
``portfolio_value``, in cents). The governor persists a start-of-day baseline
and a rolling peak to a local JSON file, then evaluates two limits plus an
owner-operated manual halt:

  * DAILY_LOSS  — equity down >= ``max_daily_loss_pct`` from start-of-day.
  * DRAWDOWN    — equity down >= ``max_drawdown_pct`` from the rolling peak.
  * MANUAL_HALT — a flag file the owner can drop to stop all new buys at once.

When halted, the caller must block *new buys* but is still free to place exits
(sells), which is how you reduce risk rather than add to it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_STATE_PATH = "data/runtime/risk_governor_state.json"
DEFAULT_HALT_FLAG_PATH = "data/runtime/TRADING_HALTED"


@dataclass
class GovernorDecision:
    """Outcome of a governor check. ``halted`` gates new buys, not exits."""

    halted: bool
    reasons: List[str]
    current_equity_cents: int
    sod_equity_cents: int
    peak_equity_cents: int

    @property
    def daily_pnl_cents(self) -> int:
        return self.current_equity_cents - self.sod_equity_cents

    @property
    def daily_loss_pct(self) -> float:
        """Positive == down from start-of-day; negative == up."""
        if self.sod_equity_cents <= 0:
            return 0.0
        return (self.sod_equity_cents - self.current_equity_cents) / self.sod_equity_cents * 100.0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity_cents <= 0:
            return 0.0
        return (self.peak_equity_cents - self.current_equity_cents) / self.peak_equity_cents * 100.0

    def to_dict(self) -> Dict:
        return {
            "halted": self.halted,
            "reasons": self.reasons,
            "current_equity_cents": self.current_equity_cents,
            "sod_equity_cents": self.sod_equity_cents,
            "peak_equity_cents": self.peak_equity_cents,
            "daily_pnl_cents": self.daily_pnl_cents,
            "daily_loss_pct": round(self.daily_loss_pct, 2),
            "drawdown_pct": round(self.drawdown_pct, 2),
        }


def evaluate_risk(
    current: int,
    sod: int,
    peak: int,
    max_daily_loss_pct: float,
    max_drawdown_pct: float,
) -> GovernorDecision:
    """Pure limit check. Boundaries are inclusive (exactly at the limit halts)."""
    reasons: List[str] = []
    if sod > 0 and current <= sod * (1.0 - max_daily_loss_pct / 100.0):
        down = (sod - current) / sod * 100.0
        reasons.append(
            f"DAILY_LOSS: down {down:.1f}% from start-of-day (limit {max_daily_loss_pct:.0f}%)"
        )
    if peak > 0 and current <= peak * (1.0 - max_drawdown_pct / 100.0):
        down = (peak - current) / peak * 100.0
        reasons.append(
            f"DRAWDOWN: down {down:.1f}% from peak (limit {max_drawdown_pct:.0f}%)"
        )
    return GovernorDecision(
        halted=bool(reasons),
        reasons=reasons,
        current_equity_cents=current,
        sod_equity_cents=sod,
        peak_equity_cents=peak,
    )


def roll_state(prev: Optional[Dict], current: int, today: str) -> Dict:
    """Advance the persisted baseline.

    New day (or no prior state): start-of-day resets to today's opening equity;
    the rolling peak carries forward (max of prior peak and current). Same day:
    start-of-day is frozen; the peak only ratchets up.
    """
    if not prev or prev.get("date") != today:
        prev_peak = int(prev.get("peak_equity_cents", 0)) if prev else 0
        return {
            "date": today,
            "sod_equity_cents": int(current),
            "peak_equity_cents": max(prev_peak, int(current)),
        }
    return {
        "date": today,
        "sod_equity_cents": int(prev["sod_equity_cents"]),
        "peak_equity_cents": max(int(prev.get("peak_equity_cents", current)), int(current)),
    }


class RiskGovernor:
    """Stateful wrapper: persists baseline/peak, fetches equity, evaluates limits."""

    def __init__(
        self,
        kalshi_client=None,
        state_path: str = DEFAULT_STATE_PATH,
        halt_flag_path: str = DEFAULT_HALT_FLAG_PATH,
        max_daily_loss_pct: float = 10.0,
        max_drawdown_pct: float = 15.0,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.kalshi_client = kalshi_client
        self.state_path = Path(state_path)
        self.halt_flag_path = Path(halt_flag_path)
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _today(self) -> str:
        return self._clock().strftime("%Y-%m-%d")

    def load_state(self) -> Dict:
        try:
            return json.loads(self.state_path.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def save_state(self, state: Dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    # --- manual kill switch -------------------------------------------------
    def is_manually_halted(self) -> bool:
        return self.halt_flag_path.exists()

    def trip_manual_halt(self, reason: str = "manual halt") -> None:
        self.halt_flag_path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_flag_path.write_text(reason)

    def clear_manual_halt(self) -> None:
        try:
            self.halt_flag_path.unlink()
        except FileNotFoundError:
            pass

    # --- equity -------------------------------------------------------------
    async def current_equity_cents(self) -> int:
        if self.kalshi_client is None:
            raise RuntimeError("RiskGovernor needs a kalshi_client to read live equity")
        bal = await self.kalshi_client.get_balance()
        cash = int(bal.get("balance", 0) or 0)
        portfolio = int(bal.get("portfolio_value", 0) or 0)
        return cash + portfolio

    # --- the check ----------------------------------------------------------
    async def check(self, current_equity_cents: Optional[int] = None) -> GovernorDecision:
        if current_equity_cents is None:
            current_equity_cents = await self.current_equity_cents()

        state = roll_state(self.load_state(), current_equity_cents, self._today())
        self.save_state(state)

        decision = evaluate_risk(
            current_equity_cents,
            state["sod_equity_cents"],
            state["peak_equity_cents"],
            self.max_daily_loss_pct,
            self.max_drawdown_pct,
        )

        if self.is_manually_halted():
            try:
                reason = self.halt_flag_path.read_text().strip() or "manual halt flag present"
            except OSError:
                reason = "manual halt flag present"
            decision.halted = True
            decision.reasons = [f"MANUAL_HALT: {reason}"] + decision.reasons

        return decision
