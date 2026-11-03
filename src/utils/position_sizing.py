"""Canonical position-sizing math.

This module is the single home for the Kelly criterion kernel. It used to be
reimplemented (with subtle differences) in safe_compounder, and twice in
portfolio_optimization — which is exactly how sizing bugs are born. Strategy
modules keep their own *policy* (fractional multipliers, regime/time-decay
adjustments, caps), but the underlying bet math lives here.

Behavior is pinned by tests/test_position_sizing.py. Do not change the math
here without updating those characterization tests deliberately.
"""

from __future__ import annotations

import math


def kelly_fraction(prob_win: float, payout_ratio: float) -> float:
    """Kelly fraction for a binary bet: f* = (p*b - q) / b.

    Args:
        prob_win: Probability the bet wins, in [0, 1].
        payout_ratio: Net odds ``b`` — profit per unit staked if the bet wins
            (e.g. a YES contract bought at $0.40 pays $0.60 profit on $0.40
            staked: b = 0.6/0.4 = 1.5).

    Returns:
        The fraction of bankroll to stake, floored at 0.0 (never bet a
        negative edge). Invalid inputs (non-positive odds or probability)
        return 0.0 rather than raising.
    """
    if payout_ratio <= 0 or prob_win <= 0:
        return 0.0
    # Fail closed on a non-finite or out-of-range probability (> 1 would size
    # a bet larger than the bankroll). Valid inputs are untouched.
    if not math.isfinite(prob_win) or prob_win > 1.0:
        return 0.0
    prob_lose = 1.0 - prob_win
    f = (prob_win * payout_ratio - prob_lose) / payout_ratio
    return max(0.0, f)


def binary_market_payout_odds(market_probability: float, bet_yes: bool = True) -> float:
    """Net payout odds ``b`` for a binary market priced at ``market_probability``.

    For a YES bet at price P: profit is (1-P) on P staked  -> b = (1-P)/P.
    For a NO  bet at price P: profit is P on (1-P) staked  -> b = P/(1-P).

    Degenerate prices (P=0 for YES, P=1 for NO) raise ZeroDivisionError;
    callers decide their own fallback for untradeable prices. A NaN or
    out-of-[0,1] price raises ValueError.
    """
    if not (0.0 <= market_probability <= 1.0):  # also False for NaN
        raise ValueError(f"market_probability must be in [0, 1], got {market_probability!r}")
    if bet_yes:
        return (1 - market_probability) / market_probability
    return market_probability / (1 - market_probability)
