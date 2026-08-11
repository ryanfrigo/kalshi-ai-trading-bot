"""
Stop-Loss Calculator Module

Implements the 5-10% stop-loss logic recommended by Grok4 performance analysis.
Provides consistent stop-loss calculation across all trading strategies.

Key Features:
- 5-10% stop-loss based on entry price and confidence
- Adaptive stop-loss based on market volatility
- Take-profit targets to lock in gains
- Time-based exit strategies

PRICE CONVENTION — read before editing
--------------------------------------
Every price here is the **price of the contract actually held** ("own-side"
price), never the YES price of the market. On Kalshi you buy a contract: a NO
position is a NO contract bought at the NO ask, and it is worth $1 if NO
resolves. So a position of EITHER side loses value when its own contract price
falls, and gains when it rises. Stop-loss is therefore always BELOW entry and
triggers on a fall; take-profit is always ABOVE entry and triggers on a rise.
There is no YES/NO asymmetry, and the `side` arguments are retained only for
call-site compatibility and logging.

This replaces an earlier YES-space convention (for NO: stop above entry,
trigger on a rise). That convention was self-consistent but every caller fed it
own-side prices — `Position.entry_price` is the fill price of the side traded
and `track.py` passes `current_no_price` for NO positions — so a NO position at
0.95 was compared against a stop of 0.95*1.07 and stopped out on its first
tracking pass, every time.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import math


class StopLossCalculator:
    """
    Centralized stop-loss calculation following Grok4 recommendations.
    
    Implements 5-10% stop-losses to prevent large losses as identified
    in the performance analysis.
    """
    
    # Grok4 recommended parameters
    MIN_STOP_LOSS_PCT = 0.05    # 5% minimum stop-loss
    MAX_STOP_LOSS_PCT = 0.10    # 10% maximum stop-loss  
    DEFAULT_STOP_LOSS_PCT = 0.07 # 7% default stop-loss
    
    # Take-profit targets
    MIN_TAKE_PROFIT_PCT = 0.15   # 15% minimum take-profit
    MAX_TAKE_PROFIT_PCT = 0.30   # 30% maximum take-profit
    DEFAULT_TAKE_PROFIT_PCT = 0.20 # 20% default take-profit
    
    @classmethod
    def calculate_stop_loss_levels(
        cls,
        entry_price: float,
        side: str,
        confidence: Optional[float] = None,
        market_volatility: Optional[float] = None,
        time_to_expiry_days: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate stop-loss and take-profit levels following Grok4 recommendations.
        
        Args:
            entry_price: Position entry price (0.01 to 0.99)
            side: Position side ("YES" or "NO")  
            confidence: AI confidence level (0.0 to 1.0)
            market_volatility: Estimated market volatility
            time_to_expiry_days: Days until market expires
            
        Returns:
            Dictionary with stop_loss_price, take_profit_price, max_hold_hours
        """
        
        # Validate inputs
        entry_price = max(0.01, min(0.99, entry_price))
        confidence = confidence or 0.7
        market_volatility = market_volatility or 0.2
        time_to_expiry_days = time_to_expiry_days or 30.0
        
        # Calculate stop-loss percentage based on confidence
        # Higher confidence = tighter stop-loss (more aggressive)
        # Lower confidence = wider stop-loss (more defensive)
        if confidence >= 0.8:
            stop_loss_pct = cls.MIN_STOP_LOSS_PCT  # 5% for high confidence
        elif confidence >= 0.6:
            stop_loss_pct = cls.DEFAULT_STOP_LOSS_PCT  # 7% for medium confidence  
        else:
            stop_loss_pct = cls.MAX_STOP_LOSS_PCT  # 10% for low confidence
            
        # Adjust for market volatility
        # Higher volatility = wider stops to avoid getting stopped out by noise
        volatility_adjustment = min(1.5, 1.0 + (market_volatility - 0.2))
        adjusted_stop_loss_pct = stop_loss_pct * volatility_adjustment
        
        # Calculate take-profit percentage (inverse of stop-loss logic)
        # Higher confidence = wider take-profit targets
        if confidence >= 0.8:
            take_profit_pct = cls.MAX_TAKE_PROFIT_PCT  # 30% for high confidence
        elif confidence >= 0.6:
            take_profit_pct = cls.DEFAULT_TAKE_PROFIT_PCT  # 20% for medium confidence
        else:
            take_profit_pct = cls.MIN_TAKE_PROFIT_PCT  # 15% for low confidence
            
        # Own-side price convention (see module docstring): a position of either
        # side loses when its own contract price falls. Stop below, target above.
        stop_loss_price = entry_price * (1 - adjusted_stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct)


        # Ensure prices are within valid bounds (1¢ to 99¢)
        stop_loss_price = max(0.01, min(0.99, stop_loss_price))
        take_profit_price = max(0.01, min(0.99, take_profit_price))
        
        # Calculate maximum hold time based on time to expiry
        # Hold for maximum 50% of time to expiry, or 72 hours, whichever is less
        max_hold_hours = min(72, time_to_expiry_days * 24 * 0.5)
        max_hold_hours = max(6, max_hold_hours)  # Minimum 6 hours
        
        return {
            'stop_loss_price': round(stop_loss_price, 2),
            'take_profit_price': round(take_profit_price, 2),
            'max_hold_hours': int(max_hold_hours),
            'stop_loss_pct': round(adjusted_stop_loss_pct * 100, 1),
            'take_profit_pct': round(take_profit_pct * 100, 1),
            'target_confidence_change': 0.15  # Exit if confidence drops 15%
        }
    
    @classmethod
    def calculate_simple_stop_loss(
        cls,
        entry_price: float,
        side: str,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT
    ) -> float:
        """
        Simple stop-loss calculation for quick use.
        
        Args:
            entry_price: Position entry price
            side: Position side ("YES" or "NO")
            stop_loss_pct: Stop-loss percentage (default 7%)
            
        Returns:
            Stop-loss price
        """
        # Own-side price convention (see module docstring): stop always below entry.
        stop_loss_price = entry_price * (1 - stop_loss_pct)

        return max(0.01, min(0.99, round(stop_loss_price, 2)))
    
    @classmethod
    def is_stop_loss_triggered(
        cls,
        position_side: str,
        entry_price: float,
        current_price: float,
        stop_loss_price: float
    ) -> bool:
        """
        Check if stop-loss should be triggered.
        
        Args:
            position_side: "YES" or "NO"
            entry_price: Original entry price
            current_price: Current market price
            stop_loss_price: Calculated stop-loss price
            
        Returns:
            True if stop-loss should be triggered
        """
        # Own-side price convention (see module docstring): both sides lose when
        # the held contract's own price falls, so the test is side-independent.
        return current_price <= stop_loss_price
    
    @classmethod
    def calculate_pnl_at_stop_loss(
        cls,
        entry_price: float,
        stop_loss_price: float,
        quantity: int,
        side: str
    ) -> float:
        """
        Calculate P&L if stop-loss is triggered.
        
        Returns:
            Expected P&L (negative for loss)
        """
        # Own-side price convention (see module docstring): P&L per contract is
        # exit minus entry for either side, so a stop below entry yields a loss.
        # The previous NO branch returned entry - stop, which reported a POSITIVE
        # number for a stopped-out NO position (visible in exit reasons logged as
        # "stop_loss_triggered_pnl_12.60" on trades that actually lost money).
        pnl_per_share = stop_loss_price - entry_price

        return pnl_per_share * quantity


# Convenience function for backward compatibility
def calculate_stop_loss_levels(
    entry_price: float,
    side: str,
    confidence: Optional[float] = None,
    **kwargs
) -> Dict[str, float]:
    """Convenience function that uses the StopLossCalculator class."""
    return StopLossCalculator.calculate_stop_loss_levels(
        entry_price=entry_price,
        side=side,
        confidence=confidence,
        **kwargs
    ) 