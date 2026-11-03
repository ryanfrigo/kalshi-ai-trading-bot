"""
Cash Reserves Management Module

Implements a minimum cash reserve requirement (0.5% of portfolio; 1.0% optimal).
Provides systematic cash management across all trading strategies to prevent liquidity crises.

Key Features:
- Minimum cash reserve enforcement (thresholds set in CashReservesManager)
- Emergency trading halt when reserves critical
- Integration with position limits system
- Real-time cash monitoring and alerts
- Automatic position closure for reserves
"""

import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.utils.logging_setup import get_trading_logger
from src.config.settings import settings


@dataclass
class CashReserveResult:
    """Result of cash reserve checking."""
    can_trade: bool
    reason: str
    current_cash: float
    portfolio_value: float
    cash_reserve_pct: float
    required_reserve_pct: float
    emergency_status: bool
    recommended_actions: List[str]


@dataclass
class CashEmergencyAction:
    """Emergency action for cash reserves."""
    action_type: str  # "close_positions", "halt_trading", "raise_alert"
    urgency: str     # "critical", "warning", "info"
    positions_to_close: int
    expected_cash_freed: float
    reason: str


class CashReservesManager:
    """
    Centralized cash reserves management following Grok4 recommendations.
    
    Implements systematic cash management to prevent liquidity crises and
    maintain operational flexibility for opportunistic trades.
    """
    
    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("cash_reserves")
        
        # UPDATED: Minimal cash reserve requirements for maximum deployment
        self.minimum_reserve_pct = 0.5       # DECREASED: Only 0.5% minimum (was 1%)
        self.optimal_reserve_pct = 1.0       # DECREASED: Only 1% optimal target (was 2%)
        self.emergency_threshold_pct = 0.2   # DECREASED: 0.2% emergency halt (was 0.5%)
        self.critical_threshold_pct = 0.05   # DECREASED: 0.05% critical threshold (was 0.1%)
        
        # Additional safety parameters - MORE AGGRESSIVE
        self.max_single_trade_impact = 5.0   # INCREASED: Allow 5% portfolio impact per trade (was 3%)
        self.buffer_for_opportunities = 0.5  # DECREASED: Only 0.5% buffer (was 1%)
        
    async def check_cash_reserves(
        self,
        proposed_trade_value: Optional[float] = None,
        portfolio_value: Optional[float] = None
    ) -> CashReserveResult:
        """
        Check if cash reserves meet requirements for trading.
        
        Args:
            proposed_trade_value: Dollar value of proposed trade. ``None`` means
                a reserve-only check (no trade). An explicit value must be a
                finite number > 0, otherwise the trade is refused.
            portfolio_value: Total portfolio value (fetched if not provided)
            
        Returns:
            CashReserveResult with decision and recommendations
        """
        if proposed_trade_value is None:
            proposed_trade_value = 0.0
        else:
            try:
                valid = math.isfinite(float(proposed_trade_value)) and float(proposed_trade_value) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                return CashReserveResult(
                    can_trade=False,
                    reason=f"Invalid proposed trade value {proposed_trade_value!r}: must be a finite number > 0",
                    current_cash=0.0,
                    portfolio_value=0.0,
                    cash_reserve_pct=0.0,
                    required_reserve_pct=self.minimum_reserve_pct,
                    emergency_status=False,
                    recommended_actions=["Fix the upstream trade sizing; refusing to trade"],
                )

        try:
            # Get current portfolio state
            if portfolio_value is None:
                portfolio_value = await self._get_portfolio_value()
            
            current_cash = await self._get_available_cash()
            current_reserve_pct = (current_cash / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # Calculate cash after proposed trade
            cash_after_trade = current_cash - proposed_trade_value
            reserve_after_trade = (cash_after_trade / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            recommendations = []
            can_trade = True
            reason = "Cash reserves adequate"
            emergency_status = False
            
            # Check 1: Current reserve level
            if current_reserve_pct < self.critical_threshold_pct:
                can_trade = False
                emergency_status = True
                reason = f"CRITICAL: Cash reserves {current_reserve_pct:.1f}% below critical threshold {self.critical_threshold_pct:.1f}%"
                recommendations.append("EMERGENCY: Close positions immediately to build cash reserves")
                recommendations.append("HALT all new trading until reserves restored")
                
            elif current_reserve_pct < self.emergency_threshold_pct:
                can_trade = False
                emergency_status = True
                reason = f"EMERGENCY: Cash reserves {current_reserve_pct:.1f}% below emergency threshold {self.emergency_threshold_pct:.1f}%"
                recommendations.append("Close 2-3 positions immediately")
                recommendations.append(f"Suspend new trading until above {self.minimum_reserve_pct:.1f}%")
                
            elif reserve_after_trade < self.minimum_reserve_pct:
                can_trade = False
                reason = f"Trade would reduce reserves to {reserve_after_trade:.1f}%, below minimum {self.minimum_reserve_pct:.1f}%"
                recommendations.append(f"Reduce trade size or close positions to maintain {self.minimum_reserve_pct:.1f}% reserves")
                
            elif current_reserve_pct < self.minimum_reserve_pct:
                can_trade = False
                reason = f"Current reserves {current_reserve_pct:.1f}% below minimum {self.minimum_reserve_pct:.1f}%"
                recommendations.append("Build cash reserves before new trades")
                recommendations.append("Consider closing lowest-performing positions")
                
            # Check 2: Trade size impact
            trade_impact_pct = (proposed_trade_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            # Add small tolerance for floating point precision issues
            if trade_impact_pct > (self.max_single_trade_impact + 0.01):
                can_trade = False
                reason = f"Trade impact {trade_impact_pct:.1f}% exceeds maximum {self.max_single_trade_impact:.1f}%"
                recommendations.append(f"Reduce trade size to maximum ${portfolio_value * self.max_single_trade_impact / 100:.2f}")
            
            # Check 3: Opportunity buffer
            if reserve_after_trade < (self.minimum_reserve_pct + self.buffer_for_opportunities):
                if can_trade:  # Only warn if not already blocked
                    recommendations.append(f"Warning: Reserves would be {reserve_after_trade:.1f}%, limiting future opportunities")
            
            # Positive recommendations
            if current_reserve_pct >= self.optimal_reserve_pct:
                recommendations.append("Excellent cash position - ready for opportunities")
            elif current_reserve_pct >= self.minimum_reserve_pct:
                recommendations.append("Good cash reserves - trading permitted")
            
            return CashReserveResult(
                can_trade=can_trade,
                reason=reason,
                current_cash=current_cash,
                portfolio_value=portfolio_value,
                cash_reserve_pct=current_reserve_pct,
                required_reserve_pct=self.minimum_reserve_pct,
                emergency_status=emergency_status,
                recommended_actions=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error checking cash reserves: {e}")
            return CashReserveResult(
                can_trade=False,
                reason=f"Error checking cash reserves: {e}",
                current_cash=0.0,
                portfolio_value=0.0,
                cash_reserve_pct=0.0,
                required_reserve_pct=self.minimum_reserve_pct,
                emergency_status=True,
                recommended_actions=["Review system errors", "Manual cash verification needed"]
            )
    
    async def handle_cash_emergency(self) -> CashEmergencyAction:
        """
        Handle cash reserve emergency by determining required actions.
        
        Returns:
            CashEmergencyAction with specific emergency response
        """
        try:
            portfolio_value = await self._get_portfolio_value()
            current_cash = await self._get_available_cash()
            current_reserve_pct = (current_cash / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # Calculate required cash to reach minimum reserves
            required_cash = portfolio_value * (self.minimum_reserve_pct / 100)
            cash_shortfall = required_cash - current_cash
            
            if current_reserve_pct >= self.minimum_reserve_pct:
                return CashEmergencyAction(
                    action_type="no_action",
                    urgency="info",
                    positions_to_close=0,
                    expected_cash_freed=0.0,
                    reason="Cash reserves adequate"
                )
            
            # Determine urgency level
            if current_reserve_pct < self.critical_threshold_pct:
                urgency = "critical"
                action_type = "halt_trading"
            elif current_reserve_pct < self.emergency_threshold_pct:
                urgency = "critical"
                action_type = "close_positions"
            else:
                urgency = "warning"
                action_type = "raise_alert"
            
            # Calculate positions to close
            positions = await self.db_manager.get_open_positions()
            positions_to_close = 0
            expected_cash_freed = 0.0
            
            # Estimate cash from closing positions (simplified)
            if positions:
                avg_position_value = 50.0  # Conservative estimate
                positions_needed = max(1, int(cash_shortfall / avg_position_value))
                positions_to_close = min(positions_needed, len(positions))
                expected_cash_freed = positions_to_close * avg_position_value
            
            return CashEmergencyAction(
                action_type=action_type,
                urgency=urgency,
                positions_to_close=positions_to_close,
                expected_cash_freed=expected_cash_freed,
                reason=f"Need ${cash_shortfall:.2f} to reach {self.minimum_reserve_pct:.1f}% reserves"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling cash emergency: {e}")
            return CashEmergencyAction(
                action_type="halt_trading",
                urgency="critical",
                positions_to_close=0,
                expected_cash_freed=0.0,
                reason=f"Emergency handling error: {e}"
            )
    
    async def get_cash_status(self) -> Dict[str, Any]:
        """Get comprehensive cash reserves status."""
        try:
            portfolio_value = await self._get_portfolio_value()
            current_cash = await self._get_available_cash()
            current_reserve_pct = (current_cash / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # Determine status
            if current_reserve_pct >= self.optimal_reserve_pct:
                status = "EXCELLENT"
            elif current_reserve_pct >= self.minimum_reserve_pct:
                status = "GOOD"
            elif current_reserve_pct >= self.emergency_threshold_pct:
                status = "WARNING"
            elif current_reserve_pct >= self.critical_threshold_pct:
                status = "EMERGENCY"
            else:
                status = "CRITICAL"
            
            # Calculate targets
            optimal_cash = portfolio_value * (self.optimal_reserve_pct / 100)
            minimum_cash = portfolio_value * (self.minimum_reserve_pct / 100)
            
            return {
                'status': status,
                'current_cash': current_cash,
                'portfolio_value': portfolio_value,
                'reserve_percentage': current_reserve_pct,
                'minimum_required': self.minimum_reserve_pct,
                'optimal_target': self.optimal_reserve_pct,
                'cash_shortfall': max(0, minimum_cash - current_cash),
                'cash_to_optimal': max(0, optimal_cash - current_cash),
                'trading_permitted': current_reserve_pct >= self.minimum_reserve_pct,
                'emergency_status': current_reserve_pct < self.emergency_threshold_pct,
                'max_trade_size': max(0, current_cash - minimum_cash),
                'recommendations': self._get_cash_recommendations(current_reserve_pct)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cash status: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    async def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions). Raises if it cannot be read."""
        try:
            # Get available cash
            balance_response = await self.kalshi_client.get_balance()
            available_cash = balance_response.get('balance', 0) / 100
            
            # Get current positions value
            positions = await self._fetch_all_positions()
            total_position_value = 0.0

            for position in positions:
                value = self._position_value(position)
                if value:
                    total_position_value += value
            
            return available_cash + total_position_value
            
        except Exception as e:
            # No fallback value: an invented denominator (it used to be $100)
            # overstates the reserve when cash is larger and lets small trades
            # through. Callers fail closed on the raise.
            self.logger.error(f"Error calculating portfolio value: {e}")
            raise
    
    POSITIONS_PAGE_LIMIT = 200
    POSITIONS_MAX_PAGES = 20

    async def _fetch_all_positions(self) -> List[Any]:
        """Every ``market_positions`` row, following the cursor.

        Raises (so every caller fails closed) on an unreadable
        page, a repeated cursor, a failed later page, or more than
        POSITIONS_MAX_PAGES pages. Never returns a partial list: an undervalued
        book would overstate the reserve %.
        """
        rows: List[Any] = []
        seen = set()
        cursor = None
        for _ in range(self.POSITIONS_MAX_PAGES):
            resp = await self.kalshi_client.get_positions(limit=self.POSITIONS_PAGE_LIMIT, cursor=cursor)
            page = resp.get('market_positions') if isinstance(resp, dict) else None
            if not isinstance(page, list):
                # Unreadable shape: don't pretend we hold nothing.
                raise ValueError("positions response missing 'market_positions' list")
            rows.extend(page)
            cursor = resp.get('cursor')
            if not cursor:
                return rows
            if cursor in seen:
                raise ValueError("repeated cursor while paging positions")
            seen.add(cursor)
        raise ValueError(f"more than {self.POSITIONS_MAX_PAGES} pages of positions")

    @staticmethod
    def _position_value(position: Any) -> float:
        """Dollar value of one v2 ``market_positions`` row; 0.0 for malformed
        rows or flat positions (skipped, never poisons the whole portfolio).

        Uses ``market_exposure_dollars`` when it is a finite number >= 0,
        otherwise estimates $0.50 per contract from signed ``position_fp``.
        """
        if not isinstance(position, dict):
            return 0.0
        try:
            contracts = abs(float(position.get("position_fp", 0) or 0))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(contracts) or contracts <= 0:
            return 0.0
        try:
            exposure = float(position.get("market_exposure_dollars"))
            if math.isfinite(exposure) and exposure >= 0:
                return exposure
        except (TypeError, ValueError):
            pass
        return contracts * 0.50  # conservative estimate

    async def _get_available_cash(self) -> float:
        """Get available cash balance."""
        try:
            balance_response = await self.kalshi_client.get_balance()
            return balance_response.get('balance', 0) / 100
        except Exception as e:
            self.logger.error(f"Error getting available cash: {e}")
            return 0.0
    
    def _get_cash_recommendations(self, reserve_pct: float) -> List[str]:
        """Get recommendations based on cash reserve level."""
        recommendations = []
        
        if reserve_pct < self.critical_threshold_pct:
            recommendations.append("🚨 CRITICAL: Close positions immediately")
            recommendations.append("🚨 HALT all trading until reserves restored")
            recommendations.append("🚨 Consider depositing additional funds")
        elif reserve_pct < self.emergency_threshold_pct:
            recommendations.append("⚠️ EMERGENCY: Close 2-3 positions immediately")
            recommendations.append(f"⚠️ Suspend new trading until above {self.minimum_reserve_pct:.1f}%")
        elif reserve_pct < self.minimum_reserve_pct:
            recommendations.append("⚠️ Close some positions to build reserves")
            recommendations.append(f"⚠️ Avoid new trades until above {self.minimum_reserve_pct:.1f}%")
        elif reserve_pct < self.optimal_reserve_pct:
            recommendations.append("✅ Reserves adequate but could be improved")
            recommendations.append(f"✅ Consider building toward {self.optimal_reserve_pct:.1f}% optimal")
        else:
            recommendations.append("🎯 Excellent cash position")
            recommendations.append("🎯 Ready for opportunistic trades")
        
        return recommendations


# Convenience functions for easy integration
async def check_can_trade_with_cash_reserves(
    trade_value: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> Tuple[bool, str]:
    """Simple check if a trade can be made within cash reserve requirements."""
    manager = CashReservesManager(db_manager, kalshi_client)
    result = await manager.check_cash_reserves(trade_value)
    return result.can_trade, result.reason


async def get_max_trade_size_for_reserves(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> float:
    """Get maximum trade size that maintains cash reserves."""
    manager = CashReservesManager(db_manager, kalshi_client)
    status = await manager.get_cash_status()
    return status.get('max_trade_size', 0.0)


async def is_cash_emergency(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> bool:
    """Check if we're in a cash emergency situation."""
    manager = CashReservesManager(db_manager, kalshi_client)
    status = await manager.get_cash_status()
    # Fail closed: an ERROR status or a missing key counts as an emergency.
    if status.get('status') == 'ERROR' or 'emergency_status' not in status:
        return True
    return bool(status['emergency_status'])
