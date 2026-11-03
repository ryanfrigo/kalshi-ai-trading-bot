"""Immediate-trade evaluation.

Converts raw markets into scored MarketOpportunity objects (with fast AI
predictions) and decides single-market immediate trades. Split out of the
former 1,300-line portfolio_optimization.py.
"""

import logging
import math
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple

from src.utils.database import DatabaseManager, Market
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.utils.market_prices import get_market_prices
from src.utils.position_sizing import binary_market_payout_odds, kelly_fraction

from src.strategies.portfolio.models import MarketOpportunity


def _coerce_volume(value) -> float:
    """float(volume), or NaN when missing/invalid so the liquidity gate fails closed."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return v if math.isfinite(v) else float('nan')


async def create_market_opportunities_from_markets(
    markets: List[Market],
    xai_client: XAIClient,
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager = None,
    total_capital: float = 10000
) -> List[MarketOpportunity]:
    """
    Convert Market objects to MarketOpportunity objects with all required metrics.
    """
    logger = get_trading_logger("portfolio_opportunities")
    opportunities = []
    
    # Limit markets to prevent excessive AI costs and focus on best opportunities
    max_markets_to_analyze = 10  # REDUCED: More selective (was 20, now 10) to focus on highest quality
    if len(markets) > max_markets_to_analyze:
        # Sort by volume and take top markets
        markets = sorted(markets, key=lambda m: m.volume, reverse=True)[:max_markets_to_analyze]
        logger.info(f"Limited to top {max_markets_to_analyze} markets by volume for AI analysis")
    
    for market in markets:
        try:
            # Get current market data
            market_data = await kalshi_client.get_market(market.market_id)
            if not market_data:
                continue
            
            # FIXED: Extract from nested 'market' object (same fix as immediate trading)
            market_info = market_data.get('market', {})
            # get_market_prices normalizes API v2 (dollars) and legacy (cents).
            # Never default a missing price: a made-up 50c prices every trade wrong.
            yes_bid, yes_ask, _no_bid, _no_ask = get_market_prices(market_info)
            if yes_bid <= 0 or yes_ask <= 0:
                logger.warning(f"No usable YES quote for {market.market_id}, skipping")
                continue
            market_prob = (yes_bid + yes_ask) / 2
            
            # Skip markets with extreme prices (too risky for portfolio)
            if market_prob < 0.05 or market_prob > 0.95:
                continue
            
            # Get REAL AI prediction using fast analysis
            predicted_prob, confidence = await _get_fast_ai_prediction(
                market, xai_client, market_prob
            )
            
            # If AI analysis failed, skip this market
            if predicted_prob is None or confidence is None:
                logger.warning(f"AI analysis failed for {market.market_id}, skipping")
                continue
            
            # Calculate metrics
            edge = predicted_prob - market_prob
            expected_return = abs(edge) * confidence
            volatility = np.sqrt(market_prob * (1 - market_prob))
            max_loss = market_prob if edge > 0 else (1 - market_prob)
            
            # Time to expiry
            time_to_expiry = 30.0  # Default 30 days
            if hasattr(market, 'expiration_ts') and market.expiration_ts:
                import time
                time_to_expiry = (market.expiration_ts - time.time()) / 86400
                time_to_expiry = max(0.1, time_to_expiry)
            
            # Apply Grok4 edge filtering - 10% minimum edge requirement
            from src.utils.edge_filter import EdgeFilter
            edge_result = EdgeFilter.calculate_edge(predicted_prob, market_prob, confidence)
            
            if edge_result.passes_filter:  # Must pass 10% edge filter
                opportunity = MarketOpportunity(
                    market_id=market.market_id,
                    market_title=market.title,
                    predicted_probability=predicted_prob,
                    market_probability=market_prob,
                    confidence=confidence,
                    edge=edge,
                    volatility=volatility,
                    expected_return=expected_return,
                    max_loss=max_loss,
                    time_to_expiry=time_to_expiry,
                    correlation_score=0.0,
                    kelly_fraction=0.0,
                    fractional_kelly=0.0,
                    risk_adjusted_fraction=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    max_drawdown_contribution=0.0,
                    volume=_coerce_volume(getattr(market, 'volume', None))
                )
                
                # Add edge filter results to opportunity
                opportunity.edge = edge_result.edge_magnitude  # Use filtered edge
                opportunity.edge_percentage = edge_result.edge_percentage
                opportunity.recommended_side = edge_result.side
                
                opportunities.append(opportunity)
                logger.info(f"✅ EDGE APPROVED: {market.market_id} - Edge: {edge_result.edge_percentage:.1%} ({edge_result.side}), Confidence: {confidence:.1%}, Reason: {edge_result.reason}")
                
                # 🚀 IMMEDIATE TRADING: Place trade for strong opportunities
                if db_manager:
                    await _evaluate_immediate_trade(opportunity, db_manager, kalshi_client, total_capital)
            else:
                logger.info(f"❌ EDGE FILTERED: {market.market_id} - {edge_result.reason}")
            
        except Exception as e:
            logger.error(f"Error creating opportunity from {market.market_id}: {e}")
            continue
    
    logger.info(f"Created {len(opportunities)} opportunities from {len(markets)} markets")
    return opportunities

async def _evaluate_immediate_trade(
    opportunity: MarketOpportunity, 
    db_manager: DatabaseManager, 
    kalshi_client: KalshiClient, 
    total_capital: float
) -> None:
    """
    Evaluate if an opportunity should be traded immediately.
    For strong opportunities, place trade right away instead of waiting for batch optimization.
    """
    logger = get_trading_logger("immediate_trading")  # Move logger definition to the top
    
    try:
        # Use enhanced edge filtering for immediate trading decisions
        from src.utils.edge_filter import EdgeFilter
        
        # Check if opportunity meets immediate trading criteria using edge filter
        should_trade, trade_reason, edge_result = EdgeFilter.should_trade_market(
            ai_probability=opportunity.predicted_probability,
            market_probability=opportunity.market_probability,
            confidence=opportunity.confidence,
            additional_filters={
                # NaN (unknown) volume fails the gate downstream; never default to passing.
                'volume': opportunity.volume,
                'min_volume': settings.trading.min_volume,
                'time_to_expiry_days': opportunity.time_to_expiry,
                'max_time_to_expiry': settings.trading.max_time_to_expiry_days
            }
        )
        
        # Additional criteria for immediate execution - MORE AGGRESSIVE
        strong_opportunity = (
            should_trade and
            edge_result.edge_percentage >= 0.10 and  # DECREASED: 10% edge for immediate execution (was 18%)
            opportunity.confidence >= 0.60 and       # DECREASED: 60% confidence (was 75%)
            opportunity.expected_return >= 0.05      # DECREASED: 5% expected return (was 8%)
        )
        
        if not strong_opportunity:
            return  # Not strong enough for immediate action
        
        # Check position limits and get maximum allowed position size
        from src.utils.position_limits import check_can_add_position
        
        # Get portfolio value for position sizing
        try:
            balance_response = await kalshi_client.get_balance()
            available_cash = balance_response.get('balance', 0) / 100  # Convert cents to dollars
            
            # Get current positions to calculate total portfolio value
            # Kalshi API v2 returns portfolio_value in balance response (in cents)
            total_position_value = balance_response.get('portfolio_value', 0) / 100  # Convert cents to dollars

            # Log active positions for visibility
            positions_response = await kalshi_client.get_positions()
            event_positions = positions_response.get('event_positions', []) if isinstance(positions_response, dict) else []
            active_positions = [p for p in event_positions if float(p.get('event_exposure_dollars', '0')) > 0]
            if active_positions:
                logger.info(f"📊 Active positions: {len(active_positions)}")
                for pos in active_positions:
                    ticker = pos.get('event_ticker', '?')
                    exposure = float(pos.get('event_exposure_dollars', '0'))
                    logger.info(f"  📌 {ticker}: exposure=${exposure:.2f}")
            
            total_portfolio_value = available_cash + total_position_value
            logger.info(f"💰 Portfolio value: Cash=${available_cash:.2f} + Positions=${total_position_value:.2f} = Total=${total_portfolio_value:.2f}")
            
        except Exception as e:
            logger.warning(f"Could not get portfolio value, using available cash: {e}")
            total_portfolio_value = total_capital
            available_cash = total_capital
        
        # Calculate Kelly-optimal position size
        kelly_fraction = _calculate_simple_kelly(opportunity)
        kelly_multiplier = settings.trading.kelly_fraction  # Use configured Kelly fraction (0.75)
        kelly_position_size = kelly_fraction * kelly_multiplier * total_portfolio_value
        
        # Safety cap: never exceed configured max per position
        max_single_position_pct = settings.trading.max_single_position  # Safety cap from config
        safety_cap = total_portfolio_value * max_single_position_pct
        
        # Cash availability constraint
        cash_limit = available_cash * 0.8  # Don't use more than 80% of available cash
        
        # Initial position size: Kelly-optimal, but capped for safety and cash availability
        initial_position_size = min(
            kelly_position_size,  # Kelly-optimal size
            safety_cap,          # Safety cap (5% max)
            cash_limit           # Available cash constraint
        )
        
        # Check position limits with actual calculated size
        can_add_position, limit_reason = await check_can_add_position(
            initial_position_size, db_manager, kalshi_client
        )
        
        if not can_add_position:
            # Instead of blocking, try to find a smaller position size that fits
            logger.info(f"⚠️ Position size ${initial_position_size:.2f} exceeds limits, attempting to reduce...")
            
            # Try progressively smaller position sizes
            for reduction_factor in [0.8, 0.6, 0.4, 0.2, 0.1]:
                reduced_position_size = initial_position_size * reduction_factor
                can_add_reduced, reduced_reason = await check_can_add_position(
                    reduced_position_size, db_manager, kalshi_client
                )
                
                if can_add_reduced:
                    initial_position_size = reduced_position_size
                    logger.info(f"✅ Position size reduced to ${initial_position_size:.2f} to fit limits")
                    break
            else:
                # If even the smallest size doesn't fit, check if it's due to position count
                from src.utils.position_limits import PositionLimitsManager
                limits_manager = PositionLimitsManager(db_manager, kalshi_client)
                current_positions = await limits_manager._get_position_count()
                
                if current_positions >= limits_manager.max_positions:
                    logger.info(f"❌ POSITION COUNT LIMIT: {current_positions}/{limits_manager.max_positions} positions - cannot add new position")
                    return
                else:
                    logger.info(f"❌ POSITION SIZE LIMIT: Even minimum size ${initial_position_size * 0.1:.2f} exceeds limits")
                    return
        
        logger.info(f"✅ POSITION LIMITS OK FOR IMMEDIATE TRADE: ${initial_position_size:.2f}")
        
        # Check if we already have a position in this market
        import aiosqlite
        async with aiosqlite.connect(db_manager.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM positions WHERE market_id = ? AND status != 'voided'",
                (opportunity.market_id,)
            )
            result = await cursor.fetchone()
            position_count = result[0] if result else 0
        
        if position_count > 0:
            logger.info(f"⏭️ Skipping immediate trade for {opportunity.market_id} - position already exists")
            return
        
        # 🚀 STRONG OPPORTUNITY - TRADE IMMEDIATELY!
        logger.info(f"🚀 IMMEDIATE TRADE: {opportunity.market_id} - Edge: {opportunity.edge:.1%}, Confidence: {opportunity.confidence:.1%}")
        
        # Use the position size that was already calculated and validated above
        position_size = initial_position_size
        
        logger.info(f"💰 Using validated position size: ${position_size:.2f}")
        
        # Final cash reserves check with actual calculated size
        from src.utils.cash_reserves import check_can_trade_with_cash_reserves
        
        can_trade_reserves, reserves_reason = await check_can_trade_with_cash_reserves(
            position_size, db_manager, kalshi_client
        )
        
        if not can_trade_reserves:
            logger.info(f"❌ CASH RESERVES CHECK FAILED: {opportunity.market_id} - {reserves_reason}")
            return
        
        logger.info(f"✅ CASH RESERVES APPROVED: ${position_size:.2f} - {reserves_reason}")
        
        # NO DOLLAR MINIMUM - we'll ensure at least 1 contract below
        
        # Determine side based on edge direction
        side = "NO" if opportunity.edge < 0 else "YES"  # Negative edge = market overpriced = bet NO
        
        # 🚨 VALIDATE MARKET IS STILL TRADEABLE before executing
        try:
            market_data = await kalshi_client.get_market(opportunity.market_id)
            
            # FIXED: Extract from nested 'market' object in API response
            market_info = market_data.get('market', {})
            market_status = market_info.get('status')
            _yes_bid, yes_ask, _no_bid, no_ask = get_market_prices(market_info)
            
            logger.info(f"🔍 Market validation for {opportunity.market_id}: status={market_status}, YES={yes_ask:.4f}, NO={no_ask:.4f}")
            
            # FIXED: Kalshi uses 'active' for tradeable markets, not 'open'
            if market_status not in ['active', 'open']:
                logger.warning(f"⏭️ Skipping {opportunity.market_id} - Market status: {market_status} (not active/open)")
                return
            
            if not (yes_ask and no_ask and yes_ask > 0 and no_ask > 0):
                logger.warning(f"⏭️ Skipping {opportunity.market_id} - No valid prices (YES={yes_ask:.4f}, NO={no_ask:.4f})")
                return

            # Skip collection/series tickers — both sides at $1.00 means it's not
            # a directly tradeable market (see GitHub issue #42)
            if yes_ask >= 0.99 and no_ask >= 0.99:
                logger.warning(f"⏭️ Skipping {opportunity.market_id} - Collection/series ticker (YES={yes_ask:.4f}, NO={no_ask:.4f})")
                return
                
            logger.info(f"✅ Market validation passed for {opportunity.market_id} - Status: {market_status}, proceeding with trade!")
            
        except Exception as e:
            logger.error(f"⏭️ Skipping {opportunity.market_id} - Market validation failed: {e}")
            import traceback
            logger.error(f"Full error: {traceback.format_exc()}")
            return
        
        # Price the trade at the LIVE ask execute_position will actually lift, not the
        # discovery-time snapshot. If the ask moved so far that less than the 10%
        # immediate-trade edge remains, refuse.
        if side == "YES":
            entry_price = yes_ask
            remaining_edge = opportunity.predicted_probability - yes_ask
        else:
            entry_price = no_ask
            remaining_edge = (1 - opportunity.predicted_probability) - no_ask
        if remaining_edge < 0.10:
            logger.info(f"⏭️ Skipping {opportunity.market_id} - live {side} ask {entry_price:.2f} leaves only {remaining_edge:.1%} edge")
            return
        shares = int(position_size / entry_price)
        if shares < 1:
            # One contract costs more than the position cap allows; never round up past it.
            logger.info(f"⏭️ Skipping {opportunity.market_id} - position cap ${position_size:.2f} cannot buy 1 {side} contract @ ${entry_price:.2f}")
            return

        # Verify we can afford the position
        min_cost = shares * entry_price
        if min_cost > available_cash:
            logger.info(f"⏭️ Cannot afford minimum 1 contract: ${min_cost:.2f} > ${available_cash:.2f}")
            return
            
        logger.info(f"📊 Trade details: {shares} {side} shares @ ${entry_price:.2f} = ${min_cost:.2f}")
        
        # Calculate proper stop-loss levels using Grok4 recommendations
        from src.utils.stop_loss_calculator import StopLossCalculator
        
        exit_levels = StopLossCalculator.calculate_stop_loss_levels(
            entry_price=entry_price,
            side=side,
            confidence=opportunity.confidence,
            market_volatility=opportunity.volatility,
            time_to_expiry_days=opportunity.time_to_expiry
        )
        
        # Create position directly
        from src.utils.database import Position
        from src.jobs.execute import execute_position
        
        position = Position(
            market_id=opportunity.market_id,
            side=side,
            quantity=shares,
            entry_price=entry_price,
            live=False,  # Will be set to True ONLY after successful execution
            timestamp=datetime.now(),
            rationale=f"IMMEDIATE TRADE: Edge={opportunity.edge_percentage:.1%} ({opportunity.recommended_side}), Conf={opportunity.confidence:.1%}, Kelly={kelly_fraction:.1%}, Stop={exit_levels['stop_loss_pct']}%",
            strategy="immediate_portfolio_optimization",
            
            # Enhanced exit strategy using Grok4 recommendations
            stop_loss_price=exit_levels['stop_loss_price'],
            take_profit_price=exit_levels['take_profit_price'],
            max_hold_hours=exit_levels['max_hold_hours'],
            target_confidence_change=exit_levels['target_confidence_change']
        )
        
        # Execute immediately
        position_id = await db_manager.add_position(position)
        if position_id:
            # Set the position ID so execute_position can update the database
            position.id = position_id
            
            # Execute the trade - respect the global trading mode setting
            live_mode = getattr(settings.trading, 'live_trading_enabled', False)
            # A failed buy leaves its row open (live=0): the order may have been accepted
            # but not yet visible, so voiding here could allow a second buy. The
            # stale-row pass voids it later only if the exchange is flat.
            success = await execute_position(position, live_mode, db_manager, kalshi_client)
            if success:
                logger.info(f"✅ IMMEDIATE TRADE EXECUTED: {opportunity.market_id} - ${position_size:.0f} position")
            else:
                logger.error(f"❌ IMMEDIATE TRADE FAILED: {opportunity.market_id}")
        
    except Exception as e:
        logger.error(f"Error in immediate trade evaluation for {opportunity.market_id}: {e}")

def _calculate_simple_kelly(opportunity: MarketOpportunity) -> float:
    """Calculate simple Kelly fraction for immediate trading.

    Uses the shared kernel (src/utils/position_sizing.py); this wrapper owns
    the side selection (YES on positive edge, NO otherwise), the 20% cap, and
    the legacy 5% fallback for degenerate market prices.
    """
    try:
        # A NaN/inf/out-of-[0,1] price or probability is corrupt data, not a
        # degenerate price: size 0, never the 5% fallback.
        for v in (opportunity.market_probability, opportunity.predicted_probability):
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                return 0.0

        if opportunity.edge > 0:  # Betting YES
            p = opportunity.predicted_probability
            b = binary_market_payout_odds(opportunity.market_probability, bet_yes=True)
        else:  # Betting NO
            p = 1 - opportunity.predicted_probability
            b = binary_market_payout_odds(opportunity.market_probability, bet_yes=False)

        if b <= 0:
            return 0.05  # degenerate price (e.g. NO at P=0): keep legacy fallback

        return min(kelly_fraction(p, b), 0.2)  # Cap at 20%

    except Exception:
        return 0.05  # Default 5% allocation (includes division by zero on P=0/P=1)


async def _get_fast_ai_prediction(
    market: Market,
    xai_client: XAIClient,
    market_price: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get a fast AI prediction for a market without expensive analysis.
    Returns (predicted_probability, confidence) or (None, None) if failed.
    """
    try:
        # Create a simplified prompt for faster analysis
        prompt = f"""
        QUICK PREDICTION REQUEST
        
        Market: {market.title}
        Current YES price: {market_price:.2f}
        
        Provide a FAST prediction in JSON format:
        {{
            "probability": [0.0-1.0],
            "confidence": [0.0-1.0],
            "reasoning": "brief 1-2 sentence explanation"
        }}
        
        Focus on: probability estimate and your confidence level.
        """
        
        # Use AI analysis for portfolio optimization - higher tokens for reasoning models  
        response_text = await xai_client.get_completion(
            prompt,
            max_tokens=3000,  # Higher for reasoning models like grok-4
            temperature=0.1   # Low temperature for consistency
        )
        
        # Check if AI response is None (API exhausted or failed)
        if response_text is None:
            logging.getLogger("portfolio_opportunities").info(f"AI analysis unavailable for {market.market_id} due to API limits")
            return None, None
        
        # Parse JSON from the response text
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                response = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                response = json.loads(response_text)
            
            if response and isinstance(response, dict):
                probability = response.get('probability')
                confidence = response.get('confidence')
                
                # Validate values
                if (isinstance(probability, (int, float)) and 0 <= probability <= 1 and
                    isinstance(confidence, (int, float)) and 0 <= confidence <= 1):
                    return float(probability), float(confidence)
            
        except (json.JSONDecodeError, ValueError) as json_error:
            logging.getLogger("portfolio_opportunities").warning(f"Failed to parse JSON from AI response for {market.market_id}: {json_error}")
            logging.getLogger("portfolio_opportunities").debug(f"Raw response: {response_text}")
        
        return None, None
        
    except Exception as e:
        logging.getLogger("portfolio_opportunities").error(f"Error in fast AI prediction for {market.market_id}: {e}")
        return None, None

