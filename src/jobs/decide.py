"""
Trading decision job - analyzes markets and generates trading decisions.

This module wires the LLM-driven directional strategy. It uses the
single-model OpenRouter pipeline (xai_client → openrouter_client) for
the actual decision call. A multi-agent ensemble path used to live
here but was never wired through trade.py — it's been removed; if you
want to bring back parallel multi-model voting, fork this module and
build it for real (see src/agents/ for the runner skeletons).
"""

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime

from src.utils.database import DatabaseManager, Market, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.xai_client import XAIClient
from src.clients.kalshi_client import KalshiClient


def _calculate_dynamic_quantity(
    balance: float,
    market_price: float,
    confidence_delta: float,
) -> int:
    """
    Calculates trade quantity based on portfolio balance and confidence delta.
    
    Args:
        balance: Current available portfolio balance.
        market_price: The price of the contract (e.g., 0.90 for 90 cents).
        confidence_delta: The difference between LLM confidence and market price.
        
    Returns:
        The number of contracts to purchase.
    """
    if market_price <= 0:
        return 0
        
    # Use a percentage of the balance for the trade
    base_investment_pct = settings.trading.default_position_size / 100
    
    # Scale investment by how much our confidence differs from the market
    investment_scaler = 1 + (settings.trading.position_size_multiplier * confidence_delta)
    
    investment_amount = (balance * base_investment_pct) * investment_scaler
    
    # Do not exceed the max position size
    max_investment = (balance * settings.trading.max_position_size_pct) / 100
    final_investment = min(investment_amount, max_investment)
    
    quantity = int(final_investment // market_price)
    
    get_trading_logger("decision_engine").info(
        "Calculated dynamic position size.",
        investment_amount=final_investment,
        quantity=quantity
    )
    
    return quantity


async def make_decision_for_market(
    market: Market,
    db_manager: DatabaseManager,
    xai_client: XAIClient,
    kalshi_client: KalshiClient,
) -> Optional[Position]:
    """
    Analyzes a single market and makes a trading decision with performance optimizations.
    Now includes cost controls and deduplication.
    """
    logger = get_trading_logger("decision_engine")
    logger.info(f"Analyzing market: {market.title} ({market.market_id})")

    try:
        # CHECK 1: Daily budget enforcement
        daily_cost = await db_manager.get_daily_ai_cost()
        if daily_cost >= settings.trading.daily_ai_budget:
            logger.warning(
                f"Daily AI budget of ${settings.trading.daily_ai_budget} exceeded. "
                f"Current cost: ${daily_cost:.3f}. Skipping analysis."
            )
            return None

        # CHECK 2: Recent analysis deduplication
        if await db_manager.was_recently_analyzed(
            market.market_id, 
            settings.trading.analysis_cooldown_hours
        ):
            logger.info(f"Market {market.market_id} was recently analyzed. Skipping to save costs.")
            return None

        # CHECK 3: Daily analysis limit per market
        analysis_count_today = await db_manager.get_market_analysis_count_today(market.market_id)
        if analysis_count_today >= settings.trading.max_analyses_per_market_per_day:
            logger.info(f"Market {market.market_id} already analyzed {analysis_count_today} times today. Skipping.")
            return None

        # CHECK 4: Volume threshold for AI analysis
        if market.volume < settings.trading.min_volume_for_ai_analysis:
            logger.info(f"Market {market.market_id} volume {market.volume} below AI analysis threshold. Skipping.")
            return None

        # CHECK 5: Category filtering
        if market.category.lower() in [cat.lower() for cat in settings.trading.exclude_low_liquidity_categories]:
            logger.info(f"Market {market.market_id} in excluded category '{market.category}'. Skipping.")
            return None

        # Get real-time portfolio balance
        balance_response = await kalshi_client.get_balance()
        available_balance = balance_response.get("balance", 0) / 100  # Convert cents to dollars
        portfolio_data = {"available_balance": available_balance}
        
        logger.info(f"Current available balance: ${available_balance:.2f}")

        # Initialize tracking variables
        total_analysis_cost = 0.0
        decision_action = "SKIP"
        confidence = 0.0

        # --- High-Confidence, Near-Expiry Strategy ---
        hours_to_expiry = (market.expiration_ts - time.time()) / 3600
        if (
            settings.trading.enable_high_confidence_strategy and
            hours_to_expiry <= settings.trading.high_confidence_expiry_hours
        ):
            logger.info("Market is near expiry, evaluating for high-confidence strategy.")
            
            # Check for high-odds YES bet
            if market.yes_price >= settings.trading.high_confidence_market_odds:
                # Skip expensive news search for high-confidence strategy to control costs
                news_summary = f"Near-expiry high-confidence analysis. Market at {market.yes_price:.2f}"
                
                decision = await xai_client.get_trading_decision(
                    market_data={"title": market.title, "yes_price": market.yes_price},
                    portfolio_data=portfolio_data,
                    news_summary=news_summary
                )
                
                # Estimate cost for high-confidence analysis (typically lower due to shorter prompts)
                estimated_cost = 0.01  # Rough estimate for simple analysis
                total_analysis_cost += estimated_cost

                if decision.side == "YES" and decision.confidence >= settings.trading.high_confidence_threshold:
                    logger.info(f"High-confidence YES opportunity found for {market.market_id}.")
                    
                    decision_action = "BUY"
                    confidence = decision.confidence
                    
                    # Record analysis before creating position
                    await db_manager.record_market_analysis(
                        market.market_id, decision_action, confidence, total_analysis_cost, "high_confidence"
                    )
                    
                    confidence_delta = decision.confidence - market.yes_price
                    quantity = _calculate_dynamic_quantity(available_balance, market.yes_price, confidence_delta)

                    if quantity > 0:
                        # Calculate exit strategy using Grok4 recommendations  
                        from src.utils.stop_loss_calculator import StopLossCalculator
                        
                        exit_strategy = StopLossCalculator.calculate_stop_loss_levels(
                            entry_price=market.yes_price,
                            side=decision.side,
                            confidence=confidence,
                            market_volatility=estimate_market_volatility(market),
                            time_to_expiry_days=get_time_to_expiry_days(market)
                        )
                        
                        position = Position(
                            market_id=market.market_id,
                            side=decision.side,
                            entry_price=market.yes_price,
                            quantity=quantity,
                            timestamp=datetime.now(),
                            rationale="High-confidence near-expiry YES bet.",
                            confidence=decision.confidence,
                            live=False,
                            strategy="directional_trading",
                            
                            # Enhanced exit strategy fields using Grok4 recommendations
                            stop_loss_price=exit_strategy['stop_loss_price'],
                            take_profit_price=exit_strategy['take_profit_price'],
                            max_hold_hours=exit_strategy['max_hold_hours'],
                            target_confidence_change=exit_strategy['target_confidence_change']
                        )
                        return position

        # --- Standard LLM Decision-Making ---
        # Feature flags
        sentiment_analysis = getattr(settings, 'sentiment_analysis', False) or (
            hasattr(settings, 'sentiment') and settings.sentiment.enabled
        )
        logger.info(
            "Proceeding with LLM decision analysis.",
            sentiment_enabled=sentiment_analysis,
        )
        
        # Cost-optimized market data fetching
        full_market_data_response = await kalshi_client.get_market(market.market_id)
        full_market_data = full_market_data_response.get("market", {})
        rules = full_market_data.get("rules", "No rules available.")
        
        market_data = {
            "ticker": market.market_id, "title": market.title, "rules": rules,
            "yes_price": market.yes_price, "no_price": market.no_price,
            "volume": market.volume, "expiration_ts": market.expiration_ts,
        }

        # COST OPTIMIZATION: Skip expensive news search for low-volume markets
        if (settings.trading.skip_news_for_low_volume and
            market.volume < settings.trading.news_search_volume_threshold):
            logger.info(f"Skipping news search for low volume market {market.market_id} (volume: {market.volume})")
            news_summary = f"Low volume market ({market.volume}). Analysis based on market data only."
            estimated_search_cost = 0.0
        else:
            # Try sentiment pipeline first if enabled
            if sentiment_analysis:
                try:
                    from src.data.sentiment_analyzer import SentimentAnalyzer
                    analyzer = SentimentAnalyzer()
                    news_summary = await asyncio.wait_for(
                        analyzer.get_market_sentiment_summary(market.title),
                        timeout=30.0
                    )
                    estimated_search_cost = analyzer.total_cost
                    logger.info(f"Sentiment pipeline returned for {market.market_id}")
                except Exception as e:
                    logger.warning(f"Sentiment pipeline failed for {market.market_id}: {e}, falling back to xAI search")
                    news_summary = None
                    estimated_search_cost = 0.0

            if not sentiment_analysis or news_summary is None:
                # Fall back to xAI search
                try:
                    news_summary = await asyncio.wait_for(
                        xai_client.search(market.title, max_length=200),
                        timeout=15.0
                    )
                    estimated_search_cost = 0.02
                except asyncio.TimeoutError:
                    logger.warning(f"Search timeout for market {market.market_id}, using fallback")
                    news_summary = f"Search timeout. Analyzing {market.title} based on market data only."
                    estimated_search_cost = 0.0
                except Exception as e:
                    logger.warning(f"Search failed for market {market.market_id}, continuing without news", error=str(e))
                    news_summary = f"News search unavailable. Analysis based on market data only."
                    estimated_search_cost = 0.0

        total_analysis_cost += estimated_search_cost

        # Check if we're approaching cost limits before making the decision
        if total_analysis_cost > settings.trading.max_ai_cost_per_decision:
            logger.warning(f"Analysis cost ${total_analysis_cost:.3f} exceeds per-decision limit. Skipping.")
            await db_manager.record_market_analysis(
                market.market_id, "SKIP", 0.0, total_analysis_cost, "cost_limited"
            )
            return None

        # --- LLM Decision (single-model via OpenRouter fallback chain) ---
        decision = await xai_client.get_trading_decision(
            market_data=market_data,
            portfolio_data=portfolio_data,
            news_summary=news_summary,
        )
        total_analysis_cost += 0.015

        if not decision:
            logger.warning(f"No decision was made for market {market.market_id}. Skipping.")
            await db_manager.record_market_analysis(
                market.market_id, "SKIP", 0.0, total_analysis_cost, "no_decision"
            )
            return None

        decision_action = decision.action
        confidence = decision.confidence

        logger.info(
            f"Generated decision for {market.market_id}: {decision.action} {decision.side} "
            f"at {decision.limit_price}c with confidence {decision.confidence} (cost: ${total_analysis_cost:.3f})"
        )

        # Record the analysis
        await db_manager.record_market_analysis(
            market.market_id, decision_action, confidence, total_analysis_cost
        )

        if decision.action == "BUY" and decision.confidence >= settings.trading.min_confidence_to_trade:
            price = market.yes_price if decision.side == "YES" else market.no_price
            
            # Apply Grok4 edge filtering - 10% minimum edge requirement
            from src.utils.edge_filter import EdgeFilter
            
            # Calculate market probabilities and AI confidence
            market_prob = market.yes_price if decision.side == "YES" else market.no_price
            ai_prob = decision.confidence
            
            # Check edge filter
            should_trade, trade_reason, edge_result = EdgeFilter.should_trade_market(
                ai_probability=ai_prob,
                market_probability=market_prob,
                confidence=decision.confidence,
                additional_filters={
                    'volume': market.volume,
                    'min_volume': settings.trading.min_volume,
                    'time_to_expiry_days': get_time_to_expiry_days(market),
                    'max_time_to_expiry': settings.trading.max_time_to_expiry_days
                }
            )
            
            if not should_trade:
                logger.info(f"❌ EDGE FILTER REJECTED: {market.market_id} - {trade_reason}")
                await db_manager.record_market_analysis(
                    market.market_id, "EDGE_FILTERED", decision.confidence, total_analysis_cost, trade_reason
                )
                return None
                
            logger.info(f"✅ EDGE FILTER APPROVED: {market.market_id} - {trade_reason}")
            
            # Check position limits before calculating quantity
            from src.utils.position_limits import check_can_add_position
            
            # Calculate initial position size
            confidence_delta = decision.confidence - price
            initial_quantity = _calculate_dynamic_quantity(available_balance, price, confidence_delta)
            initial_position_value = initial_quantity * price
            
            # Check if position can be added within limits and adjust if needed
            can_add_position, limit_reason = await check_can_add_position(
                initial_position_value, db_manager, kalshi_client
            )
            
            if not can_add_position:
                # Instead of blocking, try to find a smaller position size that fits
                logger.info(f"⚠️ Position size ${initial_position_value:.2f} exceeds limits, attempting to reduce...")
                
                # Try progressively smaller position sizes
                for reduction_factor in [0.8, 0.6, 0.4, 0.2, 0.1]:
                    reduced_position_value = initial_position_value * reduction_factor
                    reduced_quantity = int(reduced_position_value / price)
                    
                    if reduced_quantity < 1:
                        break  # Can't have less than 1 contract
                    
                    can_add_reduced, reduced_reason = await check_can_add_position(
                        reduced_position_value, db_manager, kalshi_client
                    )
                    
                    if can_add_reduced:
                        initial_position_value = reduced_position_value
                        initial_quantity = reduced_quantity
                        logger.info(f"✅ Position size reduced to ${initial_position_value:.2f} ({initial_quantity} contracts) to fit limits")
                        break
                else:
                    # If even the smallest size doesn't fit, check if it's due to position count
                    from src.utils.position_limits import PositionLimitsManager
                    limits_manager = PositionLimitsManager(db_manager, kalshi_client)
                    current_positions = await limits_manager._get_position_count()
                    
                    if current_positions >= limits_manager.max_positions:
                        logger.info(f"❌ POSITION COUNT LIMIT: {current_positions}/{limits_manager.max_positions} positions - cannot add new position")
                        await db_manager.record_market_analysis(
                            market.market_id, "POSITION_LIMITS", decision.confidence, total_analysis_cost, "Position count limit reached"
                        )
                        return None
                    else:
                        logger.info(f"❌ POSITION SIZE LIMIT: Even minimum size ${initial_position_value * 0.1:.2f} exceeds limits")
                        await db_manager.record_market_analysis(
                            market.market_id, "POSITION_LIMITS", decision.confidence, total_analysis_cost, "Position size limit exceeded"
                        )
                        return None
            
            logger.info(f"✅ POSITION LIMITS OK: ${initial_position_value:.2f} ({initial_quantity} contracts)")
            
            # Check cash reserves before proceeding with trade
            from src.utils.cash_reserves import check_can_trade_with_cash_reserves
            
            trade_value = initial_quantity * price
            can_trade_cash, cash_reason = await check_can_trade_with_cash_reserves(
                trade_value, db_manager, kalshi_client
            )
            
            if not can_trade_cash:
                logger.info(f"❌ CASH RESERVES INSUFFICIENT: {market.market_id} - {cash_reason}")
                await db_manager.record_market_analysis(
                    market.market_id, "CASH_RESERVES", decision.confidence, total_analysis_cost, cash_reason
                )
                return None
            
            logger.info(f"✅ CASH RESERVES OK: {market.market_id} - {cash_reason}")
            quantity = initial_quantity

            if quantity > 0:
                rationale = getattr(decision, 'reasoning', 'No reasoning provided by LLM.')
                # Calculate exit strategy using Grok4 recommendations
                from src.utils.stop_loss_calculator import StopLossCalculator
                
                exit_strategy = StopLossCalculator.calculate_stop_loss_levels(
                    entry_price=price,
                    side=decision.side,
                    confidence=confidence,
                    market_volatility=estimate_market_volatility(market),
                    time_to_expiry_days=get_time_to_expiry_days(market)
                )
                
                position = Position(
                    market_id=market.market_id,
                    side=decision.side,
                    entry_price=price,
                    quantity=quantity,
                    timestamp=datetime.now(),
                    rationale=rationale,
                    confidence=confidence,
                    live=False,
                    
                    # Enhanced exit strategy fields using Grok4 recommendations
                    stop_loss_price=exit_strategy['stop_loss_price'],
                    take_profit_price=exit_strategy['take_profit_price'],
                    max_hold_hours=exit_strategy['max_hold_hours'],
                    target_confidence_change=exit_strategy['target_confidence_change']
                )
                return position

        return None

    except Exception as e:
        logger.error(
            f"Failed to process market {market.market_id}: {market.title}",
            error=str(e),
            exc_info=True
        )
        # Record failed analysis
        try:
            await db_manager.record_market_analysis(
                market.market_id, "ERROR", 0.0, 0.01, "error"
            )
        except:
            pass  # Don't fail on logging failure
        return None


def calculate_dynamic_exit_strategy(
    confidence: float,
    market_volatility: float,
    time_to_expiry: float,
    current_price: float,
    edge_magnitude: float
) -> Dict:
    """
    Calculate dynamic exit strategy based on market conditions.
    
    This implements sophisticated exit logic that adapts to:
    - Market volatility (higher vol = tighter stops)
    - Time to expiry (longer time = looser stops)
    - Confidence level (higher confidence = wider stops)
    - Edge magnitude (bigger edge = longer hold time)
    """
    try:
        # Base parameters
        base_stop_loss_distance = 0.15  # 15 cents default
        base_take_profit_distance = 0.25  # 25 cents default
        base_max_hold_hours = 72  # 3 days default
        
        # Adjust based on volatility
        vol_multiplier = max(0.5, min(2.0, market_volatility / 0.1))  # Scale around 10% vol
        stop_loss_distance = base_stop_loss_distance * vol_multiplier
        take_profit_distance = base_take_profit_distance * vol_multiplier
        
        # Adjust based on confidence
        confidence_factor = max(0.5, min(2.0, confidence / 0.75))  # Scale around 75% confidence
        stop_loss_distance /= confidence_factor  # Higher confidence = tighter stops
        take_profit_distance *= confidence_factor  # Higher confidence = wider targets
        
        # Adjust based on time to expiry
        time_factor = max(0.3, min(3.0, time_to_expiry / 7))  # Scale around 7 days
        max_hold_hours = min(base_max_hold_hours * time_factor, time_to_expiry * 24 * 0.8)  # Max 80% of time to expiry
        
        # Calculate actual prices
        stop_loss_price = max(0.01, current_price - stop_loss_distance)
        take_profit_price = min(0.99, current_price + take_profit_distance)
        
        # Confidence change threshold (exit if confidence drops significantly)
        target_confidence_change = max(0.1, 0.3 - (edge_magnitude * 0.5))  # Bigger edge = more tolerance
        
        return {
            'stop_loss_price': round(stop_loss_price, 2),
            'take_profit_price': round(take_profit_price, 2),
            'max_hold_hours': int(max_hold_hours),
            'target_confidence_change': round(target_confidence_change, 2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating exit strategy: {e}")
        # Return conservative defaults
        return {
            'stop_loss_price': max(0.01, current_price - 0.10),
            'take_profit_price': min(0.99, current_price + 0.20),
            'max_hold_hours': 48,
            'target_confidence_change': 0.2
        }


def estimate_market_volatility(market: Market) -> float:
    """
    Estimate market volatility based on price level and market characteristics.
    """
    try:
        # Get current price to estimate volatility
        current_price = getattr(market, 'yes_price', 50) / 100  # Convert to 0-1
        
        # Binary option volatility formula
        intrinsic_vol = np.sqrt(current_price * (1 - current_price))
        
        # Adjust based on volume (higher volume = lower volatility)
        volume_factor = max(0.5, min(2.0, 1000 / (market.volume + 100)))
        
        # Adjust based on time to expiry
        time_to_expiry = get_time_to_expiry_days(market)
        time_factor = max(0.5, min(2.0, np.sqrt(time_to_expiry / 7)))
        
        estimated_vol = intrinsic_vol * volume_factor * time_factor
        
        # Keep in reasonable range
        return max(0.05, min(0.50, estimated_vol))
        
    except Exception as e:
        logger.error(f"Error estimating volatility for {market.market_id}: {e}")
        return 0.15  # Default 15%


def get_time_to_expiry_days(market: Market) -> float:
    """
    Get time to expiry in days.
    """
    try:
        if hasattr(market, 'expiration_ts') and market.expiration_ts:
            return max(0.1, (market.expiration_ts - time.time()) / 86400)
        elif hasattr(market, 'expiration_ts') and market.expiration_ts:
            expiry_time = datetime.fromtimestamp(market.expiration_ts)
            return max(0.1, (expiry_time - datetime.now()).total_seconds() / 86400)
        else:
            return 7.0  # Default 7 days
    except Exception as e:
        logger.error(f"Error calculating time to expiry: {e}")
        return 7.0
