"""
Position Tracking Job

This job monitors open positions and implements smart exit strategies:
- Market resolution (original)
- Stop-loss exits
- Take-profit exits  
- Time-based exits
- Confidence-based exits
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from src.utils.database import DatabaseManager, Position, TradeLog
from src.config.settings import settings
from src.utils.logging_setup import setup_logging, get_trading_logger
from src.clients.kalshi_client import KalshiClient

async def should_exit_position(
    position: Position, 
    current_yes_price: float, 
    current_no_price: float, 
    market_status: str,
    market_result: str = None
) -> tuple[bool, str, float]:
    """
    Determine if position should be exited based on smart exit strategies.
    
    Returns:
        (should_exit, exit_reason, exit_price)
    """
    current_price = current_yes_price if position.side == "YES" else current_no_price
    
    # 1. Market resolution (original logic)
    if market_status == 'closed':
        # If market resolved, use the result to determine exit price.
        # Case-normalise both sides: Kalshi returns result as lowercase
        # "yes"/"no" while Position.side is stored uppercase "YES"/"NO", so a
        # raw == comparison is ALWAYS False and every resolved position settled
        # at 0.0 — booking a total loss on winners too. (paper_trader.py already
        # did result.lower() for the same field.)
        if market_result:
            resolved_side = str(market_result).strip().lower()
            held_side = str(position.side).strip().lower()
            exit_price = 1.0 if resolved_side == held_side else 0.0
        else:
            # Fallback to current price if no result available
            exit_price = current_price
        return True, "market_resolution", exit_price
    
    # 2. ENHANCED Stop-loss exit using proper logic for YES/NO positions
    if position.stop_loss_price:
        from src.utils.stop_loss_calculator import StopLossCalculator
        
        should_trigger = StopLossCalculator.is_stop_loss_triggered(
            position_side=position.side,
            entry_price=position.entry_price,
            current_price=current_price,
            stop_loss_price=position.stop_loss_price
        )
        
        if should_trigger:
            # Calculate the actual loss to log it
            expected_pnl = StopLossCalculator.calculate_pnl_at_stop_loss(
                entry_price=position.entry_price,
                stop_loss_price=position.stop_loss_price,
                quantity=position.quantity,
                side=position.side
            )
            return True, f"stop_loss_triggered_pnl_{expected_pnl:.2f}", current_price
    
    # 3. Take-profit exit (enhanced logic for YES/NO)
    if position.take_profit_price:
        # Own-side price convention (see src/utils/stop_loss_calculator.py):
        # `current_price` above is already the held side's own price, and a
        # position of either side gains when that price rises. The previous NO
        # branch fired on a *fall*, i.e. it took profit while losing money.
        take_profit_triggered = current_price >= position.take_profit_price

        if take_profit_triggered:
            return True, "take_profit", current_price
    
    # 4. Time-based exit
    if position.max_hold_hours:
        hours_held = (datetime.now() - position.timestamp).total_seconds() / 3600
        if hours_held >= position.max_hold_hours:
            return True, "time_based", current_price
    
    # 5. Emergency exit for positions without stop-loss (legacy positions)
    if not position.stop_loss_price:
        # Calculate emergency stop-loss at 10% loss
        from src.utils.stop_loss_calculator import StopLossCalculator
        emergency_stop = StopLossCalculator.calculate_simple_stop_loss(
            entry_price=position.entry_price,
            side=position.side,
            stop_loss_pct=0.10  # 10% emergency stop
        )
        
        emergency_triggered = StopLossCalculator.is_stop_loss_triggered(
            position_side=position.side,
            entry_price=position.entry_price,
            current_price=current_price,
            stop_loss_price=emergency_stop
        )
        
        if emergency_triggered:
            return True, "emergency_stop_loss_10pct", current_price
    
    # 6. Confidence-based exit (placeholder - would need re-analysis)
    # This would require periodic re-analysis, which we're avoiding for cost reasons
    # Could be implemented as a separate, less frequent job
    
    return False, "", current_price

async def calculate_dynamic_exit_levels(position: Position) -> dict:
    """Calculate smart exit levels using Grok4 recommendations."""
    from src.utils.stop_loss_calculator import StopLossCalculator
    
    # Use the centralized stop-loss calculator
    exit_levels = StopLossCalculator.calculate_stop_loss_levels(
        entry_price=position.entry_price,
        side=position.side,
        confidence=position.confidence or 0.7,
        market_volatility=0.2,  # Default volatility estimate
        time_to_expiry_days=30.0  # Default time estimate
    )
    
    return exit_levels

async def run_tracking(db_manager: Optional[DatabaseManager] = None):
    """
    Enhanced position tracking with smart exit strategies and sell limit orders.
    
    Args:
        db_manager: Optional DatabaseManager instance for testing.
    """
    logger = get_trading_logger("position_tracking")
    logger.info("Starting enhanced position tracking job with sell limit orders.")

    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.initialize()

    kalshi_client = KalshiClient()

    try:
        # Step 1: Place sell limit orders for profit-taking and stop-loss
        from src.jobs.execute import place_profit_taking_orders, place_stop_loss_orders
        
        logger.info("🎯 Checking for profit-taking opportunities...")
        profit_results = await place_profit_taking_orders(
            db_manager=db_manager,
            kalshi_client=kalshi_client,
            profit_threshold=0.20  # 20% profit target
        )
        
        logger.info("🛡️ Checking for stop-loss protection...")
        stop_loss_results = await place_stop_loss_orders(
            db_manager=db_manager,
            kalshi_client=kalshi_client,
            stop_loss_threshold=-0.15  # 15% stop loss
        )
        
        total_sell_orders = profit_results['orders_placed'] + stop_loss_results['orders_placed']
        if total_sell_orders > 0:
            logger.info(f"📈 SELL LIMIT ORDERS SUMMARY: {total_sell_orders} orders placed")
            logger.info(f"   Profit-taking: {profit_results['orders_placed']} orders")
            logger.info(f"   Stop-loss: {stop_loss_results['orders_placed']} orders")
        
        # Step 2: Continue with existing position tracking (market resolution, etc.)
        open_positions = await db_manager.get_open_live_positions()

        if not open_positions:
            logger.info("No open positions to track.")
            return

        logger.info(f"Found {len(open_positions)} open positions to track.")

        resolution_exits = 0       # markets that auto-settled — no sell order needed
        exit_sell_orders_placed = 0  # sell orders we successfully placed
        exit_sell_failures = 0       # exits we tried to execute but couldn't place a sell
        for position in open_positions:
            try:
                # Get current market data
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get('market', {})

                if not market_data:
                    logger.warning(f"Could not retrieve market data for {position.market_id}. Skipping.")
                    continue

                # Get current prices. The legacy yes_price / no_price fields are
                # not returned by Kalshi API v2 (only *_dollars), so reading them
                # yielded 0.0 for every position and every exit decision below
                # was made against a zero mark. Use the repo's normalizer and
                # mark at mid-book, falling back to the last trade.
                from src.utils.market_prices import get_market_prices
                yes_bid, yes_ask, no_bid, no_ask = get_market_prices(market_data)
                if yes_bid + yes_ask > 0:
                    current_yes_price = (yes_bid + yes_ask) / 2
                else:
                    current_yes_price = float(market_data.get('last_price_dollars') or 0)
                if no_bid + no_ask > 0:
                    current_no_price = (no_bid + no_ask) / 2
                else:
                    current_no_price = (1.0 - current_yes_price) if current_yes_price > 0 else 0.0

                # Last resort: the pre-v2 cent-denominated fields, for any
                # payload that still carries them.
                if current_yes_price <= 0 and market_data.get('yes_price'):
                    current_yes_price = float(market_data['yes_price']) / 100
                if current_no_price <= 0 and market_data.get('no_price'):
                    current_no_price = float(market_data['no_price']) / 100
                market_status = market_data.get('status', 'unknown')
                market_result = market_data.get('result')  # Market resolution result
                
                # Exit levels must bracket the entry price under the own-side
                # convention (see src/utils/stop_loss_calculator.py): stop BELOW
                # entry, target ABOVE it. Levels that fail that test are either
                # missing or mis-anchored — e.g. rows written while positions
                # were priced at a phantom $0.50 carry stop 0.535 / target 0.40
                # against a real ~0.955 fill, and a target below the current
                # mark fires a bogus "take_profit" on the very next pass.
                # Re-anchor from the position's actual entry price.
                levels_incoherent = (
                    (position.stop_loss_price or 0) >= position.entry_price
                    or (position.take_profit_price or 1.0) <= position.entry_price
                )
                if (not position.stop_loss_price and not position.take_profit_price) or levels_incoherent:
                    if levels_incoherent:
                        logger.warning(
                            f"Re-anchoring incoherent exit levels for {position.market_id}: "
                            f"entry={position.entry_price:.3f} stop={position.stop_loss_price} "
                            f"target={position.take_profit_price}"
                        )
                    else:
                        logger.info(f"Setting up exit strategy for position {position.market_id}")

                    exit_levels = await calculate_dynamic_exit_levels(position)

                    # Applied in-memory for this pass; add_position persists the
                    # levels for new positions.
                    position.stop_loss_price = exit_levels["stop_loss_price"]
                    position.take_profit_price = exit_levels["take_profit_price"]
                    position.max_hold_hours = exit_levels["max_hold_hours"]
                    position.target_confidence_change = exit_levels["target_confidence_change"]

                # Check if position should be exited (market resolution, time-based, etc.)
                should_exit, exit_reason, exit_price = await should_exit_position(
                    position, current_yes_price, current_no_price, market_status, market_result
                )

                if should_exit:
                    logger.info(
                        f"Exiting position {position.market_id} due to {exit_reason}. "
                        f"Entry: {position.entry_price:.3f}, Exit: {exit_price:.3f}"
                    )

                    # For non-resolution exits, place a real sell order on Kalshi
                    # before touching the DB. Kalshi auto-settles resolved markets,
                    # so we skip order placement only in the market_resolution case.
                    is_resolution = (exit_reason == "market_resolution")

                    if not is_resolution:
                        # Sanity guard: a $0 exit on an active market means we're
                        # working from bad market data. Refuse to write a phantom
                        # close — was the source of issue #49.
                        if exit_price <= 0.0:
                            logger.error(
                                f"Refusing to close {position.market_id}: exit_price={exit_price:.3f} "
                                f"on non-resolution exit ({exit_reason}). Likely missing market data; "
                                f"will retry next cycle."
                            )
                            exit_sell_failures += 1
                            continue

                        from src.jobs.execute import place_sell_limit_order
                        sell_ok = await place_sell_limit_order(
                            position=position,
                            limit_price=exit_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client,
                        )
                        if not sell_ok:
                            logger.error(
                                f"Sell order failed for {position.market_id} ({exit_reason}); "
                                f"position remains open. Will retry next cycle."
                            )
                            exit_sell_failures += 1
                            continue
                        exit_sell_orders_placed += 1

                    # Calculate PnL
                    pnl = (exit_price - position.entry_price) * position.quantity

                    # Create trade log
                    trade_log = TradeLog(
                        market_id=position.market_id,
                        side=position.side,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        quantity=position.quantity,
                        pnl=pnl,
                        entry_timestamp=position.timestamp,
                        exit_timestamp=datetime.now(),
                        rationale=f"{position.rationale} | EXIT: {exit_reason}"
                    )

                    # Record the exit. For non-resolution exits the sell order
                    # may still be resting unfilled — the local DB now optimistically
                    # treats it as closed, which mirrors the existing behavior of
                    # place_profit_taking_orders / place_stop_loss_orders.
                    await db_manager.add_trade_log(trade_log)
                    await db_manager.update_position_status(position.id, 'closed')

                    if is_resolution:
                        resolution_exits += 1
                    logger.info(
                        f"Position for market {position.market_id} closed via {exit_reason}. "
                        f"PnL: ${pnl:.2f}"
                    )
                else:
                    # Log current position status for monitoring
                    current_price = current_yes_price if position.side == "YES" else current_no_price
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    hours_held = (datetime.now() - position.timestamp).total_seconds() / 3600
                    
                    logger.debug(
                        f"Position {position.market_id} status: "
                        f"Entry: {position.entry_price:.3f}, Current: {current_price:.3f}, "
                        f"Unrealized P&L: ${unrealized_pnl:.2f}, Hours held: {hours_held:.1f}"
                    )

            except Exception as e:
                logger.error(f"Failed to process position for market {position.market_id}.", error=str(e))

        logger.info(
            f"Position tracking completed. "
            f"Profit/SL sell orders: {total_sell_orders}, "
            f"Resolution exits: {resolution_exits}, "
            f"Exit sell orders placed: {exit_sell_orders_placed}, "
            f"Exit failures (still open): {exit_sell_failures}"
        )

    except Exception as e:
        logger.error("Error in position tracking job.", error=str(e), exc_info=True)
    finally:
        await kalshi_client.close()

if __name__ == "__main__":
    setup_logging()
    asyncio.run(run_tracking())
