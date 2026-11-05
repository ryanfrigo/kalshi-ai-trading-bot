"""
Trade Execution Job

This job takes a position and executes it as a trade.
"""
import asyncio
import math
import uuid
from datetime import datetime
from typing import Optional, Dict

from src.utils.database import DatabaseManager, Position, TradeLog
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.kalshi_client import KalshiClient, KalshiAPIError
from src.utils.market_prices import get_market_prices, is_tradeable_market

async def execute_position(
    position: Position, 
    live_mode: bool, 
    db_manager: DatabaseManager, 
    kalshi_client: KalshiClient
) -> bool:
    """
    Executes a single trade position.
    
    Args:
        position: The position to execute.
        live_mode: Whether to execute a live or simulated trade.
        db_manager: The database manager instance.
        kalshi_client: The Kalshi client instance.
        
    Returns:
        True if execution was successful, False otherwise.
    """
    logger = get_trading_logger("trade_execution")
    logger.info(f"🎯 Executing position for market: {position.market_id}")
    logger.info(f"🎛️ Live mode: {live_mode}")
    
    if live_mode:
        logger.warning(f"💰 PLACING LIVE ORDER - Real money will be used for {position.market_id}")
        try:
            # Get current market prices to determine the appropriate price field
            market_data = await kalshi_client.get_market(position.market_id)
            market = market_data.get('market', {})
            
            # For market orders, use the ask price based on which side we're buying
            side_lower = position.side.lower()
            client_order_id = str(uuid.uuid4())
            
            # Prepare order parameters
            order_params = {
                "ticker": position.market_id,
                "client_order_id": client_order_id,
                "side": side_lower,
                "action": "buy",
                "count": position.quantity,
                "type_": "market"
            }
            
            # Add the appropriate price field based on side
            # For market orders, we use the ask price (what we're willing to pay)
            # get_market_prices normalizes both API v2 (dollar floats) and legacy (cent ints)
            # to dollar values (0.0–1.0). place_order expects cents (int), so multiply by 100.
            _yes_bid, yes_ask_dollars, _no_bid, no_ask_dollars = get_market_prices(market)

            # --- Price sanity checks (issue #42) ---
            # Guard 1: Collection/aggregate tickers return $1.00/$1.00 and are not
            # directly tradeable.  Placing an order against them yields HTTP 400
            # invalid_price.  Reject early if both asks are at or above $0.99.
            if not is_tradeable_market(market):
                logger.warning(
                    f"⚠️  Skipping {position.market_id}: collection/aggregate ticker "
                    f"(yes_ask={yes_ask_dollars:.4f}, no_ask={no_ask_dollars:.4f}). "
                    "Both asks >= $0.99 — market is not directly tradeable (issue #42)."
                )
                return False

            # Guard 2: Reject any price that would convert to 0¢ or >= 100¢ — these
            # are outside the valid Kalshi order range and will be rejected by the API.
            ask_dollars = yes_ask_dollars if side_lower == "yes" else no_ask_dollars
            ask_cents = int(round(ask_dollars * 100))
            if ask_cents <= 0 or ask_cents >= 100:
                logger.warning(
                    f"⚠️  Skipping {position.market_id}: {side_lower} ask price "
                    f"{ask_dollars:.4f} converts to {ask_cents}¢ which is outside the "
                    "valid Kalshi range (1–99¢). Cannot place order (issue #42)."
                )
                return False
            # --- End price sanity checks ---

            if side_lower == "yes":
                yes_ask_cents = int(round(yes_ask_dollars * 100))
                if not (1 <= yes_ask_cents <= 99):
                    logger.warning(
                        f"Price sanity check failed for {position.market_id}: "
                        f"yes_ask_cents={yes_ask_cents} is outside valid range 1–99 "
                        f"(raw yes_ask_dollars={yes_ask_dollars:.4f}). Skipping order."
                    )
                    return False
                order_params["yes_price"] = yes_ask_cents
            else:  # side_lower == "no"
                no_ask_cents = int(round(no_ask_dollars * 100))
                if not (1 <= no_ask_cents <= 99):
                    logger.warning(
                        f"Price sanity check failed for {position.market_id}: "
                        f"no_ask_cents={no_ask_cents} is outside valid range 1–99 "
                        f"(raw no_ask_dollars={no_ask_dollars:.4f}). Skipping order."
                    )
                    return False
                order_params["no_price"] = no_ask_cents

            # --- Balance guard: never place an order we cannot afford ---
            # Fail closed: if the balance check itself errors, skip the order
            # rather than risk placing it against a stale bankroll figure.
            order_cost_cents = ask_cents * position.quantity
            try:
                balance_response = await kalshi_client.get_balance()
                available_cents = int(balance_response.get("balance", 0) or 0)
            except Exception as e:
                logger.error(
                    f"❌ Could not verify balance before order for {position.market_id} "
                    f"({e}); skipping order (fail-closed)."
                )
                return False
            if order_cost_cents > available_cents:
                logger.warning(
                    f"⚠️  Insufficient balance for {position.market_id}: order needs "
                    f"~{order_cost_cents}¢ ({position.quantity} @ {ask_cents}¢) but only "
                    f"{available_cents}¢ available. Skipping order."
                )
                return False
            # --- End balance guard ---

            logger.info(f"Placing order with params: {order_params}")
            order_response = await kalshi_client.place_order(**order_params)
            
            # For a market order, the fill price is not guaranteed.
            # A more robust implementation would query the /fills endpoint
            # to confirm the execution price after the fact.
            # For now, we will optimistically assume it fills at the entry price.
            fill_price = position.entry_price

            await db_manager.update_position_to_live(position.id, fill_price)
            logger.info(f"✅ LIVE ORDER PLACED for {position.market_id}. Order ID: {order_response.get('order', {}).get('order_id')}")
            logger.info(f"💰 Real money used: ${position.quantity * fill_price:.2f}")
            return True

        except KalshiAPIError as e:
            logger.error(f"❌ FAILED to place LIVE order for {position.market_id}: {e}")
            return False
    else:
        # Simulate the trade
        await db_manager.update_position_to_live(position.id, position.entry_price)
        logger.info(f"📝 PAPER TRADE SIMULATED for {position.market_id} - No real money used")
        logger.info(f"📊 Would have used: ${position.quantity * position.entry_price:.2f}")
        return True


def _filled_count(order: Dict) -> float:
    """fill_count from a Kalshi order response; 0.0 when absent or unreadable."""
    for key in ("fill_count_fp", "fill_count"):
        try:
            v = float(order.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v >= 0:
            return v
    return 0.0


async def _record_ioc_exit(position: Position, order: Dict, limit_price: float,
                           db_manager: DatabaseManager, logger,
                           reason: str = "partial IOC fill under risk halt",
                           track: bool = False) -> bool:
    """True only when the IOC sell filled the whole position.

    A partial fill logs the sold part and shrinks the open position to what is
    left; no fill leaves it untouched. Either way the caller keeps it open.
    With ``track`` the rest is a resting GTC order: its id is stored in the same
    transaction as the partial exit.
    """
    filled = round(min(_filled_count(order), position.quantity), 2)
    if filled >= round(position.quantity, 2):
        return True
    if filled > 0:
        columns = _resting_exit_columns(position, order, logger) if track else {}
        await db_manager.record_partial_exit(position.id, TradeLog(
            market_id=position.market_id, side=position.side,
            entry_price=position.entry_price, exit_price=limit_price,
            quantity=filled, pnl=(limit_price - position.entry_price) * filled,
            entry_timestamp=position.timestamp, exit_timestamp=datetime.now(),
            rationale=f"{position.rationale} | EXIT: {reason}"),
            round(position.quantity - filled, 2), **columns)
    elif track:
        await _track_resting_exit(position, order, db_manager, logger)
    logger.warning(
        f"Exit sell for {position.market_id} filled {filled} of {position.quantity} "
        f"({reason}); the rest stays open.")
    return False


def log_paper_exit(position: Position, limit_price: float, reason: str) -> bool:
    """Paper-mode exit: a log line and NOTHING else.

    No exchange request and no DB write. The DB cannot tell a paper-simulated row from a
    real holding (execute_position's paper branch also stores live=1), so closing or
    logging a P&L for the row would corrupt a real position's management.
    """
    cents = int(round(limit_price * 100))
    get_trading_logger("paper_exit").info(
        f"PAPER: would sell {position.quantity} {position.side} {position.market_id} "
        f"at {cents}c ({reason})")
    return True


async def place_sell_limit_order(
    position: Position,
    limit_price: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    order_out: Optional[Dict] = None,
    live: bool = False,
) -> bool:
    """
    Place a sell limit order to close an existing position.
    
    Args:
        position: The position to close
        limit_price: The limit price for the sell order (in dollars)
        db_manager: Database manager
        kalshi_client: Kalshi API client
        order_out: If given, filled with the order response so the caller can
            reconcile fills (see reconcile_exit_fill)
        live: must be True to send. False (default) refuses before any request.
    
    Returns:
        True if order placed successfully, False otherwise
    """
    logger = get_trading_logger("sell_limit_order")

    # Lowest-layer paper guard: no caller can send a sell without live=True.
    if not live:
        logger.warning(f"Not selling {position.market_id}: paper mode (live=False); "
                       f"no order sent.")
        return False

    try:
        # Convert price to cents for Kalshi API; round, since int() truncates
        # float error (0.29 * 100 == 28.999...) and would sell 1c low.
        limit_price_cents = int(round(limit_price * 100))
        if not 1 <= limit_price_cents <= 99:
            logger.warning(f"Not selling {position.market_id}: limit {limit_price_cents}c "
                           f"is outside 1-99c.")
            return False
        if position.exit_order_id and not await _cancel_exit_above(
                position, limit_price_cents, kalshi_client, db_manager, logger):
            logger.warning(f"Not selling {position.market_id}: exit order "
                           f"{position.exit_order_id} is still being tracked.")
            return False

        # Never sell more than we hold on the exchange, net of our resting sells:
        # a DB row can outlive the contracts, and an oversell opens the other side.
        sellable = await _sellable_contracts(kalshi_client, position.market_id, position.side)
        if sellable is None:
            logger.warning(f"Not selling {position.market_id}: could not confirm held "
                           f"contracts or resting sells on the exchange.")
            return False
        count = min(position.quantity, math.floor(sellable * 100) / 100)
        if count <= 0:
            logger.warning(f"Not selling {position.market_id}: nothing sellable "
                           f"(held minus resting sells = {sellable}).")
            return False

        import uuid
        client_order_id = str(uuid.uuid4())
        
        # For sell orders, we need to use the opposite side logic:
        # - If we have YES position, we sell YES shares (action="sell", side="yes")
        # - If we have NO position, we sell NO shares (action="sell", side="no")
        side = position.side.lower()  # "YES" -> "yes", "NO" -> "no"
        
        order_params = {
            "ticker": position.market_id,
            "client_order_id": client_order_id,
            "side": side,
            "action": "sell",  # We're selling our existing position
            "count": count,
            "type_": "limit"
        }
        
        # Add the appropriate price parameter based on what we're selling
        if side == "yes":
            order_params["yes_price"] = limit_price_cents
        else:
            order_params["no_price"] = limit_price_cents
        
        logger.info(f"🎯 Placing SELL LIMIT order: {count} {side.upper()} at {limit_price_cents}¢ for {position.market_id}")
        
        # Place the sell limit order
        response = await kalshi_client.place_order(**order_params)
        if order_out is not None and isinstance(response, dict):
            order_out.update(response)
        
        if response and 'order' in response:
            order_id = response['order'].get('order_id', client_order_id)

            if response.get('reduce_only_ioc'):
                # Under a limit halt the sell went out immediate-or-cancel, so the
                # unfilled remainder is gone. Count only what filled.
                return await _record_ioc_exit(
                    position, response['order'], limit_price, db_manager, logger)
            
            # Record the sell order in the database (we could add a sell_orders table if needed)
            logger.info(f"✅ SELL LIMIT ORDER placed successfully! Order ID: {order_id}")
            logger.info(f"   Market: {position.market_id}")
            logger.info(f"   Side: {side.upper()} (selling {count} shares)")
            logger.info(f"   Limit Price: {limit_price_cents}¢")
            logger.info(f"   Expected Proceeds: ${limit_price * count:.2f}")
            
            return True
        else:
            logger.error(f"❌ Failed to place sell limit order: {response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error placing sell limit order for {position.market_id}: {e}")
        return False


async def _cancel_exit_above(position: Position, limit_cents: int, kalshi_client: KalshiClient,
                             db_manager: DatabaseManager, logger) -> bool:
    """Cancel the tracked resting exit when it is priced above a new exit at ``limit_cents``.

    A GTC exit the market moved away from never fills, and while its id is stored
    every later exit is refused. Fills on it are booked first, then it is cancelled
    and read again so a fill that raced the cancel is booked too. True only when the
    order is gone, every fill is booked and the id is cleared; ``position`` is then
    refreshed in place from the DB (the caller books the new sell against it). Any
    read, cancel or booking problem returns False and leaves the id set.
    """
    oid = position.exit_order_id

    async def refresh(expected_oid) -> bool:
        fresh = await db_manager.get_position_by_market_and_side(position.market_id,
                                                                 position.side)
        if fresh is None or fresh.id != position.id or fresh.exit_order_id != expected_oid:
            return False
        vars(position).update(vars(fresh))
        return True

    try:
        order = (await kalshi_client.get_order(oid)).get("order")
        cents = int(round(float(order[f"{position.side.lower()}_price_dollars"]) * 100))
        if order["status"] != "resting" or cents <= limit_cents:
            return False
        if (await reconcile_resting_exit(position, kalshi_client, db_manager)
                not in ("unchanged", "shrunk") or not await refresh(oid)):
            return False
        logger.info(f"Cancelling exit {oid} on {position.market_id} at {cents}c: "
                    f"above the new exit at {limit_cents}c.")
        await kalshi_client.cancel_order(oid)
        return (await reconcile_resting_exit(position, kalshi_client, db_manager)
                in ("cleared", "shrunk") and await refresh(None))
    except Exception as e:
        logger.error(f"Exit {oid} on {position.market_id} not replaced ({e!r}).")
        return False


def _order_outcome(o: Dict) -> Optional[str]:
    """Outcome ('yes'/'no') a resting order is positioned for; None if unreadable.

    Kalshi's Order schema: outcome_side ("buy-yes and sell-no produce 'yes';
    buy-no and sell-yes produce 'no'"), book_side ('bid' = yes, 'ask' = no), and
    the deprecated side + action (https://docs.kalshi.com/api-reference/orders/get-orders).
    """
    outcome = str(o.get("outcome_side") or "").lower()
    if outcome in ("yes", "no"):
        return outcome
    book = str(o.get("book_side") or "").lower()
    if book in ("bid", "ask"):
        return "yes" if book == "bid" else "no"
    side, action = str(o.get("side") or "").lower(), str(o.get("action") or "").lower()
    if side in ("yes", "no") and action in ("buy", "sell"):
        return side if action == "buy" else ("no" if side == "yes" else "yes")
    return None


async def _sellable_contracts(kalshi_client: KalshiClient, ticker: str,
                              side: str) -> Optional[float]:
    """Contracts held on ``side`` of ``ticker`` minus our resting orders against it.

    A resting order counts against a YES holding when it is positioned for NO
    (sell YES, or buy NO, which nets the same contracts), and vice versa. None
    when an exchange read fails or a resting order can't be read, so the caller
    sends nothing (fail closed); an unreadable order is logged by its fields.
    """
    logger = get_trading_logger("sellable_guard")
    held = await kalshi_client._held_on_side(ticker, side)
    if isinstance(held, bool) or not isinstance(held, (int, float)) or not math.isfinite(held):
        logger.error(f"Exit blocked on {ticker}: held {side} contracts unreadable ({held!r}).")
        return None
    try:
        resp = await kalshi_client.get_orders(ticker=ticker, status="resting")
    except Exception as e:
        logger.error(f"Exit blocked on {ticker}: resting-orders read failed ({e}).")
        return None
    orders = resp.get("orders") if isinstance(resp, dict) else None
    if not isinstance(orders, list):
        logger.error(f"Exit blocked on {ticker}: resting-orders response has no 'orders' list.")
        return None
    if resp.get("cursor") and len(orders) >= 100:  # a full default page: there may be more
        logger.error(f"Exit blocked on {ticker}: resting orders span more than one page.")
        return None
    against = "no" if side.lower() == "yes" else "yes"
    resting = 0.0
    for o in orders:
        if not isinstance(o, dict) or o.get("ticker") != ticker:
            continue
        outcome = _order_outcome(o)
        if outcome is None:
            logger.error(f"Exit blocked on {ticker}: resting order {o.get('order_id')} has no "
                         f"readable outcome_side, book_side or side+action; keys={sorted(o)}.")
            return None
        if outcome != against:
            continue
        left = None
        for key in ("remaining_count_fp", "remaining_count"):
            try:
                left = float(o.get(key))
                break
            except (TypeError, ValueError):
                continue
        if left is None or not math.isfinite(left) or left < 0:
            logger.error(f"Exit blocked on {ticker}: resting order {o.get('order_id')} has no "
                         f"readable remaining_count_fp; keys={sorted(o)}.")
            return None
        resting += left
    return held - resting


async def _read_ticker_state(kalshi_client: KalshiClient, ticker: str, logger):
    """GET-only read of what the exchange shows on ``ticker``.

    Returns ``(held_signed, resting_orders, resting_qty)`` or None when any read
    fails or is unreadable (several position rows, a non-finite count, a full
    orders page). Callers treat None as "unknown" and change nothing.
    """
    try:
        from src.utils.market_prices import signed_position
        resp = await kalshi_client.get_positions(ticker=ticker)
        rows = resp.get("market_positions") if isinstance(resp, dict) else None
        if not isinstance(rows, list):
            logger.error(f"{ticker}: positions response unreadable.")
            return None
        mine = [r for r in rows if isinstance(r, dict) and r.get("ticker") == ticker]
        if len(mine) > 1:
            logger.error(f"{ticker}: {len(mine)} position rows.")
            return None
        held = signed_position(mine[0]) if mine else 0.0
        if not math.isfinite(held):
            logger.error(f"{ticker}: exchange position {held!r} is not finite.")
            return None
        resp = await kalshi_client.get_orders(ticker=ticker, status="resting")
        orders = resp.get("orders") if isinstance(resp, dict) else None
        if not isinstance(orders, list):
            logger.error(f"{ticker}: resting-orders response unreadable.")
            return None
        if resp.get("cursor") and len(orders) >= 100:
            logger.error(f"{ticker}: resting orders span more than one page.")
            return None
        n, qty = 0, 0.0
        for o in orders:
            if isinstance(o, dict) and o.get("ticker") != ticker:
                continue
            n += 1  # an unreadable entry still counts as an order
            for key in ("remaining_count_fp", "remaining_count"):
                try:
                    q = float(o.get(key))  # type: ignore[union-attr]
                except (TypeError, ValueError, AttributeError):
                    continue
                if math.isfinite(q):
                    qty += q
                break
        return held, n, qty
    except Exception as e:
        logger.error(f"{ticker}: exchange read failed ({e}).")
        return None


STALE_FAILED_ROW_MINUTES = 10


async def reconcile_stale_failed_rows(db_manager: DatabaseManager, kalshi_client: KalshiClient,
                                      min_age_minutes: int = STALE_FAILED_ROW_MINUTES) -> int:
    """Void open ``live=0`` rows older than ``min_age_minutes`` that the exchange shows flat.

    Such rows come from an ambiguous failed buy and otherwise block their market
    and hold a position slot forever. Only GET requests are made. A row whose
    ticker shows a holding or a resting order is NOT adopted or marked live: one
    ERROR line names the ticker, side and exchange quantity and the row is left
    as is. A read failure also leaves the row open. Returns the number voided.
    """
    from datetime import timedelta
    logger = get_trading_logger("stale_failed_rows")
    voided = 0
    try:
        rows = await db_manager.get_open_non_live_positions()
    except Exception as e:
        logger.error(f"Stale-row pass skipped: could not read rows ({e}).")
        return 0
    cutoff = datetime.now() - timedelta(minutes=min_age_minutes)
    for row in rows:
        try:
            if row.id is None or row.timestamp > cutoff:
                continue
            state = await _read_ticker_state(kalshi_client, row.market_id, logger)
            if state is None:
                continue  # unknown: leave open
            held, n_orders, resting_qty = state
            if held == 0 and not n_orders:
                await db_manager.update_position_status(row.id, 'voided')
                voided += 1
                logger.info(f"Voided stale failed row {row.id} on {row.market_id}: exchange flat.")
            else:
                logger.error(
                    f"STALE FAILED ROW NOT RECONCILED: {row.market_id} side={row.side} "
                    f"exchange held={held:g} resting_orders={n_orders} resting_qty={resting_qty:g}; "
                    f"row {row.id} (live=0) left open for the owner.")
        except Exception as e:
            logger.error(f"Stale-row check failed for row {getattr(row, 'id', None)}: {e}")
    return voided


async def reconcile_exit_fill(position: Position, response: Dict, sell_ok: bool,
                              limit_price: float, db_manager: DatabaseManager,
                              reason: str) -> str:
    """Record what an exit sell filled, read from the order response, never assumed.

    Returns "filled" (exit logged, position closed), "partial" (sold part logged,
    position shrunk), "resting" (GTC order on the book, nothing recorded) or
    "none" (no order went out, or it filled nothing).
    """
    logger = get_trading_logger("exit_fill")
    order = response.get("order") if isinstance(response, dict) else None
    if not isinstance(order, dict):
        logger.warning(f"No {reason} sell sent for {position.market_id}; position unchanged.")
        return "none"
    if response.get("reduce_only_ioc"):
        # place_sell_limit_order already ran _record_ioc_exit: True = all filled.
        full = sell_ok
    elif _filled_count(order) > 0:
        full = await _record_ioc_exit(position, order, limit_price, db_manager, logger,
                                      reason=f"partial fill ({reason})", track=True)
    else:
        logger.info(f"{reason} sell for {position.market_id} is resting; "
                    f"not recorded as sold until it fills.")
        await _track_resting_exit(position, order, db_manager, logger)
        return "resting"
    if full:
        await db_manager.add_trade_log(TradeLog(
            market_id=position.market_id, side=position.side,
            entry_price=position.entry_price, exit_price=limit_price,
            quantity=position.quantity,
            pnl=(limit_price - position.entry_price) * position.quantity,
            entry_timestamp=position.timestamp, exit_timestamp=datetime.now(),
            rationale=f"{position.rationale} | EXIT: {reason}"))
        await db_manager.update_position_status(position.id, 'closed')
        logger.info(f"✅ {reason} sell for {position.market_id} filled; position closed.")
        return "filled"
    if _filled_count(order) > 0:
        logger.info(f"{reason} sell for {position.market_id} partly filled; "
                    f"sold part logged, the rest stays open.")
        return "partial"
    logger.info(f"{reason} sell for {position.market_id} filled nothing; position unchanged.")
    return "none"


async def _track_resting_exit(position: Position, order: Dict, db_manager: DatabaseManager,
                              logger) -> None:
    """Store a GTC exit's order id and recorded fill count so later fills get booked."""
    columns = _resting_exit_columns(position, order, logger)
    if columns:
        await db_manager.set_exit_order(position.id, columns["exit_order_id"],
                                        columns["exit_order_filled"])


def _resting_exit_columns(position: Position, order: Dict, logger) -> Dict:
    """The exit_order_* columns that track a resting GTC exit; {} without an order_id."""
    if not order.get("order_id"):
        logger.warning(f"Exit sell for {position.market_id} has no order_id; "
                       f"a later fill cannot be reconciled.")
        return {}
    return {"exit_order_id": order["order_id"], "exit_order_filled": _filled_count(order),
            "exit_order_read_failures": 0}


_MAX_EXIT_READ_FAILURES = 3


async def _exit_read_failed(position: Position, db_manager: DatabaseManager, logger) -> str:
    """Count one bad read of the stored exit order; drop the id on the third in a row."""
    oid = position.exit_order_id
    failures = (position.exit_order_read_failures or 0) + 1
    if failures >= _MAX_EXIT_READ_FAILURES:
        await db_manager.set_exit_order(position.id, None, None)
        logger.error(f"Exit order {oid} on {position.market_id} unreadable {failures} times "
                     f"in a row: late fills of {oid} will not be booked; check the trade log.")
        return "abandoned"
    await db_manager.set_exit_order(position.id, oid, position.exit_order_filled, failures)
    return "unchanged"


async def reconcile_resting_exit(position: Position, kalshi_client: KalshiClient,
                                 db_manager: DatabaseManager) -> str:
    """Book fills of the position's stored exit order since we last looked.

    Returns "closed", "shrunk", "cleared" (order done, nothing new filled),
    "unchanged" or "abandoned". A read or field problem books nothing and counts
    one failure; a good read resets the count. The third failure in a row drops
    the stored id, so the row falls back to the sellable guard.
    """
    logger = get_trading_logger("exit_fill")
    oid, ticker = position.exit_order_id, position.market_id
    try:
        order = (await kalshi_client.get_order(oid)).get("order")
        filled = float(order["fill_count_fp"])
        price = float(order[f"{position.side.lower()}_price_dollars"])
        status = order["status"]
    except Exception as e:
        logger.error(f"Exit order {oid} on {ticker} unreadable ({e!r}); position unchanged.")
        return await _exit_read_failed(position, db_manager, logger)
    new = round(min(filled - (position.exit_order_filled or 0), position.quantity), 2)
    if (order.get("ticker") != ticker or status not in ("resting", "canceled", "executed")
            or not math.isfinite(filled) or not math.isfinite(price) or new < 0):
        logger.error(f"Exit order {oid} on {ticker} inconsistent (ticker={order.get('ticker')}, "
                     f"status={status}, filled={filled}); position unchanged.")
        return await _exit_read_failed(position, db_manager, logger)
    done = status != "resting"
    if new == 0:
        if done:
            await db_manager.set_exit_order(position.id, None, None)
            logger.info(f"Exit order {oid} on {ticker} {status} with no new fill; cleared.")
            return "cleared"
        if position.exit_order_read_failures:
            await db_manager.set_exit_order(position.id, oid, position.exit_order_filled)
        return "unchanged"
    log = TradeLog(market_id=ticker, side=position.side, entry_price=position.entry_price,
                   exit_price=price, quantity=new, pnl=(price - position.entry_price) * new,
                   entry_timestamp=position.timestamp, exit_timestamp=datetime.now(),
                   rationale=f"{position.rationale} | EXIT: resting exit {oid} filled later")
    if new >= round(position.quantity, 2):
        await db_manager.record_partial_exit(position.id, log, position.quantity, status="closed",
                                             exit_order_id=None, exit_order_filled=None,
                                             exit_order_read_failures=0)
        logger.info(f"Exit order {oid} on {ticker} filled; position closed.")
        return "closed"
    await db_manager.record_partial_exit(position.id, log, round(position.quantity - new, 2),
                                         exit_order_id=None if done else oid,
                                         exit_order_filled=None if done else filled,
                                         exit_order_read_failures=0)
    logger.info(f"Exit order {oid} on {ticker} filled {new} more; position shrunk.")
    return "shrunk"


# Threshold compares allow float error: (0.45 - 0.50) / 0.50 is
# -0.09999999999999998, which would miss an exact -10% stop.
_PCT_EPS = 1e-9


async def place_profit_taking_orders(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    profit_threshold: float = 0.25,  # 25% profit target
    live: bool = False,
) -> Dict[str, int]:
    """
    Place sell limit orders for positions that have reached profit targets.
    
    Args:
        db_manager: Database manager
        kalshi_client: Kalshi API client
        profit_threshold: Minimum profit percentage to trigger sell order
    
    Returns:
        Dictionary with results: {'orders_placed': int, 'positions_processed': int}; paper mode adds 'simulated': int
        (live=False, the default, sends nothing and writes nothing: exits are log-only)
    """
    logger = get_trading_logger("profit_taking")
    
    results = {'orders_placed': 0, 'positions_processed': 0}
    
    try:
        # Get all open live positions
        positions = await db_manager.get_open_live_positions()
        
        if not positions:
            logger.info("No open positions to process for profit taking")
            return results
        
        logger.info(f"📊 Checking {len(positions)} positions for profit-taking opportunities")
        
        for position in positions:
            try:
                results['positions_processed'] += 1
                
                # Get current market data
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get('market', {})
                
                if not market_data:
                    logger.warning(f"Could not get market data for {position.market_id}")
                    continue
                
                # Price the held side at its bid, what a sell can get. v2 sends
                # *_dollars fields only; there is no yes_price/no_price key.
                yes_bid, _, no_bid, _ = get_market_prices(market_data)
                current_price = yes_bid if position.side == "YES" else no_bid
                
                # Calculate current profit
                if current_price > 0:
                    profit_pct = (current_price - position.entry_price) / position.entry_price
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    logger.debug(f"Position {position.market_id}: Entry=${position.entry_price:.3f}, Current=${current_price:.3f}, Profit={profit_pct:.1%}, PnL=${unrealized_pnl:.2f}")
                    
                    # Check if we should place a profit-taking sell order
                    if profit_pct >= profit_threshold - _PCT_EPS:
                        # Calculate sell limit price (slightly below current to ensure execution)
                        sell_price = current_price * 0.98  # 2% below current price for quick execution
                        
                        logger.info(f"💰 PROFIT TARGET HIT: {position.market_id} - {profit_pct:.1%} profit (${unrealized_pnl:.2f})")
                        
                        if not live:
                            if log_paper_exit(position, sell_price, "profit_take"):
                                results['simulated'] = results.get('simulated', 0) + 1
                            continue
                        # Place sell limit order
                        order = {}
                        success = await place_sell_limit_order(
                            position=position,
                            limit_price=sell_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client,
                            order_out=order,
                            live=live,
                        )
                        # Logs the outcome; closes or shrinks the position only on a fill.
                        await reconcile_exit_fill(position, order, success, sell_price,
                                                  db_manager, "profit_take")
                        if order:
                            results['orders_placed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing position {position.market_id} for profit taking: {e}")
                continue
        
        logger.info(f"🎯 Profit-taking summary: {results['orders_placed']} orders placed from {results['positions_processed']} positions")
        return results
        
    except Exception as e:
        logger.error(f"Error in profit-taking order placement: {e}")
        return results


async def place_stop_loss_orders(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    stop_loss_threshold: float = -0.10,  # 10% stop loss
    live: bool = False,
) -> Dict[str, int]:
    """
    Place sell limit orders for positions that need stop-loss protection.
    
    Args:
        db_manager: Database manager
        kalshi_client: Kalshi API client
        stop_loss_threshold: Maximum loss percentage before triggering stop loss
    
    Returns:
        Dictionary with results: {'orders_placed': int, 'positions_processed': int}; paper mode adds 'simulated': int
        (live=False, the default, sends nothing and writes nothing: exits are log-only)
    """
    logger = get_trading_logger("stop_loss_orders")
    
    results = {'orders_placed': 0, 'positions_processed': 0}
    
    try:
        # Get all open live positions
        positions = await db_manager.get_open_live_positions()
        
        if not positions:
            logger.info("No open positions to process for stop-loss orders")
            return results
        
        logger.info(f"🛡️ Checking {len(positions)} positions for stop-loss protection")
        
        for position in positions:
            try:
                results['positions_processed'] += 1
                
                # Get current market data
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get('market', {})
                
                if not market_data:
                    logger.warning(f"Could not get market data for {position.market_id}")
                    continue
                
                # Price the held side at its bid, what a sell can get. v2 sends
                # *_dollars fields only; there is no yes_price/no_price key.
                yes_bid, _, no_bid, _ = get_market_prices(market_data)
                current_price = yes_bid if position.side == "YES" else no_bid
                
                # Calculate current loss
                if current_price > 0:
                    loss_pct = (current_price - position.entry_price) / position.entry_price
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    # Check if we need stop-loss protection
                    if loss_pct <= stop_loss_threshold + _PCT_EPS:  # Negative loss percentage
                        # Sell at the bid (marketable). A price off entry rests above
                        # the bid after a gap, never fills, and blocks later exits.
                        stop_price = current_price
                        
                        logger.info(f"🛡️ STOP LOSS TRIGGERED: {position.market_id} - {loss_pct:.1%} loss (${unrealized_pnl:.2f})")
                        
                        if not live:
                            if log_paper_exit(position, stop_price, "stop_loss"):
                                results['simulated'] = results.get('simulated', 0) + 1
                            continue
                        # Place stop-loss sell order
                        order = {}
                        success = await place_sell_limit_order(
                            position=position,
                            limit_price=stop_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client,
                            order_out=order,
                            live=live,
                        )
                        # Logs the outcome; closes or shrinks the position only on a fill.
                        await reconcile_exit_fill(position, order, success, stop_price,
                                                  db_manager, "stop_loss")
                        if order:
                            results['orders_placed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing position {position.market_id} for stop loss: {e}")
                continue
        
        logger.info(f"🛡️ Stop-loss summary: {results['orders_placed']} orders placed from {results['positions_processed']} positions")
        return results
        
    except Exception as e:
        logger.error(f"Error in stop-loss order placement: {e}")
        return results
