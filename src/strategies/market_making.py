"""
Market Making Strategy - Advanced Liquidity Provision

This strategy implements sophisticated market making by:
1. Placing limit orders on both YES and NO sides
2. Calculating optimal spreads based on volatility and edge
3. Managing inventory risk and position sizing
4. Rebalancing orders based on market conditions

Key advantages:
- Profits from spreads without directional risk
- Doesn't tie up capital like taking positions
- Provides liquidity while capturing edge
- Scales efficiently across many markets
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
import numpy as np

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


@dataclass
class LimitOrder:
    """Represents a limit order in the market making strategy."""
    market_id: str
    side: str  # "YES" or "NO"
    price: float  # Price in dollars (0.00-1.00)
    quantity: int
    order_type: str = "limit"
    status: str = "pending"  # pending, placed, filled, cancelled
    order_id: Optional[str] = None
    placed_at: Optional[datetime] = None
    expected_profit: float = 0.0
    
    
@dataclass
class MarketMakingOpportunity:
    """Represents a market making opportunity with calculated spreads."""
    market_id: str
    market_title: str
    current_yes_price: float
    current_no_price: float
    ai_predicted_prob: float
    ai_confidence: float
    
    # Calculated optimal prices
    optimal_yes_bid: float
    optimal_yes_ask: float
    optimal_no_bid: float
    optimal_no_ask: float
    
    # Expected profits
    yes_spread_profit: float
    no_spread_profit: float
    total_expected_profit: float
    
    # Risk metrics
    inventory_risk: float
    volatility_estimate: float
    
    # Order sizing
    optimal_yes_size: int
    optimal_no_size: int


class AdvancedMarketMaker:
    """
    Advanced market making strategy that provides liquidity while capturing edge.
    
    This implements cutting-edge market making techniques:
    - Dynamic spread calculation based on volatility and edge
    - Inventory risk management  
    - Optimal order sizing using Kelly Criterion
    - Cross-market arbitrage opportunities
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.logger = get_trading_logger("market_maker")
        
        # Market making parameters
        self.min_spread = getattr(settings.trading, 'min_spread_for_making', 0.03)  # $0.03 minimum
        self.max_spread = getattr(settings.trading, 'max_bid_ask_spread', 0.10)  # $0.10 maximum
        self.target_inventory = 0.0  # Neutral inventory target
        self.inventory_penalty = getattr(settings.trading, 'max_inventory_risk', 0.01)
        self.volatility_multiplier = 2.0  # Volatility adjustment factor
        
        # Order management
        self.active_orders: Dict[str, List[LimitOrder]] = {}  # market_id -> orders
        self.filled_orders: List[LimitOrder] = []
        self.total_pnl = 0.0
        
        # Performance tracking
        self.markets_traded = 0
        self.total_volume = 0
        self.win_rate = 0.0

    async def analyze_market_making_opportunities(
        self, 
        markets: List[Market]
    ) -> List[MarketMakingOpportunity]:
        """
        Analyze markets for market making opportunities.
        
        Returns list of opportunities ranked by expected profitability.
        """
        opportunities = []
        
        for market in markets:
            try:
                # Get current market data
                market_data = await self.kalshi_client.get_market(market.market_id)
                if not market_data:
                    continue
                    
                # Handle both new and old field formats
                current_yes_price = float(market_data.get('yes_bid_dollars', 0) or market_data.get('yes_price', 0) or 0)
                current_no_price = float(market_data.get('no_bid_dollars', 0) or market_data.get('no_price', 0) or 0)
                
                # Convert cents to dollars if needed
                if current_yes_price > 1.0:
                    current_yes_price = current_yes_price / 100.0
                if current_no_price > 1.0:
                    current_no_price = current_no_price / 100.0
                
                # Skip if prices are extreme (hard to make markets) - relaxed thresholds
                if current_yes_price < 0.02 or current_yes_price > 0.98:
                    continue
                
                # Get AI prediction for edge calculation
                analysis = await self._get_ai_analysis(market)
                if not analysis:
                    continue
                    
                ai_prob = analysis.get('probability', 0.5)
                ai_confidence = analysis.get('confidence', 0.5)
                
                # Apply edge filtering before creating market making opportunity
                from src.utils.edge_filter import EdgeFilter
                
                # Check if either side meets edge requirements
                yes_edge_result = EdgeFilter.calculate_edge(ai_prob, current_yes_price, ai_confidence)
                no_edge_result = EdgeFilter.calculate_edge(1 - ai_prob, current_no_price, ai_confidence)
                
                # Only proceed if at least one side meets edge requirements
                if yes_edge_result.passes_filter or no_edge_result.passes_filter:
                    # Calculate market making opportunity
                    opportunity = await self._calculate_market_making_opportunity(
                        market, current_yes_price, current_no_price, ai_prob, ai_confidence
                    )
                    
                    if opportunity and opportunity.total_expected_profit > 0:
                        opportunities.append(opportunity)
                        self.logger.info(f"✅ MARKET MAKING APPROVED: {market.market_id} - YES edge: {yes_edge_result.edge_percentage:.1%}, NO edge: {no_edge_result.edge_percentage:.1%}")
                else:
                    self.logger.info(f"❌ MARKET MAKING FILTERED: {market.market_id} - Insufficient edge on both sides")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing market {market.market_id}: {e}")
                continue
        
        # Sort by expected profitability
        opportunities.sort(key=lambda x: x.total_expected_profit, reverse=True)
        return opportunities

    async def _calculate_market_making_opportunity(
        self,
        market: Market,
        yes_price: float,
        no_price: float, 
        ai_prob: float,
        ai_confidence: float
    ) -> Optional[MarketMakingOpportunity]:
        """
        Calculate optimal market making prices and expected profits.
        """
        try:
            # Calculate edge (difference between AI prediction and market price)
            yes_edge = ai_prob - yes_price
            no_edge = (1 - ai_prob) - no_price
            
            # Estimate volatility from price and time to expiry
            volatility = self._estimate_volatility(yes_price, market)
            
            # Calculate optimal spreads
            base_spread = max(self.min_spread, min(self.max_spread, volatility * self.volatility_multiplier))
            
            # Adjust spread based on edge and confidence
            edge_adjustment = abs(yes_edge) * ai_confidence
            adjusted_spread = base_spread * (1 + edge_adjustment)
            
            # Calculate optimal bid/ask prices
            if yes_edge > 0:  # AI thinks YES is underpriced
                optimal_yes_bid = yes_price + (adjusted_spread / 2)
                optimal_yes_ask = yes_price + adjusted_spread
                optimal_no_bid = no_price - adjusted_spread
                optimal_no_ask = no_price - (adjusted_spread / 2)
            else:  # AI thinks NO is underpriced  
                optimal_yes_bid = yes_price - adjusted_spread
                optimal_yes_ask = yes_price - (adjusted_spread / 2)
                optimal_no_bid = no_price + (adjusted_spread / 2)
                optimal_no_ask = no_price + adjusted_spread
            
            # Ensure prices are within bounds
            optimal_yes_bid = max(0.01, min(0.99, optimal_yes_bid))
            optimal_yes_ask = max(0.01, min(0.99, optimal_yes_ask))
            optimal_no_bid = max(0.01, min(0.99, optimal_no_bid))
            optimal_no_ask = max(0.01, min(0.99, optimal_no_ask))
            
            # Calculate expected profits
            yes_spread_profit = (optimal_yes_ask - optimal_yes_bid) * ai_confidence
            no_spread_profit = (optimal_no_ask - optimal_no_bid) * ai_confidence
            total_expected_profit = yes_spread_profit + no_spread_profit
            
            # Calculate optimal position sizes using Kelly Criterion
            yes_size, no_size = self._calculate_optimal_sizes(
                yes_edge, no_edge, volatility, ai_confidence
            )
            
            return MarketMakingOpportunity(
                market_id=market.market_id,
                market_title=market.title,
                current_yes_price=yes_price,
                current_no_price=no_price,
                ai_predicted_prob=ai_prob,
                ai_confidence=ai_confidence,
                optimal_yes_bid=optimal_yes_bid,
                optimal_yes_ask=optimal_yes_ask,
                optimal_no_bid=optimal_no_bid,
                optimal_no_ask=optimal_no_ask,
                yes_spread_profit=yes_spread_profit,
                no_spread_profit=no_spread_profit,
                total_expected_profit=total_expected_profit,
                inventory_risk=volatility,
                volatility_estimate=volatility,
                optimal_yes_size=yes_size,
                optimal_no_size=no_size
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity for {market.market_id}: {e}")
            return None

    def _estimate_volatility(self, price: float, market: Market) -> float:
        """
        Estimate market volatility based on price level and time to expiry.
        
        Uses the theoretical volatility of binary options.
        """
        try:
            # Get time to expiry in days
            if hasattr(market, 'expiration_ts') and market.expiration_ts:
                expiry_time = datetime.fromtimestamp(market.expiration_ts)
                time_to_expiry = (expiry_time - datetime.now()).total_seconds() / 86400
                time_to_expiry = max(0.1, time_to_expiry)  # Minimum 0.1 days
            else:
                time_to_expiry = 7.0  # Default 7 days
            
            # Binary option volatility formula: σ = sqrt(p(1-p)/t)
            # Where p is probability and t is time to expiry
            intrinsic_vol = np.sqrt(price * (1 - price) / time_to_expiry)
            
            # Scale to reasonable range
            return max(0.01, min(0.20, intrinsic_vol))
            
        except Exception as e:
            self.logger.error(f"Error estimating volatility: {e}")
            return 0.05  # Default 5%

    def _calculate_optimal_sizes(
        self, 
        yes_edge: float, 
        no_edge: float, 
        volatility: float,
        confidence: float
    ) -> Tuple[int, int]:
        """
        Calculate position sizes from quoting edge.

        NOTE: despite the comments below, this is NOT the Kelly criterion —
        it is a linear edge scaler (win_prob - 0.5) / 0.5 capped at 25%.
        The canonical Kelly kernel lives in src/utils/position_sizing.py;
        if this sizing is ever revisited, use that instead of "fixing" the
        formula here in place.
        """
        try:
            # Available capital for market making
            available_capital = getattr(settings.trading, 'max_position_size', 1000)
            
            # Kelly fraction calculation
            # f* = (bp - q) / b where b=odds, p=win_prob, q=lose_prob
            
            # For YES side
            if yes_edge > 0:
                win_prob = 0.5 + (yes_edge * confidence)
                kelly_yes = max(0, min(0.25, (win_prob - 0.5) / 0.5))  # Cap at 25%
                yes_size = int(available_capital * kelly_yes)
            else:
                yes_size = int(available_capital * 0.05)  # Small size for unfavorable
            
            # For NO side  
            if no_edge > 0:
                win_prob = 0.5 + (no_edge * confidence)
                kelly_no = max(0, min(0.25, (win_prob - 0.5) / 0.5))
                no_size = int(available_capital * kelly_no)
            else:
                no_size = int(available_capital * 0.05)
            
            # Ensure minimum sizes
            yes_size = max(10, yes_size)  # Minimum $10
            no_size = max(10, no_size)
            
            return yes_size, no_size
            
        except Exception as e:
            self.logger.error(f"Error calculating sizes: {e}")
            return 50, 50  # Default sizes

    async def execute_market_making_strategy(
        self, 
        opportunities: List[MarketMakingOpportunity]
    ) -> Dict:
        """
        Execute market making strategy on top opportunities.
        """
        results = {
            'orders_placed': 0,
            'total_exposure': 0.0,
            'expected_profit': 0.0,
            'markets_count': 0
        }
        
        # Limit to top opportunities based on available capital
        max_markets = settings.trading.max_positions
        top_opportunities = opportunities[:max_markets]
        
        for opportunity in top_opportunities:
            try:
                await self._place_market_making_orders(opportunity)
                
                results['orders_placed'] += 2  # YES and NO orders
                results['total_exposure'] += opportunity.optimal_yes_size + opportunity.optimal_no_size
                results['expected_profit'] += opportunity.total_expected_profit
                results['markets_count'] += 1
                
                self.logger.info(
                    f"Market making orders placed for {opportunity.market_title}: "
                    f"Expected profit: ${opportunity.total_expected_profit:.2f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error executing market making for {opportunity.market_id}: {e}")
                continue
        
        return results

    async def _place_market_making_orders(self, opportunity: MarketMakingOpportunity):
        """
        Place the actual limit orders for market making.
        """
        orders = []
        
        # Create YES bid order (we buy YES at lower price)
        yes_bid_order = LimitOrder(
            market_id=opportunity.market_id,
            side="YES",
            price=opportunity.optimal_yes_bid * 100,  # Convert to cents
            quantity=opportunity.optimal_yes_size,
            expected_profit=opportunity.yes_spread_profit
        )
        
        # Create NO bid order (we buy NO at lower price)  
        no_bid_order = LimitOrder(
            market_id=opportunity.market_id,
            side="NO", 
            price=opportunity.optimal_no_bid * 100,
            quantity=opportunity.optimal_no_size,
            expected_profit=opportunity.no_spread_profit
        )
        
        orders.extend([yes_bid_order, no_bid_order])
        
        # Place orders with Kalshi (simulated for now)
        for order in orders:
            await self._place_limit_order(order)
        
        # Track active orders
        if opportunity.market_id not in self.active_orders:
            self.active_orders[opportunity.market_id] = []
        self.active_orders[opportunity.market_id].extend(orders)

    async def _place_limit_order(self, order: LimitOrder):
        """
        Place a limit order with the exchange.
        """
        try:
            # Check if we're in live mode
            live_mode = getattr(settings.trading, 'live_trading_enabled', False)
            
            if live_mode:
                # Place actual limit order with Kalshi
                import uuid
                client_order_id = str(uuid.uuid4())
                
                # Convert side to match Kalshi API
                side = order.side.lower()  # "YES" -> "yes", "NO" -> "no"
                
                # Set price parameters based on side
                order_params = {
                    "ticker": order.market_id,
                    "client_order_id": client_order_id,
                    "side": side,
                    "action": "buy",  # Market making involves buying at our bid prices
                    "count": order.quantity,
                    "type_": "limit"
                }
                
                # Add the appropriate price parameter
                if side == "yes":
                    order_params["yes_price"] = int(order.price)  # Price in cents
                else:
                    order_params["no_price"] = int(order.price)
                
                # Place the order
                response = await self.kalshi_client.place_order(**order_params)
                
                if response and 'order' in response:
                    order.status = "placed"
                    order.placed_at = datetime.now()
                    order.order_id = response['order'].get('order_id', client_order_id)
                    
                    # Convert cents back to dollars for display
                    display_price = order.price / 100 if order.price > 1.0 else order.price
                    self.logger.info(
                        f"✅ LIVE limit order placed: {order.side} {order.quantity} at ${display_price:.2f} "
                        f"for market {order.market_id} (Order ID: {order.order_id})"
                    )
                else:
                    self.logger.error(f"Failed to place live order: {response}")
                    order.status = "failed"
            else:
                # Simulate order placement for paper trading
                order.status = "placed"
                order.placed_at = datetime.now()
                order.order_id = f"sim_{order.market_id}_{order.side}_{int(datetime.now().timestamp())}"
                
                self.logger.info(
                    f"📝 SIMULATED limit order placed: {order.side} {order.quantity} at {order.price:.1f}¢ "
                    f"for market {order.market_id}"
                )
            
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            order.status = "failed"

    async def _get_ai_analysis(self, market: Market) -> Optional[Dict]:
        """
        Get AI analysis for market making edge calculation.
        """
        try:
            # Use existing AI analysis but optimized for market making
            prompt = f"""
            MARKET MAKING ANALYSIS REQUEST
            
            Market: {market.title}
            
            Provide a quick assessment for market making in JSON format:
            {{
                "probability": [0.0-1.0 probability estimate],
                "confidence": [0.0-1.0 confidence level],
                "volatility_factors": "brief description",
                "stability": [0.0-1.0 price stability estimate]
            }}
            
            Focus on: probability estimate and confidence in that estimate.
            """
            
            # Use AI analysis for market making - higher tokens for reasoning models
            response = await self.xai_client.get_completion(
                prompt, 
                max_tokens=3000,  # Higher for reasoning models like grok-4
                temperature=0.1   # Lower for consistency
            )
            
            # Check if AI response is None (API exhausted or failed)
            if response is None:
                self.logger.info(f"AI analysis unavailable for {market.market_id} due to API limits, using conservative defaults")
                return {
                    'probability': 0.5,  # Neutral probability
                    'confidence': 0.2,   # Low confidence
                    'volatility_factors': "API unavailable",
                    'stability': 0.3     # Low stability estimate
                }
            
            # Try to parse JSON response
            try:
                import json
                import re
                
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    
                    if isinstance(parsed_response, dict) and 'probability' in parsed_response:
                        # Validate the response
                        probability = parsed_response.get('probability')
                        confidence = parsed_response.get('confidence')
                        
                        if (isinstance(probability, (int, float)) and 0 <= probability <= 1 and
                            isinstance(confidence, (int, float)) and 0 <= confidence <= 1):
                            return parsed_response
                        else:
                            self.logger.warning(f"Invalid AI response format for {market.market_id}")
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse AI response for {market.market_id}: {e}")
            
            # If AI analysis fails, provide conservative defaults
            self.logger.warning(f"AI analysis failed for {market.market_id}, using conservative defaults")
            return {
                'probability': 0.5,  # Neutral probability
                'confidence': 0.3,   # Low confidence (will result in small positions)
                'volatility_factors': 'AI analysis failed',
                'stability': 0.5
            }
                
        except Exception as e:
            self.logger.error(f"Error getting AI analysis: {e}")
            # Return conservative defaults instead of None
            return {
                'probability': 0.5,
                'confidence': 0.3,
                'volatility_factors': 'Error in analysis',
                'stability': 0.5
            }

    async def monitor_and_update_orders(self):
        """
        Monitor active orders and update/cancel as needed.
        """
        for market_id, orders in self.active_orders.items():
            try:
                for order in orders:
                    if order.status == "placed":
                        # Check if order is still competitive
                        should_update = await self._should_update_order(order)
                        if should_update:
                            await self._update_order(order)
                            
            except Exception as e:
                self.logger.error(f"Error monitoring orders for {market_id}: {e}")

    async def _should_update_order(self, order: LimitOrder) -> bool:
        """
        Determine if an order should be updated based on market conditions.
        """
        try:
            # Get current market data
            market_data = await self.kalshi_client.get_market(order.market_id)
            if not market_data:
                return False
            
            current_yes_price = market_data.get('yes_price', 0) / 100
            order_price = order.price / 100
            
            # Update if market has moved significantly
            price_diff = abs(current_yes_price - order_price)
            return price_diff > 0.05  # 5 cent threshold
            
        except Exception as e:
            self.logger.error(f"Error checking order update: {e}")
            return False

    async def _update_order(self, order: LimitOrder):
        """
        Update an existing order with new price/quantity.
        """
        try:
            # Cancel old order and place new one
            order.status = "cancelled"
            
            # Recalculate optimal price
            # This would need to recalculate the market making opportunity
            
            self.logger.info(f"Updated order {order.order_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating order: {e}")

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary of market making strategy.
        """
        try:
            active_count = sum(len(orders) for orders in self.active_orders.values())
            filled_count = len(self.filled_orders)
            
            return {
                'total_pnl': self.total_pnl,
                'active_orders': active_count,
                'filled_orders': filled_count,
                'markets_traded': self.markets_traded,
                'win_rate': self.win_rate,
                'total_volume': self.total_volume
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}


async def run_market_making_strategy(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient, 
    xai_client: XAIClient
) -> Dict:
    """
    Main entry point for market making strategy.
    """
    logger = get_trading_logger("market_making_main")
    
    try:
        # Initialize market maker
        market_maker = AdvancedMarketMaker(db_manager, kalshi_client, xai_client)
        
        # Get eligible markets (remove time restrictions!)
        markets = await db_manager.get_eligible_markets(
            volume_min=30000,  # Higher volume for market making (needs more liquidity)
            max_days_to_expiry=365  # Accept any timeline
        )
        
        if not markets:
            logger.warning("No eligible markets found for market making")
            return {'error': 'No markets available'}
        
        logger.info(f"Analyzing {len(markets)} markets for market making opportunities")
        
        # Analyze opportunities
        opportunities = await market_maker.analyze_market_making_opportunities(markets)
        
        if not opportunities:
            logger.warning("No profitable market making opportunities found")
            return {'opportunities': 0}
        
        logger.info(f"Found {len(opportunities)} profitable market making opportunities")
        
        # Execute strategy
        results = await market_maker.execute_market_making_strategy(opportunities)
        
        # Add performance summary
        results['performance'] = market_maker.get_performance_summary()
        
        logger.info(f"Market making strategy completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in market making strategy: {e}")
        return {'error': str(e)} 