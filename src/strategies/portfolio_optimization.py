"""
Advanced Portfolio Optimization - Kelly Criterion Extensions

This module implements cutting-edge portfolio optimization techniques:
1. Kelly Criterion Extension (KCE) for prediction markets
2. Risk Parity allocation 
3. Dynamic position sizing based on market conditions
4. Cross-correlation analysis between markets
5. Multi-objective optimization (return vs risk vs drawdown)

Based on latest research:
- Kelly Criterion Extension for dynamic markets (Kim, 2024)
- Fractional Kelly strategies for risk management
- Portfolio optimization for prediction markets

Key innovations:
- Uses Kelly Criterion for fund managers (not direct asset investment)
- Adapts to both favorable and unfavorable market conditions  
- Implements dynamic rebalancing based on market state
- Risk-adjusted allocation rather than equal capital weights
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from scipy.optimize import minimize, minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from src.utils.database import DatabaseManager, Market, Position
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.utils.market_prices import get_market_prices


@dataclass
class MarketOpportunity:
    """Represents a trading opportunity with all required metrics for optimization."""
    market_id: str
    market_title: str
    predicted_probability: float
    market_probability: float
    confidence: float
    edge: float  # predicted_prob - market_prob
    volatility: float
    expected_return: float
    max_loss: float
    time_to_expiry: float  # in days
    correlation_score: float  # correlation with portfolio
    
    # Kelly metrics
    kelly_fraction: float
    fractional_kelly: float  # Conservative Kelly
    risk_adjusted_fraction: float
    
    # Portfolio metrics  
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_contribution: float


@dataclass
class PortfolioAllocation:
    """Optimal portfolio allocation across opportunities."""
    allocations: Dict[str, float]  # market_id -> allocation fraction
    total_capital_used: float
    expected_portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    max_portfolio_drawdown: float
    diversification_ratio: float
    
    # Risk metrics
    portfolio_var_95: float  # Value at Risk
    portfolio_cvar_95: float  # Conditional Value at Risk
    
    # Kelly metrics
    aggregate_kelly_fraction: float
    portfolio_growth_rate: float


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization using Kelly Criterion Extensions and modern portfolio theory.
    
    This implements the latest research in prediction market portfolio optimization:
    - Kelly Criterion Extension (KCE) for dynamic market conditions
    - Risk parity allocation for balanced risk exposure  
    - Multi-factor optimization considering correlation, volatility, and drawdown
    - Dynamic rebalancing based on market regime detection
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
        self.logger = get_trading_logger("portfolio_optimizer")
        
        # Portfolio parameters
        self.total_capital = getattr(settings.trading, 'total_capital', 10000)
        self.max_position_fraction = getattr(settings.trading, 'max_single_position', 0.25)
        self.min_position_size = getattr(settings.trading, 'min_position_size', 5)
        self.kelly_fraction_multiplier = getattr(settings.trading, 'kelly_fraction', 0.25)  # Fractional Kelly
        
        # Risk management
        self.max_portfolio_volatility = getattr(settings.trading, 'max_volatility', 0.20)
        self.max_correlation = getattr(settings.trading, 'max_correlation', 0.70)
        self.target_sharpe_ratio = getattr(settings.trading, 'target_sharpe', 2.0)
        
        # Market regime detection
        self.market_state = "normal"  # normal, volatile, trending
        self.regime_lookback = 30  # days
        
        # Performance tracking
        self.historical_allocations = []
        self.realized_returns = []
        self.portfolio_metrics = {}

    async def optimize_portfolio(
        self, 
        opportunities: List[MarketOpportunity]
    ) -> PortfolioAllocation:
        """
        Main portfolio optimization using advanced Kelly Criterion and risk parity.
        
        Process:
        1. Calculate Kelly fractions for each opportunity
        2. Apply risk adjustments and correlations
        3. Optimize using multi-objective function
        4. Apply risk constraints and position limits
        5. Return optimal allocation
        """
        self.logger.info(f"Optimizing portfolio across {len(opportunities)} opportunities")
        
        if not opportunities:
            return self._empty_allocation()
        
        # Limit opportunities to prevent optimization complexity
        max_opportunities = getattr(settings.trading, 'max_opportunities_per_batch', 50)
        if len(opportunities) > max_opportunities:
            # Sort by confidence * expected_return and take top N
            opportunities = sorted(
                opportunities, 
                key=lambda x: x.confidence * x.expected_return, 
                reverse=True
            )[:max_opportunities]
            self.logger.info(f"Limited to top {max_opportunities} opportunities for optimization")
        
        try:
            # Step 1: Enhance opportunities with portfolio metrics
            enhanced_opportunities = await self._enhance_opportunities_with_metrics(opportunities)
            
            # Step 2: Detect market regime and adjust parameters
            await self._detect_market_regime()
            
            # Step 3: Calculate Kelly fractions
            kelly_fractions = self._calculate_kelly_fractions(enhanced_opportunities)
            
            # Step 3.5: Update opportunities with Kelly fractions
            for opp in enhanced_opportunities:
                kelly_val = kelly_fractions.get(opp.market_id, 0.0)
                # Update the opportunity object in place
                opp.kelly_fraction = kelly_val
                opp.fractional_kelly = kelly_val * 0.5  # Conservative Kelly
                opp.risk_adjusted_fraction = opp.fractional_kelly
            
            # Step 4: Apply correlation adjustments
            correlation_matrix = await self._estimate_correlation_matrix(enhanced_opportunities)
            adjusted_fractions = self._apply_correlation_adjustments(kelly_fractions, correlation_matrix)
            
            # Step 5: Multi-objective optimization
            optimal_allocation = self._multi_objective_optimization(
                enhanced_opportunities, adjusted_fractions, correlation_matrix
            )
            
            # Step 6: Apply risk constraints
            final_allocation = self._apply_risk_constraints(optimal_allocation, enhanced_opportunities)
            
            # Step 7: Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                final_allocation, enhanced_opportunities, correlation_matrix
            )
            
            result = PortfolioAllocation(
                allocations=final_allocation,
                **portfolio_metrics
            )
            
            self.logger.info(
                f"Portfolio optimization complete: "
                f"Capital used: ${result.total_capital_used:.0f}, "
                f"Expected return: {result.expected_portfolio_return:.1%}, "
                f"Sharpe ratio: {result.portfolio_sharpe:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return self._empty_allocation()

    async def _enhance_opportunities_with_metrics(
        self, 
        opportunities: List[MarketOpportunity]
    ) -> List[MarketOpportunity]:
        """
        Enhance opportunities with additional portfolio metrics.
        """
        enhanced = []
        
        for opp in opportunities:
            try:
                # Calculate advanced metrics
                sharpe_ratio = self._calculate_sharpe_ratio(opp)
                sortino_ratio = self._calculate_sortino_ratio(opp)
                max_dd_contribution = self._estimate_max_drawdown_contribution(opp)
                
                # Create enhanced opportunity
                enhanced_opp = MarketOpportunity(
                    market_id=opp.market_id,
                    market_title=opp.market_title,
                    predicted_probability=opp.predicted_probability,
                    market_probability=opp.market_probability,
                    confidence=opp.confidence,
                    edge=opp.edge,
                    volatility=opp.volatility,
                    expected_return=opp.expected_return,
                    max_loss=opp.max_loss,
                    time_to_expiry=opp.time_to_expiry,
                    correlation_score=0.0,  # Will be calculated later
                    kelly_fraction=0.0,
                    fractional_kelly=0.0,
                    risk_adjusted_fraction=0.0,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    max_drawdown_contribution=max_dd_contribution
                )
                
                enhanced.append(enhanced_opp)
                
            except Exception as e:
                self.logger.error(f"Error enhancing opportunity {opp.market_id}: {e}")
                continue
        
        return enhanced

    def _calculate_kelly_fractions(self, opportunities: List[MarketOpportunity]) -> Dict[str, float]:
        """
        Calculate Kelly fractions using the Kelly Criterion Extension (KCE).
        
        Implements the advanced Kelly Criterion that adapts to market conditions:
        - Standard Kelly for high-confidence, low-correlation opportunities
        - Fractional Kelly for moderate confidence
        - Kelly Criterion Extension for dynamic market environments
        """
        kelly_fractions = {}
        
        for opp in opportunities:
            try:
                # Calculate basic Kelly fraction: f* = (bp - q) / b
                # Where p = win probability, q = lose probability, b = odds
                
                win_prob = opp.predicted_probability
                lose_prob = 1 - win_prob
                
                # Calculate odds from market price
                if opp.market_probability > 0 and opp.market_probability < 1:
                    odds = (1 - opp.market_probability) / opp.market_probability
                else:
                    odds = 1.0
                
                # Standard Kelly calculation
                if opp.edge > 0 and win_prob > 0.5:
                    kelly_standard = (odds * win_prob - lose_prob) / odds
                else:
                    kelly_standard = 0.0
                
                # Apply Kelly Criterion Extension for dynamic markets
                # Adjust for market regime and time decay
                regime_multiplier = self._get_regime_multiplier()
                time_decay_factor = max(0.1, min(1.0, opp.time_to_expiry / 30))  # Decay over 30 days
                
                kelly_kce = kelly_standard * regime_multiplier * time_decay_factor
                
                # Apply confidence adjustment
                confidence_adjusted = kelly_kce * opp.confidence
                
                # Apply fractional Kelly (typically 25-50% of full Kelly)
                fractional_kelly = confidence_adjusted * self.kelly_fraction_multiplier
                
                # Ensure reasonable bounds
                final_kelly = max(0.0, min(self.max_position_fraction, fractional_kelly))
                
                # Store calculations
                opp.kelly_fraction = kelly_standard
                opp.fractional_kelly = fractional_kelly
                opp.risk_adjusted_fraction = final_kelly
                
                kelly_fractions[opp.market_id] = final_kelly
                
                self.logger.debug(
                    f"Kelly calculation for {opp.market_id}: "
                    f"Standard: {kelly_standard:.3f}, "
                    f"KCE: {kelly_kce:.3f}, "
                    f"Final: {final_kelly:.3f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error calculating Kelly for {opp.market_id}: {e}")
                kelly_fractions[opp.market_id] = 0.0
        
        return kelly_fractions

    async def _estimate_correlation_matrix(
        self, 
        opportunities: List[MarketOpportunity]
    ) -> np.ndarray:
        """
        Estimate correlation matrix between markets for portfolio optimization.
        
        Uses multiple approaches:
        1. Category-based correlations
        2. Time-based correlations (similar expiry dates)
        3. Content similarity analysis
        4. Historical price correlations where available
        """
        n = len(opportunities)
        if n <= 1:
            return np.eye(1)
            
        correlation_matrix = np.eye(n)
        
        try:
            for i, opp1 in enumerate(opportunities):
                for j, opp2 in enumerate(opportunities):
                    if i != j:
                        correlation = await self._estimate_pairwise_correlation(opp1, opp2)
                        correlation_matrix[i, j] = correlation
                        
                        # Update correlation score in opportunity
                        opp1.correlation_score = max(opp1.correlation_score, abs(correlation))
            
            # Ensure matrix is positive semidefinite
            correlation_matrix = self._ensure_positive_semidefinite(correlation_matrix)
            
        except Exception as e:
            self.logger.error(f"Error estimating correlation matrix: {e}")
            correlation_matrix = np.eye(n)  # Fall back to identity matrix
        
        return correlation_matrix

    async def _estimate_pairwise_correlation(
        self, 
        opp1: MarketOpportunity, 
        opp2: MarketOpportunity
    ) -> float:
        """
        Estimate correlation between two market opportunities.
        """
        try:
            correlation = 0.0
            
            # 1. Category-based correlation (if markets are in same category)
            category_corr = await self._get_category_correlation(opp1.market_id, opp2.market_id)
            
            # 2. Time-based correlation (similar expiry times)
            time_diff = abs(opp1.time_to_expiry - opp2.time_to_expiry)
            time_corr = max(0, 1 - (time_diff / 30))  # Decay over 30 days
            
            # 3. Content similarity (use AI to assess)
            content_corr = await self._get_content_similarity(opp1, opp2)
            
            # 4. Volatility similarity
            vol_diff = abs(opp1.volatility - opp2.volatility)
            vol_corr = max(0, 1 - vol_diff)
            
            # Combine correlations with weights
            correlation = (
                0.4 * category_corr +
                0.2 * time_corr +
                0.3 * content_corr +
                0.1 * vol_corr
            )
            
            # Cap maximum correlation
            correlation = min(self.max_correlation, correlation)
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Error estimating pairwise correlation: {e}")
            return 0.1  # Small default correlation

    def _apply_correlation_adjustments(
        self, 
        kelly_fractions: Dict[str, float], 
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Adjust Kelly fractions based on correlations to reduce portfolio risk.
        
        High correlations reduce effective diversification, so we scale down allocations
        to highly correlated markets.
        """
        adjusted_fractions = kelly_fractions.copy()
        market_ids = list(kelly_fractions.keys())
        
        try:
            for i, market_id in enumerate(market_ids):
                # Calculate average correlation with other markets
                avg_correlation = np.mean([
                    abs(correlation_matrix[i, j]) 
                    for j in range(len(market_ids)) 
                    if i != j
                ])
                
                # Adjust fraction based on correlation
                # Higher correlation -> lower allocation
                correlation_penalty = 1 - (avg_correlation * 0.5)  # Max 50% penalty
                
                adjusted_fractions[market_id] *= correlation_penalty
                
                self.logger.debug(
                    f"Correlation adjustment for {market_id}: "
                    f"Avg corr: {avg_correlation:.3f}, "
                    f"Penalty: {correlation_penalty:.3f}"
                )
        
        except Exception as e:
            self.logger.error(f"Error applying correlation adjustments: {e}")
        
        return adjusted_fractions

    def _multi_objective_optimization(
        self,
        opportunities: List[MarketOpportunity],
        kelly_fractions: Dict[str, float],
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Multi-objective optimization balancing return, risk, and diversification.
        
        Objective function combines:
        1. Expected return (maximize)
        2. Portfolio volatility (minimize)  
        3. Maximum drawdown (minimize)
        4. Diversification ratio (maximize)
        5. Sharpe ratio (maximize)
        """
        try:
            n = len(opportunities)
            if n == 0:
                return {}
            
            # Initial allocation from Kelly fractions
            initial_weights = np.array([kelly_fractions.get(opp.market_id, 0) for opp in opportunities])
            
            # Normalize to sum to 1 or less
            if initial_weights.sum() > 1.0:
                initial_weights = initial_weights / initial_weights.sum()
            
            # If initial weights are all zero or very small, use simple fallback
            if initial_weights.sum() < 0.001:
                self.logger.warning("Initial weights too small, using simple allocation fallback")
                return self._simple_allocation_fallback(opportunities)
            
            # Expected returns vector
            expected_returns = np.array([opp.expected_return for opp in opportunities])
            
            # Volatilities vector
            volatilities = np.array([opp.volatility for opp in opportunities])
            
            # Create covariance matrix from correlations and volatilities
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            def objective_function(weights):
                """Multi-objective function to minimize."""
                try:
                    # Ensure weights are valid
                    if np.any(weights < 0) or np.sum(weights) > 1.0001:  # Small tolerance
                        return 1e6
                    
                    # Portfolio return
                    portfolio_return = np.dot(weights, expected_returns)
                    
                    # Portfolio volatility
                    portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                    
                    # Diversification ratio (higher is better)
                    weighted_vol = np.dot(weights, volatilities)
                    diversification_ratio = weighted_vol / (portfolio_vol + 1e-8)
                    
                    # Sharpe ratio approximation
                    sharpe = (portfolio_return / (portfolio_vol + 1e-8))
                    
                    # Maximum drawdown approximation
                    max_dd = self._estimate_portfolio_max_drawdown(weights, opportunities)
                    
                    # Multi-objective score (higher is better, so we minimize negative)
                    # FIXED: Much less conservative - focus on returns, light risk penalty
                    score = -(
                        10.0 * portfolio_return +     # Heavy weight on returns
                        2.0 * sharpe +                # Moderate Sharpe weight
                        0.5 * diversification_ratio - # Light diversification bonus
                        0.2 * portfolio_vol -         # Light volatility penalty (was 1.0)
                        0.1 * max_dd                  # Very light drawdown penalty (was 1.0)
                    )
                    
                    return score
                    
                except Exception as e:
                    return 1e6  # Large penalty for errors
            
            # Try optimization with more relaxed constraints
            try:
                # Constraints - force meaningful allocation
                constraints = [
                    {'type': 'ineq', 'fun': lambda w: 0.80 - np.sum(w)},  # Sum <= 80% (reasonable max)
                    {'type': 'ineq', 'fun': lambda w: np.sum(w) - 0.05},  # Sum >= 5% (force minimum allocation)
                ]
                
                # Bounds for each weight
                bounds = [(0, min(self.max_position_fraction, 0.3)) for _ in range(n)]  # Cap at 30%
                
                # Optimize
                result = minimize(
                    objective_function,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-6}  # More relaxed tolerance
                )
                
                if result.success and np.sum(result.x) > 0.001:
                    optimal_weights = result.x
                    self.logger.info(f"Optimization successful with sum: {np.sum(optimal_weights):.3f}")
                else:
                    self.logger.warning(f"Optimization failed: {result.message}")
                    raise Exception("Optimization failed")
                    
            except Exception as e:
                self.logger.warning(f"Scipy optimization failed: {e}, using simple fallback")
                return self._simple_allocation_fallback(opportunities)
            
            # Convert back to dictionary
            optimal_allocation = {
                opp.market_id: float(optimal_weights[i]) 
                for i, opp in enumerate(opportunities)
                if optimal_weights[i] > 0.001  # Filter tiny allocations
            }
            
            return optimal_allocation
            
        except Exception as e:
            self.logger.error(f"Error in multi-objective optimization: {e}")
            # Fall back to simple allocation
            return self._simple_allocation_fallback(opportunities)

    def _simple_allocation_fallback(self, opportunities: List[MarketOpportunity]) -> Dict[str, float]:
        """
        Simple fallback allocation when optimization fails.
        Allocates based on expected return * confidence, subject to position limits.
        """
        try:
            if not opportunities:
                return {}
            
            # Calculate scores for each opportunity
            scores = []
            for opp in opportunities:
                # Score based on expected return, confidence, and edge
                score = opp.expected_return * opp.confidence * max(0, abs(opp.edge))
                scores.append((opp.market_id, score))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Simple allocation: give more to higher scoring opportunities
            allocation = {}
            total_allocation = 0.0
            max_positions = min(5, len(opportunities))  # Limit to top 5 positions
            
            for i, (market_id, score) in enumerate(scores[:max_positions]):
                if score <= 0:
                    continue
                    
                # Allocate more to higher ranked opportunities
                weight = max(0.05, 0.25 - (i * 0.03))  # Start at 25%, decrease by 3% each rank
                
                # Don't exceed total capital
                if total_allocation + weight <= 0.8:  # Max 80% allocation
                    allocation[market_id] = weight
                    total_allocation += weight
                else:
                    remaining = 0.8 - total_allocation
                    if remaining > 0.01:  # Only if meaningful allocation left
                        allocation[market_id] = remaining
                    break
            
            self.logger.info(f"Simple fallback allocation: {len(allocation)} positions, {total_allocation:.1%} capital")
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error in simple allocation fallback: {e}")
            return {}

    def _apply_risk_constraints(
        self, 
        allocation: Dict[str, float], 
        opportunities: List[MarketOpportunity]
    ) -> Dict[str, float]:
        """
        Apply final risk constraints and position sizing limits.
        """
        try:
            constrained_allocation = {}
            total_allocation = 0.0
            
            for market_id, fraction in allocation.items():
                # Find the opportunity
                opp = next((o for o in opportunities if o.market_id == market_id), None)
                if not opp:
                    continue
                
                # Apply minimum position size
                dollar_allocation = fraction * self.total_capital
                if dollar_allocation < self.min_position_size:
                    continue
                
                # Apply maximum position fraction
                final_fraction = min(fraction, self.max_position_fraction)
                
                # Check if this would exceed total capital
                if total_allocation + final_fraction <= 1.0:
                    constrained_allocation[market_id] = final_fraction
                    total_allocation += final_fraction
                else:
                    # Scale down to fit remaining capital
                    remaining = 1.0 - total_allocation
                    if remaining > 0.001:
                        constrained_allocation[market_id] = remaining
                        total_allocation = 1.0
                    break
            
            self.logger.info(f"Risk constraints applied. Total allocation: {total_allocation:.3f}")
            self.logger.info(f"Final constrained allocations: {constrained_allocation}")
            
            return constrained_allocation
            
        except Exception as e:
            self.logger.error(f"Error applying risk constraints: {e}")
            return allocation

    def _calculate_portfolio_metrics(
        self,
        allocation: Dict[str, float],
        opportunities: List[MarketOpportunity],
        correlation_matrix: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive portfolio metrics.
        """
        try:
            if not allocation:
                return self._empty_portfolio_metrics()
            
            # Get vectors for allocated opportunities
            allocated_opps = [opp for opp in opportunities if opp.market_id in allocation]
            weights = np.array([allocation[opp.market_id] for opp in allocated_opps])
            returns = np.array([opp.expected_return for opp in allocated_opps])
            volatilities = np.array([opp.volatility for opp in allocated_opps])
            
            # Portfolio return
            portfolio_return = np.dot(weights, returns)
            
            # Portfolio volatility
            n = len(allocated_opps)
            if n > 1:
                allocated_corr_matrix = correlation_matrix[:n, :n]
                covariance_matrix = np.outer(volatilities, volatilities) * allocated_corr_matrix
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            else:
                portfolio_vol = volatilities[0] * weights[0] if len(volatilities) > 0 else 0.0
            
            # Sharpe ratio
            portfolio_sharpe = portfolio_return / (portfolio_vol + 1e-8)
            
            # Diversification ratio
            weighted_vol = np.dot(weights, volatilities)
            diversification_ratio = weighted_vol / (portfolio_vol + 1e-8)
            
            # Capital usage
            total_capital_used = sum(allocation.values()) * self.total_capital
            
            # Risk metrics (simplified)
            portfolio_var_95 = portfolio_vol * 1.645  # 95% VaR
            portfolio_cvar_95 = portfolio_var_95 * 1.2  # Approximate CVaR
            
            # Kelly metrics
            aggregate_kelly = sum(
                opp.kelly_fraction * allocation[opp.market_id] 
                for opp in allocated_opps
            )
            
            portfolio_growth_rate = portfolio_return - (portfolio_vol ** 2) / 2  # Geometric return approximation
            
            # Maximum drawdown
            max_portfolio_drawdown = self._estimate_portfolio_max_drawdown(weights, allocated_opps)
            
            return {
                'total_capital_used': total_capital_used,
                'expected_portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'portfolio_sharpe': portfolio_sharpe,
                'max_portfolio_drawdown': max_portfolio_drawdown,
                'diversification_ratio': diversification_ratio,
                'portfolio_var_95': portfolio_var_95,
                'portfolio_cvar_95': portfolio_cvar_95,
                'aggregate_kelly_fraction': aggregate_kelly,
                'portfolio_growth_rate': portfolio_growth_rate
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return self._empty_portfolio_metrics()

    # Helper methods
    
    async def _detect_market_regime(self):
        """Detect current market regime for Kelly adjustments."""
        # Simplified regime detection - in production would use more sophisticated methods
        self.market_state = "normal"  # Default
    
    def _get_regime_multiplier(self) -> float:
        """Get Kelly multiplier based on market regime."""
        regime_multipliers = {
            "normal": 1.0,
            "volatile": 0.7,  # Reduce Kelly in volatile markets
            "trending": 1.2   # Increase Kelly in trending markets
        }
        return regime_multipliers.get(self.market_state, 1.0)
    
    def _calculate_sharpe_ratio(self, opp: MarketOpportunity) -> float:
        """Calculate Sharpe ratio for opportunity."""
        return opp.expected_return / (opp.volatility + 1e-8)
    
    def _calculate_sortino_ratio(self, opp: MarketOpportunity) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        # Simplified - assumes normal distribution
        downside_vol = opp.volatility * 0.7  # Approximation
        return opp.expected_return / (downside_vol + 1e-8)
    
    def _estimate_max_drawdown_contribution(self, opp: MarketOpportunity) -> float:
        """Estimate maximum drawdown contribution."""
        # Simplified approximation
        return opp.volatility * 2.0
    
    async def _get_category_correlation(self, market_id1: str, market_id2: str) -> float:
        """Get correlation based on market categories."""
        # Simplified - would query market metadata
        return 0.1  # Default low correlation
    
    async def _get_content_similarity(self, opp1: MarketOpportunity, opp2: MarketOpportunity) -> float:
        """Get content similarity using AI."""
        # Simplified - would use embeddings or AI analysis
        return 0.1  # Default low similarity
    
    def _ensure_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive semidefinite."""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 0.001)  # Ensure positive
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except:
            return np.eye(matrix.shape[0])
    
    def _estimate_portfolio_max_drawdown(self, weights: np.ndarray, opportunities: List[MarketOpportunity]) -> float:
        """Estimate portfolio maximum drawdown."""
        # Simplified approximation
        individual_mdd = np.array([opp.max_drawdown_contribution for opp in opportunities])
        return np.dot(weights, individual_mdd) * 0.8  # Diversification benefit
    
    def _empty_allocation(self) -> PortfolioAllocation:
        """Return empty portfolio allocation."""
        return PortfolioAllocation(
            allocations={},
            **self._empty_portfolio_metrics()
        )
    
    def _empty_portfolio_metrics(self) -> Dict:
        """Return empty portfolio metrics."""
        return {
            'total_capital_used': 0.0,
            'expected_portfolio_return': 0.0,
            'portfolio_volatility': 0.0,
            'portfolio_sharpe': 0.0,
            'max_portfolio_drawdown': 0.0,
            'diversification_ratio': 1.0,
            'portfolio_var_95': 0.0,
            'portfolio_cvar_95': 0.0,
            'aggregate_kelly_fraction': 0.0,
            'portfolio_growth_rate': 0.0
        }


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
            market_prob = market_info.get('yes_price', 50) / 100
            
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
                    max_drawdown_contribution=0.0
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
                'volume': getattr(opportunity, 'volume', 1000),
                'min_volume': 1000,
                'time_to_expiry_days': opportunity.time_to_expiry,
                'max_time_to_expiry': 365
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
                "SELECT COUNT(*) FROM positions WHERE market_id = ?",
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
        
        # Calculate proper entry price (what we expect to pay)
        if side == "YES":
            entry_price = opportunity.market_probability  # Price for YES shares
            shares = max(1, int(position_size / entry_price))  # Minimum 1 contract
        else:
            entry_price = 1 - opportunity.market_probability  # Price for NO shares  
            shares = max(1, int(position_size / entry_price))  # Minimum 1 contract
        
        # Verify we can afford at least 1 contract
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
                
            logger.info(f"✅ Market validation passed for {opportunity.market_id} - Status: {market_status}, proceeding with trade!")
            
        except Exception as e:
            logger.error(f"⏭️ Skipping {opportunity.market_id} - Market validation failed: {e}")
            import traceback
            logger.error(f"Full error: {traceback.format_exc()}")
            return
        
        # Execute immediately
        position_id = await db_manager.add_position(position)
        if position_id:
            # Set the position ID so execute_position can update the database
            position.id = position_id
            
            # Execute the trade - respect the global trading mode setting
            live_mode = getattr(settings.trading, 'live_trading_enabled', False)
            success = await execute_position(position, live_mode, db_manager, kalshi_client)
            if success:
                logger.info(f"✅ IMMEDIATE TRADE EXECUTED: {opportunity.market_id} - ${position_size:.0f} position")
            else:
                logger.error(f"❌ IMMEDIATE TRADE FAILED: {opportunity.market_id}")
        
    except Exception as e:
        logger.error(f"Error in immediate trade evaluation for {opportunity.market_id}: {e}")

def _calculate_simple_kelly(opportunity: MarketOpportunity) -> float:
    """Calculate simple Kelly fraction for immediate trading."""
    try:
        # Simple Kelly: (bp - q) / b
        # where b = odds, p = win probability, q = lose probability
        if opportunity.edge > 0:  # Betting YES
            p = opportunity.predicted_probability
            q = 1 - p
            b = (1 - opportunity.market_probability) / opportunity.market_probability
        else:  # Betting NO
            p = 1 - opportunity.predicted_probability  
            q = opportunity.predicted_probability
            b = opportunity.market_probability / (1 - opportunity.market_probability)
        
        kelly = (b * p - q) / b
        return max(0, min(kelly, 0.2))  # Cap at 20%
        
    except:
        return 0.05  # Default 5% allocation


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


async def run_portfolio_optimization(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: XAIClient
) -> PortfolioAllocation:
    """
    Main entry point for portfolio optimization.
    """
    logger = get_trading_logger("portfolio_optimization_main")
    
    try:
        # Initialize optimizer
        optimizer = AdvancedPortfolioOptimizer(db_manager, kalshi_client, xai_client)
        
        # Get markets
        markets = await db_manager.get_eligible_markets(
            volume_min=20000,  # Balanced volume for actual trading opportunities
            max_days_to_expiry=365  # Accept any timeline with dynamic exits
        )
        if not markets:
            logger.warning("No eligible markets for portfolio optimization")
            return optimizer._empty_allocation()
        
        # Convert to opportunities (no immediate trading in batch mode)
        opportunities = await create_market_opportunities_from_markets(
            markets, xai_client, kalshi_client, None, 0
        )
        
        if not opportunities:
            logger.warning("No valid opportunities for portfolio optimization")
            return optimizer._empty_allocation()
        
        logger.info(f"Running portfolio optimization on {len(opportunities)} opportunities")
        
        # Optimize portfolio
        allocation = await optimizer.optimize_portfolio(opportunities)
        
        logger.info(
            f"Portfolio optimization complete: "
            f"{len(allocation.allocations)} positions, "
            f"${allocation.total_capital_used:.0f} allocated"
        )
        
        return allocation
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        return AdvancedPortfolioOptimizer(db_manager, kalshi_client, xai_client)._empty_allocation() 