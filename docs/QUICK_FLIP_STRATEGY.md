# Quick Flip Scalping Strategy 🎯

A high-frequency scalping strategy that targets rapid profits on low-priced Kalshi contracts.

## Strategy Overview

The Quick Flip strategy implements automated scalping by:

1. **Finding Low-Priced Opportunities**: Identifies contracts priced 1¢-20¢ 
2. **AI Movement Analysis**: Uses AI to predict short-term price movements
3. **Immediate Execution**: Buys contracts and instantly places sell orders
4. **Quick Exits**: Holds positions for maximum 30 minutes

## How It Works

### Entry Criteria
- Contract prices between 1¢-20¢ (configurable)
- AI confidence ≥ 60% for upward movement
- Sufficient liquidity for entry and exit
- Market must be actively tradeable

### Execution Flow
```
1. Buy YES/NO at low price (e.g., 3¢)
2. Immediately place sell limit order (e.g., 6¢)
3. Monitor for fill or time expiry
4. Cut losses with market order if held >30 min
```

### Example Scenarios

| Entry Price | Exit Price | Quantity | Profit | Return |
|-------------|------------|----------|--------|--------|
| 1¢          | 2¢         | 50       | $0.50  | 100%   |
| 5¢          | 10¢        | 20       | $1.00  | 100%   |
| 3¢          | 7¢         | 30       | $1.20  | 133%   |

## Configuration

### Default Settings
```python
QuickFlipConfig(
    min_entry_price=1,           # Minimum entry price (cents)
    max_entry_price=20,          # Maximum entry price (cents)
    min_profit_margin=1.0,       # 100% minimum return
    max_position_size=100,       # Max contracts per position
    max_concurrent_positions=50, # Max simultaneous positions
    capital_per_trade=50.0,      # Max capital per trade
    confidence_threshold=0.6,    # 60% AI confidence required
    max_hold_minutes=30          # Maximum hold time
)
```

### Adjustable Parameters
- **Entry Range**: Customize minimum/maximum prices to target
- **Profit Margins**: Set minimum required returns (default 100%)
- **Position Sizing**: Control risk per trade and total exposure
- **AI Confidence**: Filter trades by prediction confidence
- **Time Limits**: Maximum hold time before cutting losses

## Integration with Trading System

### Capital Allocation
- **20%** of total portfolio allocated to quick flip strategy
- Runs parallel with market making (30%) and directional trading (40%)
- Independent risk management and position tracking

### Risk Management
- ✅ **Position Limits**: Maximum contracts and capital per trade
- ✅ **Time Stops**: Automatic exit after 30 minutes
- ✅ **Confidence Filtering**: Only high-confidence AI predictions
- ✅ **Diversification**: Spread across multiple markets
- ✅ **Capital Controls**: Limited allocation prevents overexposure

## Usage

### Testing the Strategy
```bash
# Test opportunity identification (no trades)
python test_quick_flip_strategy.py

# Run paper trading test
python -c "
from test_quick_flip_strategy import test_quick_flip_full_strategy
import asyncio
asyncio.run(test_quick_flip_full_strategy())
"
```

### Running in Production
The strategy is automatically included in the unified trading system:

```python
from src.strategies.unified_trading_system import run_unified_trading_system

# Quick flip runs automatically as part of unified system
results = await run_unified_trading_system(db_manager, kalshi_client, xai_client)
```

### Manual Execution
```python
from src.strategies.quick_flip_scalping import run_quick_flip_strategy, QuickFlipConfig

config = QuickFlipConfig(
    min_entry_price=1,
    max_entry_price=15,
    capital_per_trade=25.0
)

results = await run_quick_flip_strategy(
    db_manager=db_manager,
    kalshi_client=kalshi_client, 
    xai_client=xai_client,
    available_capital=500.0,
    config=config
)
```

## Strategy Advantages

### High Return Potential
- A 1¢ contract that moves to 2¢ is a 100% gross return before fees. This strategy has no committed backtest.
- **Quick turnover** - capital not tied up long-term
- **Many opportunities** - low-priced contracts are common

### Low Risk Profile  
- **Small position sizes** limit downside per trade
- **Time-based stops** prevent large losses
- **Diversification** across many markets reduces risk

### Scalability
- **Automated execution** - no manual intervention needed
- **Parallel processing** - handles many positions simultaneously  
- **AI-driven** - scales with market opportunities

## Performance Monitoring

### Key Metrics
- **Fill Rate**: Percentage of sell orders that execute
- **Average Hold Time**: How long positions are held
- **Profit Factor**: Ratio of winning to losing trades
- **Capital Efficiency**: Return per dollar allocated

### Optimization Opportunities
- **AI Model Tuning**: Improve movement prediction accuracy
- **Parameter Adjustment**: Optimize entry/exit criteria
- **Market Selection**: Focus on highest-probability markets
- **Timing Analysis**: Identify best execution windows

## Risk Considerations

### Market Risks
- **Low Liquidity**: Difficulty exiting positions quickly
- **Price Gaps**: Markets can move against positions rapidly
- **Time Decay**: Some contracts lose value approaching expiry

### Strategy Risks
- **Over-Trading**: Too many simultaneous positions
- **AI Accuracy**: Predictions may be incorrect
- **Execution Risk**: Orders may not fill as expected

### Mitigation Strategies
- Conservative position sizing (≤$50 per trade)
- Strict time limits (≤30 minutes)
- High confidence thresholds (≥60%)
- Diversification across markets
- Continuous monitoring and adjustment

## Future Enhancements

### Planned Improvements
- **Dynamic Pricing**: Adjust exit prices based on market movement
- **Volume Analysis**: Factor in liquidity when sizing positions
- **News Integration**: Use breaking news for timing entries
- **Machine Learning**: Improve AI prediction models

### Advanced Features
- **Pair Trading**: Simultaneously trade YES/NO sides
- **Market Making Integration**: Combine with spread strategies
- **Cross-Market Arbitrage**: Exploit price differences
- **Options-Style Strategies**: Complex multi-leg positions

---

## Quick Start

1. **Test the strategy**: `python test_quick_flip_strategy.py`
2. **Review configuration** in `QuickFlipConfig`
3. **Start unified trading** - quick flip runs automatically
4. **Monitor performance** through logs and analytics

The quick flip strategy adds a new dimension to your trading system, targeting rapid profits on price movements while maintaining strict risk controls. 🚀 