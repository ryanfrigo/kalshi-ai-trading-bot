# Trading System Dashboard 📊

A comprehensive Streamlit-based dashboard for monitoring and analyzing all aspects of your automated trading system.

## 🚀 Features

### 📈 Overview Page
- **Real-time metrics**: Portfolio balance, total trades, P&L, active positions
- **Strategy performance summary**: Visual charts comparing all strategies
- **Recent activity**: Latest positions and trades
- **Quick insights**: At-a-glance system health

### 🎯 Strategy Performance
- **Detailed analytics**: P&L, win rates, trade counts by strategy
- **Individual strategy drill-down**: Comprehensive metrics per strategy
- **Risk-return analysis**: Bubble charts showing performance vs. risk
- **Capital deployment**: How funds are allocated across strategies

### 🤖 LLM Analysis & Review
- **Query tracking**: Every LLM request and response logged
- **Usage statistics**: Token consumption, costs, query frequency
- **Strategy breakdown**: LLM usage by trading strategy
- **Response analysis**: Review AI predictions and confidence levels
- **Cost monitoring**: Track AI expenses over time

### 💼 Positions & Trades
- **Active positions**: Real-time position tracking with filters
- **Trade history**: Completed trades with performance metrics
- **Position analytics**: Value distribution and risk analysis
- **Strategy attribution**: See which strategy created each position

### ⚠️ Risk Management
- **Portfolio utilization**: How much capital is deployed
- **Position sizing**: Average and maximum position sizes
- **Risk alerts**: Automated warnings for high-risk situations
- **Strategy risk**: Risk breakdown by trading approach

### 🔧 System Health
- **Connection status**: Kalshi API, database, LLM integration
- **Activity timeline**: Recent system events and queries
- **Configuration**: Current system settings
- **Recommendations**: Automated system optimization suggestions

## 🛠️ Setup & Installation

### Prerequisites
```bash
# Install dashboard requirements
pip install -e ".[dashboard]"

# Or install individually
pip install streamlit pandas plotly
```

### Quick Start
```bash
# Launch the dashboard
python cli.py dashboard
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📊 Dashboard Sections

### Navigation
Use the sidebar to navigate between different sections:
- **📈 Overview**: System summary and key metrics
- **🎯 Strategy Performance**: Detailed strategy analysis
- **🤖 LLM Analysis**: AI query review and optimization
- **💼 Positions & Trades**: Position tracking and history
- **⚠️ Risk Management**: Risk monitoring and alerts
- **🔧 System Health**: System status and diagnostics

### Data Refresh
- **Auto-refresh**: Data updates automatically (1-5 minute cache)
- **Real-time**: Most metrics update in real-time
- **Manual refresh**: Use browser refresh for instant updates

## 🤖 LLM Query Tracking

The dashboard automatically tracks and displays:

### What's Logged
- **Every LLM query**: Complete prompt and response
- **Strategy attribution**: Which strategy made the query
- **Query type**: Movement prediction, market analysis, etc.
- **Market context**: Which market was being analyzed
- **Costs**: Token usage and monetary cost
- **Extracted data**: Confidence levels, decisions, etc.

### Benefits
- **Review AI reasoning**: See exactly what the LLM is thinking
- **Optimize prompts**: Identify successful vs. failed queries
- **Cost management**: Track and control AI expenses
- **Strategy improvement**: Understand which strategies use AI effectively
- **Debug issues**: Identify problematic queries or responses

### Example Analysis
```
🤖 Quick Flip Scalping | Movement Prediction | 14:23:15

Strategy: quick_flip_scalping
Type: movement_prediction  
Market: KXELECTION-2024-TRUMP

Prompt: "QUICK SCALP ANALYSIS for Will Trump win 2024..."

Response: "TARGET_PRICE: 7
CONFIDENCE: 0.75
REASON: Recent polling surge suggests upward momentum..."

Cost: $0.0023
```

## 📈 Strategy Performance Analytics

### Key Metrics Tracked
- **Total P&L**: Profit/loss by strategy
- **Win Rate**: Percentage of winning trades
- **Average Trade Size**: Capital per position
- **Risk Metrics**: Volatility, maximum drawdown
- **Capital Efficiency**: Returns per dollar deployed

### Performance Comparison
- **Side-by-side**: Compare all strategies at once
- **Risk-return plots**: Visualize performance vs. risk
- **Capital allocation**: See resource distribution
- **Trend analysis**: Performance over time

### Strategy Insights
- **Best performers**: Identify top-earning strategies
- **Optimization opportunities**: Spot underperforming areas
- **Resource allocation**: Guide capital distribution decisions
- **Risk assessment**: Monitor strategy-specific risks

## ⚠️ Risk Management Features

### Automated Alerts
- **High utilization**: >90% of capital deployed
- **Large positions**: Single position >20% of portfolio
- **Missing stops**: Positions without stop-loss protection
- **Concentration risk**: Too many positions in similar markets

### Risk Monitoring
- **Real-time tracking**: Continuous risk assessment
- **Portfolio level**: Overall portfolio risk metrics
- **Strategy level**: Risk by trading approach
- **Position level**: Individual trade risk analysis

### Visual Indicators
- **Color coding**: Green/yellow/red risk levels
- **Progress bars**: Utilization and exposure metrics
- **Charts**: Risk distribution and trends
- **Alerts**: Pop-up warnings for critical issues

## 🔧 Customization

### Configuration
The dashboard automatically adapts to your trading system configuration. No manual setup required for:
- Database connections
- Strategy detection
- Risk parameters
- Performance calculations

### Extending the Dashboard
To add new features:

1. **New metrics**: Add to `load_performance_data()`
2. **New visualizations**: Create charts in relevant sections
3. **Custom alerts**: Modify risk management rules
4. **Additional pages**: Add new sections to the sidebar

## 📱 Mobile-Friendly

The dashboard is responsive and works on:
- **Desktop browsers**: Full functionality
- **Tablets**: Optimized layout
- **Mobile phones**: Essential metrics available

## 🔒 Security

- **Local only**: Dashboard runs locally on your machine
- **No external data**: All data stays in your local database
- **No cloud services**: Pure local analysis
- **API keys**: Never exposed in the dashboard

## 🚀 Performance

- **Efficient caching**: Data cached for optimal performance
- **Lazy loading**: Charts load as needed
- **Minimal overhead**: Dashboard doesn't impact trading system
- **Fast updates**: Sub-second refresh for most metrics

## 💡 Tips & Best Practices

### Optimization
- **Regular review**: Check dashboard daily for insights
- **Strategy comparison**: Use performance analytics to guide allocation
- **LLM analysis**: Review AI queries weekly for optimization opportunities
- **Risk monitoring**: Set up alerts for key risk thresholds

### Troubleshooting
- **Missing data**: Ensure trading system is running and database is accessible
- **Slow loading**: Check database size and consider archiving old data
- **Connection issues**: Verify Kalshi API credentials and connectivity
- **LLM data**: New queries will appear after strategies run with updated clients

## 🆘 Support

### Common Issues
1. **"No data available"**: System needs to run and collect data first
2. **"LLM queries empty"**: Update XAI client instantiation with db_manager
3. **"Dashboard won't start"**: Install requirements with `pip install -e ".[dashboard]"`
4. **"Charts not loading"**: Refresh browser or check console for errors

### Getting Help
- Check the console output for detailed error messages
- Ensure all dependencies are installed
- Verify database file exists and is accessible
- Review system logs for trading system issues

---

## 🎯 Quick Start Summary

1. **Install**: `pip install -e ".[dashboard]"`
2. **Launch**: `python cli.py dashboard`
3. **Browse**: Navigate to different sections via sidebar
4. **Analyze**: Review strategy performance and LLM usage
5. **Optimize**: Use insights to improve trading strategies

The dashboard provides everything you need to monitor, analyze, and optimize your automated trading system! 🚀 