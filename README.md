# 🤖 Kalshi AI Trading Bot

A sophisticated, multi-agent AI-powered trading system for Kalshi prediction markets. This system uses advanced LLM reasoning, portfolio optimization, and real-time market analysis to make intelligent trading decisions.

## ⚠️ **IMPORTANT DISCLAIMER**

**This is experimental software for educational and research purposes only.**

- **Trading involves substantial risk of loss**
- **Only trade with capital you can afford to lose**
- **Past performance does not guarantee future results**
- **This software is not financial advice**
- **Use at your own risk**

The authors are not responsible for any financial losses incurred through the use of this software.

## 🚀 Features

### Core Trading System
- **Multi-Agent AI Decision Engine**: Uses Forecaster, Critic, and Trader agents for comprehensive market analysis
- **Real-time Market Scanning**: Continuously monitors Kalshi markets for opportunities
- **Portfolio Optimization**: Kelly Criterion and risk parity allocation strategies
- **Live Trading**: Direct integration with Kalshi API for real-time order execution
- **Performance Analytics**: Comprehensive tracking and analysis of trading performance

### Advanced Features
- **Beast Mode Trading**: Aggressive multi-strategy trading system
- **Market Making**: Automated spread trading and liquidity provision
- **Dynamic Exit Strategies**: Intelligent position management and risk control
- **Cost Optimization**: Smart AI usage to minimize analysis costs
- **Real-time Dashboard**: Web-based monitoring and control interface

### AI Integration
- **Grok-4 Integration**: Primary AI model for market analysis and decision making
- **Multi-Model Support**: Fallback to alternative AI models when needed
- **Confidence Calibration**: AI confidence scoring and validation
- **News Analysis**: Real-time news integration for market context

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Market Data   │    │   AI Analysis   │    │   Trade Exec    │
│   Ingestion     │───▶│   Engine        │───▶│   & Tracking    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │    │   Portfolio     │    │   Performance   │
│   Storage       │    │   Optimization  │    │   Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Kalshi API account
- xAI API key (for Grok-4 access)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/kalshi-ai-trading-bot.git
   cd kalshi-ai-trading-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Generate your Kalshi API Key and xAI API Key** 

   Save your Kalshi private key as "kalshi_private_key" (no file extension) in the root directory of this project

5. **Configure environment variables**

   Create a `.env` file in the root directory:
   ```bash
   KALSHI_API_KEY=your_kalshi_api_key
   XAI_API_KEY=your_xai_api_key
   ```

6. **Initialize the database**
   ```bash
   python src/utils/database.py
   ```
   or, if the above comman failed, try
   ```bash
   python -m src.utils.database # Run database.py as a module
   ```

## 🚀 Quick Start

### Basic Trading Bot
```bash
# Start the main trading system
python beast_mode_bot.py
```

### Dashboard Interface
```bash
# Launch the web dashboard
python launch_dashboard.py
```

### Performance Analysis
```bash
# Run performance analysis
python performance_analysis.py
```

## 📈 Trading Strategies

### 1. Multi-Agent Decision Making
The system uses three AI agents:
- **Forecaster**: Estimates true probability using market data and news
- **Critic**: Identifies potential flaws or missing context
- **Trader**: Makes final BUY/SKIP decisions with position sizing

### 2. Portfolio Optimization
- **Kelly Criterion**: Optimal position sizing based on edge and odds
- **Risk Parity**: Balanced risk allocation across positions
- **Dynamic Rebalancing**: Automatic portfolio adjustments

### 3. Market Making
- **Spread Trading**: Profiting from bid-ask spreads
- **Liquidity Provision**: Providing market liquidity
- **Inventory Management**: Risk-controlled position management

## ⚙️ Configuration

### Trading Parameters
Key configuration options in `src/config/settings.py`:

```python
# Position sizing
max_position_size_pct: float = 5.0
max_daily_loss_pct: float = 15.0
max_positions: int = 15

# Market filtering
min_volume: float = 200.0
max_time_to_expiry_days: int = 30
min_confidence_to_trade: float = 0.50

# AI settings
primary_model: str = "grok-4"
ai_temperature: float = 0
ai_max_tokens: int = 8000
```

### Risk Management
- Maximum daily loss limits
- Position size constraints
- Correlation and volatility limits
- Dynamic exit strategies

## 📊 Performance Monitoring

### Real-time Metrics
- Total P&L and win rate
- Sharpe ratio and drawdown analysis
- AI confidence calibration
- Cost per trade analysis

### Dashboard Features
- Live trading activity
- Portfolio overview
- Performance charts
- Risk metrics
- AI decision logs

## 🔧 Development

### Project Structure
```
src/
├── clients/          # API clients (Kalshi, xAI)
├── config/           # Configuration management
├── jobs/             # Trading jobs and tasks
├── strategies/       # Trading strategies
└── utils/            # Utilities and helpers
```

### Testing
```bash
# Run all tests
python run_tests.py

# Run specific test
pytest tests/test_decide.py
```

### Code Quality
```bash
# Format code
black src/
isort src/

# Type checking
mypy src/
```

## ⚠️ Important Notes

### Risk Disclaimer
- This is experimental software for educational purposes
- Trading involves substantial risk of loss
- Only trade with capital you can afford to lose
- Past performance does not guarantee future results

### API Limits
- Respect Kalshi API rate limits
- Monitor xAI API usage and costs
- Implement proper error handling

### Security
- Never commit API keys or private keys
- Use environment variables for sensitive data
- Regularly rotate API credentials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kalshi for providing the prediction market platform
- xAI for the Grok-4 AI model
- The open-source community for various libraries and tools

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the test files for usage examples

---

**Disclaimer**: This software is for educational and research purposes. Trading involves risk, and you should only trade with capital you can afford to lose. The authors are not responsible for any financial losses incurred through the use of this software.
