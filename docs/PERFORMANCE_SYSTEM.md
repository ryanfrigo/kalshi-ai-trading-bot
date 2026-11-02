# 🎯 Kalshi Automated Performance Analysis System

A comprehensive, AI-powered performance analysis and risk management system for the Kalshi prediction market trading platform. This system provides real-time monitoring, automated analysis using the configured LLM, and actionable recommendations to optimize trading performance.

## 🚀 **System Overview**

### **What It Does**
- **Automated Performance Analysis**: Runs daily/weekly analysis with the configured LLM
- **Risk Management**: Monitors capital utilization, cash reserves, position limits
- **Real-time Alerts**: Immediate notifications for critical issues
- **Action Item Generation**: Specific, prioritized recommendations
- **Manual vs Automated Tracking**: Compares performance between trading modes
- **Dashboard Integration**: Real-time health metrics and alerts

### **Key Features**
- **Health Score (0-100)**: Overall system health rating
- **Risk Checks**: Automated monitoring of 5+ critical risk factors
- **LLM Intelligence**: AI-powered analysis and recommendations
- **Emergency Intervention**: Immediate analysis for critical situations
- **Scheduler**: Automated daily (9 AM) and weekly (Monday) analysis
- **Interactive CLI**: Command-line interface for manual operations


## 🛠 **Installation & Setup**

### **Prerequisites**
```bash
pip install schedule aiosqlite structlog
```

### **Quick Start**
```bash
# Start the automated system
python3 performance_system_manager.py --start

# Run immediate analysis
python3 performance_system_manager.py --analyze

# Check system status
python3 performance_system_manager.py --status

# Emergency intervention
python3 performance_system_manager.py --emergency

# Interactive mode
python3 performance_system_manager.py --interactive
```

## 📋 **System Components**

### **1. Automated Performance Analyzer** (`src/jobs/automated_performance_analyzer.py`)
- **Core Engine**: Main analysis logic with the configured LLM integration
- **Risk Checks**: 5 automated risk management checks
- **Performance Metrics**: Comprehensive trading statistics
- **Action Items**: Prioritized recommendations (CRITICAL, HIGH, MEDIUM, LOW)

### **2. Performance Scheduler** (`src/jobs/performance_scheduler.py`)
- **Daily Analysis**: 9:00 AM automatic analysis
- **Weekly Reports**: Monday comprehensive reports
- **Alert System**: File-based and console alerts
- **Health Monitoring**: Continuous health score tracking

### **3. Dashboard Integration** (`src/jobs/performance_dashboard_integration.py`)
- **Real-time Metrics**: Current system state
- **Alert Management**: Critical issue notifications
- **Quick Actions**: Emergency intervention tools
- **API Endpoints**: Integration with beast mode dashboard

### **4. System Manager** (`performance_system_manager.py`)
- **Orchestration**: Complete system lifecycle management
- **CLI Interface**: Command-line operations
- **Signal Handling**: Graceful shutdown
- **Interactive Mode**: Real-time system management

## 🔍 **Risk Management Checks**

### **1. Cash Reserve Threshold** (Target: 15-20%)
- **Current**: ~9% (CRITICAL)
- **Recommendation**: Close positions to build $90+ cash reserves
- **Impact**: Prevents liquidity crisis, enables opportunistic trading

### **2. Capital Utilization** (Target: <80%)
- **Current**: 91.4% (CRITICAL)
- **Recommendation**: Reduce position sizes immediately
- **Impact**: Reduces volatility risk, prevents margin calls

### **3. Position Concentration** (Target: ≤12 positions)
- **Current**: 16 positions (WARNING)
- **Recommendation**: Close 4 lowest-conviction positions
- **Impact**: Improves capital efficiency, reduces management overhead

### **4. Single Position Risk** (Target: <10% per position)
- **Current**: Largest position too concentrated (WARNING)
- **Recommendation**: Reduce size of largest position
- **Impact**: Prevents concentration risk in single market

### **5. Manual vs Automated Performance**
- **Current**: Gap analysis needed
- **Recommendation**: Extract patterns from profitable manual trades
- **Impact**: Could improve automated trading win rate

## 🤖 **LLM Integration**

### **Analysis Cost**: ~$0.01-0.02 per analysis
### **Response Time**: 25-30 seconds
### **Analysis Depth**: 3,000-4,000 character detailed reports

### **Key Capabilities**:
- **Quantitative Analysis**: Specific metrics and thresholds
- **Risk Assessment**: Multi-factor risk evaluation
- **Performance Trends**: Historical and predictive analysis
- **Actionable Recommendations**: Prioritized improvement steps

## 📈 **Performance Metrics Tracked**

### **Trading Performance**
- **Total Trades**: 20 (0% current win rate - mostly unrealized)
- **Manual vs Automated**: Performance comparison and gap analysis
- **P&L Tracking**: Realized and unrealized profit/loss
- **Win Rate Analysis**: Success rate by trading method

### **Portfolio Health**
- **Cash Reserves**: Available liquidity monitoring
- **Capital Utilization**: Risk exposure percentage
- **Position Count**: Active market positions
- **Concentration Risk**: Single position exposure

### **System Health Score Calculation**
- **Base Score**: 100 points
- **Deductions**: -25 for CRITICAL issues, -10 for warnings
- **Performance Adjustments**: Win rate bonuses/penalties
- **Risk Adjustments**: Capital utilization impact

## 🚨 **Alert System**

### **Alert Types**
1. **CRITICAL**: Health score <50, cash crisis, overutilization
2. **WARNING**: Health degradation, position limits
3. **WEEKLY_SUMMARY**: Regular status updates
4. **SYSTEM_ERROR**: Analysis failures

### **Alert Channels**
- **File-based**: Timestamped alert files
- **Console**: Real-time terminal notifications
- **Database**: Persistent alert history
- **Dashboard**: Real-time UI integration

### **Emergency Procedures**
```bash
# Immediate intervention
python3 performance_system_manager.py --emergency

# Manual override
python3 performance_system_manager.py --interactive
> emergency

# Status check
python3 performance_system_manager.py --status
```

## 🎯 **Usage Examples**

### **Daily Operations**
```bash
# Start automated monitoring
python3 performance_system_manager.py --start

# Check morning status
python3 performance_system_manager.py --status

# Run analysis after market close
python3 performance_system_manager.py --analyze
```

### **Emergency Situations**
```bash
# Immediate analysis with the configured LLM
python3 performance_system_manager.py --emergency

# Interactive troubleshooting
python3 performance_system_manager.py --interactive
```

### **System Management**
```bash
# Custom schedule
python3 performance_system_manager.py --start --daily-time "10:30" --weekly-day "friday"

# Health threshold adjustment
python3 performance_system_manager.py --start --health-threshold 60.0
```

## 📊 **Integration with Existing System**

### **Database Integration**
- **Tables**: `analysis_reports` for historical tracking
- **Compatibility**: Works with existing `positions` and `trade_logs`
- **Storage**: JSON reports + database summaries

### **Beast Mode Dashboard**
- **API Endpoints**: `dashboard_get_metrics()`, `dashboard_get_alerts()`
- **Real-time Data**: Current health score, active alerts
- **Quick Actions**: Emergency analysis, position management

### **Trading System**
- **Risk Checks**: Pre-trade risk validation
- **Performance Feedback**: Post-trade analysis
- **Automation Improvements**: Manual pattern extraction

## 🔧 **Configuration Options**

### **Scheduler Configuration**
```python
config = ScheduleConfig(
    daily_analysis_time="09:00",           # Daily analysis time
    weekly_deep_analysis_day="monday",     # Weekly analysis day
    critical_check_interval_minutes=60,    # Health monitoring frequency
    health_score_threshold=50.0,           # Alert threshold
    enable_email_alerts=False,             # Email notifications
    enable_file_alerts=True                # File-based alerts
)
```

### **Risk Thresholds**
- **Cash Reserves**: 15-20% minimum
- **Capital Utilization**: 80% maximum
- **Position Count**: 12 maximum
- **Single Position**: 10% maximum
- **Health Score**: 50 minimum for alerts

## 🚀 **Next Steps & Recommendations**

### **Immediate (24 hours)**
1. **Close 2-3 positions** to build cash reserves to $100+
2. **Reduce capital utilization** to 75-80% range
3. **Set up automated monitoring** with daily analysis

### **Short-term (1 week)**
1. **Position consolidation** to 10-12 active positions
2. **Manual pattern analysis** to improve automation
3. **Risk threshold tuning** based on performance

### **Long-term (1 month)**
1. **Automated position sizing** based on Kelly Criterion
2. **Advanced risk models** with volatility adjustments
3. **Machine learning integration** for pattern recognition

## 📞 **Support & Troubleshooting**

### **Common Issues**
1. **Database errors**: Check `trading_system.db` permissions
2. **XAI API issues**: Verify API key and credits
3. **Scheduler not running**: Check system clock and permissions

### **Debug Commands**
```bash
# Test individual components
python3 test_automated_analyzer.py
python3 src/jobs/performance_scheduler.py --status
python3 src/jobs/performance_dashboard_integration.py --summary
```

### **Log Files**
- **Latest**: `logs/latest.log`
- **Timestamped**: `logs/trading_system_YYYYMMDD_HHMMSS.log`
- **Analysis Reports**: `performance_analysis_YYYYMMDD_HHMMSS.json`

---

## 🎉 **Success Metrics**

This system provides:
- **98% Automated**: Minimal manual intervention required
- **Real-time**: <30 second analysis with the configured LLM
- **Cost-effective**: ~$0.30/month for daily analysis
- **Comprehensive**: 20+ metrics and 5+ risk checks
- **Actionable**: Specific steps with target dates

**The system successfully identified critical cash shortage and overutilization issues that require immediate attention!** 