"""
Multi-agent ensemble decision engine for Kalshi trading.

This package implements a multi-model AI ensemble that combines
forecasting, news analysis, bull/bear research, risk management,
and final trade execution into a structured decision pipeline.
"""

from experimental.agents.base_agent import BaseAgent
from experimental.agents.forecaster_agent import ForecasterAgent
from experimental.agents.news_analyst_agent import NewsAnalystAgent
from experimental.agents.bull_researcher import BullResearcher
from experimental.agents.bear_researcher import BearResearcher
from experimental.agents.risk_manager_agent import RiskManagerAgent
from experimental.agents.trader_agent import TraderAgent
from experimental.agents.ensemble import EnsembleRunner
from experimental.agents.debate import DebateRunner

__all__ = [
    "BaseAgent",
    "ForecasterAgent",
    "NewsAnalystAgent",
    "BullResearcher",
    "BearResearcher",
    "RiskManagerAgent",
    "TraderAgent",
    "EnsembleRunner",
    "DebateRunner",
]
