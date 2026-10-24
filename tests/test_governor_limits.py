"""The governor must enforce the *configured* limits on every path.

Constructing RiskGovernor() directly falls back to 10% daily / 15% drawdown, while settings
say 5% / 20% (Ryan-approved). That meant the agent's own trade path halted at twice the agreed
daily loss. `governor_from_settings` is the single source of truth; this pins it.
"""
from src.risk.risk_governor import RiskGovernor, governor_from_settings
from src.config import settings as settings_module
settings = settings_module.settings


def test_factory_uses_the_configured_limits():
    gov = governor_from_settings(kalshi_client=None)
    assert gov.max_daily_loss_pct == settings.trading.max_daily_loss_pct == 5.0
    assert gov.max_drawdown_pct == settings_module.max_drawdown * 100.0 == 20.0


def test_factory_differs_from_the_bare_defaults_it_replaced():
    bare = RiskGovernor(kalshi_client=None)
    # the trap this fix closes: the defaults were looser on daily loss and tighter on drawdown
    assert bare.max_daily_loss_pct == 10.0 and bare.max_drawdown_pct == 15.0
    assert bare.max_daily_loss_pct != settings.trading.max_daily_loss_pct
