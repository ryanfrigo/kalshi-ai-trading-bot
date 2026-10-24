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


def test_position_cap_comes_from_settings_not_a_literal():
    """cap_count's own default agrees with settings today; the tool must not depend on that."""
    from src.agent.toolbelt import configured_position_pct, cap_count
    assert configured_position_pct() == settings.trading.max_position_size_pct / 100.0
    # 10% of a $10,000 equity at 50c is 2,000 contracts
    n, reason = cap_count(9999, 50, 1_000_000, 10_000_000, configured_position_pct())
    assert n == 2000 and "position cap" in reason
