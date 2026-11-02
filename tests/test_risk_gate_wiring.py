"""place_order -> RiskGovernor wiring, and governor edge cases a halt could slip through.

HTTP is mocked; POSTs are counted. The kill switch is exercised end to end with a real
RiskGovernor on tmp_path state, so no real files or exchange are touched.
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.clients.kalshi_client import KalshiAPIError, KalshiClient
from src.config.settings import settings
from src.risk import risk_governor as rg
from src.risk.risk_governor import GovernorDecision, RiskGovernor, roll_state

TICKER = "KXTEST-26OCT30"


def _client(equity_cents=100_000, positions=None):
    """Client whose balance feeds the real governor; POSTs are recorded."""
    client = KalshiClient.__new__(KalshiClient)
    client.posts = []

    async def fake(method, path, params=None, json_data=None, **kw):
        if method == "POST":
            client.posts.append(json_data)
            return {"order_id": "abc"}
        if path.endswith("/portfolio/balance"):
            return {"balance": equity_cents, "portfolio_value": 0}
        return {"market_positions": positions or []}

    client._make_authenticated_request = fake
    return client


def _place(client, action="buy", side="yes", count=1, price=50):
    return asyncio.run(client.place_order(
        ticker=TICKER, client_order_id="cid", side=side, action=action, count=count,
        type_="limit", yes_price=price if side == "yes" else None,
        no_price=price if side == "no" else None))


def _real_governor(tmp_path, **kw):
    """Patch RiskGovernor in place_order's import so state/flag live in tmp_path."""
    real = RiskGovernor

    def factory(**kwargs):
        kwargs.update(state_path=str(tmp_path / "state.json"),
                      halt_flag_path=str(tmp_path / "HALT"))
        return real(**kwargs)

    return patch("src.risk.risk_governor.RiskGovernor", side_effect=factory)


# ---- kill switch, end to end -------------------------------------------------
def test_manual_halt_flag_blocks_buy_and_sell_with_zero_posts(tmp_path):
    (tmp_path / "HALT").write_text("operator stop")
    c = _client(positions=[{"ticker": TICKER, "position_fp": "10"}])
    with _real_governor(tmp_path):
        for action in ("buy", "sell"):
            with pytest.raises(KalshiAPIError, match="MANUAL_HALT: operator stop"):
                _place(c, action=action)
    assert c.posts == []


def test_clearing_manual_halt_lets_order_through(tmp_path):
    (tmp_path / "HALT").write_text("x")
    c = _client()
    with _real_governor(tmp_path):
        with pytest.raises(KalshiAPIError):
            _place(c)
        (tmp_path / "HALT").unlink()
        _place(c)
    assert len(c.posts) == 1


def test_corrupt_state_file_blocks_orders_and_is_left_untouched(tmp_path):
    (tmp_path / "state.json").write_text("{not json")
    c = _client()
    with _real_governor(tmp_path):
        with pytest.raises(KalshiAPIError, match="STATE_CORRUPT"):
            _place(c)
    assert c.posts == []
    assert (tmp_path / "state.json").read_text() == "{not json"


def test_real_drawdown_halt_blocks_buy(tmp_path):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (tmp_path / "state.json").write_text(
        f'{{"date": "{today}", "sod_equity_cents": 100000, "peak_equity_cents": 100000}}')
    c = _client(equity_cents=70_000)  # -30%: past both daily-loss and drawdown limits
    with _real_governor(tmp_path):
        with pytest.raises(KalshiAPIError, match="RISK_HALT"):
            _place(c, count=1, price=10)
    assert c.posts == []


# ---- limits actually passed to the governor ----------------------------------
def test_governor_built_with_configured_daily_loss_and_module_drawdown(tmp_path):
    seen = {}
    ok = GovernorDecision(False, [], 100_000, 100_000, 100_000)

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)

        async def check(self):
            return ok

    with patch("src.risk.risk_governor.RiskGovernor", Spy), \
         patch.object(settings.trading, "max_daily_loss_pct", 3.5), \
         patch("src.config.settings.max_drawdown", 0.12):
        _place(_client())
    assert seen["max_daily_loss_pct"] == 3.5
    assert seen["max_drawdown_pct"] == pytest.approx(12.0)


# ---- position cap boundary ---------------------------------------------------
def _ok(equity):
    return GovernorDecision(False, [], equity, equity, equity)


def test_buy_cost_exactly_at_cap_passes_and_one_cent_over_blocks():
    cap_pct = settings.trading.max_position_size_pct
    equity = 100_000  # $1,000 -> cap = $1000 * pct/100
    cap_dollars = equity / 100 * cap_pct / 100
    count = 10
    price_at_cap = int(round(cap_dollars / count * 100))  # cents per contract
    c = _client()
    from unittest.mock import AsyncMock
    with patch("src.risk.risk_governor.RiskGovernor.check", new=AsyncMock(return_value=_ok(equity))):
        _place(c, count=count, price=price_at_cap)
        assert len(c.posts) == 1
        with pytest.raises(KalshiAPIError, match="POSITION_CAP"):
            _place(c, count=count, price=price_at_cap + 1)
    assert len(c.posts) == 1


def test_position_cap_does_not_apply_to_sells():
    from unittest.mock import AsyncMock
    c = _client()
    with patch("src.risk.risk_governor.RiskGovernor.check", new=AsyncMock(return_value=_ok(1_000))):
        _place(c, action="sell", count=500, price=90)  # huge vs $10 equity; not a buy
    assert len(c.posts) == 1


# ---- governor unit edges -----------------------------------------------------
def test_to_dict_reports_pnl_and_percentages():
    d = rg.evaluate_risk(8_000, 10_000, 12_000, 10.0, 15.0).to_dict()
    assert d["halted"] is True
    assert d["daily_pnl_cents"] == -2_000
    assert d["daily_loss_pct"] == 20.0
    assert d["drawdown_pct"] == pytest.approx(33.33)


def test_percentages_are_zero_without_baseline():
    d = GovernorDecision(False, [], 5_000, 0, 0)
    assert d.daily_loss_pct == 0.0 and d.drawdown_pct == 0.0


def test_check_reads_equity_as_cash_plus_portfolio(tmp_path):
    class C:
        async def get_balance(self):
            return {"balance": 6_000, "portfolio_value": 4_000}

    gov = RiskGovernor(C(), str(tmp_path / "s.json"), str(tmp_path / "H"))
    assert asyncio.run(gov.current_equity_cents()) == 10_000
    assert asyncio.run(gov.check()).current_equity_cents == 10_000


def test_check_without_client_or_equity_raises(tmp_path):
    gov = RiskGovernor(None, str(tmp_path / "s.json"), str(tmp_path / "H"))
    with pytest.raises(RuntimeError):
        asyncio.run(gov.check())


def test_manual_halt_reason_listed_first_alongside_limit_breach(tmp_path):
    t = datetime(2026, 11, 2, tzinfo=timezone.utc)
    gov = RiskGovernor(None, str(tmp_path / "s.json"), str(tmp_path / "H"), clock=lambda: t)
    gov.save_state({"date": "2026-11-02", "sod_equity_cents": 10_000, "peak_equity_cents": 10_000})
    gov.trip_manual_halt("by hand")
    d = asyncio.run(gov.check(current_equity_cents=5_000))
    assert d.halted and d.reasons[0] == "MANUAL_HALT: by hand"
    assert any(r.startswith("DAILY_LOSS:") for r in d.reasons)


def test_empty_halt_flag_file_still_halts_with_default_reason(tmp_path):
    gov = RiskGovernor(None, str(tmp_path / "s.json"), str(tmp_path / "H"))
    (tmp_path / "H").write_text("")
    d = asyncio.run(gov.check(current_equity_cents=1_000))
    assert d.halted and d.reasons == ["MANUAL_HALT: manual halt flag present"]


def test_peak_never_decreases_within_a_day_even_if_equity_dips_then_recovers(tmp_path):
    t = datetime(2026, 11, 2, tzinfo=timezone.utc)
    gov = RiskGovernor(None, str(tmp_path / "s.json"), str(tmp_path / "H"), clock=lambda: t)
    for eq in (10_000, 12_000, 9_000):
        asyncio.run(gov.check(current_equity_cents=eq))
    assert gov.load_state()["peak_equity_cents"] == 12_000
    assert gov.load_state()["sod_equity_cents"] == 10_000


@pytest.mark.xfail(strict=True, reason=(
    "BUG: src/risk/risk_governor.py roll_state/check freeze start-of-day at the first "
    "reading of the day even when that reading is 0 (e.g. get_balance() returned {} -> "
    "current_equity_cents()==0). sod=0 then disables the DAILY_LOSS check "
    "(`if sod > 0`) until the date changes; the first zero reading also fails open."))
def test_zero_start_of_day_is_replaced_by_first_real_equity_same_day():
    state = roll_state({"date": "2026-11-02", "sod_equity_cents": 0, "peak_equity_cents": 0},
                       10_000, "2026-11-02")
    assert state["sod_equity_cents"] == 10_000
