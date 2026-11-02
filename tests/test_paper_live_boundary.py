"""Paper-vs-live boundary audit.

README promise: paper / dry mode never touches the exchange; live needs an explicit
``--live``. Every entry point is driven here in its DEFAULT mode (no ``--live``)
with a REAL ``KalshiClient`` whose httpx transport is an ``httpx.MockTransport``
that records every request. Nothing leaves the process, no credential is read
(``dotenv.load_dotenv`` is stubbed, the key loader/signer are patched).

For each entry point we assert:
  * zero non-GET requests on the wire, and
  * zero calls to place_order / cancel_order (the client has no amend/batch method;
    ``test_inventory_*`` pins that, so a new one fails loudly).
A ``--live`` control is run beside the dry run where one exists, to prove the
sensor can see a mutation (a test that never sees a POST proves nothing).

Paths that CAN send a mutating request without ``--live`` are strict xfails that
carry the exact command. Full table: /private/tmp/paper-live-report.md
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs

import httpx
import pytest

import cli
from src.clients.kalshi_client import KalshiClient

REPO = Path(__file__).resolve().parent.parent
ORDERS = "/trade-api/v2/portfolio/events/orders"
MARKET = {"status": "active", "yes_bid_dollars": 0.48, "yes_ask_dollars": 0.50,
          "no_bid_dollars": 0.50, "no_ask_dollars": 0.52}


class Exchange:
    """Fake Kalshi. Records every request; accepts an order POST (so a leak shows
    up as a recorded mutation instead of an opaque error)."""

    def __init__(self):
        self.log = []                 # (method, path)
        self.held = {}                # ticker -> position_fp
        self.markets = {"T-A": dict(MARKET)}
        self.default_market = dict(MARKET)
        self.market_list = []         # GET /markets (list)

    @property
    def mutations(self):
        return [r for r in self.log if r[0] != "GET"]

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        q = parse_qs(request.url.query.decode())
        self.log.append((method, path))
        if method != "GET":
            if method == "POST" and path == ORDERS:
                body = json.loads(request.content)
                return httpx.Response(201, json={"order_id": "O-1", "fill_count": "0.00",
                                                 "status": "resting", "ticker": body["ticker"]})
            return httpx.Response(405, json={"error": "fake exchange: mutation refused"})
        if path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"balance": 100_000, "portfolio_value": 0})
        if path.endswith("/portfolio/positions"):
            t = q.get("ticker", [None])[0]
            rows = [{"ticker": k, "position_fp": v} for k, v in self.held.items()
                    if t is None or k == t]
            return httpx.Response(200, json={"market_positions": rows, "cursor": ""})
        if path.endswith("/orderbook"):
            return httpx.Response(200, json={
                "orderbook": {"yes": [[48, 100]], "no": [[50, 100]]},
                "orderbook_fp": {"yes_dollars": [["0.48", "100"]], "no_dollars": [["0.50", "100"]]}})
        if path.startswith("/trade-api/v2/markets/"):
            t = path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"market": {"ticker": t, **self.markets.get(t, self.default_market)}})
        if path == "/trade-api/v2/markets":
            return httpx.Response(200, json={"markets": self.market_list, "cursor": ""})
        if path.endswith("/portfolio/orders"):
            return httpx.Response(200, json={"orders": [], "cursor": ""})
        if path.endswith("/portfolio/fills"):
            return httpx.Response(200, json={"fills": [], "cursor": ""})
        if path.endswith("/portfolio/settlements"):
            return httpx.Response(200, json={"settlements": [], "cursor": ""})
        if path == "/trade-api/v2/events":
            return httpx.Response(200, json={"events": [], "cursor": ""})
        return httpx.Response(404, json={"error": f"unexpected {path}"})


class Calls:
    """Counts calls to every client method that can mutate."""

    def __init__(self):
        self.n = {"place_order": 0, "cancel_order": 0}


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """chdir to tmp (DB / data/runtime are relative), no live env, every
    KalshiClient() is a real client on the recording transport."""
    monkeypatch.chdir(tmp_path)
    for k in ("LIVE_TRADING_ENABLED", "RUN_LIVE_TESTS"):
        monkeypatch.delenv(k, raising=False)
    # `cli run` / BeastModeBot mutate process-global settings; restore them on teardown.
    from src.config import settings as cfg
    for name in ("max_drawdown", "max_sector_exposure"):
        monkeypatch.setattr(cfg, name, getattr(cfg, name))
    for name in ("min_confidence_to_trade", "max_position_size_pct", "kelly_fraction",
                 "live_trading_enabled", "paper_trading_mode"):
        monkeypatch.setattr(cfg.settings.trading, name, getattr(cfg.settings.trading, name))
    ex, calls = Exchange(), Calls()
    real_init, real_place, real_cancel = KalshiClient.__init__, KalshiClient.place_order, KalshiClient.cancel_order

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self.api_key = self.api_key or "test-key"
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(ex.handle))

    async def place(self, *a, **k):
        calls.n["place_order"] += 1
        return await real_place(self, *a, **k)

    async def cancel(self, *a, **k):
        calls.n["cancel_order"] += 1
        return await real_cancel(self, *a, **k)

    real_sleep = asyncio.sleep

    async def fast_sleep(_s, *a, **k):
        await real_sleep(0)

    import dotenv
    with patch.object(KalshiClient, "__init__", init), \
         patch.object(KalshiClient, "_load_private_key", lambda self: None), \
         patch.object(KalshiClient, "_sign_request", lambda self, *a: "sig"), \
         patch.object(KalshiClient, "place_order", place), \
         patch.object(KalshiClient, "cancel_order", cancel), \
         patch.object(dotenv, "load_dotenv", lambda *a, **k: False), \
         patch("src.clients.kalshi_client.asyncio.sleep", fast_sleep):
        ex.calls = calls
        yield ex


def run_cli(*argv):
    """Drive the real parser + subcommand. SystemExit / errors are tolerated:
    the audited property is what reached the wire, not that the command succeeded."""
    with patch.object(sys, "argv", ["cli.py", *argv]):
        try:
            cli.main()
        except SystemExit:
            pass
        except Exception as e:                       # noqa: BLE001
            return e
    return None


def assert_clean(ex: Exchange):
    assert ex.mutations == [], f"mutating request(s) in default mode: {ex.mutations}"
    assert ex.calls.n == {"place_order": 0, "cancel_order": 0}, ex.calls.n


# --------------------------------------------------------------------------
# cli.py: order-capable subcommands, default mode, + --live control
# --------------------------------------------------------------------------

def test_cli_trade_default_is_dry(wire):
    run_cli("trade", "--ticker", "T-A", "--side", "no", "--count", "3", "--price", "0.52")
    assert wire.log, "command never reached the exchange: test would prove nothing"
    assert_clean(wire)


def test_cli_trade_live_control_posts(wire):
    run_cli("trade", "--ticker", "T-A", "--side", "no", "--count", "3", "--price", "0.52", "--live")
    assert [m for m, _ in wire.mutations] == ["POST"], wire.log


def test_cli_close_default_is_dry(wire):
    wire.held["T-A"] = "5.00"
    run_cli("close", "--ticker", "T-A")
    assert any(p.endswith("/portfolio/positions") for _, p in wire.log)
    assert_clean(wire)


def test_cli_close_live_control_posts(wire):
    wire.held["T-A"] = "5.00"
    run_cli("close", "--ticker", "T-A", "--live")
    assert [m for m, _ in wire.mutations] == ["POST"], wire.log


def test_cli_close_all_default_is_dry(wire):
    wire.held.update({"T-A": "5.00", "T-B": "-2.00"})
    run_cli("close-all")
    assert any(p.endswith("/orderbook") for _, p in wire.log)
    assert_clean(wire)


def test_cli_close_all_live_control_posts(wire):
    wire.held.update({"T-A": "5.00"})
    run_cli("close-all", "--live", "--yes")
    assert [m for m, _ in wire.mutations] == ["POST"], wire.log


def test_cli_close_all_live_without_yes_needs_typed_confirmation(wire, monkeypatch):
    wire.held["T-A"] = "5.00"
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    run_cli("close-all", "--live")
    assert_clean(wire)


def test_cli_verify_never_orders(wire, tmp_path):
    research = tmp_path / "r.json"
    research.write_text(json.dumps({"catalyst": "x", "sources": [], "est_prob": 0.5}))
    run_cli("verify", "--ticker", "T-A", "--research-file", str(research))
    assert_clean(wire)


def test_cli_daily_default_is_dry(wire):
    run_cli("daily")
    assert wire.log
    assert_clean(wire)


def test_cli_run_safe_compounder_default_is_dry(wire):
    err = run_cli("run", "--safe-compounder")
    assert wire.log, err
    assert_clean(wire)


def test_cli_run_safe_compounder_paper_flag_is_dry(wire):
    run_cli("run", "--safe-compounder", "--paper")
    assert_clean(wire)


@pytest.mark.parametrize("argv", [
    ["brief"], ["settle"], ["fills"], ["status"], ["learnings"], ["policy"],
    ["improve"], ["edge"], ["report"], ["scores"], ["history"], ["backtest"],
])
def test_cli_other_subcommands_send_no_mutation(wire, argv):
    run_cli(*argv)
    assert_clean(wire)


def test_cli_only_close_all_places_orders_directly_and_only_behind_live():
    """Static backstop for the subcommands not driven above (health, dashboard):
    cli.py's only direct place_order is in cmd_close_all, after the dry-run `continue`."""
    src = (REPO / "cli.py").read_text()
    assert src.count("place_order(") == 1
    body = src[src.index("def cmd_close_all"):]
    assert body.index("if not live_mode:") < body.index("client.place_order(")
    tree = ast.parse(src)
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("cmd_health", "cmd_dashboard")]:
        names = {a.attr for a in ast.walk(fn) if isinstance(a, ast.Attribute)}
        assert not names & {"place_order", "cancel_order", "_make_authenticated_request"}, fn.name


# --------------------------------------------------------------------------
# `run` / beast_mode_bot: mode plumbing
# --------------------------------------------------------------------------

def _captured_bot(argv):
    """cli.main(argv) with BeastModeBot.run captured: returns (live_mode, settings flags)."""
    import beast_mode_bot
    from src.config.settings import settings
    seen = {}

    async def fake_run(self):
        seen.update(live=self.live_mode, enabled=settings.trading.live_trading_enabled,
                    paper=settings.trading.paper_trading_mode)

    with patch.object(beast_mode_bot.BeastModeBot, "run", fake_run):
        run_cli(*argv)
    return seen


@pytest.mark.parametrize("argv", [["run"], ["run", "--paper"], ["run", "--beast"],
                                  ["run", "--beast", "--paper"], ["dashboard"]])
def test_run_default_mode_is_paper(wire, argv):
    seen = _captured_bot(argv)
    if argv[0] == "dashboard":                       # dashboard shells out when launch_dashboard.py exists
        return
    assert seen == {"live": False, "enabled": False, "paper": True}, seen


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "True "])
def test_env_var_alone_cannot_switch_cli_run_to_live(wire, monkeypatch, value):
    """LIVE_TRADING_ENABLED reaches the settings object, but BeastModeBot overwrites it
    from the CLI flag before anything trades."""
    from src.config.settings import TradingConfig, settings
    monkeypatch.setenv("LIVE_TRADING_ENABLED", value)
    monkeypatch.setattr(settings.trading, "live_trading_enabled", TradingConfig().live_trading_enabled)
    monkeypatch.setattr(settings.trading, "paper_trading_mode", TradingConfig().paper_trading_mode)
    seen = _captured_bot(["run"])
    assert seen == {"live": False, "enabled": False, "paper": True}, seen


def test_run_live_and_paper_together_is_refused(wire):
    seen = _captured_bot(["run", "--live", "--paper"])
    assert seen == {}


def test_run_live_flag_is_the_switch(wire):
    seen = _captured_bot(["run", "--live"])
    assert seen["live"] is True and seen["enabled"] is True


def test_beast_mode_bot_module_default_is_paper(wire):
    import beast_mode_bot
    from src.config.settings import settings
    beast_mode_bot.BeastModeBot()
    assert settings.trading.live_trading_enabled is False


def test_env_var_is_a_documented_but_unreferenced_switch():
    """Only settings.py reads LIVE_TRADING_ENABLED, and nothing outside tests/ sets it."""
    hits = []
    for p in list(REPO.glob("*.py")) + list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").glob("*.py")):
        if "LIVE_TRADING_ENABLED" in p.read_text():
            hits.append(str(p.relative_to(REPO)))
    assert hits == ["src/config/settings.py"], hits


# --------------------------------------------------------------------------
# strict xfails: default-mode paths that DO send a mutating request
# --------------------------------------------------------------------------

def _dump(tmp_path):
    """Every row of every table, for byte-identical before/after comparison."""
    import sqlite3
    con = sqlite3.connect(str(tmp_path / "trading_system.db"))
    out = {}
    for (t,) in con.execute("select name from sqlite_master where type='table' order by name"):
        out[t] = con.execute(f"select * from {t} order by 1").fetchall()
    return out


async def _seed_live_row(tmp_path, ticker="T-PT", take_profit=0.48):
    from src.utils.database import DatabaseManager, Position
    db = DatabaseManager(db_path=str(tmp_path / "trading_system.db"))
    await db.initialize()
    # a row exactly as a prior `--live` run (or a paper "simulated" fill, see
    # execute_position's else-branch) leaves it: open, live=1
    await db.add_position(Position(ticker, "YES", 0.40, 4, datetime.now(), live=True,
                                   confidence=0.7, stop_loss_price=0.37, take_profit_price=take_profit))
    return db


@pytest.mark.parametrize("argv", [["run"], ["run", "--paper"], ["run", "--beast"]])
async def test_run_default_position_tracking_sends_no_mutation(wire, tmp_path, argv):
    import beast_mode_bot
    await _seed_live_row(tmp_path)
    wire.held["T-PT"] = "4.00"
    wire.markets["T-PT"] = {"status": "active", "yes_price": 60, "yes_bid_dollars": 0.60,
                            "yes_ask_dollars": 0.62, "no_bid_dollars": 0.38, "no_ask_dollars": 0.40}
    before = _dump(tmp_path)
    bot = beast_mode_bot.BeastModeBot(live_mode=False)          # exactly what cmd_run builds without --live
    assert bot.live_mode is False

    real_sleep = asyncio.sleep

    async def stop_after_one(_s, *a, **k):
        bot.shutdown_event.set()
        await real_sleep(0)

    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "trading_system.db"))
    with patch("beast_mode_bot.asyncio.sleep", stop_after_one):
        await bot._run_position_tracking(db, MagicMock())
    assert wire.mutations == [], f"paper-mode run sent {wire.mutations}"
    assert wire.calls.n == {"place_order": 0, "cancel_order": 0}
    assert _dump(tmp_path) == before, "paper run wrote to the DB"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "OWNER DECISION: script is live by default. `LIVE_TRADING_ENABLED=true python scripts/test_quick_flip_strategy.py`: this script (and any "
    "caller of experimental QuickFlip / strategies.market_making / portfolio.immediate that does "
    "not go through BeastModeBot) takes live mode from settings.trading.live_trading_enabled, "
    "which an env var alone sets. No --live is involved."))
async def test_env_var_alone_live_for_non_beastmode_entry_points(wire, monkeypatch, tmp_path):
    from experimental.strategies.quick_flip_scalping import (
        QuickFlipOpportunity, QuickFlipScalpingStrategy)
    from src.config.settings import TradingConfig, settings
    from src.utils.database import DatabaseManager
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setattr(settings.trading, "live_trading_enabled", TradingConfig().live_trading_enabled)
    db = DatabaseManager(db_path=str(tmp_path / "trading_system.db"))
    await db.initialize()
    c = KalshiClient()
    strat = QuickFlipScalpingStrategy(db, c, MagicMock())
    await strat._execute_single_quick_flip(QuickFlipOpportunity(
        "T-A", "t", "YES", 0.50, 0.80, 2, 0.3, 0.9, "x", 10))
    await c.close()
    assert wire.mutations == [], f"env var alone produced {wire.mutations}"


# ---- scripts that import an order method -----------------------------------

def _script_sandbox(tmp_path, name):
    """Copy a script into tmp/scripts so its REPO-relative state/journal writes land in tmp."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "data" / "runtime").mkdir(parents=True)
    (root / "src").symlink_to(REPO / "src")
    dst = root / "scripts" / name
    dst.write_text((REPO / "scripts" / name).read_text())
    return root, dst


def _run_script(path, *argv):
    with patch.object(sys, "argv", [str(path), *argv]):
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit:
            pass


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "OWNER DECISION: script is live by default. `python scripts/closeout_live.py` (no flag): LIVE unless --dry. "
    "It buys 5 contracts per candidate with place_order."))
def test_script_closeout_live_default_must_not_buy(wire, tmp_path):
    root, path = _script_sandbox(tmp_path, "closeout_live.py")
    (root / "data" / "aaa" / "pricings").mkdir(parents=True)
    (root / "data" / "aaa" / "pricings" / "s.json").write_text(json.dumps({
        "series": "KXAAA", "target_date": "2999-01-01",
        "rows": [{"ticker": "KXAAA-X", "fair_yes": 0.999, "yes_ask": 0.95, "yes_bid": 0.90}]}))
    _run_script(path)
    assert wire.mutations == [], f"{wire.mutations}"


def test_script_closeout_live_dry_flag_is_clean(wire, tmp_path):
    root, path = _script_sandbox(tmp_path, "closeout_live.py")
    (root / "data" / "aaa" / "pricings").mkdir(parents=True)
    (root / "data" / "aaa" / "pricings" / "s.json").write_text(json.dumps({
        "series": "KXAAA", "target_date": "2999-01-01",
        "rows": [{"ticker": "KXAAA-X", "fair_yes": 0.999, "yes_ask": 0.95, "yes_bid": 0.90}]}))
    _run_script(path, "--dry")
    assert_clean(wire)


def _maker_market():
    soon = datetime.now(timezone.utc).replace(microsecond=0)
    from datetime import timedelta
    return {"ticker": "KXNFLGAME-M", "event_ticker": "E1", "volume": 5000, "yes_bid": 94, "yes_ask": 97,
            "close_time": (soon + timedelta(hours=5)).isoformat().replace("+00:00", "Z"),
            "status": "active"}


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "OWNER DECISION: script is live by default. `python scripts/maker_closeout.py` (no flag): LIVE unless --dry. "
    "It rests 15-contract buys with place_order."))
def test_script_maker_closeout_default_must_not_buy(wire, tmp_path):
    root, path = _script_sandbox(tmp_path, "maker_closeout.py")
    m = _maker_market()
    wire.market_list = [m]
    wire.markets["KXNFLGAME-M"] = {"status": "active", "close_time": m["close_time"]}
    _run_script(path, "--max-markets", "1")
    assert wire.mutations == [], f"{wire.mutations}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "OWNER DECISION: script is live by default. `python scripts/ornn_stop_check.py` (no flag, no dry mode): sells the held ORNN position "
    "with place_order whenever its print rules fire. Risk-reducing by design, but a real order "
    "with no --live."))
def test_script_ornn_stop_check_default_must_not_sell(wire, tmp_path):
    root, path = _script_sandbox(tmp_path, "ornn_stop_check.py")
    now = datetime.now(timezone.utc)
    (root / "data" / "ornn").mkdir(parents=True)
    (root / "data" / "ornn" / "a100-sxm4.json").write_text(json.dumps({"data": [
        {"timestamp": f"{now:%Y-%m}-01T16:00:00Z", "index_value": 0.95}]}))

    class _P:
        async def wait(self):
            return 0

    async def fake_exec(*a, **k):
        return _P()

    wire.held[f"KXA100MS-26{now.strftime('%b').upper()}-1.000"] = "10.00"
    with patch("asyncio.create_subprocess_exec", fake_exec):
        _run_script(path)
    assert wire.mutations == [], f"{wire.mutations}"


def test_script_exit_preview_is_read_only(wire, tmp_path):
    """scripts/exit_preview.py forwards GETs only; its full proof lives in test_exit_preview.py."""
    src = (REPO / "scripts" / "exit_preview.py").read_text()
    assert "place_order is recorded and never" in src


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_every_tool_default_args_sends_no_mutation(wire, tmp_path):
    import src.mcp_server as server
    wire.held["T-A"] = "5.00"
    for fn, kw in [
        (server.brief, {}), (server.settle, {}), (server.learnings, {}), (server.edge, {}),
        (server.policy, {}), (server.status, {}), (server.scores, {}), (server.history, {}),
        (server.hunt, {"top": 1, "books": False}),
        (server.trade, dict(ticker="T-A", side="no", count=2, price=0.52)),
        (server.close, dict(ticker="T-A")),
    ]:
        try:
            await fn(**kw)
        except Exception:                                  # noqa: BLE001
            pass
    assert wire.log
    assert_clean(wire)


@pytest.mark.asyncio
async def test_mcp_confirm_is_the_only_switch_control(wire):
    import src.mcp_server as server
    await server.trade(ticker="T-A", side="no", count=2, price=0.52, confirm=True)
    assert [m for m, _ in wire.mutations] == ["POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["true", "1"])
async def test_mcp_env_var_cannot_enable_orders(wire, monkeypatch, value):
    import src.mcp_server as server
    monkeypatch.setenv("LIVE_TRADING_ENABLED", value)
    wire.held["T-A"] = "5.00"
    await server.trade(ticker="T-A", side="no", count=2, price=0.52)
    await server.close(ticker="T-A")
    assert_clean(wire)


# --------------------------------------------------------------------------
# paper_trader.py
# --------------------------------------------------------------------------

def test_paper_trader_never_reaches_an_order_method():
    src = (REPO / "paper_trader.py").read_text()
    for name in ("place_order", "cancel_order", "execute_position", "run_tracking",
                 "_make_authenticated_request", "BeastModeBot", "SafeCompounder"):
        assert name not in src, name


@pytest.mark.asyncio
async def test_paper_trader_scan_sends_no_mutation(wire, tmp_path, monkeypatch):
    import paper_trader
    args = [a for a in ("main", "run_scan", "scan_once") if hasattr(paper_trader, a)]
    assert args, "paper_trader entry point renamed: update this test"
    fn = getattr(paper_trader, "run_scan", None) or getattr(paper_trader, "scan_once", None)
    if fn is not None:
        with patch("src.clients.xai_client.XAIClient", MagicMock()), \
             patch("src.jobs.ingest.run_ingestion", AsyncMock()), \
             patch("src.jobs.decide.make_decision_for_market", AsyncMock(return_value=None)):
            try:
                await fn()
            except Exception:                              # noqa: BLE001
                pass
    assert_clean(wire)


# --------------------------------------------------------------------------
# inventory: every order-method call site is known, so a new one fails here
# --------------------------------------------------------------------------

KNOWN_ORDER_SITES = {
    "cli.py", "scripts/ornn_stop_check.py", "scripts/closeout_live.py", "scripts/maker_closeout.py",
    "scripts/exit_preview.py", "src/clients/kalshi_client.py", "src/strategies/market_making.py",
    "src/agent/toolbelt.py", "src/strategies/safe_compounder.py", "src/jobs/execute.py",
    "src/risk/risk_governor.py", "src/utils/database.py",
}


def test_inventory_order_call_sites_are_all_audited():
    pat = re.compile(r"\.(place_order|cancel_order|amend_order|batch_create_orders|batch_cancel_orders)\(")
    found = set()
    files = list(REPO.glob("*.py")) + list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").glob("*.py"))
    for p in files:
        if pat.search(p.read_text()):
            found.add(str(p.relative_to(REPO)))
    assert found <= KNOWN_ORDER_SITES, f"new order call site(s) need a paper/live audit: {found - KNOWN_ORDER_SITES}"


def test_inventory_client_has_exactly_two_mutating_methods():
    src = (REPO / "src" / "clients" / "kalshi_client.py").read_text()
    verbs = re.findall(r'"(POST|DELETE|PUT|PATCH)"', src)
    assert sorted(verbs) == ["DELETE", "POST"], verbs
    for name in ("amend_order", "batch_create_orders", "batch_cancel_orders"):
        assert not hasattr(KalshiClient, name)


def test_inventory_only_run_daily_sh_opts_in_to_live_and_only_on_explicit_arg():
    """launchd / shell wrappers: list any that opt in to --live (informational gate: must be reviewed)."""
    hits = []
    for p in list((REPO / "deploy").rglob("*")) + list((REPO / "scripts").glob("*.sh")):
        if p.is_file() and "--live" in p.read_text(errors="ignore"):
            hits.append(str(p.relative_to(REPO)))
    # run_daily.sh emits --live only when KALSHI_DAILY_MODE=live (default dry).
    assert hits == ["scripts/run_daily.sh"], hits
    sh = (REPO / "scripts" / "run_daily.sh").read_text()
    assert re.search(r'if \[ "\$MODE" = "live" \]; then\s+FLAG="--live"', sh)
    # the wrapper's own switch is the KALSHI_DAILY_MODE env var, and its default is dry
    assert 'MODE="${KALSHI_DAILY_MODE:-dry}"' in sh, "default MODE changed: re-audit"


# --------------------------------------------------------------------------
# run_tracking / exit helpers: paper by default, live only with live=True
# --------------------------------------------------------------------------

_PT_MARKET = {"status": "active", "yes_price": 60, "yes_bid_dollars": 0.60,
              "yes_ask_dollars": 0.62, "no_bid_dollars": 0.38, "no_ask_dollars": 0.40}


async def _tracking_setup(wire, tmp_path, **kw):
    from src.utils.database import DatabaseManager
    await _seed_live_row(tmp_path, **kw)
    wire.held["T-PT"] = "4.00"
    wire.markets["T-PT"] = dict(_PT_MARKET)
    return DatabaseManager(db_path=str(tmp_path / "trading_system.db"))


async def test_run_tracking_default_sends_no_mutation_and_writes_nothing(wire, tmp_path):
    from src.jobs.track import run_tracking
    db = await _tracking_setup(wire, tmp_path)
    before = _dump(tmp_path)
    await run_tracking(db)
    assert _dump(tmp_path) == before
    assert wire.log and wire.mutations == []
    assert wire.calls.n == {"place_order": 0, "cancel_order": 0}


async def test_run_tracking_live_true_control_sells(wire, tmp_path):
    from src.jobs.track import run_tracking
    db = await _tracking_setup(wire, tmp_path)
    await run_tracking(db, live=True)
    assert [m for m, _ in wire.mutations] == ["POST"]


async def test_place_sell_limit_order_refuses_without_live(wire, tmp_path):
    """Lowest layer: a future caller that forgets the flag cannot send."""
    from src.jobs.execute import place_sell_limit_order
    from src.utils.database import Position
    wire.held["T-A"] = "4.00"
    c = KalshiClient()
    pos = Position("T-A", "YES", 0.40, 4, datetime.now(), live=True, id=1)
    order = {}
    ok = await place_sell_limit_order(pos, 0.55, MagicMock(), c, order_out=order)
    assert ok is False and order == {}
    assert wire.log == [] and wire.mutations == []
    ok = await place_sell_limit_order(pos, 0.55, MagicMock(), c, live=True)
    assert ok is True and [m for m, _ in wire.mutations] == ["POST"]
    await c.close()


@pytest.mark.parametrize("helper,market", [
    ("place_profit_taking_orders", _PT_MARKET),
    ("place_stop_loss_orders", {"status": "active", "yes_bid_dollars": 0.20, "yes_ask_dollars": 0.22,
                                "no_bid_dollars": 0.78, "no_ask_dollars": 0.80}),
])
async def test_profit_and_stop_helpers_default_paper(wire, tmp_path, helper, market):
    import src.jobs.execute as ex_mod
    db = await _tracking_setup(wire, tmp_path)
    wire.markets["T-PT"] = dict(market)
    before = _dump(tmp_path)
    c = KalshiClient()
    res = await getattr(ex_mod, helper)(db, c)
    await c.close()
    assert res["simulated"] == 1 and res["orders_placed"] == 0
    assert _dump(tmp_path) == before
    assert wire.mutations == [] and wire.calls.n["place_order"] == 0


@pytest.mark.parametrize("name,market,pos_kw,reason", [
    ("stop", {"status": "active", "yes_bid_dollars": 0.20, "yes_ask_dollars": 0.22,
              "no_bid_dollars": 0.78, "no_ask_dollars": 0.80}, {}, "stop_loss"),
    ("profit", _PT_MARKET, {}, "profit_take"),
    # exit-level path: +15% hits the row's take_profit_price (0.45) but not the helper's 20%
    ("exit_level", {"status": "active", "yes_bid_dollars": 0.46, "yes_ask_dollars": 0.48,
                    "no_bid_dollars": 0.52, "no_ask_dollars": 0.54}, {"take_profit": 0.45}, "take_profit"),
    # resolution: used to close the row and log pnl with no sell
    ("resolution", {"status": "closed", "result": "yes", "yes_bid_dollars": 0.0,
                    "no_bid_dollars": 0.0}, {}, "market_resolution"),
])
async def test_paper_run_leaves_db_byte_identical_and_sends_nothing(wire, tmp_path, name, market, pos_kw, reason):
    import src.jobs.execute as ex_mod
    from src.jobs.track import run_tracking
    db = await _tracking_setup(wire, tmp_path, **pos_kw)
    wire.markets["T-PT"] = dict(market)
    before = _dump(tmp_path)
    assert before["positions"], "seed row missing"
    seen = []
    real = ex_mod.log_paper_exit
    with patch.object(ex_mod, "log_paper_exit", lambda p, px, r: (seen.append(r), real(p, px, r))[1]):
        await run_tracking(db)                     # paper (default)
    assert _dump(tmp_path) == before, f"{name}: paper run wrote to the DB"
    assert wire.mutations == [] and wire.calls.n == {"place_order": 0, "cancel_order": 0}
    assert wire.log, "never reached the exchange"
    if name != "resolution":
        assert seen and seen[0] == reason, f"{name}: expected first paper exit {reason!r}, got {seen}"
    # control: the same scenario under live=True acts, so this test can fail
    await run_tracking(db, live=True)
    assert wire.mutations or _dump(tmp_path) != before, f"{name}: live control did nothing"
