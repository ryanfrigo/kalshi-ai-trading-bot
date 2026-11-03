"""The governor's default state file and halt flag must not depend on the cwd.

Before this was anchored, a process launched from another directory started
with fresh state and an unseen kill switch, so the drawdown halt silently reset.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RISK_GOVERNOR = REPO / "src" / "risk" / "risk_governor.py"


def _fresh_module():
    # Load a private copy: the autouse conftest fixture redirects the constants
    # on the shared module, and this test must see the real defaults.
    name = "_risk_governor_fresh"
    spec = importlib.util.spec_from_file_location(name, RISK_GOVERNOR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_default_paths_are_absolute_and_anchored_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _fresh_module()

    state = Path(mod.DEFAULT_STATE_PATH)
    halt = Path(mod.DEFAULT_HALT_FLAG_PATH)
    assert state.is_absolute()
    assert halt.is_absolute()
    assert state == REPO / "data" / "runtime" / "risk_governor_state.json"
    assert halt == REPO / "data" / "runtime" / "TRADING_HALTED"


def test_governor_built_with_defaults_uses_repo_paths_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _fresh_module()

    gov = mod.RiskGovernor()
    assert gov.state_path == REPO / "data" / "runtime" / "risk_governor_state.json"
    assert gov.halt_flag_path == REPO / "data" / "runtime" / "TRADING_HALTED"


def test_explicit_path_arguments_still_win(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _fresh_module()

    gov = mod.RiskGovernor(state_path=str(tmp_path / "s.json"),
                           halt_flag_path=str(tmp_path / "H"))
    assert gov.state_path == tmp_path / "s.json"
    assert gov.halt_flag_path == tmp_path / "H"


# --- KALSHI_RUNTIME_DIR and the report's audit log -------------------------

import json

import pytest


def test_env_var_redirects_state_halt_and_audit(tmp_path, monkeypatch):
    from src.risk import risk_governor as rg

    d = tmp_path / "elsewhere"
    monkeypatch.setenv("KALSHI_RUNTIME_DIR", str(d))

    assert rg.resolve_state_path() == d / "risk_governor_state.json"
    assert rg.resolve_halt_flag_path() == d / "TRADING_HALTED"
    assert rg.resolve_audit_path() == d / "governor_audit.jsonl"
    gov = rg.RiskGovernor()
    assert gov.state_path == d / "risk_governor_state.json"
    assert gov.halt_flag_path == d / "TRADING_HALTED"


def test_env_var_read_at_call_time_not_import_time(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_RUNTIME_DIR", str(tmp_path / "late"))
    mod = _fresh_module()  # imported before... env already set; now change it
    monkeypatch.setenv("KALSHI_RUNTIME_DIR", str(tmp_path / "later"))
    assert mod.resolve_audit_path() == tmp_path / "later" / "governor_audit.jsonl"


def test_unset_env_keeps_repo_root_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KALSHI_RUNTIME_DIR", raising=False)
    mod = _fresh_module()

    runtime = REPO / "data" / "runtime"
    assert mod.resolve_state_path() == runtime / "risk_governor_state.json"
    assert mod.resolve_halt_flag_path() == runtime / "TRADING_HALTED"
    assert mod.resolve_audit_path() == runtime / "governor_audit.jsonl"
    # An empty value is "unset", not "the cwd".
    monkeypatch.setenv("KALSHI_RUNTIME_DIR", "  ")
    assert mod.resolve_audit_path() == runtime / "governor_audit.jsonl"


def test_cli_report_reads_audit_from_anchor_when_cwd_elsewhere(tmp_path, monkeypatch, capsys):
    import argparse

    import cli
    from src.risk import risk_governor as rg

    anchor = tmp_path / "anchor"
    anchor.mkdir()
    (anchor / "governor_audit.jsonl").write_text(json.dumps({
        "action": "rebase_peak",
        "old": {"peak_equity_cents": 123456},
        "new": {"date": "2026-09-25"},
    }) + "\n")
    monkeypatch.setattr(rg, "DEFAULT_AUDIT_PATH", str(anchor / "governor_audit.jsonl"))

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)  # no data/runtime here: a cwd-relative read finds nothing

    seen = {}
    import src.agent.report as report

    def fake_render(**kw):
        seen.update(kw)
        return "md"

    monkeypatch.setattr(report, "render_track_record", fake_render)

    async def fake_brief(client):
        return {"equity_cents": 1}

    import src.agent.toolbelt as toolbelt
    monkeypatch.setattr(toolbelt, "account_brief", fake_brief)

    class FakeClient:
        async def close(self):
            pass

    import src.clients.kalshi_client as kc
    monkeypatch.setattr(kc, "KalshiClient", FakeClient)

    cli.cmd_report(argparse.Namespace(stdout=True))
    assert seen["equity"]["rebase"] == {"date": "2026-09-25", "prior_peak_cents": 123456}


def test_env_var_tilde_expands_under_home(monkeypatch):
    from src.risk import risk_governor as rg

    monkeypatch.setenv("KALSHI_RUNTIME_DIR", "~/x")
    assert rg.resolve_halt_flag_path() == Path.home() / "x" / "TRADING_HALTED"


@pytest.mark.parametrize("name", ["resolve_state_path", "resolve_halt_flag_path", "resolve_audit_path"])
def test_relative_env_var_raises(monkeypatch, name):
    from src.risk import risk_governor as rg

    monkeypatch.setenv("KALSHI_RUNTIME_DIR", "rel")
    with pytest.raises(ValueError, match="absolute"):
        getattr(rg, name)()


def test_governor_with_relative_env_var_raises(monkeypatch):
    from src.risk import risk_governor as rg

    monkeypatch.setenv("KALSHI_RUNTIME_DIR", "rel")
    with pytest.raises(ValueError):
        rg.RiskGovernor()
