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
