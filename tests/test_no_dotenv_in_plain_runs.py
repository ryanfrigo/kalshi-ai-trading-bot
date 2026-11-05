"""A plain pytest run must not load the checkout's .env (see tests/conftest.py)."""
import os
import subprocess
import sys
from pathlib import Path

import dotenv
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_load_dotenv_is_stubbed_in_a_plain_run(tmp_path, monkeypatch):
    if os.getenv("RUN_LIVE_TESTS", "").lower() in {"1", "true", "yes"}:
        pytest.skip("live runs opt in to the real .env")
    env_file = tmp_path / ".env"
    env_file.write_text("DOTENV_GUARD_SENTINEL=loaded\n")
    monkeypatch.delenv("DOTENV_GUARD_SENTINEL", raising=False)
    assert dotenv.load_dotenv(env_file) is False
    assert "DOTENV_GUARD_SENTINEL" not in os.environ


def test_settings_still_calls_load_dotenv_at_import():
    # The guard exists because of this line. If it goes, the guard can go too.
    assert "\nload_dotenv()\n" in (ROOT / "src/config/settings.py").read_text()


def test_guard_runs_before_any_src_import():
    source = (ROOT / "tests/conftest.py").read_text()
    guard = source.index("dotenv.load_dotenv = ")
    assert "from src" not in source[:guard]
    assert "import src" not in source[:guard]


@pytest.mark.parametrize("flag,loaded", [("", "no"), ("1", "yes")])
def test_conftest_guard_end_to_end_in_a_fresh_process(tmp_path, flag, loaded):
    # A fresh interpreter importing conftest, then a bare load_dotenv on a sentinel file.
    env_file = tmp_path / ".env"
    env_file.write_text("DOTENV_GUARD_SENTINEL=loaded\n")
    code = (
        "import importlib.util, os, sys, dotenv\n"
        f"spec = importlib.util.spec_from_file_location('c', {str(ROOT / 'tests/conftest.py')!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"dotenv.load_dotenv({str(env_file)!r})\n"
        "print('yes' if os.environ.get('DOTENV_GUARD_SENTINEL') else 'no')\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in {"DOTENV_GUARD_SENTINEL", "RUN_LIVE_TESTS"}}
    if flag:
        env["RUN_LIVE_TESTS"] = flag
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.stdout.strip().splitlines()[-1] == loaded, out.stderr
