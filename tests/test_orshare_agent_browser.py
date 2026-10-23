"""The scheduled job must be able to find agent-browser.

Under launchd the PATH is minimal, so a bare `agent-browser` call raised FileNotFoundError
on 51 separate daily runs while working perfectly in an interactive shell — the OpenRouter
share family was silently never captured. `_find_agent_browser` resolves the binary
explicitly, so this pins that behaviour.
"""
import os

import scripts.orshare_data as osh


def test_binary_override_wins(monkeypatch):
    monkeypatch.setenv("AGENT_BROWSER_BIN", "/tmp/some-agent-browser")
    assert osh._find_agent_browser() == "/tmp/some-agent-browser"


def test_candidates_cover_the_nvm_and_brew_locations():
    cands = osh._agent_browser_candidates()
    assert any(".nvm/versions/node/" in c for c in cands)   # how it is installed here
    assert any("homebrew" in c or "/usr/local/bin" in c for c in cands)


def test_resolution_ignores_path_and_finds_a_real_binary(monkeypatch):
    # simulate launchd: no useful PATH, no override — the glob must still find it
    monkeypatch.delenv("AGENT_BROWSER_BIN", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(osh.shutil, "which", lambda name: None)
    found = osh._find_agent_browser()
    assert found.endswith("agent-browser")
    assert os.path.isabs(found)
