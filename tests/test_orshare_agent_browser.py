"""The scheduled job must be able to find agent-browser.

Under launchd the PATH is minimal, so a bare `agent-browser` call raised FileNotFoundError on 51
separate daily runs while working perfectly in an interactive shell — the OpenRouter-share family
was silently never captured. These tests pin the resolution order *without* requiring the binary to
be installed: the first version asserted a real install was found and failed in CI, which is a test
depending on machine state rather than testing the logic.
"""
import os

import pytest

from src.utils import agent_browser as ab


def test_binary_override_wins(monkeypatch):
    monkeypatch.setenv("AGENT_BROWSER_BIN", "/tmp/some-agent-browser")
    assert ab.find() == "/tmp/some-agent-browser"


def test_candidates_cover_the_nvm_and_brew_locations():
    cands = ab.candidates()
    assert any(".nvm/versions/node/" in c for c in cands)   # how it is installed here
    assert any("homebrew" in c or "/usr/local/bin" in c for c in cands)


def test_path_hit_is_used_when_present(monkeypatch):
    monkeypatch.delenv("AGENT_BROWSER_BIN", raising=False)
    monkeypatch.setattr(ab.shutil, "which", lambda name: f"/pretend/{name}")
    assert ab.find() == "/pretend/agent-browser"


def test_falls_back_to_the_install_locations_when_path_fails(tmp_path, monkeypatch):
    # simulate launchd: no override, nothing on PATH — the glob must still resolve one
    monkeypatch.delenv("AGENT_BROWSER_BIN", raising=False)
    monkeypatch.setattr(ab.shutil, "which", lambda name: None)
    fake = tmp_path / "agent-browser"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setattr(ab, "candidates", lambda: [str(fake)])
    found = ab.find()
    assert found == str(fake) and os.path.isabs(found)


def test_raises_a_clear_error_when_nothing_is_found(monkeypatch):
    monkeypatch.delenv("AGENT_BROWSER_BIN", raising=False)
    monkeypatch.setattr(ab.shutil, "which", lambda name: None)
    monkeypatch.setattr(ab, "candidates", lambda: [])
    with pytest.raises(RuntimeError, match="agent-browser not found"):
        ab.find()
