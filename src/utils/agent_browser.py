"""Locate the agent-browser CLI reliably.

launchd and cron run with a minimal PATH, so a bare ``agent-browser`` lookup fails there even
though it works in an interactive shell. That exact bug silently broke the OpenRouter-share
capture on 51 scheduled runs. Any script that shells out to the CLI must resolve it through
here rather than assuming PATH.
"""
from __future__ import annotations

import glob
import os
import shutil


def candidates() -> list:
    """Plausible locations, most-specific first. Pure: patterns only."""
    return [
        os.path.expanduser("~/.nvm/versions/node/*/bin/agent-browser"),
        os.path.expanduser("~/.local/bin/agent-browser"),
        "/opt/homebrew/bin/agent-browser",
        "/usr/local/bin/agent-browser",
    ]


def find() -> str:
    """Resolve the binary: AGENT_BROWSER_BIN, then PATH, then the usual install locations."""
    override = os.environ.get("AGENT_BROWSER_BIN")
    if override:
        return override
    found = shutil.which("agent-browser")
    if found:
        return found
    for pattern in candidates():
        hits = sorted(glob.glob(pattern), reverse=True)   # newest node version first
        if hits:
            return hits[0]
    raise RuntimeError(
        "agent-browser not found: install it, put it on PATH, or set AGENT_BROWSER_BIN"
    )
