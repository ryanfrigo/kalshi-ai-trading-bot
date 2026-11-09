# Security Policy

This project signs requests to a real-money exchange. Treat your credentials accordingly.

## Reporting a Vulnerability

Please **do not open a public issue** for security problems. Use
[GitHub private vulnerability reporting](https://github.com/ryanfrigo/kalshi-ai-trading-bot/security/advisories/new)
instead. You'll get a response within a few days. Anything that could cause
unintended order placement, credential exposure, or position oversizing is in scope.

## Handling Your Credentials

- **Never commit** `.env`, `kalshi_private_key`, or any `*.pem` file. They are
  gitignored, and CI runs a secret scan (gitleaks) on every push — but the scan
  is a backstop, not a license to be careless.
- If a key ever lands in a commit (even one you amended away), **rotate it
  immediately** in your Kalshi account settings. Public git history is forever.
- Use [Kalshi's demo environment](https://demo.kalshi.co) for development and
  experiments before pointing anything at production.
- Live orders need an explicit `--live` flag (or `confirm=true` over MCP). Everything
  defaults to dry-run. Read the execution path and test in demo/paper mode first.

## Test Suite Safety

A plain `pytest` run is safe: tests that touch the real Kalshi API or spend LLM
credits are marked `live` and skipped unless you explicitly opt in with
`RUN_LIVE_TESTS=1 pytest -m live`. Some live tests **place real orders** — only
run them against an account whose losses you accept.
