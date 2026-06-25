# Kalshi MCP server

A thin [Model Context Protocol](https://modelcontextprotocol.io) server that lets any
MCP client — Claude Desktop, Claude Code, or anything else that speaks MCP — drive this
toolkit's **existing, governor-gated** tools. No official Kalshi MCP exists; this is the
first-mover distribution layer.

It is a thin wrapper: every tool calls a function that already lives in `src/agent/*`
and `src/risk/*`. The server adds **no new authority** — the two mutating tools route
through the same `place_guarded_order` / `close_position` paths the CLI uses, which
enforce the risk governor (daily-loss + drawdown kill switch + manual halt) and the 10%
per-position cap.

## Trust model

- **Runs locally, on your own key.** The server is launched on your machine and talks to
  Kalshi using *your* `KALSHI_API_KEY` plus your private-key file, read from the
  environment exactly as the CLI does.
- **Your key never leaves your machine.** This server opens no outbound channel of its
  own; it only talks to Kalshi's API, the same as `cli.py`.
- **Read-only by default; confirm to trade.** Eight tools are read-only. The two mutating
  tools (`trade`, `close`) default to a **dry-run preview** — they run the full guard
  stack and report what *would* happen, but place no order. You must pass
  `confirm=true` to place a live order, and even then it goes through the governor + cap.

## Install

The `mcp` SDK is an **optional** dependency. Install it into your environment:

```bash
pip install 'mcp>=1.2'
# or, from this repo:  pip install -e '.[mcp]'
```

## Tools

| Tool        | Kind        | What it does                                                                 |
|-------------|-------------|------------------------------------------------------------------------------|
| `brief`     | read-only   | Governor verdict, equity, positions, resting orders                          |
| `status`    | read-only   | Balance, portfolio value, active event positions                             |
| `settle`    | read-only*  | Pull Kalshi settlements; summarize realized win-rate / P&L (*writes local log)|
| `learnings` | read-only*  | Reconcile outcomes → calibration + edge + candidate learnings (`dry=true` default)|
| `edge`      | read-only   | **The headline:** Brier / log-loss / edge-vs-book + a gated verdict          |
| `scores`    | read-only   | Category scores, win rates, allocation limits                                |
| `history`   | read-only   | Closed-trade history from the local database                                 |
| `hunt`      | read-only   | Broad live-book scan for edge candidates (research material, not a buy list) |
| `trade`     | **mutating**| Place ONE guarded order — **dry-run unless `confirm=true`**                   |
| `close`     | **mutating**| Sell (close) a position — **dry-run unless `confirm=true`**                  |

`trade` and `close` are annotated `destructiveHint`; all others `readOnlyHint`, so a
well-behaved MCP client can warn before invoking the mutating ones.

## Client config

Add this to your MCP client's `mcpServers` config (e.g. Claude Desktop's
`claude_desktop_config.json`, or a Claude Code `.mcp.json`). Point `cwd` at your clone so
the server finds your `.env` / private key, and set `KALSHI_API_KEY` (and
`KALSHI_PRIVATE_KEY_PATH` if your key file isn't the default `kalshi_private_key.pem`).

```json
{
  "mcpServers": {
    "kalshi": {
      "command": "/path/to/kalshi-ai-trading-bot/.venv/bin/python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/kalshi-ai-trading-bot",
      "env": {
        "KALSHI_API_KEY": "your_kalshi_api_key_id",
        "KALSHI_PRIVATE_KEY_PATH": "kalshi_private_key.pem"
      }
    }
  }
}
```

Once the console script is installed (`pip install -e '.[mcp]'`), you can instead use:

```json
{
  "mcpServers": {
    "kalshi": {
      "command": "kalshi-mcp",
      "cwd": "/path/to/kalshi-ai-trading-bot",
      "env": { "KALSHI_API_KEY": "your_kalshi_api_key_id" }
    }
  }
}
```

## Safety reminders

- A `trade` / `close` call with `confirm` unset (or `false`) **never places an order** —
  it returns the dry-run preview (guard checks + the size after the 10% cap).
- The governor halt, manual kill switch (`data/runtime/TRADING_HALTED`), and 10% cap are
  authoritative for live orders. The MCP layer cannot and does not bypass them.
- This server uses real money when you confirm a trade. Treat `confirm=true` like the
  CLI's `--live`.
