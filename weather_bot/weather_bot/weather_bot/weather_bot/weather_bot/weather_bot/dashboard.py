# ============================================================
# dashboard.py  -  Terminal trade log + ASCII PnL chart
# Run standalone:  python weather_bot/dashboard.py
# Auto-refreshes from the JSON log files every 10s
# ============================================================
import json
import os
import time
from datetime import datetime, timezone
from typing import List, Dict
import config as cfg


def _load(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _color(text: str, code: str) -> str:
    """ANSI color helper."""
    return f"\033[{code}m{text}\033[0m"


def _green(t): return _color(t, "92")
def _red(t):   return _color(t, "91")
def _cyan(t):  return _color(t, "96")
def _bold(t):  return _color(t, "1")
def _yellow(t): return _color(t, "93")


def render_pnl_graph(pnl_history: List[Dict], width: int = 60, height: int = 10):
    """Simple ASCII line chart of cumulative PnL over time."""
    if len(pnl_history) < 2:
        print(_yellow("  [PnL chart: waiting for closed trades...]"))
        return

    values = [p["pnl"] for p in pnl_history]
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        max_v = min_v + 0.01  # avoid divide-by-zero

    # Downsample to width
    step = max(1, len(values) // width)
    sampled = values[::step][-width:]

    chart = [[" "] * len(sampled) for _ in range(height)]
    for x, v in enumerate(sampled):
        row = int((v - min_v) / (max_v - min_v) * (height - 1))
        row = height - 1 - row
        chart[row][x] = "*"

    print(_bold("\n  PnL Chart (cumulative)"))
    print(f"  ${max_v:.2f} |")
    for row in chart:
        line = "".join(row)
        colored = line.replace("*", _green("*") if values[-1] >= 0 else _red("*"))
        print(f"       |", colored)
    print(f"  ${min_v:.2f} +" + "-" * len(sampled))
    print(f"         0{'':>{len(sampled)-10}}Now")


def render_trade_log(trade_log: List[Dict], last_n: int = 15):
    """Print last N trades with color-coded PnL and reason."""
    print(_bold("\n  ========== CLAWDBOT TRADE LOG =========="))
    recent = [t for t in trade_log][-last_n:]
    recent.reverse()  # newest first
    for t in recent:
        ts = t.get("timestamp", "")[:19].replace("T", " ")
        t_type = t.get("type", "").upper()
        ticker = t.get("ticker", "")[:30]
        side = t.get("side", "").upper()
        city = t.get("city_name", "")
        price = t.get("price") or t.get("exit_price", 0)
        pnl = t.get("pnl", None)
        reason = t.get("reason", "")

        if t_type == "ENTRY":
            tag = _cyan(f"[ENTER {side}]")
            price_str = _cyan(f"@ ${price:.2f}")
            pnl_str = ""
        else:
            tag = _green(f"[EXIT  {side}]") if (pnl or 0) >= 0 else _red(f"[EXIT  {side}]")
            price_str = f"@ ${price:.2f}"
            pnl_str_raw = f"PnL: ${pnl:+.4f}" if pnl is not None else ""
            pnl_str = _green(pnl_str_raw) if (pnl or 0) >= 0 else _red(pnl_str_raw)

        print(f"  {ts}  {tag}  {city:<8} {ticker:<32} {price_str}  {pnl_str}")
        if reason:
            # Truncate long reason to 100 chars
            short_reason = reason[:100] + "..." if len(reason) > 100 else reason
            print(f"    {_yellow('WHY:')} {short_reason}")
    print()


def render_open_positions(positions: Dict):
    """Print currently open paper positions."""
    print(_bold("  === OPEN POSITIONS ==="))
    if not positions:
        print("  (none)")
        return
    for ticker, pos in positions.items():
        entry = pos.get("entry_price", 0)
        peak  = pos.get("peak_price", entry)
        cost  = pos.get("cost_usd", 0)
        city  = pos.get("city_name", "")
        side  = pos.get("side", "").upper()
        target = entry * cfg.EXIT_MULTIPLIER
        print(f"  {city:<8} {ticker:<35} {side}  entry=${entry:.2f}  "
              f"peak=${peak:.2f}  target=${target:.2f}  cost=${cost:.2f}")
    print()


def render_summary(trade_log: List[Dict], positions: Dict, pnl_history: List[Dict]):
    running_pnl = pnl_history[-1]["pnl"] if pnl_history else 0
    exits = [t for t in trade_log if t.get("type") == "exit"]
    wins  = [t for t in exits if t.get("pnl", 0) > 0]
    win_rate = (len(wins) / len(exits) * 100) if exits else 0
    pnl_color = _green if running_pnl >= 0 else _red
    print(_bold("  === SUMMARY ==="))
    print(f"  Running PnL : {pnl_color(f'${running_pnl:+.4f}')}")
    print(f"  Total Trades: {len(exits)}")
    print(f"  Win Rate    : {win_rate:.1f}%  ({len(wins)}/{len(exits)})")
    print(f"  Open Pos    : {len(positions)}")
    print(f"  Paper Mode  : {_yellow('YES - NO REAL MONEY')}")
    print()


def display(clear: bool = True):
    if clear:
        os.system("cls" if os.name == "nt" else "clear")
    trade_log   = _load(cfg.LOG_FILE, [])
    positions   = _load(cfg.POSITIONS_FILE, {})
    pnl_history = _load(cfg.PNL_FILE, [])

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(_bold(f"\n  CLAWDBOT Weather Trader | Paper Mode | {now}"))
    print("  " + "="*60)
    render_summary(trade_log, positions, pnl_history)
    render_pnl_graph(pnl_history)
    render_open_positions(positions)
    render_trade_log(trade_log)


if __name__ == "__main__":
    print("Clawdbot Dashboard - refreshes every 10s. Ctrl+C to quit.")
    while True:
        display()
        time.sleep(10)