"""
Generate a static HTML dashboard for paper trading signals.
Can be served via GitHub Pages or opened locally.
"""

import os
from datetime import datetime, timezone
from .tracker import get_all_signals, get_stats


def generate_html(output_path: str = None) -> str:
    """Generate the dashboard HTML and optionally write to file."""
    stats = get_stats()
    signals = get_all_signals()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build cumulative P&L series for chart
    settled = [s for s in reversed(signals) if s["outcome"] != "pending"]
    cum_pnl = []
    running = 0.0
    for s in settled:
        running += s["pnl"] or 0
        cum_pnl.append({"x": s["settled_at"] or s["timestamp"], "y": round(running, 2)})

    # Signal rows
    rows_html = ""
    for s in signals:
        outcome_badge = {
            "win": '<span class="badge win">WIN</span>',
            "loss": '<span class="badge loss">LOSS</span>',
            "pending": '<span class="badge pending">PENDING</span>',
        }.get(s["outcome"], s["outcome"])

        pnl_str = f"${s['pnl']:.2f}" if s["pnl"] is not None else "—"
        pnl_class = "pos" if (s["pnl"] or 0) > 0 else "neg" if (s["pnl"] or 0) < 0 else ""

        ts = s["timestamp"][:16].replace("T", " ")

        rows_html += f"""
        <tr>
            <td>{ts}</td>
            <td title="{s['market_id']}">{_trunc(s['market_title'], 50)}</td>
            <td>{s['side']}</td>
            <td>{s['entry_price']:.0%}</td>
            <td>{s['confidence']:.0%}</td>
            <td>{s['strategy']}</td>
            <td>{outcome_badge}</td>
            <td class="{pnl_class}">{pnl_str}</td>
            <td title="{_escape(s['reasoning'])}">{_trunc(s['reasoning'], 40)}</td>
        </tr>"""

    # Chart data as JSON
    import json
    chart_json = json.dumps(cum_pnl)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kalshi AI Bot — Paper Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9;
           --green: #3fb950; --red: #f85149; --yellow: #d29922; --blue: #58a6ff; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: var(--bg); color: var(--text); padding: 1.5rem; }}
  h1 {{ color: #fff; margin-bottom: .25rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 1.5rem; font-size: .9rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
            gap: 1rem; margin-bottom: 1.5rem; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
           padding: 1rem; text-align: center; }}
  .stat .value {{ font-size: 1.8rem; font-weight: 700; }}
  .stat .label {{ font-size: .75rem; color: #8b949e; margin-top: .25rem; }}
  .stat .value.pos {{ color: var(--green); }}
  .stat .value.neg {{ color: var(--red); }}
  .chart-wrap {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
                 padding: 1rem; margin-bottom: 1.5rem; max-height: 300px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  th {{ background: #21262d; text-align: left; padding: .6rem .8rem; font-size: .75rem;
        text-transform: uppercase; color: #8b949e; position: sticky; top: 0; }}
  td {{ padding: .5rem .8rem; border-top: 1px solid var(--border); font-size: .85rem; }}
  tr:hover {{ background: #1c2128; }}
  .badge {{ padding: 2px 8px; border-radius: 12px; font-size: .75rem; font-weight: 600; }}
  .badge.win {{ background: #23302a; color: var(--green); }}
  .badge.loss {{ background: #30201f; color: var(--red); }}
  .badge.pending {{ background: #2a2415; color: var(--yellow); }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}
  .table-wrap {{ overflow-x: auto; }}
  a {{ color: var(--blue); text-decoration: none; }}
  footer {{ margin-top: 2rem; text-align: center; color: #484f58; font-size: .8rem; }}
</style>
</head>
<body>
<h1>📊 Paper Trading Dashboard</h1>
<p class="subtitle">Kalshi AI Trading Bot — Signal tracker &amp; hypothetical P&amp;L · Updated {now}</p>

<div class="stats">
  <div class="stat"><div class="value">{stats['total_signals']}</div><div class="label">Total Signals</div></div>
  <div class="stat"><div class="value">{stats['settled']}</div><div class="label">Settled</div></div>
  <div class="stat"><div class="value">{stats['pending']}</div><div class="label">Pending</div></div>
  <div class="stat"><div class="value">{stats['win_rate']}%</div><div class="label">Win Rate</div></div>
  <div class="stat"><div class="value {'pos' if stats['total_pnl']>=0 else 'neg'}">${stats['total_pnl']:.2f}</div><div class="label">Total P&amp;L</div></div>
  <div class="stat"><div class="value">${stats['avg_return']:.4f}</div><div class="label">Avg Return / Trade</div></div>
  <div class="stat"><div class="value pos">${stats['best_trade']:.2f}</div><div class="label">Best Trade</div></div>
  <div class="stat"><div class="value neg">${stats['worst_trade']:.2f}</div><div class="label">Worst Trade</div></div>
</div>

<div class="chart-wrap">
  <canvas id="pnlChart"></canvas>
</div>

<div class="table-wrap">
<table>
<thead><tr>
  <th>Time</th><th>Market</th><th>Side</th><th>Entry</th><th>Conf</th>
  <th>Strategy</th><th>Outcome</th><th>P&amp;L</th><th>Reasoning</th>
</tr></thead>
<tbody>{rows_html if rows_html else '<tr><td colspan="9" style="text-align:center;padding:2rem;color:#8b949e;">No signals yet. Run the bot with <code>--paper</code> to start tracking.</td></tr>'}
</tbody>
</table>
</div>

<footer>
  Generated by <a href="https://github.com/ryanfrigo/kalshi-ai-trading-bot">kalshi-ai-trading-bot</a> · Paper trading mode · Not financial advice
</footer>

<script>
const data = {chart_json};
if (data.length > 0) {{
  new Chart(document.getElementById('pnlChart'), {{
    type: 'line',
    data: {{
      labels: data.map(d => d.x.slice(0,10)),
      datasets: [{{
        label: 'Cumulative P&L ($)',
        data: data.map(d => d.y),
        borderColor: data[data.length-1].y >= 0 ? '#3fb950' : '#f85149',
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 2,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 10 }}, grid: {{ color: '#21262d' }} }},
        y: {{ ticks: {{ color: '#8b949e', callback: v => '$'+v }}, grid: {{ color: '#21262d' }} }}
      }}
    }}
  }});
}}
</script>
</body>
</html>"""

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

    return html


def _trunc(s: str, n: int) -> str:
    if not s:
        return ""
    return s[:n] + "…" if len(s) > n else s


def _escape(s: str) -> str:
    if not s:
        return ""
    return s.replace('"', '&quot;').replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "paper_dashboard.html")
    generate_html(out)
    print(f"Dashboard generated: {out}")
