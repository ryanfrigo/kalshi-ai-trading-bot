# The Kalshi Bot Operator's Playbook

### Running an AI trading bot on Kalshi without lighting your money on fire

*A field guide from the maintainer of [kalshi-ai-trading-bot](https://github.com/ryanfrigo/kalshi-ai-trading-bot) — one of the most-starred open-source Kalshi bots on GitHub.*

> ⚠️ **Read this first.** Nothing here is financial advice, and no bot in this guide is guaranteed to make money. I ran one of these strategies live and drew my account down ~86% before I understood what I'm about to teach you. This guide is about *operating the software correctly* — getting it connected, keeping it alive, and reading what it's telling you. It will not pick winners for you. Paper-trade first. Only ever risk money you are fully prepared to lose.

> **Want the rest of the operator's manual?** This free playbook covers getting connected, the silence, and staying online. The paid companion — *Build & Run an AI Kalshi Bot* — covers AI spend, Kalshi API drift, the dashboard, and the paper-first protocol. → **https://kalshi-community.vercel.app/guide**

---

## Why this guide exists

Most people who clone the repo hit the same five walls, in the same order:

1. It won't authenticate (the private-key wall).
2. It runs but never places a trade ("is it broken?").
3. It dies the moment they close their laptop.
4. The AI bill arrives and there's nothing to show for it.
5. Kalshi changes an API field and everything breaks.

> The free field-by-field **Kalshi API map** (what the live responses actually contain, and why snapshot prices lie): **https://kalshi-community.vercel.app/reference/kalshi-fields**

None of these are exotic. They're the same questions in the issue tracker over and over. This guide is the answer key — the things I wish someone had handed me on day one.

---

## Chapter 1 — From clone to first heartbeat

The goal of day one is *not* to make money. It's to see a green `health` check and a paper trade land in the dashboard. That's it.

### 1.1 Get the two keys

You need exactly two things before anything works:

- **A Kalshi API key + private key** — from [kalshi.com/account/settings](https://kalshi.com/account/settings). Kalshi gives you an **API key ID** (a string) and a **private key** (an RSA `.pem` file it downloads to your machine). You need *both*. The `.pem` is the part everyone fumbles — see 1.3.
- **An OpenRouter key** — from [openrouter.ai](https://openrouter.ai/). Only needed for the AI directional strategy. The Safe Compounder strategy needs **no** LLM key at all, so if you just want to see the machine run, skip this for now.

### 1.2 Install

```bash
git clone https://github.com/ryanfrigo/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot
python setup_env.py        # creates .venv, installs everything
cp env.template .env       # then open .env and paste your keys
```

> **Use Python 3.12.** Newer point releases (3.13/3.14) can fail to build some pinned dependencies. If `setup_env.py` errors on a wheel build, that's almost always the cause — install 3.12 and point the venv at it.

### 1.3 The private-key wall (this causes ~90% of all 401 errors)

This is the single most common support question, so slow down here.

Your Kalshi `API_KEY` goes in `.env`. But the **private key is a file**, not an env var. The bot expects to find that `.pem` file **in the project root** — the same folder as `cli.py`. People paste the *contents* of the key into `.env`, or drop the file in the wrong directory, and get a wall of HTTP 401s.

Checklist:

- [ ] The `.pem` file Kalshi gave you is sitting in the project root (next to `cli.py`).
- [ ] Its filename matches what the config expects (check `src/config/settings.py` / `.env` for the key-path setting — don't rename it to something the bot isn't looking for).
- [ ] You did **not** paste the key's text into `.env`. `.env` holds the **API key ID** only.
- [ ] You're running commands **from the project root**, not from inside `src/` or your home folder. Wrong directory = "file not found" = 401.

### 1.4 Prove it's alive

```bash
python cli.py health
```

Green across the board means your keys, signing, and connectivity all work. **Do not move on until this passes.** Every downstream problem is harder to debug when you're not sure the foundation is solid.

### 1.5 Your first (paper) trade

```bash
python cli.py run --safe-compounder      # no LLM needed — pure math, dry-run preview
# or
python cli.py run --paper                # AI directional, paper mode
```

Then, in a second terminal:

```bash
python cli.py dashboard
```

If you see the dashboard populate, you've cleared the hardest part. Most people never get here cleanly — you just did.

---

## Chapter 2 — "It's been running for days and placed zero trades. Is it broken?"

Almost certainly **not broken — it's being conservative**, exactly as designed. This is the #2 source of confusion, and it's worth understanding *deeply*, because the instinct is to "fix" it by removing the very guardrails that are protecting you.

### 2.1 The bot is a sieve, on purpose

Every candidate market has to pass *all* of the filters before a single dollar moves. With the defaults, that means:

- **Confidence ≥ 45%** on the directional read,
- **Position ≤ 10%** of capital,
- **Sector concentration ≤ 30%**,
- **Drawdown circuit-breaker at 20%**,
- **quarter-Kelly** sizing on top of all that.

The **Safe Compounder** is even pickier: NO-side only, YES last ≤ 20¢, NO ask > 80¢, edge > 3¢, and it skips sports / entertainment / "mention" markets entirely. On a quiet day, *nothing* clears that bar. Zero trades is a valid, healthy output. A bot that trades constantly is usually a bot with its safeties off.

### 2.2 How to tell "conservative" from "actually broken"

Run the diagnostics instead of guessing:

```bash
python cli.py health      # connectivity still good?
python cli.py status      # what does it think it's holding?
python cli.py scores      # what did it score, and why did each market get rejected?
```

`scores` is the one that ends the anxiety: it shows you the markets it looked at and which filter each one failed. If you see it scoring markets and rejecting them on threshold, **it's working** — the market just didn't offer an edge worth taking.

It's *actually* broken only if: `health` fails, the logs show exceptions every cycle, or it stopped logging entirely (see the "silently paused" trap below).

### 2.3 The "silently paused" trap

There's a failure mode where a daily-limit or state file gets into a bad state and the bot quietly stops trading without crashing. If you've had activity and then *nothing* — not even rejections in the logs — check your daily AI-cost limit hasn't tripped and that the bot didn't pause itself. Restarting the cycle usually clears it.

### 2.4 If you want it to trade more (carefully)

Loosen **one** knob at a time, in `src/config/settings.py`, and watch the effect for a day before touching the next:

- Lower the confidence threshold a few points — **not** to zero.
- Widen the edge requirement on Safe Compounder cautiously.

Never disable the drawdown circuit-breaker or the position cap to "see more action." That's the exact move that turns a slow week into a blown account.

---

## Chapter 3 — Keeping it alive 24/7

A trading bot on your laptop dies when the laptop sleeps, when Wi-Fi drops, or when you close the lid. To run continuously you want it on a small always-on server. This is the walkthrough everyone asks for.

### 3.1 The cheapest reliable setup: a $4–6/mo VPS

A basic DigitalOcean / Hetzner / Linode droplet (1 GB RAM is plenty) runs this fine.

```bash
# On the droplet:
git clone https://github.com/ryanfrigo/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot
python setup_env.py
cp env.template .env        # paste your keys
# copy your Kalshi .pem into the project root (scp it up — never paste it into a chat or commit it)
python cli.py health        # confirm it works on the server before going further
```

### 3.2 Make it survive disconnects

The mistake is running `python cli.py run ...` over SSH and then closing the terminal — which kills the process. Two ways to detach it properly:

**Option A — `tmux` (simplest):**

```bash
tmux new -s bot
python cli.py run --safe-compounder --live --loop --interval 300
# detach with Ctrl-b then d   — the bot keeps running
# reattach later with:  tmux attach -t bot
```

**Option B — `systemd` (survives reboots, restarts on crash):** create a service unit that runs the same command with `Restart=always` and `WorkingDirectory` set to the project root. This is the "set it and forget it" option — the bot comes back automatically after a server reboot or a crash.

> The `--loop --interval 300` flags make it run a cycle every 5 minutes forever, instead of running once and exiting. Without `--loop`, your "24/7 bot" runs for one cycle and stops.

### 3.3 The two things people forget

- **Log rotation.** A bot logging every 5 minutes will eventually fill a small disk. Cap your log size (logrotate, or the bot's file-logging settings).
- **A kill switch you can reach from your phone.** Before you walk away, know how to stop it and flatten everything from an SSH session:

```bash
# stop the bot (Ctrl-c in the tmux session, or: systemctl stop your-bot)
python cli.py close-all --live    # limit-sell every open position at best bid
python cli.py status              # confirm
```

`close-all` queries Kalshi directly, so it works even if your local database is stale. Sells are limit-priced and may rest unfilled on thin books — check `status` after a minute.

---

## What's in the full guide

This free playbook covers getting connected, understanding the silence, and staying online — the three walls everyone hits. The complete **Build & Run an AI Kalshi Bot** guide also covers:

- **Controlling AI spend** — the daily-budget circuit breaker, avoiding the "one prediction ate my whole budget" trap, and picking a cheaper OpenRouter model without gutting quality.
- **Surviving Kalshi API drift** — the recurring field changes that break bots (`*_dollars` price fields, `volume_fp`, ticker collisions) and how to pin a release and recover in minutes.
- **Reading the dashboard like an operator** — what every metric means, when to intervene, and when to walk away.
- **The full paper-first protocol** I now follow after the 86% lesson.

**Get the full guide →** https://kalshi-community.vercel.app/guide

---

*Built something with the toolkit, or stuck on a wall this guide didn't cover? Open a thread in [Discussions](https://github.com/ryanfrigo/kalshi-ai-trading-bot/discussions). If this saved you an afternoon, a ⭐ on the [repo](https://github.com/ryanfrigo/kalshi-ai-trading-bot) helps other operators find it.*
