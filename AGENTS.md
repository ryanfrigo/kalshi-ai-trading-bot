# AGENTS.md

This repository is intended to be worked on by both humans and AI agents.
Use this file as the first-stop operating guide.

## Core Rules

- Work on a feature branch, not directly on `main`.
- Make one coherent change at a time.
- Keep secrets out of commits, logs, docs, and screenshots.
- Prefer small, reviewable pull requests with a clear test summary.
- If you change behavior, update the docs and tests together.

## Safe Editing Workflow

1. Inspect the current code and docs first.
2. Make the smallest change that solves the task.
3. Run the relevant tests locally.
4. Summarize behavior changes and test results in the PR.
5. Push the branch and open a PR.

## Local Entry Points

- `python cli.py run --paper` starts the paper bot.
- `python cli.py dashboard` opens the dashboard.
- `python cli.py status` prints portfolio and position status.
- `python paper_trader.py --loop` runs the signal logger in a loop.
- `python paper_trader.py --dashboard` regenerates the paper dashboard.

## Testing Expectations

- Run focused tests for the area you touched.
- Run the broader repo test set before opening a PR if the change is cross-cutting.
- Include both backend and frontend checks when the change touches the dashboard.
- Never rely on manual inspection alone if a test can verify the behavior.

## PR Expectations

- Explain what changed.
- Explain why the change was needed.
- List all relevant test commands and their results.
- Mention any known follow-up work or limitations.
- Include screenshots only when they add value for UI changes.

## For AI Agents

- Prefer non-mutating exploration first.
- Do not overwrite user work.
- Keep assumptions explicit.
- If a task needs a new file or doc, put it in the repo rather than leaving it in chat.
- When the repo has a dedicated future-plan document, update that instead of inventing a parallel plan.

