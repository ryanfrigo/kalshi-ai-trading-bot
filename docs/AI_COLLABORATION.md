# AI Collaboration Guide

This repository is designed to be workable by coding assistants as well as humans.

## Agent Workflow

1. Read `AGENTS.md` and the relevant code path first.
2. Inspect existing docs before creating new ones.
3. Keep changes scoped to one concern when possible.
4. Run the appropriate tests before claiming success.
5. Open a PR with a concise explanation and a real test summary.

## Branching and PRs

- Always work on a branch.
- Do not push directly to `main`.
- Use a PR for every code change.
- Keep PRs focused and easy to review.

Suggested workflow:

```bash
git checkout -b codex/<short-topic>
pytest
gh pr create --draft --fill
```

## What to Include in a PR

- Short description of the user-visible change.
- Technical summary of the implementation.
- Commands used for testing.
- Test results, including failures fixed during the work.
- Follow-up work if the change only solves part of the problem.

## When Changing Behavior

- Update docs and tests together.
- Call out any compatibility impact.
- If the change affects the dashboard or runtime behavior, include a browser check.
- If the change affects signal quality or performance, include timing or throughput observations when available.

## Safety

- Never leak API keys or private credentials.
- Do not overwrite work you did not create.
- If the repo has a stale process or old branch tip, verify the running state before assuming the code is live.
