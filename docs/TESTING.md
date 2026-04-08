# Testing Guide

This document is the canonical testing checklist for the repo.
It is written so that both humans and AI coding assistants can use it directly.

## Test Strategy

- Run focused tests for the code you touched.
- Run the broader suite if the change affects shared behavior.
- Prefer deterministic unit tests for parsing, routing, and scoring logic.
- Use integration tests for pipeline and orchestration changes.
- Use browser-based checks when validating the dashboard.

## Backend Tests

Typical commands:

```bash
pytest tests/test_decide.py
pytest tests/test_ensemble.py
pytest tests/test_agents.py
pytest tests/test_live_order_execution.py
pytest tests/test_end_to_end.py
```

If you changed core orchestration or risk logic, run the full suite:

```bash
pytest
```

## Dashboard / Browser Checks

- Start the bot or paper signal logger.
- Open the dashboard in a browser.
- Verify the key panels show live updates.
- Check the browser console for errors and warnings.

If you are validating UI changes, capture the browser state before and after the first update cycle.

## What to Verify

- Paper mode runs without exceptions.
- Live mode still respects risk and category filters.
- News and sentiment paths can fail without breaking the decision loop.
- Paper signals are logged and settled correctly.
- Dashboard stats match the underlying SQLite or API-backed state.

## Performance Checks

When a change touches scanning, aggregation, or model routing:

- Measure whether the change increases end-to-end decision latency.
- Confirm the bot still completes a full analysis cycle within the expected interval.
- Check that cache TTLs are respected.
- Confirm the new code does not introduce repeated or duplicate work.

## PR Acceptance Criteria

- Tests that cover the changed behavior pass locally.
- The PR description lists the exact commands run and the results.
- If any test was skipped, the reason is documented.
- Any new failure mode has a clear fallback or a logged error path.

