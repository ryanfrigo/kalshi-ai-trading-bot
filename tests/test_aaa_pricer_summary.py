"""The score summary must show the EFFECTIVE sample, not just the row count.

One realized print generates a whole strike ladder, so scores.jsonl rows are heavily
correlated. A summary that reports only the row count invites the exact "assert skill
from a handful of settlements" error this toolkit refuses, so `aaa_pricer.py summary`
must print the distinct-target (event) count beside every Brier comparison.
"""
import json
from pathlib import Path

import scripts.aaa_pricer as ap


def _write(rows, tmp_path: Path) -> Path:
    d = tmp_path / "data" / "aaa"
    d.mkdir(parents=True)
    p = d / "scores.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def _row(series, target, ticker, fair, book, y):
    return {"series": series, "target": target, "ticker": ticker,
            "fair_yes": fair, "book_mid": book, "y": y,
            "brier_model": (fair - y) ** 2, "brier_book": (book - y) ** 2}


def test_summary_reports_events_not_just_rows(tmp_path, monkeypatch, capsys):
    # 3 strikes x 2 prints = 6 rows, but only 2 independent events.
    rows = [
        _row("KXTEST", "2026-01-01", f"KXTEST-{i}", 0.9, 0.8, 1.0) for i in range(3)
    ] + [
        _row("KXTEST", "2026-01-02", f"KXTEST-B{i}", 0.2, 0.5, 0.0) for i in range(3)
    ]
    monkeypatch.setattr(ap, "REPO", tmp_path)
    _write(rows, tmp_path)

    assert ap.cmd_summary() == 0
    out = capsys.readouterr().out
    assert "events" in out
    assert "KXTEST" in out
    assert "6" in out and "2" in out      # six rows, two events
