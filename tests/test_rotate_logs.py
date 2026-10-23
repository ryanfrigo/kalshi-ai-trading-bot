"""Log rotation must compress, never delete, and must skip the live log."""
import os
import time
from pathlib import Path

import scripts.rotate_logs as rl


def _mk(p: Path, name: str, age_days: float):
    p.mkdir(parents=True, exist_ok=True)
    f = p / name
    f.write_text("x" * 5000)
    t = time.time() - age_days * 86400
    os.utime(f, (t, t))
    return f


def test_rotates_old_logs_and_keeps_the_plain_file_content(tmp_path):
    old = _mk(tmp_path, "trading_system_old.log", 30)
    keep = _mk(tmp_path, "latest.log", 30)          # never rotated
    fresh = _mk(tmp_path, "today.log", 0.5)         # too new

    rotated = rl.rotate(tmp_path, days=7)

    assert rotated == ["trading_system_old.log"]
    assert not old.exists() and old.with_suffix(".log.gz").exists()
    assert keep.exists() and fresh.exists()          # untouched
    # content survives the round trip
    import gzip
    assert gzip.open(old.with_suffix(".log.gz"), "rt").read() == "x" * 5000


def test_nested_directories_are_covered(tmp_path):
    _mk(tmp_path / "daily_runs", "daily_20260101_000000.log", 40)
    assert rl.rotate(tmp_path, days=7) == ["daily_20260101_000000.log"]
