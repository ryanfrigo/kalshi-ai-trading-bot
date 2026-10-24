#!/usr/bin/env python
"""Compress log files older than a cutoff (never delete anything).

A bot that logs every cycle will eventually fill a disk, and a full disk fails the
trading loop *quietly*. Plain-text logs compress roughly 10-20x, so rotating them costs
nothing but an occasional gunzip and buys back the space.

Policy: gzip ``*.log`` older than ``--days`` (default 7), leaving ``latest.log`` alone so
the most recent run is always readable. **Nothing is ever deleted** — retention is a
separate decision, and these logs are part of the record.
"""
import argparse
import gzip
import os
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def rotate(root: Path, days: int = 7, now: float | None = None, keep: str = "latest.log") -> list:
    """Gzip ``*.log`` under ``root`` older than ``days``. Returns the names rotated.

    Pure enough to test: takes an explicit ``now`` and never deletes.
    """
    now = time.time() if now is None else now
    cutoff = now - days * 86400
    rotated = []
    for path in sorted(Path(root).rglob("*.log")):
        if path.name == keep or not path.is_file():
            continue
        if path.stat().st_mtime > cutoff:
            continue
        target = path.with_suffix(path.suffix + ".gz")
        with open(path, "rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        # only remove the plain file once the gzipped copy is safely written
        if target.exists() and target.stat().st_size > 0:
            os.remove(path)
            rotated.append(path.name)
    return rotated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(REPO / "logs"))
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    rotated = rotate(Path(args.dir), args.days)
    total = sum(p.stat().st_size for p in Path(args.dir).rglob("*.gz"))
    print(f"rotated {len(rotated)} log file(s) older than {args.days}d "
          f"(none deleted; {total / 1e6:.1f} MB now gzipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
