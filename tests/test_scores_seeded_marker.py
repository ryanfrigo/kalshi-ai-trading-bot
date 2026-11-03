from src.strategies.category_scorer import KNOWN_DATA, CategoryScorer, is_seeded_row


def _row(cat, wins, total, pnl, score=50.0):
    return {"category": cat, "score": score, "win_count": wins, "total_count": total,
            "total_pnl": pnl}


def test_seeded_rows_marked_and_footer_shown(tmp_path):
    d = KNOWN_DATA["FED"]
    rows = [_row("FED", d["wins"], d["total"], d["total_pnl"]), _row("MYCAT", 3, 5, 1.0)]
    out = CategoryScorer(db_path=str(tmp_path / "x.db")).format_scores_table(rows)
    assert "FED*" in out and "MYCAT*" not in out
    assert "starting estimates hard-coded in category_scorer.py (KNOWN_DATA)" in out


def test_changed_counts_not_marked_and_no_footer(tmp_path):
    d = KNOWN_DATA["FED"]
    row = _row("FED", d["wins"] + 1, d["total"] + 1, d["total_pnl"])
    assert not is_seeded_row(row)
    out = CategoryScorer(db_path=str(tmp_path / "x.db")).format_scores_table([row])
    assert "*" not in out
