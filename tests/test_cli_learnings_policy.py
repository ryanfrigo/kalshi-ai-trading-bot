"""`cli.py learnings` and `cli.py policy`, on fakes and local files only.

The Kalshi fake has one read method, so an order attempt would raise.
"""
import argparse
import json
from pathlib import Path

import pytest

import cli
import src.clients.kalshi_client as kc
from src.agent.journal import DEFAULT_JOURNAL_PATH, load_journal, write_journal
from src.agent.learnings import (DEFAULT_LEARNINGS_PATH, calibration_table, edge_breakdown,
                                 load_learnings, reconcile_outcomes)
from src.agent.policy import build_settled_records, derive_policy
from src.agent.settle import DEFAULT_SETTLEMENTS_PATH, settlement_pnl


def _raw(ticker, result="no", no="10", cost="9", revenue=1000, fee="0"):
    return {"ticker": ticker, "event_ticker": ticker.split("-")[0], "yes_count_fp": "0", "no_count_fp": no,
            "market_result": result, "yes_total_cost_dollars": "0", "no_total_cost_dollars": cost,
            "fee_cost": fee, "revenue": revenue, "settled_time": "2026-10-02T00:00:00Z"}


def _buy(ticker, category="KXA", est=0.95, price=0.90, count=10, oid=None, action="buy", **extra):
    return {"ts": "2026-10-01T00:00:00Z", "ticker": ticker, "side": "no", "action": action, "count": count,
            "price": price, "est_prob": est, "edge": 0.05, "category": category,
            "order_id": oid or f"o-{ticker}", "outcome": None, **extra}


class FakeKalshi:
    """Serves settlements in pages. Reads only."""

    def __init__(self, rows=(), page=200, error=None):
        self.rows, self.page, self.error = list(rows), page, error
        self.closed = 0

    async def _make_authenticated_request(self, method, path, params=None):
        assert (method, path) == ("GET", "/trade-api/v2/portfolio/settlements")
        if self.error:
            raise self.error
        start = int(params.get("cursor") or 0)
        rows = self.rows[start:start + self.page]
        more = start + self.page < len(self.rows)
        return {"settlements": rows, "cursor": str(start + self.page) if more else ""}

    async def close(self):
        self.closed += 1


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def _run(client, dry=False, as_json=False):
        monkeypatch.setattr(kc, "KalshiClient", lambda: client)
        cli.cmd_learnings(argparse.Namespace(dry=dry, json=as_json))
        out = capsys.readouterr().out
        return json.loads(out) if as_json else out

    return _run


def _losers(n=5):
    """n settled losers in one category: enough to flag a stop."""
    return [_raw(f"KXL-{i}", result="yes", revenue=0) for i in range(n)]


def _loser_journal(n=5):
    return [_buy(f"KXL-{i}", category="KXL") for i in range(n)]


def test_learnings_fills_outcomes_and_appends_candidates(run):
    write_journal(_loser_journal())
    out = run(FakeKalshi(_losers()), as_json=True)
    assert out["dry"] is False and out["reconciled"] == 5 and out["settled_total"] == 5
    records = load_journal()
    assert all(r["outcome"]["won"] is False and r["outcome"]["pnl"] == -9.0 for r in records)
    kinds = {c["kind"] for c in load_learnings()}
    assert kinds == {"stop", "overconfidence"}
    assert len(load_learnings()) == len(out["new_learnings"]) == len(out["candidates"])


def test_learnings_second_run_writes_nothing_new(run):
    write_journal(_loser_journal())
    run(FakeKalshi(_losers()), as_json=True)
    before = Path(DEFAULT_JOURNAL_PATH).read_bytes(), Path(DEFAULT_LEARNINGS_PATH).read_bytes()
    out = run(FakeKalshi(_losers()), as_json=True)
    assert out["reconciled"] == 0 and out["new_learnings"] == [] and out["candidates"]
    assert before == (Path(DEFAULT_JOURNAL_PATH).read_bytes(), Path(DEFAULT_LEARNINGS_PATH).read_bytes())


def test_learnings_dry_previews_the_same_and_writes_nothing(run):
    write_journal(_loser_journal())
    before = Path(DEFAULT_JOURNAL_PATH).read_bytes()
    dry = run(FakeKalshi(_losers()), dry=True, as_json=True)
    assert Path(DEFAULT_JOURNAL_PATH).read_bytes() == before
    assert not Path(DEFAULT_LEARNINGS_PATH).exists()
    wet = run(FakeKalshi(_losers()), as_json=True)
    assert dry["dry"] is True and dry["reconciled"] == wet["reconciled"] == 5
    assert dry["new_learnings"] == wet["new_learnings"] and dry["new_learnings"]


def test_learnings_dry_filters_out_what_the_file_already_holds(run):
    write_journal(_loser_journal())
    run(FakeKalshi(_losers()), as_json=True)
    out = run(FakeKalshi(_losers()), dry=True, as_json=True)
    assert out["candidates"] and out["new_learnings"] == []


def test_learnings_failed_read_closes_the_client_and_writes_nothing(run):
    write_journal(_loser_journal())
    before = Path(DEFAULT_JOURNAL_PATH).read_bytes()
    client = FakeKalshi(error=RuntimeError("503"))
    with pytest.raises(RuntimeError, match="503"):
        run(client)
    assert client.closed == 1
    assert Path(DEFAULT_JOURNAL_PATH).read_bytes() == before
    assert not Path(DEFAULT_LEARNINGS_PATH).exists()


def test_learnings_closes_the_client_on_success(run):
    client = FakeKalshi()
    run(client, as_json=True)
    assert client.closed == 1


def test_learnings_reads_past_the_first_page(run):
    rows = _losers(5)
    write_journal(_loser_journal(5))
    out = run(FakeKalshi(rows, page=2), as_json=True)
    assert out["reconciled"] == 5


def test_learnings_skips_a_settlement_with_no_position(run):
    write_journal([_buy("KXA-1")])
    out = run(FakeKalshi([_raw("KXA-1", no="0")]), as_json=True)
    assert out["reconciled"] == 0 and load_journal()[0]["outcome"] is None


@pytest.mark.parametrize("flags", [{"voided": True}, {"status": "pending", "order_outcome": "outcome_unknown"}])
def test_learnings_never_settles_a_void_or_pending_record(run, flags):
    write_journal([_buy("KXA-1", **flags)])
    out = run(FakeKalshi([_raw("KXA-1")]), as_json=True)
    assert out["reconciled"] == 0 and out["settled_total"] == 0
    assert load_journal()[0]["outcome"] is None


def test_learnings_doesnt_rewrite_the_journal_when_nothing_settled(run, monkeypatch):
    write_journal([_buy("KXA-1")])
    import src.agent.journal as journal
    monkeypatch.setattr(journal, "write_journal", lambda *a, **k: pytest.fail("no write expected"))
    run(FakeKalshi(), as_json=True)


def test_learnings_human_output_on_an_empty_journal(run):
    out = run(FakeKalshi())
    assert "Reconciled this run: 0 new outcome(s)" in out
    assert "Settled journal trades: 0 / 0" in out
    assert "(no settled trades with est_prob yet)" in out
    assert "(none flagged" in out
    assert "Appended 0 new learning(s)" in out
    assert "DRY" not in out


def test_learnings_human_output_shows_tables_and_tags(run):
    write_journal(_loser_journal())
    first = run(FakeKalshi(_losers()))
    assert "Reconciled this run: 5 new outcome(s)" in first and "Settled journal trades: 5 / 5" in first
    assert "95%" in first  # predicted, against 0% actual
    assert "by category:" in first and "by side:" in first and "by method:" not in first
    assert "KXL" in first and "$ -45.00" in first
    assert "[NEW] (" in first and "[seen]" not in first
    second = run(FakeKalshi(_losers()))
    assert "[seen] (" in second and "[NEW]" not in second
    assert "Appended 0 new learning(s)" in second


def test_learnings_human_counts_only_settled_records(run):
    write_journal([_buy("KXA-1"), _buy("KXA-2")])
    assert "Settled journal trades: 1 / 2" in run(FakeKalshi([_raw("KXA-1")]))


def test_learnings_human_dry_says_so_and_has_no_appended_line(run):
    out = run(FakeKalshi(), dry=True)
    assert "(DRY: no writes)" in out and "Appended" not in out


def test_learnings_human_prints_na_for_missing_edge_and_price(run):
    rec = _buy("KXA-1")
    rec.update(edge=None, price=None, est_prob=None)
    write_journal([rec])
    out = run(FakeKalshi([_raw("KXA-1")]))
    assert out.count("n/a") == 4  # category and side rows, entry edge and real edge each


def test_learnings_human_shows_a_method_table_when_a_record_has_one(run):
    write_journal([_buy("KXA-1", method="fade")])
    out = run(FakeKalshi([_raw("KXA-1")]))
    assert "by method:" in out and "fade" in out


# --- _render_policy


def _policy(**over):
    return {"meta": {"generated_date": "2026-11-05", "settled_n": 12}, "blocks": [], "warnings": [],
            "haircuts": [], **over}


def test_render_empty_policy_says_none_for_each_section(capsys):
    cli._render_policy(_policy())
    out = capsys.readouterr().out
    assert "EDGE POLICY — 2026-11-05" in out and "derived from 12 settled trades" in out
    assert "(none — no n>=5 category/method is net-negative)" in out
    assert "(none — every n>=5 band is well-calibrated)" in out
    assert out.count("(none") == 3


def test_render_copes_with_missing_meta_and_sections(capsys):
    cli._render_policy({})
    out = capsys.readouterr().out
    assert "EDGE POLICY — ?" in out and "derived from 0 settled trades" in out


def test_render_lists_worst_block_and_warning_first(capsys):
    cli._render_policy(_policy(
        blocks=[{"dimension": "category", "label": "SMALL", "n": 5, "pnl": -1.0},
                {"dimension": "method", "label": "BIG", "n": 7, "pnl": -90.5}],
        warnings=[{"label": "yes", "n": 5, "pnl": -2.0}, {"label": "no", "n": 6, "pnl": -8.0}]))
    out = capsys.readouterr().out
    assert out.index("method=BIG") < out.index("category=SMALL")
    assert "n=  7" in out and "pnl=$   -90.50" in out
    assert out.index("side=no") < out.index("side=yes")


def test_render_cuts_a_long_label(capsys):
    cli._render_policy(_policy(blocks=[{"dimension": "category", "label": "X" * 40, "n": 5, "pnl": -1.0}]))
    out = capsys.readouterr().out
    assert "X" * 22 in out and "X" * 23 not in out


def test_render_shows_a_haircut(capsys):
    cli._render_policy(_policy(haircuts=[{"band": "0.90-1.00", "n": 8, "gap": 0.25, "shrink_to": 0.7}]))
    out = capsys.readouterr().out
    assert "0.90-1.00" in out and "overconfident by 25%" in out and "shrink est_prob to 0.70" in out


# --- cmd_policy


@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(kc, "KalshiClient", lambda: pytest.fail("policy must not build a client"))


def _policy_out(capsys, **ns):
    cli.cmd_policy(argparse.Namespace(**{"demo": False, "json": False, **ns}))
    return capsys.readouterr().out


def test_policy_demo_shows_all_three_mechanisms(capsys, no_network):
    policy = json.loads(_policy_out(capsys, demo=True, json=True))
    assert policy["blocks"] and policy["warnings"] and policy["haircuts"]
    assert policy["meta"]["settled_n"] > 0


def test_policy_demo_works_from_any_directory(tmp_path, monkeypatch, capsys, no_network):
    """It read a cwd-relative path, so elsewhere it found no files and printed a clean bill of health."""
    expected = json.loads(_policy_out(capsys, demo=True, json=True))
    monkeypatch.chdir(tmp_path)
    assert json.loads(_policy_out(capsys, demo=True, json=True)) == expected


def test_policy_with_no_local_files_has_no_opinion(tmp_path, monkeypatch, capsys, no_network):
    monkeypatch.chdir(tmp_path)
    policy = json.loads(_policy_out(capsys, json=True))
    assert (policy["blocks"], policy["warnings"], policy["haircuts"]) == ([], [], [])
    assert policy["meta"]["settled_n"] == 0


def test_policy_reads_the_local_settlement_log_and_journal(tmp_path, monkeypatch, capsys, no_network):
    monkeypatch.chdir(tmp_path)
    write_journal(_loser_journal())
    log = Path(DEFAULT_SETTLEMENTS_PATH)
    log.write_text("".join(json.dumps(settlement_pnl(r)) + "\n" for r in _losers()))
    policy = json.loads(_policy_out(capsys, json=True))
    assert policy["meta"]["settled_n"] == 5
    assert [(b["dimension"], b["label"]) for b in policy["blocks"]] == [("category", "KXL")]
    assert [w["label"] for w in policy["warnings"]] == ["no"]
    assert policy["haircuts"]


def test_policy_doesnt_write_the_journal_or_a_policy(tmp_path, monkeypatch, capsys, no_network):
    monkeypatch.chdir(tmp_path)
    write_journal(_loser_journal())
    log = Path(DEFAULT_SETTLEMENTS_PATH)
    log.write_text("".join(json.dumps(settlement_pnl(r)) + "\n" for r in _losers()))
    before = Path(DEFAULT_JOURNAL_PATH).read_bytes()
    _policy_out(capsys)
    assert Path(DEFAULT_JOURNAL_PATH).read_bytes() == before
    assert not Path("data/runtime/edge_policy.json").exists()


def test_policy_human_output_ends_with_the_no_gate_note(tmp_path, monkeypatch, capsys, no_network):
    monkeypatch.chdir(tmp_path)
    assert "No active gate saved yet" in _policy_out(capsys)


def _saved(policy):
    gate = Path("data/runtime/edge_policy.json")
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps(policy))


def test_policy_says_when_the_saved_gate_differs(tmp_path, monkeypatch, capsys, no_network):
    monkeypatch.chdir(tmp_path)
    _saved(_policy())
    out = _policy_out(capsys)
    assert "saved active gate differs" in out and "No active gate" not in out


def test_policy_is_quiet_when_the_saved_gate_matches(tmp_path, monkeypatch, capsys, no_network):
    monkeypatch.chdir(tmp_path)
    _saved(json.loads(_policy_out(capsys, json=True)))
    out = _policy_out(capsys)
    assert "differs" not in out and "No active gate" not in out


# --- Open: one position counted once per journal record. Fixing it moves the Edge Policy's
# blocks, so it changes what the bot trades, and that's the owner's call.

_DOUBLE = "every record on a ticker takes the whole settlement's P&L, so one position counts once per record"


@pytest.mark.xfail(strict=True, reason=_DOUBLE)
def test_two_buys_of_one_position_book_its_pnl_once():
    settlement = settlement_pnl(_raw("KXA-1", result="yes", no="15", cost="13.5", revenue=0))
    records, _ = reconcile_outcomes([_buy("KXA-1", oid="a"), _buy("KXA-1", count=5, oid="b")], [settlement])
    assert edge_breakdown(records)["by_category"]["KXA"]["pnl"] == settlement["pnl"]


def _derive(journal, raws):
    settlements = [settlement_pnl(r) for r in raws]
    records = build_settled_records(journal, settlements)
    return derive_policy(edge_breakdown(records), calibration_table(records), date="2026-11-05")


def _winners(n=4):
    return [_raw(f"KXB-{i}") for i in range(n)]  # each +$1.00


def _winner_journal(n=4):
    return [_buy(f"KXB-{i}", category="KXB") for i in range(n)]


@pytest.mark.xfail(strict=True, reason=_DOUBLE + ": a net winner gets a hard block")
def test_a_category_that_made_money_isnt_blocked():
    loser = _raw("KXB-L", result="yes", no="30", cost="3", revenue=0)  # -$3.00, so the category is +$1.00
    journal = _winner_journal() + [_buy("KXB-L", category="KXB", oid=f"L{i}", price=0.1) for i in range(3)]
    assert _derive(journal, _winners() + [loser])["blocks"] == []


@pytest.mark.xfail(strict=True, reason=_DOUBLE + ": a net loser escapes its block")
def test_a_category_that_lost_money_is_blocked():
    winner = _raw("KXC-W", no="30", cost="27", revenue=3000)  # +$3.00
    losers = [_raw(f"KXC-{i}", result="yes", revenue=0, cost="1.25") for i in range(4)]  # -$5.00 in all
    journal = [_buy("KXC-W", category="KXC", oid=f"W{i}") for i in range(3)]
    journal += [_buy(f"KXC-{i}", category="KXC") for i in range(4)]
    blocks = _derive(journal, [winner] + losers)["blocks"]
    assert [b["label"] for b in blocks] == ["KXC"]


@pytest.mark.xfail(strict=True, reason="a sell record takes the settlement's outcome like a buy")
def test_a_sell_record_doesnt_take_the_settlement_outcome():
    settlement = settlement_pnl(_raw("KXA-1", result="yes", no="5", cost="4.5", revenue=0))
    records, newly = reconcile_outcomes(
        [_buy("KXA-1", oid="a"), _buy("KXA-1", count=5, oid="s", action="sell")], [settlement])
    assert newly == 1 and records[1]["outcome"] is None
