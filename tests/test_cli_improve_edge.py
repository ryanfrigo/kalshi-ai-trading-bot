"""`cli.py improve` and `cli.py edge`, on fakes and local files only.

The Kalshi fake has one read method, so an order attempt would raise.
"""
import argparse
import json
from pathlib import Path

import pytest

import cli
import src.clients.kalshi_client as kc
from src.agent.edge import edge_report
from src.agent.journal import DEFAULT_JOURNAL_PATH, load_journal, write_journal
from src.agent.policy import DEFAULT_POLICY_PATH, load_policy
from src.agent.settle import DEFAULT_SETTLEMENTS_PATH, load_settlements, settlement_pnl


def _raw(ticker, result="yes", revenue=0):
    """A NO position of 10 bought for $9. It loses $9 unless revenue says otherwise."""
    return {"ticker": ticker, "event_ticker": ticker.split("-")[0], "yes_count_fp": "0", "no_count_fp": "10",
            "market_result": result, "yes_total_cost_dollars": "0", "no_total_cost_dollars": "9",
            "fee_cost": "0", "revenue": revenue, "settled_time": "2026-10-02T00:00:00Z"}


def _buy(ticker, category="KXL", est=0.95, **extra):
    return {"ts": "2026-10-01T00:00:00Z", "ticker": ticker, "side": "no", "action": "buy", "count": 10,
            "price": 0.90, "est_prob": est, "edge": 0.05, "category": category,
            "order_id": f"o-{ticker}", "outcome": None, **extra}


JOURNAL = [_buy(f"KXL-{i}") for i in range(5)]
LOSSES = [_raw(f"KXL-{i}") for i in range(5)]
WINS = [_raw(f"KXL-{i}", result="no", revenue=1000) for i in range(5)]


class FakeKalshi:
    """Serves settlements. Reads only."""

    def __init__(self, rows=(), error=None):
        self.rows, self.error = list(rows), error
        self.closed = 0

    async def _make_authenticated_request(self, method, path, params=None):
        assert (method, path) == ("GET", "/trade-api/v2/portfolio/settlements")
        if self.error:
            raise self.error
        return {"settlements": self.rows, "cursor": ""}

    async def close(self):
        self.closed += 1


DOWN = RuntimeError("503 from exchange")


@pytest.fixture
def improve(tmp_path, monkeypatch, capsys):
    """Run cmd_improve. Returns (stdout or parsed JSON, stderr)."""
    monkeypatch.chdir(tmp_path)

    def _run(client, dry=False, as_json=False):
        monkeypatch.setattr(kc, "KalshiClient", lambda: client)
        cli.cmd_improve(argparse.Namespace(dry=dry, json=as_json))
        cap = capsys.readouterr()
        return (json.loads(cap.out) if as_json else cap.out), cap.err

    return _run


@pytest.fixture
def edge(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def _run(client, as_json=False):
        monkeypatch.setattr(kc, "KalshiClient", lambda: client)
        cli.cmd_edge(argparse.Namespace(json=as_json))
        out = capsys.readouterr().out
        return json.loads(out) if as_json else out

    return _run


def _labels(policy):
    return [b["label"] for b in policy["blocks"]]


def _save_gate(body):
    p = Path(DEFAULT_POLICY_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body if isinstance(body, str) else json.dumps(body))


def _wipe_records():
    Path(DEFAULT_SETTLEMENTS_PATH).unlink()
    Path(DEFAULT_JOURNAL_PATH).unlink()


# ---------------------------------------------------------------- improve: normal run

def test_improve_writes_settlements_journal_and_gate_and_closes_client(improve):
    write_journal(JOURNAL)
    client = FakeKalshi(LOSSES)
    out, err = improve(client)
    assert client.closed == 1
    assert len(load_settlements()) == 5
    assert all(r["outcome"] for r in load_journal())
    assert _labels(load_policy()) == ["KXL"]
    assert "LOOP: pulled 5 settlement(s), reconciled 5 new outcome(s)" in out
    assert "+ BLOCK category=KXL (n=5, $-45.00)" in out
    assert f"Saved active gate → {DEFAULT_POLICY_PATH}" in out
    assert err == ""


def test_improve_second_run_reports_no_change_and_no_duplicate_rows(improve):
    write_journal(JOURNAL)
    improve(FakeKalshi(LOSSES))
    out, _ = improve(FakeKalshi(LOSSES))
    assert len(load_settlements()) == 5
    assert "reconciled 0 new outcome(s)" in out
    assert "(no change" in out


def test_improve_json_output_carries_the_run(improve):
    write_journal(JOURNAL)
    data, _ = improve(FakeKalshi(LOSSES), as_json=True)
    assert data["dry"] is False
    assert data["gate_kept"] is False
    assert data["settlements_pulled"] == 5
    assert data["reconciled"] == 5
    assert [b["label"] for b in data["diff"]["added_blocks"]] == ["KXL"]
    assert _labels(data["policy"]) == ["KXL"]


def test_improve_with_nothing_at_all_saves_an_empty_gate(improve):
    out, err = improve(FakeKalshi())
    assert load_policy()["blocks"] == []
    assert "pulled 0 settlement(s)" in out
    assert err == ""


def test_improve_shows_a_haircut_that_moved(improve):
    _save_gate({"meta": {}, "blocks": [], "warnings": [],
                "haircuts": [{"band": "90-100%", "lo": 0.9, "hi": 1.0, "shrink_to": 0.5, "n": 5, "gap": 0.1}]})
    out, _ = improve(FakeKalshi())
    assert "~ haircut 90-100%: shrink 0.5 → None" in out


# ---------------------------------------------------------------- improve: failed pull

def test_failed_pull_note_goes_to_stderr_so_json_stays_parseable(improve):
    write_journal(JOURNAL)
    client = FakeKalshi(error=DOWN)
    data, err = improve(client, as_json=True)
    assert data["settlements_pulled"] == 0
    assert "live settlement pull unavailable: 503 from exchange" in err
    assert client.closed == 1


def test_failed_pull_uses_the_local_log(improve):
    write_journal(JOURNAL)
    improve(FakeKalshi(LOSSES))
    out, _ = improve(FakeKalshi(error=DOWN))
    assert "pulled 0 settlement(s)" in out
    assert _labels(load_policy()) == ["KXL"]


def test_client_that_will_not_build_falls_back_too(improve, monkeypatch, capsys):
    def boom():
        raise ValueError("no keys")

    monkeypatch.setattr(kc, "KalshiClient", boom)
    cli.cmd_improve(argparse.Namespace(dry=False, json=False))
    assert "no keys" in capsys.readouterr().err


# ---------------------------------------------------------------- improve: --dry

def test_dry_previews_the_run_a_real_one_would_make(improve):
    write_journal(JOURNAL)
    dry, _ = improve(FakeKalshi(LOSSES), dry=True, as_json=True)
    wet, _ = improve(FakeKalshi(LOSSES), as_json=True)
    for key in ("settlements_pulled", "reconciled", "gate_kept", "diff", "policy"):
        assert dry[key] == wet[key], key
    assert _labels(dry["policy"]) == ["KXL"]


def test_dry_writes_nothing(improve):
    write_journal(JOURNAL)
    before = Path(DEFAULT_JOURNAL_PATH).read_bytes()
    out, _ = improve(FakeKalshi(LOSSES), dry=True)
    assert "DRY: nothing written" in out
    assert "Saved active gate" not in out
    assert Path(DEFAULT_JOURNAL_PATH).read_bytes() == before
    assert not Path(DEFAULT_SETTLEMENTS_PATH).exists()
    assert not Path(DEFAULT_POLICY_PATH).exists()


def test_dry_does_not_count_a_row_already_in_the_log_twice(improve):
    """No journal: these rows reach the policy as settlement records, unmerged."""
    improve(FakeKalshi(LOSSES))
    dry, _ = improve(FakeKalshi(LOSSES), dry=True, as_json=True)
    assert dry["policy"]["meta"]["settled_n"] == 5
    assert dry["diff"]["added_blocks"] == dry["diff"]["removed_blocks"] == []


def test_dry_keeps_the_logged_row_when_a_pull_disagrees_as_a_real_run_does(improve):
    write_journal(JOURNAL)
    improve(FakeKalshi(LOSSES))
    dry, _ = improve(FakeKalshi(WINS), dry=True, as_json=True)
    wet, _ = improve(FakeKalshi(WINS), as_json=True)
    assert _labels(dry["policy"]) == _labels(wet["policy"]) == ["KXL"]


def test_dry_adds_only_the_rows_the_log_lacks(improve):
    write_journal(JOURNAL)
    improve(FakeKalshi(LOSSES[:3]))
    dry, _ = improve(FakeKalshi(LOSSES), dry=True, as_json=True)
    assert dry["policy"]["meta"]["settled_n"] == 5
    assert len(load_settlements()) == 3


# ---------------------------------------------------------------- improve: the gate must not open on lost data

def test_lost_records_and_a_failed_pull_keep_the_block(improve):
    write_journal(JOURNAL)
    improve(FakeKalshi(LOSSES))
    saved = Path(DEFAULT_POLICY_PATH).read_bytes()
    _wipe_records()
    out, err = improve(FakeKalshi(error=DOWN))
    assert Path(DEFAULT_POLICY_PATH).read_bytes() == saved
    assert "saved gate kept: it came from 5 settled trades, this run has 0" in err
    assert DEFAULT_POLICY_PATH in err
    assert f"Kept the saved gate → {DEFAULT_POLICY_PATH}" in out
    assert "Saved active gate" not in out
    assert "unblock" not in out
    assert "category=KXL" in out  # the gate shown is the one still in force


def test_a_kept_gate_file_is_not_rewritten_at_all(improve):
    _save_gate('{"meta": {"settled_n": 5},   "blocks": []}')
    improve(FakeKalshi())
    assert Path(DEFAULT_POLICY_PATH).read_text() == '{"meta": {"settled_n": 5},   "blocks": []}'


def test_kept_gate_in_json_says_so_and_shows_the_gate_in_force(improve):
    write_journal(JOURNAL)
    improve(FakeKalshi(LOSSES))
    _wipe_records()
    data, _ = improve(FakeKalshi(error=DOWN), as_json=True)
    assert data["gate_kept"] is True
    assert _labels(data["policy"]) == ["KXL"]
    assert data["diff"]["removed_blocks"] == []


def test_dry_previews_a_kept_gate_without_writing(improve):
    write_journal(JOURNAL)
    improve(FakeKalshi(LOSSES))
    _wipe_records()
    data, err = improve(FakeKalshi(error=DOWN), dry=True, as_json=True)
    assert data["gate_kept"] is True
    assert "saved gate kept" in err
    assert not Path(DEFAULT_SETTLEMENTS_PATH).exists()


def test_deleting_the_gate_file_rebuilds_from_what_is_left(improve):
    write_journal(JOURNAL)
    improve(FakeKalshi(LOSSES))
    _wipe_records()
    improve(FakeKalshi(error=DOWN))
    Path(DEFAULT_POLICY_PATH).unlink()
    out, err = improve(FakeKalshi(error=DOWN))
    assert load_policy()["blocks"] == []
    assert "Saved active gate" in out
    assert err.count("kept") == 0


def test_equal_settled_count_still_replaces_the_gate(improve):
    write_journal(JOURNAL)
    _save_gate({"meta": {"settled_n": 5}, "blocks": [{"dimension": "category", "label": "OLD", "n": 9, "pnl": -1.0}],
                "warnings": [], "haircuts": []})
    out, _ = improve(FakeKalshi(LOSSES))
    assert _labels(load_policy()) == ["KXL"]
    assert "- unblock category=OLD" in out
    assert "Saved active gate" in out


def test_more_settled_trades_can_lift_a_block(improve):
    """A real unblock is not a loss of data, so it still goes through."""
    write_journal(JOURNAL)
    _save_gate({"meta": {"settled_n": 3}, "blocks": [{"dimension": "category", "label": "OLD", "n": 3, "pnl": -1.0}],
                "warnings": [], "haircuts": []})
    improve(FakeKalshi(WINS))
    assert load_policy()["blocks"] == []
    assert load_policy()["meta"]["settled_n"] == 5


@pytest.mark.parametrize("meta", [
    "absent", {}, {"settled_n": None}, {"settled_n": "9"}, {"settled_n": True}, {"settled_n": 9.5}, 5,
])
def test_a_gate_with_no_usable_settled_count_gets_no_floor(improve, meta):
    """Nothing says how much data it came from, so nothing says data was lost."""
    gate = {"blocks": [{"dimension": "category", "label": "OLD", "n": 9, "pnl": -1.0}], "warnings": [], "haircuts": []}
    if meta != "absent":
        gate["meta"] = meta
    _save_gate(gate)
    out, err = improve(FakeKalshi())
    assert load_policy()["blocks"] == []
    assert "Saved active gate" in out
    assert "kept" not in err


# ---------------------------------------------------------------- improve: unreadable saved gate

BAD_GATES = ["5", '["x"]', "null", '{"blocks":5}', '{"blocks":["x"]}', "nonsense", "", '{"blocks":']


@pytest.mark.parametrize("dry", [False, True])
@pytest.mark.parametrize("body", BAD_GATES)
def test_unreadable_gate_stops_the_run_and_writes_nothing(improve, capsys, body, dry):
    """place_guarded_order refuses on this file, and improve must not undo that."""
    write_journal(JOURNAL)
    _save_gate(body)
    journal = Path(DEFAULT_JOURNAL_PATH).read_bytes()
    client = FakeKalshi(LOSSES)
    with pytest.raises(SystemExit) as stop:
        improve(client, dry=dry)
    assert stop.value.code == 1
    cap = capsys.readouterr()
    assert DEFAULT_POLICY_PATH in cap.err and "unreadable" in cap.err
    assert cap.out == ""
    assert Path(DEFAULT_POLICY_PATH).read_text() == body
    assert Path(DEFAULT_JOURNAL_PATH).read_bytes() == journal
    assert not Path(DEFAULT_SETTLEMENTS_PATH).exists()


def test_deleting_an_unreadable_gate_lets_improve_rebuild(improve, capsys):
    _save_gate("5")
    with pytest.raises(SystemExit):
        improve(FakeKalshi())
    capsys.readouterr()
    Path(DEFAULT_POLICY_PATH).unlink()
    out, err = improve(FakeKalshi())
    assert "Saved active gate" in out
    assert err == ""


@pytest.mark.parametrize("dry", [False, True])
def test_a_missing_gate_is_the_normal_first_run(improve, dry):
    out, err = improve(FakeKalshi(), dry=dry)
    assert "unreadable" not in err
    assert Path(DEFAULT_POLICY_PATH).exists() is (not dry)


# ---------------------------------------------------------------- edge

def test_edge_measures_and_writes_nothing(edge):
    write_journal(JOURNAL)
    before = Path(DEFAULT_JOURNAL_PATH).read_bytes()
    client = FakeKalshi(LOSSES)
    data = edge(client, as_json=True)
    assert client.closed == 1
    assert data["n_settled"] == data["n_forward"] == 5
    assert data["edge_vs_book"]["overall"]["realized_winrate"] == 0
    assert Path(DEFAULT_JOURNAL_PATH).read_bytes() == before
    assert not Path(DEFAULT_SETTLEMENTS_PATH).exists()
    assert not Path(DEFAULT_POLICY_PATH).exists()


def test_edge_human_output_matches_the_report(edge):
    write_journal(JOURNAL)
    out = edge(FakeKalshi(WINS))
    assert "Settled: 5  |  forward: 5  |  suspect: 0  |  unknown: 0" in out
    assert "overall: n=5  won=100%  implied=90%  edge=+0.1000" in out
    assert "by category:" in out and "by side:" in out
    assert "VERDICT:" in out


def test_edge_refuses_to_measure_when_the_pull_fails(edge):
    """No local fallback: a stale number here would read as the headline metric."""
    write_journal(JOURNAL)
    client = FakeKalshi(error=DOWN)
    with pytest.raises(RuntimeError, match="503"):
        edge(client)
    assert client.closed == 1


def test_edge_ignores_a_settlement_it_holds_no_position_in(edge):
    write_journal(JOURNAL)
    empty = dict(_raw("KXL-9"), no_count_fp="0", no_total_cost_dollars="0")
    data = edge(FakeKalshi([*LOSSES, empty]), as_json=True)
    assert data["n_settled"] == 5


# ---------------------------------------------------------------- _print_edge_report

def _report(records=()):
    return edge_report(list(records), "2026-11-05")


def _settled(ticker, category="KXL"):
    return _buy(ticker, category=category, outcome=settlement_pnl(_raw(ticker)))


def test_print_empty_report_says_there_is_nothing_to_score(capsys):
    cli._print_edge_report(_report(), as_json=False)
    out = capsys.readouterr().out
    assert "EDGE vs. THE BOOK — 2026-11-05" in out
    assert out.count("n/a (no scorable trades)") == 2
    assert "(no forward-settled trades with a fill price yet)" in out
    assert "(no forward-settled trades with est_prob yet)" in out
    assert "by category:" not in out


def test_print_json_is_the_report_unchanged(capsys):
    report = _report([_settled("KXL-1")])
    cli._print_edge_report(report, as_json=True)
    assert json.loads(capsys.readouterr().out) == report


def test_print_counts_and_calibration_rows(capsys):
    cli._print_edge_report(_report(_settled(f"KXL-{i}") for i in range(3)), as_json=False)
    out = capsys.readouterr().out
    assert "Settled: 3  |  forward: 3" in out
    assert "(n=3)" in out
    assert "bucket" in out
    assert "n/a" not in out


def test_print_truncates_a_long_category_to_the_column(capsys):
    long = "KXVERYLONGCATEGORYNAME"
    cli._print_edge_report(_report([_settled("A-1", category=long)]), as_json=False)
    out = capsys.readouterr().out
    assert long[:18] in out
    assert long[:19] not in out


def test_print_sorts_groups_by_label(capsys):
    cli._print_edge_report(_report([_settled("Z-1", category="ZZZ"), _settled("A-1", category="AAA")]), as_json=False)
    out = capsys.readouterr().out
    assert out.index("AAA") < out.index("ZZZ")
