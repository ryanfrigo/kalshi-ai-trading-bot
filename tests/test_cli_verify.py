"""`cli.py verify`: the book read, the LLM reply parsing and fail-closed behaviour.

Fakes only. The Kalshi fake has read methods only, so an order attempt would raise.
"""
import argparse
import json

import pytest

import cli


RESEARCH = {
    "resolution_criteria": "Resolves YES if X.", "true_yes_pct": 2.0, "catalyst": "none",
    "direction": "none", "is_frontrunner": False, "verdict": "looks like a fade",
}
SKEPTIC = {
    "adjusted_true_yes_pct": 2.0, "survives": True, "recommend": "BUY_NO", "size_hint": "full",
    "strongest_yes_path": "none", "note": "ok",
}


class FakeKalshi:
    book = {"orderbook_fp": {"yes_dollars": [["0.10", "10"]]}}
    market = {"market": {"title": "Will X happen", "yes_sub_title": "by June"}}
    market_error = None
    closed = 0

    async def get_orderbook(self, ticker, depth=10):
        return type(self).book

    async def get_market(self, ticker):
        if type(self).market_error:
            raise type(self).market_error
        return type(self).market

    async def close(self):
        type(self).closed += 1


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FakeKalshi.book = {"orderbook_fp": {"yes_dollars": [["0.10", "10"]]}}
    FakeKalshi.market = {"market": {"title": "Will X happen", "yes_sub_title": "by June"}}
    FakeKalshi.market_error = None
    FakeKalshi.closed = 0
    monkeypatch.setattr("src.clients.kalshi_client.KalshiClient", FakeKalshi)
    monkeypatch.setattr("src.utils.logging_setup.setup_logging", lambda **k: None)


def _llm(monkeypatch, replies):
    """Fake OpenRouter that returns each reply in turn, and a fake database."""
    queue = list(replies)
    prompts = []

    class FakeOR:
        def __init__(self, db_manager=None):
            self.db_manager = db_manager

        async def get_completion(self, prompt, **kwargs):
            prompts.append((prompt, kwargs))
            return queue.pop(0)

    class FakeDB:
        async def initialize(self):
            pass

    monkeypatch.setattr("src.clients.openrouter_client.OpenRouterClient", FakeOR)
    monkeypatch.setattr("src.utils.database.DatabaseManager", FakeDB)
    return prompts


def _args(**over):
    base = dict(ticker="KX-T", research_file=None, json=True, jev=False)
    base.update(over)
    return argparse.Namespace(**base)


def _verdict(capsys, **over):
    cli.cmd_verify(_args(**over))
    return json.loads(capsys.readouterr().out)


def _research_file(tmp_path, research=None, skeptic=None):
    path = tmp_path / "research.json"
    path.write_text(json.dumps({"research": research or RESEARCH, "skeptic": skeptic or SKEPTIC}))
    return str(path)


# --- the book -------------------------------------------------------------------------

def test_no_ask_comes_from_the_best_yes_bid_in_any_order(tmp_path, capsys):
    FakeKalshi.book = {"orderbook_fp": {"yes_dollars": [["0.02", "5"], ["0.10", "1"], ["0.03", "9"]]}}
    verdict = _verdict(capsys, research_file=_research_file(tmp_path))
    # NO ask 0.90, true YES 2%: edge = 100 - 2 - 90 = 8.
    assert verdict["edge_pts"] == 8
    assert FakeKalshi.closed == 1


@pytest.mark.parametrize("book", [
    {}, {"orderbook_fp": None}, {"orderbook_fp": {}}, {"orderbook_fp": {"yes_dollars": None}},
    {"orderbook_fp": {"yes_dollars": []}}, {"orderbook_fp": {"no_dollars": [["0.90", "5"]]}},
])
def test_empty_or_one_sided_book_refuses_and_closes(tmp_path, book):
    FakeKalshi.book = book
    with pytest.raises(RuntimeError, match="no executable NO ask"):
        cli.cmd_verify(_args(research_file=_research_file(tmp_path)))
    assert FakeKalshi.closed == 1


def test_the_gate_recomputes_edge_off_the_book_whatever_the_file_says(tmp_path, capsys):
    # An optimistic research file can't fudge a price with no edge left.
    FakeKalshi.book = {"orderbook_fp": {"yes_dollars": [["0.01", "5"]]}}
    verdict = _verdict(capsys, research_file=_research_file(tmp_path))
    assert verdict["recommend"] != "BUY_NO"
    assert verdict["edge_pts"] == -1


def test_a_zero_yes_bid_leaves_no_edge_to_buy(tmp_path, capsys):
    FakeKalshi.book = {"orderbook_fp": {"yes_dollars": [["0.00", "5"]]}}
    assert _verdict(capsys, research_file=_research_file(tmp_path))["recommend"] != "BUY_NO"


@pytest.mark.parametrize("bid", ["junk", "nan", "inf", "-inf"])
def test_a_malformed_book_raises_instead_of_reading_as_a_price(tmp_path, bid):
    FakeKalshi.book = {"orderbook_fp": {"yes_dollars": [[bid, "5"]]}}
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        cli.cmd_verify(_args(research_file=_research_file(tmp_path)))
    assert FakeKalshi.closed == 1


@pytest.mark.parametrize("bid", ["1.00", "1.50"])
def test_a_yes_bid_at_or_above_a_dollar_is_not_a_free_no(tmp_path, bid):
    # 1 - bid gives a NO ask of 0 or less, which read as the biggest edge possible.
    FakeKalshi.book = {"orderbook_fp": {"yes_dollars": [[bid, "5"]]}}
    with pytest.raises(RuntimeError, match="no executable NO ask"):
        cli.cmd_verify(_args(research_file=_research_file(tmp_path)))
    assert FakeKalshi.closed == 1


# --- the question ---------------------------------------------------------------------

def test_question_joins_title_and_subtitle(tmp_path, capsys):
    cli.cmd_verify(_args(json=False, research_file=_research_file(tmp_path)))
    assert "Will X happen :: by June" in capsys.readouterr().out


@pytest.mark.parametrize("market", [{}, {"market": None}, {"market": {"title": ""}}])
def test_question_falls_back_to_the_ticker(tmp_path, capsys, market):
    FakeKalshi.market = market
    cli.cmd_verify(_args(json=False, research_file=_research_file(tmp_path)))
    assert "\n  KX-T\n" in capsys.readouterr().out


def test_market_lookup_failure_doesnt_stop_verify(tmp_path, capsys):
    FakeKalshi.market_error = ConnectionError("down")
    assert _verdict(capsys, research_file=_research_file(tmp_path))["ticker"] == "KX-T"


# --- operator file --------------------------------------------------------------------

def test_operator_file_never_builds_an_ai_client(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("no AI client with a research file")

    monkeypatch.setattr("src.clients.openrouter_client.OpenRouterClient", boom)
    assert _verdict(capsys, research_file=_research_file(tmp_path))["recommend"] == "BUY_NO"


def test_half_filled_operator_file_raises(tmp_path):
    path = tmp_path / "research.json"
    path.write_text(json.dumps({"research": RESEARCH}))
    with pytest.raises(ValueError, match="skeptic"):
        cli.cmd_verify(_args(research_file=str(path)))
    assert FakeKalshi.closed == 1


def test_text_output_names_the_operator_file(tmp_path, capsys):
    file = _research_file(tmp_path)
    cli.cmd_verify(_args(json=False, research_file=file))
    out = capsys.readouterr().out
    assert f"operator file {file}" in out
    assert "NO ask: 0.90  (implied YES 10%)" in out
    assert "recommend:  BUY_NO" in out


# --- LLM replies ----------------------------------------------------------------------

def _j(obj):
    return json.dumps(obj)


@pytest.mark.parametrize("wrap", [
    lambda s: s,
    lambda s: f"```json\n{s}\n```",
    lambda s: f"```\n{s}\n```",
    lambda s: f"Here it is:\n{s}\nThanks.",
])
def test_llm_reply_shapes_parse_to_the_same_verdict(monkeypatch, capsys, wrap):
    _llm(monkeypatch, [wrap(_j(RESEARCH)), wrap(_j(SKEPTIC))])
    verdict = _verdict(capsys)
    assert verdict["recommend"] == "BUY_NO"
    assert verdict["edge_pts"] == 8


def test_llm_trailing_comma_is_repaired(monkeypatch, capsys):
    _llm(monkeypatch, [_j(RESEARCH)[:-1] + ",}", _j(SKEPTIC)])
    assert _verdict(capsys)["edge_pts"] == 8


def test_llm_prompt_asks_for_the_schema_keys_and_books_as_verify(monkeypatch, capsys):
    prompts = _llm(monkeypatch, [_j(RESEARCH), _j(SKEPTIC)])
    _verdict(capsys)
    prompt, kwargs = prompts[0]
    assert "Return ONLY a JSON object with these keys: " in prompt
    assert "true_yes_pct" in prompt
    assert kwargs["strategy"] == "verify"


def test_llm_none_raises_and_closes(monkeypatch):
    _llm(monkeypatch, [None])
    with pytest.raises(RuntimeError, match="LLM unavailable"):
        cli.cmd_verify(_args())
    assert FakeKalshi.closed == 1


def test_llm_prose_raises(monkeypatch):
    _llm(monkeypatch, ["I can't help with that."])
    with pytest.raises(RuntimeError, match="no JSON object"):
        cli.cmd_verify(_args())


@pytest.mark.parametrize("research,skeptic", [
    ("{}", "{}"),
    ("```json\n[1, 2]\n```", _j(SKEPTIC)),
    (_j(RESEARCH), "{}"),
    (_j({**RESEARCH, "true_yes_pct": "lots"}), _j(SKEPTIC)),
    (_j(RESEARCH), _j({**SKEPTIC, "adjusted_true_yes_pct": "few"})),
    (_j(RESEARCH), _j({**SKEPTIC, "survives": "false"})),
    (_j(RESEARCH), _j({**SKEPTIC, "adjusted_true_yes_pct": -500})),
    (_j({**RESEARCH, "direction": "Positive"}), _j(SKEPTIC)),
])
def test_malformed_llm_replies_refuse(monkeypatch, research, skeptic):
    _llm(monkeypatch, [research, skeptic])
    with pytest.raises((ValueError, KeyError, AttributeError, TypeError)):
        cli.cmd_verify(_args())
    assert FakeKalshi.closed == 1


@pytest.mark.parametrize("skeptic", [
    {**SKEPTIC, "survives": False},
    {**SKEPTIC, "recommend": "PASS"},
    {**SKEPTIC, "adjusted_true_yes_pct": 99.0},
])
def test_wellformed_replies_against_the_fade_pass(monkeypatch, capsys, skeptic):
    _llm(monkeypatch, [_j(RESEARCH), _j(skeptic)])
    assert _verdict(capsys)["recommend"] == "PASS"


def test_a_malformed_operator_file_refuses_too(tmp_path):
    file = _research_file(tmp_path, skeptic={**SKEPTIC, "survives": "false"})
    with pytest.raises(ValueError, match="survives"):
        cli.cmd_verify(_args(research_file=file))


# --- Jev floor ------------------------------------------------------------------------

def test_jev_floor_is_reported(tmp_path, monkeypatch, capsys):
    import src.agent.jev as jev

    def fake_floor(llm_call, question, facts=None, record=None):
        record["jev_true_yes_pct"] = 12.34
        record["facts_seen"] = facts
        return llm_call

    monkeypatch.setattr(jev, "with_jev_floor", fake_floor)
    file = _research_file(tmp_path)
    assert _verdict(capsys, research_file=file, jev=True)["jev_true_yes_pct"] == 12.3
    cli.cmd_verify(_args(research_file=file, jev=True, json=False))
    assert "Jev P(YES): 12.3%" in capsys.readouterr().out
