"""The verify gate refuses malformed LLM or operator input, so it can never read as BUY_NO."""
import pytest

from src.agent.verify import aggregate_verdict

CANDIDATE = {"ticker": "T", "question": "q", "no_ask": 0.90}
RESEARCH = {
    "resolution_criteria": "x", "true_yes_pct": 2.0, "catalyst": "none", "direction": "none",
    "is_frontrunner": False, "verdict": "fade",
}
SKEPTIC = {
    "strongest_yes_path": "none", "adjusted_true_yes_pct": 2.0, "survives": True,
    "recommend": "BUY_NO", "size_hint": "full",
}


def _gate(candidate=None, research=None, skeptic=None):
    return aggregate_verdict(
        {**CANDIDATE, **(candidate or {})}, {**RESEARCH, **(research or {})}, {**SKEPTIC, **(skeptic or {})}
    )


def test_a_valid_reply_still_buys():
    verdict = _gate()
    assert (verdict["recommend"], verdict["size_hint"], verdict["edge_pts"]) == ("BUY_NO", "full", 8)
    assert verdict["survives"] is True


@pytest.mark.parametrize("survives", ["false", "no", "true", 1, 0, None, [], "True"])
def test_survives_must_be_a_real_bool(survives):
    with pytest.raises(ValueError, match="survives"):
        _gate(skeptic={"survives": survives})


def test_survives_false_is_a_plain_pass():
    assert _gate(skeptic={"survives": False})["recommend"] == "PASS"


@pytest.mark.parametrize("value", ["false", "true", None, 0, 1, []])
def test_is_frontrunner_must_be_a_real_bool(value):
    with pytest.raises(ValueError, match="is_frontrunner"):
        _gate(research={"is_frontrunner": value})


def test_is_frontrunner_missing_is_refused():
    research = {k: v for k, v in RESEARCH.items() if k != "is_frontrunner"}
    with pytest.raises(ValueError, match="is_frontrunner"):
        aggregate_verdict(CANDIDATE, research, SKEPTIC)


def test_is_frontrunner_true_is_a_plain_pass():
    assert _gate(research={"is_frontrunner": True})["recommend"] == "PASS"


@pytest.mark.parametrize("direction", ["Positive", "POSITIVE", "live ", "Live", "neutral", "", None, 1])
def test_direction_must_be_one_of_the_schema_values(direction):
    with pytest.raises(ValueError, match="direction"):
        _gate(research={"direction": direction})


@pytest.mark.parametrize("direction,recommend", [
    ("adverse", "BUY_NO"), ("none", "BUY_NO"), ("positive", "PASS"), ("live", "PASS"),
])
def test_the_four_schema_directions_still_work(direction, recommend):
    assert _gate(research={"direction": direction})["recommend"] == recommend


@pytest.mark.parametrize("field,where", [("adjusted_true_yes_pct", "skeptic"), ("true_yes_pct", "research")])
@pytest.mark.parametrize("value", [-0.1, -500, 100.1, True, False, "2", None, float("nan"), float("inf"), []])
def test_probabilities_must_be_numbers_from_0_to_100(field, where, value):
    with pytest.raises(ValueError, match=field):
        _gate(**{where: {field: value}})


@pytest.mark.parametrize("value", [0, 0.0, 100, 100.0, 2])
def test_probability_edges_are_accepted(value):
    assert _gate(skeptic={"adjusted_true_yes_pct": value})["true_yes"] == value


@pytest.mark.parametrize("no_ask", [0, 0.0, -0.5, 1.01, True, "0.9", None, float("nan"), float("inf")])
def test_no_ask_must_be_a_price(no_ask):
    with pytest.raises(ValueError, match="no_ask"):
        _gate(candidate={"no_ask": no_ask})


def test_a_no_ask_of_a_dollar_is_a_price_with_no_edge():
    assert _gate(candidate={"no_ask": 1.0})["recommend"] == "PASS"


@pytest.mark.parametrize("research,skeptic", [([], SKEPTIC), (RESEARCH, []), (None, SKEPTIC), (RESEARCH, "x")])
def test_replies_must_be_objects(research, skeptic):
    with pytest.raises(ValueError):
        aggregate_verdict(CANDIDATE, research, skeptic)
