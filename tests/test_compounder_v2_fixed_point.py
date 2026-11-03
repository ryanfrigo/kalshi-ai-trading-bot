from src.strategies.safe_compounder import market_confidence_score


def test_confidence_keeps_fixed_point_sizes():
    m = {"volume_fp": "5000.00", "_days_to_expiry": 2, "last_price_dollars": "0.0400"}
    fp = {"yes_dollars": [["0.0300", "100.00"]], "no_dollars": [["0.9500", "100.00"]]}
    whole = {"yes_dollars": [["0.0300", "100"]], "no_dollars": [["0.9500", "100"]]}
    s_fp, r_fp = market_confidence_score("T", fp, m)
    s_whole, r_whole = market_confidence_score("T", whole, m)
    assert "no book" not in r_fp
    assert s_fp == s_whole
