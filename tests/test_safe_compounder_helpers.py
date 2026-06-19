"""Regression test: Kalshi v2 returns numeric fields as STRINGS ("0.00").

A live run placed real orders but then crashed on `if fill_count > 0` because
fill_count was the string "0.00" (str > int -> TypeError), mis-reporting placed
orders as errors. _to_int_count coerces these safely.
"""
from src.strategies.safe_compounder import _to_int_count


def test_to_int_count_handles_v2_strings_and_ints():
    assert _to_int_count("0.00") == 0
    assert _to_int_count("16.00") == 16
    assert _to_int_count(3) == 3
    assert _to_int_count(None) == 0
    assert _to_int_count("") == 0
    assert _to_int_count("garbage") == 0
