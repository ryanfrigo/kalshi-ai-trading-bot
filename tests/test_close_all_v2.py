"""close-all must read Kalshi v2 signed string positions (position_fp)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli import open_positions_from_v2  # noqa: E402


def test_v2_signed_string_no_position_is_kept_as_negative_float():
    rows = [{"ticker": "OAIAGI-26", "position_fp": "-40.00"}]
    assert open_positions_from_v2(rows) == [("OAIAGI-26", -40.0)]


def test_v2_positive_string_is_yes():
    rows = [{"ticker": "ABC-1", "position_fp": "12.00"}]
    assert open_positions_from_v2(rows) == [("ABC-1", 12.0)]


def test_zero_rows_are_dropped():
    rows = [
        {"ticker": "ZERO-FP", "position_fp": "0.00"},
        {"ticker": "ZERO-LEGACY", "position": 0},
        {"ticker": "NONE-FP", "position_fp": None},
        {"ticker": "KEEP", "position_fp": "-1.00"},
    ]
    assert open_positions_from_v2(rows) == [("KEEP", -1.0)]


def test_legacy_numeric_position_still_works():
    rows = [{"ticker": "LEGACY-1", "position": 3}, {"ticker": "LEGACY-2", "position": -2}]
    assert open_positions_from_v2(rows) == [("LEGACY-1", 3.0), ("LEGACY-2", -2.0)]


def test_empty_input():
    assert open_positions_from_v2([]) == []


def test_fractional_v2_positions_survive_the_helper():
    rows = [
        {"ticker": "BIG-FRAC", "position_fp": "2102.12"},
        {"ticker": "TINY", "position_fp": "0.50"},
    ]
    assert open_positions_from_v2(rows) == [("BIG-FRAC", 2102.12), ("TINY", 0.5)]


class _FakeKalshiClient:
    """Stands in for KalshiClient so close-all can run with no network and no orders."""

    rows: list = []
    placed: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def get_positions(self):
        return {"market_positions": self.rows}

    async def get_orderbook(self, ticker, depth=1):
        return {"orderbook": {"yes": [[50, 10]], "no": [[40, 10]]}}

    async def place_order(self, **kwargs):
        self.placed.append(kwargs)
        return {"order": {"order_id": "test-order"}}

    async def close(self):
        pass


def _run_close_all(monkeypatch, rows, live=True):
    from argparse import Namespace

    import cli
    from src.clients import kalshi_client as kc

    _FakeKalshiClient.rows = rows
    _FakeKalshiClient.placed = []
    monkeypatch.setattr(kc, "KalshiClient", _FakeKalshiClient)
    cli.cmd_close_all(Namespace(yes=True, live=live))
    return _FakeKalshiClient.placed


def test_close_all_sends_fractional_count_unchanged(monkeypatch):
    placed = _run_close_all(monkeypatch, [{"ticker": "BIG-FRAC", "position_fp": "2102.12"}])
    assert len(placed) == 1
    assert placed[0]["ticker"] == "BIG-FRAC"
    assert placed[0]["count"] == 2102.12


def test_close_all_sends_half_contract_not_zero(monkeypatch):
    placed = _run_close_all(monkeypatch, [{"ticker": "HALF", "position_fp": "-0.50"}])
    assert len(placed) == 1
    assert placed[0]["side"] == "no"
    assert placed[0]["count"] == 0.5


def test_close_all_skips_position_that_rounds_to_zero(monkeypatch):
    placed = _run_close_all(monkeypatch, [{"ticker": "DUST", "position_fp": "0.004"}])
    assert placed == []


def test_close_all_dry_run_places_no_orders(monkeypatch):
    placed = _run_close_all(monkeypatch, [{"ticker": "BIG-FRAC", "position_fp": "2102.12"}], live=False)
    assert placed == []


def test_fractional_count_formats_in_v2_payload():
    from src.clients.kalshi_client import build_order_v2_payload

    payload = build_order_v2_payload(
        ticker="BIG-FRAC", client_order_id="c1", side="no", action="sell",
        count=2102.12, type_="limit", no_price=40,
    )
    assert payload["count"] == "2102.12"
