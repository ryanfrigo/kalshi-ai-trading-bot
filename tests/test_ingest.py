"""Pin the behaviour of src/jobs/ingest.py (market ingestion).

No network, no credentials, no orders: KalshiClient is replaced by a recording
fake that serves canned pages and records every call. Any method other than the
read-only ones is logged as a violation (it raises AssertionError, but
run_ingestion swallows exceptions, so each test checks the violation list).
The DB is a real SQLite file in tmp_path.

Bugs found were first pinned as xfail(strict=True) and have since been fixed; the
tests below are their regression tests.
"""
import asyncio
import math
from datetime import datetime

import aiosqlite
import pytest

from src.config.settings import settings
from src.jobs import ingest
from src.utils.database import DatabaseManager

READ_ONLY = {"get_market", "get_markets", "_make_authenticated_request", "close"}
EXPIRY = "2026-12-31T23:59:59Z"
EXPIRY_TS = 1798761599  # 2026-12-31T23:59:59Z, hand-computed


class RecLogger:
    def __init__(self):
        self.records = []

    def _rec(self, level):
        def f(msg, *a, **k):
            self.records.append((level, msg, k))
        return f

    def __getattr__(self, name):
        if name in ("debug", "info", "warning", "error"):
            return self._rec(name)
        raise AttributeError(name)

    def messages(self, level=None):
        return [m for lv, m, _ in self.records if level in (None, lv)]


def mk(ticker="KXA-1", yes_bid="0.4000", yes_ask="0.4400", no_bid="0.5600",
       no_ask="0.6000", volume_fp="250.00", status="active", **over):
    """A Kalshi v2 market row: *_dollars and *_fp fixed-point strings only."""
    row = {
        "ticker": ticker,
        "title": f"title {ticker}",
        "yes_bid_dollars": yes_bid,
        "yes_ask_dollars": yes_ask,
        "no_bid_dollars": no_bid,
        "no_ask_dollars": no_ask,
        "volume_fp": volume_fp,
        "expiration_time": EXPIRY,
        "status": status,
        "category": "Politics",
    }
    row.update(over)
    return row


@pytest.fixture
async def db(tmp_path):
    m = DatabaseManager(str(tmp_path / "t.db"))
    await m.initialize()
    return m


async def rows(db):
    async with aiosqlite.connect(db.db_path) as c:
        c.row_factory = aiosqlite.Row
        cur = await c.execute("SELECT * FROM markets ORDER BY market_id")
        return [dict(r) for r in await cur.fetchall()]


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def process(batch, db, existing=frozenset()):
    q = asyncio.Queue()
    log = RecLogger()
    await ingest.process_and_queue_markets(batch, db, q, set(existing), log)
    return q, log


# --------------------------------------------------------------------------- parsing


async def test_v2_dollar_fields_parse_to_market_and_db(db):
    q, _ = await process([mk()], db)
    (m,) = drain(q)
    # mid = (bid+ask)/2, hand computed
    assert m.yes_price == pytest.approx(0.42)
    assert m.no_price == pytest.approx(0.58)
    assert m.volume == 250
    assert m.expiration_ts == EXPIRY_TS
    assert m.market_id == "KXA-1" and m.status == "active" and m.category == "Politics"
    (r,) = await rows(db)
    assert r["yes_price"] == pytest.approx(0.42)
    assert r["no_price"] == pytest.approx(0.58)
    assert r["volume"] == 250 and r["expiration_ts"] == EXPIRY_TS


async def test_v2_dollars_win_over_stale_v1_cent_fields(db):
    """Issue #101 guard: when *_dollars exist the v1 cent fields must be ignored."""
    row = mk()
    row.update(yes_bid=95, yes_ask=99, no_bid=1, no_ask=5)  # stale v1 cents alongside v2 dollars
    q, _ = await process([row], db)
    (m,) = drain(q)
    assert m.yes_price == pytest.approx(0.42)
    assert m.no_price == pytest.approx(0.58)


async def test_legacy_v1_cents_fallback_still_parses(db):
    """v1 read path (ingest.py:42-47) still works for cent-only payloads. Flagged: it is a
    v1 field read; remove once nothing serves v1 shapes."""
    row = {"ticker": "KXL-1", "title": "t", "yes_bid": 40, "yes_ask": 44, "no_bid": 56,
           "no_ask": 60, "volume": 300, "expiration_time": EXPIRY, "status": "active"}
    q, _ = await process([row], db)
    (m,) = drain(q)
    assert m.yes_price == pytest.approx(0.42)
    assert m.no_price == pytest.approx(0.58)
    assert m.volume == 300
    assert m.category == "unknown"


@pytest.mark.parametrize("fp,legacy,expected", [
    ("250.00", None, 250),
    ("1234.99", None, 1234),   # truncates, not rounds
    ("0.00", 500, 0),          # "0.00" is a truthy string, so volume_fp wins
    ("", 500, 500),            # empty fp falls back to v1 volume
    (None, 500, 500),
    (None, None, 0),
    (1500, None, 1500),        # numeric volume_fp tolerated
])
async def test_volume_fp_parsing(db, fp, legacy, expected):
    row = mk(volume_fp=fp)
    if legacy is not None:
        row["volume"] = legacy
    q, _ = await process([row], db)
    (r,) = await rows(db)
    assert r["volume"] == expected


@pytest.mark.parametrize("field,value,warning", [
    ("no_ask_dollars", "__missing__", "malformed"),
    ("yes_bid_dollars", None, "malformed"),
    ("no_bid_dollars", "", "malformed"),
    ("yes_ask_dollars", "0.0000", "two-sided"),   # ask == 0 with a bid: half-the-bid fabrication
    ("no_ask_dollars", "0", "two-sided"),
    ("yes_bid_dollars", "-0.0100", "two-sided"),
])
async def test_missing_none_or_zero_ask_skips_row_with_warning(db, field, value, warning):
    row = mk()
    if value == "__missing__":
        del row[field]
    else:
        row[field] = value
    q, log = await process([row, mk("KXOK-1")], db)
    assert [m.market_id for m in drain(q)] == ["KXOK-1"]
    assert [r["market_id"] for r in await rows(db)] == ["KXOK-1"]
    assert any("KXA-1" in m and warning in m for m in log.messages("warning"))


async def test_expiration_z_suffix_parsed_as_utc(db):
    q, _ = await process([mk(expiration_time="2026-01-01T00:00:00Z")], db)
    (m,) = drain(q)
    assert m.expiration_ts == 1767225600  # 2026-01-01T00:00:00Z


# --------------------------------------------------------------------------- filters


@pytest.mark.parametrize("ya,na,skipped", [
    ("1.0000", "1.0000", True),     # classic collection ticker
    ("0.9800", "0.9800", True),     # boundary: >= 0.98 on both is skipped
    ("0.9850", "0.9850", True),     # between ingest 0.98 and helper 0.99 thresholds
    ("0.9799", "0.9800", False),    # one side under the line -> kept
    ("0.9800", "0.9799", False),
    ("1.0000", "0.5000", False),    # only one side saturated -> real market
])
async def test_collection_ticker_threshold_boundary(db, ya, na, skipped):
    q, log = await process([mk(yes_ask=ya, no_ask=na, volume_fp="500")], db)
    if skipped:
        assert drain(q) == [] and await rows(db) == []
        assert any("collection" in m for m in log.messages("warning"))
    else:
        assert len(drain(q)) == 1 and len(await rows(db)) == 1


@pytest.mark.parametrize("vol,queued", [("100.00", True), ("99.00", False), ("99.99", False),
                                         ("100.99", True)])
async def test_min_volume_100_boundary_gates_queue_not_db(db, vol, queued):
    q, _ = await process([mk(volume_fp=vol)], db)
    assert (len(drain(q)) == 1) is queued
    assert len(await rows(db)) == 1  # low-volume markets are still persisted


async def test_excluded_and_preferred_categories(db, monkeypatch):
    batch = [mk("KXP-1", category="Politics"), mk("KXS-1", category="Sports"),
             mk("KXN-1", category=None)]
    batch[2].pop("category")
    monkeypatch.setattr(settings.trading, "excluded_categories", ["Sports"])
    monkeypatch.setattr(settings.trading, "preferred_categories", [])
    q, _ = await process(batch, db)
    assert sorted(m.market_id for m in drain(q)) == ["KXN-1", "KXP-1"]
    assert len(await rows(db)) == 3

    monkeypatch.setattr(settings.trading, "excluded_categories", [])
    monkeypatch.setattr(settings.trading, "preferred_categories", ["Politics"])
    q, _ = await process(batch, db)
    # category missing -> "unknown" -> not in the preferred list
    assert [m.market_id for m in drain(q)] == ["KXP-1"]


async def test_has_position_flag_from_existing_positions(db):
    q, _ = await process([mk("KXA-1"), mk("KXB-1")], db, existing={"KXB-1"})
    flags = {m.market_id: m.has_position for m in drain(q)}
    assert flags == {"KXA-1": False, "KXB-1": True}
    assert {r["market_id"]: r["has_position"] for r in await rows(db)} == {"KXA-1": 0, "KXB-1": 1}


async def test_empty_and_all_skipped_batches_write_nothing(db):
    q, log = await process([], db)
    assert drain(q) == [] and await rows(db) == []
    assert any("No new markets" in m for m in log.messages("info"))
    q, _ = await process([mk(yes_ask="1.0000", no_ask="1.0000")], db)
    assert drain(q) == [] and await rows(db) == []


async def test_upsert_updates_in_place_no_duplicates(db):
    await process([mk(yes_bid="0.4000", yes_ask="0.4400")], db)
    await process([mk(yes_bid="0.5000", yes_ask="0.5400")], db, existing={"KXA-1"})
    (r,) = await rows(db)
    assert r["yes_price"] == pytest.approx(0.52)
    assert r["has_position"] == 1


# --------------------------------------------------------------------------- bugs: bad rows


BAD_ROWS = {
    "no_ticker": lambda r: r.pop("ticker"),
    "no_title": lambda r: r.pop("title"),
    "no_expiration": lambda r: r.pop("expiration_time"),
    "bad_expiration": lambda r: r.update(expiration_time="not-a-date"),
    "no_status": lambda r: r.pop("status"),
    "bad_volume_fp": lambda r: r.update(volume_fp="abc"),
    "bad_price": lambda r: r.update(yes_bid_dollars="abc"),
}


@pytest.mark.parametrize("kind", sorted(BAD_ROWS))
async def test_malformed_row_is_skipped_not_batch_fatal(db, kind):
    bad = mk("KXBAD-1")
    BAD_ROWS[kind](bad)
    q, log = await process([mk("KXGOOD-1"), bad, mk("KXGOOD-2")], db)
    assert sorted(m.market_id for m in drain(q)) == ["KXGOOD-1", "KXGOOD-2"]
    assert [r["market_id"] for r in await rows(db)] == ["KXGOOD-1", "KXGOOD-2"]
    assert any("malformed" in m for m in log.messages("warning"))


async def test_non_dict_row_is_skipped(db):
    q, _ = await process([None, "junk", mk("KXGOOD-1")], db)
    assert [m.market_id for m in drain(q)] == ["KXGOOD-1"]


async def test_legacy_none_cent_field_does_not_crash(db):
    row = {"ticker": "KXL-1", "title": "t", "yes_bid": None, "yes_ask": 44, "no_bid": 56,
           "no_ask": 60, "volume": 300, "expiration_time": EXPIRY, "status": "active"}
    q, log = await process([row, mk("KXOK-1")], db)
    # a None field is a malformed row: skipped with a warning, never read as 0
    assert [m.market_id for m in drain(q)] == ["KXOK-1"]
    assert any("malformed" in m and "yes_bid" in m for m in log.messages("warning"))


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("field", ["yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars"])
async def test_nonfinite_price_is_not_stored_or_queued(db, field, bad):
    q, log = await process([mk(**{field: bad}), mk("KXOK-1")], db)
    assert [m.market_id for m in drain(q)] == ["KXOK-1"]
    assert [r["market_id"] for r in await rows(db)] == ["KXOK-1"]
    assert any("malformed" in m and "non-finite" in m for m in log.messages("warning"))


@pytest.mark.parametrize("quotes", [
    dict(yes_bid="0.6000", yes_ask="0", no_bid="0", no_ask="0.4500"),      # bid only on YES leg
    dict(yes_bid="0", yes_ask="0.0500", no_bid="0.9500", no_ask="0"),      # NO leg: bid, ask 0
    dict(yes_bid="0", yes_ask="0", no_bid="0", no_ask="0"),                # empty book
])
async def test_one_sided_book_is_skipped_not_half_priced(db, quotes):
    q, log = await process([mk(volume_fp="500", **quotes)], db)
    assert drain(q) == [] and await rows(db) == []
    assert any("two-sided" in m for m in log.messages("warning"))


async def test_longshot_zero_bid_keeps_old_bid_ask_mid_and_is_ingested(db):
    """bid == 0 with ask > 0 is a real state (no bids on a longshot): same pricing as before."""
    q, log = await process([mk(yes_bid="0.0000", yes_ask="0.0300", no_bid="0.9700", no_ask="1.0000",
                               volume_fp="500")], db)
    (m,) = drain(q)
    assert m.yes_price == pytest.approx(0.015)   # (0 + 0.03) / 2, unchanged from before
    assert m.no_price == pytest.approx(0.985)    # (0.97 + 1.00) / 2
    (r,) = await rows(db)
    assert r["yes_price"] == pytest.approx(0.015) and r["no_price"] == pytest.approx(0.985)
    assert not log.messages("warning")


async def test_near_certain_zero_no_bid_is_ingested_at_old_price(db):
    q, _ = await process([mk(yes_bid="0.9700", yes_ask="1.0000", no_bid="0.0000", no_ask="0.0300",
                             volume_fp="500")], db)
    (m,) = drain(q)
    assert m.yes_price == pytest.approx(0.985) and m.no_price == pytest.approx(0.015)


async def test_legacy_cents_zero_bid_longshot_is_ingested(db):
    row = {"ticker": "KXL-2", "title": "t", "yes_bid": 0, "yes_ask": 3, "no_bid": 97, "no_ask": 100,
           "volume": 300, "expiration_time": EXPIRY, "status": "active"}
    q, _ = await process([row], db)
    (m,) = drain(q)
    assert m.yes_price == pytest.approx(0.015)


async def test_row_with_no_price_fields_at_all_is_not_ingested_as_zero(db):
    row = {"ticker": "KXZ-1", "title": "t", "volume_fp": "500.00", "expiration_time": EXPIRY,
           "status": "active"}
    q, log = await process([row], db)
    assert drain(q) == [] and await rows(db) == []
    assert any("malformed" in m and "missing price field" in m for m in log.messages("warning"))


# --------------------------------------------------------------------------- run_ingestion


class FakeKalshi:
    """Recording stand-in for KalshiClient. Never touches the network."""

    instances = []

    def __init__(self, *a, **k):
        self.calls = []
        self.violations = []
        self.closed = 0
        self.single = {}               # ticker -> response dict (or Exception)
        self.event_pages = []          # sequential responses; Exception items are raised
        self.market_pages = []         # same for /markets
        FakeKalshi.instances.append(self)

    async def get_market(self, ticker):
        self.calls.append(("get_market", ticker))
        r = self.single.get(ticker, {})
        if isinstance(r, Exception):
            raise r
        return r

    async def get_markets(self, limit=100, cursor=None, **k):
        self.calls.append(("get_markets", cursor))
        if not self.market_pages:
            return {"markets": [], "cursor": None}
        r = self.market_pages.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def _make_authenticated_request(self, method, endpoint, params=None, **k):
        self.calls.append((method, endpoint, dict(params or {})))
        if method != "GET":
            self.violations.append((method, endpoint))
            raise AssertionError("non-GET request")
        if not self.event_pages:
            return {"events": []}
        r = self.event_pages.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def close(self):
        self.closed += 1

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        async def forbidden(*a, **k):
            self.violations.append(name)
            raise AssertionError(f"ingest called forbidden client method {name}")
        return forbidden

    def gets(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def client(monkeypatch):
    FakeKalshi.instances = []
    monkeypatch.setattr(ingest, "KalshiClient", FakeKalshi)
    real_sleep = asyncio.sleep

    async def fast_sleep(_):
        await real_sleep(0)
    monkeypatch.setattr(ingest.asyncio, "sleep", fast_sleep)

    yield FakeKalshi
    for inst in FakeKalshi.instances:
        assert inst.violations == [], f"order/mutating call attempted: {inst.violations}"
        assert not any(c[0] in {"POST", "PUT", "DELETE", "PATCH"} for c in inst.calls)


async def run(db, ticker=None, setup=None):
    q = asyncio.Queue()
    orig_init = FakeKalshi.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        if setup:
            setup(self)
    FakeKalshi.__init__ = init
    try:
        await ingest.run_ingestion(db, q, market_ticker=ticker)
    finally:
        FakeKalshi.__init__ = orig_init
    return q, FakeKalshi.instances[-1]


def ev(*markets):
    return {"markets": list(markets)}


async def test_single_ticker_found(db, client):
    q, c = await run(db, "KXA-1", lambda c: c.single.update({"KXA-1": {"market": mk("KXA-1")}}))
    assert [m.market_id for m in drain(q)] == ["KXA-1"]
    assert c.gets("get_market") == [("get_market", "KXA-1")]
    assert c.gets("get_markets") == [] and c.closed == 1


@pytest.mark.parametrize("resp", [{}, {"error": "nope"}, None])
async def test_single_ticker_not_found_is_quiet_noop(db, client, resp):
    q, c = await run(db, "KXA-1", lambda c: c.single.update({"KXA-1": resp}))
    assert drain(q) == [] and await rows(db) == [] and c.closed == 1


async def test_single_ticker_api_error_swallowed_and_client_closed(db, client):
    q, c = await run(db, "KXA-1", lambda c: c.single.update({"KXA-1": RuntimeError("boom")}))
    assert drain(q) == [] and c.closed == 1  # does not raise


async def test_positions_lookup_failure_aborts_before_any_fetch(db, client, monkeypatch):
    async def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(db, "get_markets_with_positions", boom)
    q, c = await run(db)
    assert c.calls == [] and drain(q) == [] and c.closed == 1


async def test_single_ticker_inactive_market_not_queued(db, client):
    q, _ = await run(db, "KXA-1", lambda c: c.single.update(
        {"KXA-1": {"market": mk("KXA-1", status="finalized")}}))
    assert drain(q) == [] and await rows(db) == []


async def test_events_pagination_dedupe_and_active_filter(db, client):
    def setup(c):
        c.event_pages = [
            {"events": [{"markets": [mk("KXA-1"), mk("KXC-1", status="closed")]},
                        {"markets": [mk("KXB-1")]}], "cursor": "p2"},
            {"events": [{"markets": [mk("KXB-1"), mk("KXD-1")]}], "cursor": ""},
        ]
    q, c = await run(db, setup=setup)
    assert sorted(m.market_id for m in drain(q)) == ["KXA-1", "KXB-1", "KXD-1"]
    ev_calls = [x for x in c.calls if x[0] == "GET"]
    assert len(ev_calls) == 2
    assert ev_calls[0][2] == {"status": "open", "limit": 100, "with_nested_markets": "true"}
    assert ev_calls[1][2]["cursor"] == "p2"
    assert [r["market_id"] for r in await rows(db)] == ["KXA-1", "KXB-1", "KXD-1"]


async def test_events_ticker_missing_or_blank_is_ignored(db, client):
    def setup(c):
        c.event_pages = [{"events": [{"markets": [mk("KXA-1"), {"status": "active"},
                                                  mk("", status="active")]}]}]
    q, _ = await run(db, setup=setup)
    assert [m.market_id for m in drain(q)] == ["KXA-1"]


async def test_few_events_triggers_markets_fallback_with_dedupe(db, client):
    def setup(c):
        c.event_pages = [{"events": [{"markets": [mk("KXA-1")]}]}]
        c.market_pages = [
            {"markets": [mk("KXA-1"), mk("KXB-1"), mk("KXC-1", status="settled")], "cursor": "n"},
            {"markets": [mk("KXD-1")], "cursor": None},
        ]
    q, c = await run(db, setup=setup)
    assert sorted(m.market_id for m in drain(q)) == ["KXA-1", "KXB-1", "KXD-1"]
    assert c.gets("get_markets") == [("get_markets", None), ("get_markets", "n")]


async def test_no_fallback_when_events_supply_100_markets(db, client):
    page = [mk(f"KX{i:03d}-1") for i in range(100)]

    def setup(c):
        c.event_pages = [{"events": [{"markets": page}]}]
    q, c = await run(db, setup=setup)
    assert len(drain(q)) == 100
    assert c.gets("get_markets") == []


async def test_99_events_markets_still_triggers_fallback(db, client):
    page = [mk(f"KX{i:03d}-1") for i in range(99)]

    def setup(c):
        c.event_pages = [{"events": [{"markets": page}]}]
    _, c = await run(db, setup=setup)
    assert len(c.gets("get_markets")) == 1  # <100 boundary


async def test_events_error_mid_pagination_keeps_page1_and_falls_back(db, client):
    def setup(c):
        c.event_pages = [{"events": [{"markets": [mk("KXA-1")]}], "cursor": "p2"},
                         RuntimeError("503 on page 2")]
        c.market_pages = [{"markets": [mk("KXA-1"), mk("KXB-1")], "cursor": None}]
    q, c = await run(db, setup=setup)
    # page-1 market kept, fallback adds only the new one (seen_tickers dedupes)
    assert sorted(m.market_id for m in drain(q)) == ["KXA-1", "KXB-1"]
    assert len(await rows(db)) == 2 and c.closed == 1


async def test_markets_fallback_api_error_is_swallowed_after_partial_progress(db, client):
    def setup(c):
        c.event_pages = [{"events": [{"markets": [mk("KXA-1")]}]}]
        c.market_pages = [{"markets": [mk("KXB-1")], "cursor": "n"}, RuntimeError("500")]
    q, c = await run(db, setup=setup)
    assert sorted(m.market_id for m in drain(q)) == ["KXA-1", "KXB-1"]  # partial pages kept
    assert c.closed == 1


async def test_both_apis_failing_ingests_nothing_without_raising(db, client):
    def setup(c):
        c.event_pages = [RuntimeError("events down")]
        c.market_pages = [RuntimeError("markets down")]
    q, c = await run(db, setup=setup)
    assert drain(q) == [] and await rows(db) == [] and c.closed == 1


async def test_malformed_event_does_not_drop_rest_of_page(db, client):
    def setup(c):
        c.event_pages = [{"events": [{"markets": None}, {"markets": [mk("KXA-1")]}]}]
        c.market_pages = []
    q, _ = await run(db, setup=setup)
    assert [m.market_id for m in drain(q)] == ["KXA-1"]


async def test_markets_fallback_row_without_status_does_not_abort_pages(db, client):
    bad = mk("KXBAD-1")
    del bad["status"]

    def setup(c):
        c.market_pages = [{"markets": [bad, mk("KXA-1")], "cursor": "n"},
                          {"markets": [mk("KXB-1")], "cursor": None}]
    q, _ = await run(db, setup=setup)
    assert sorted(m.market_id for m in drain(q)) == ["KXA-1", "KXB-1"]


async def test_events_page_cap_is_20(db, client):
    n = {"i": 0}

    def setup(c):
        async def endless(method, endpoint, params=None, **k):
            c.calls.append((method, endpoint, dict(params or {})))
            n["i"] += 1
            return {"events": [{"markets": [mk(f"KXP{n['i']:03d}-1")]}], "cursor": f"c{n['i']}"}
        c._make_authenticated_request = endless
    q, c = await run(db, setup=setup)
    assert len([x for x in c.calls if x[0] == "GET"]) == 20


async def test_markets_fallback_page_cap_is_exactly_20(db, client):
    def setup(c):
        async def endless(limit=100, cursor=None, **k):
            c.calls.append(("get_markets", cursor))
            if len(c.calls) > 60:
                raise RuntimeError("runaway")
            return {"markets": [], "cursor": f"c{len(c.calls)}"}
        c.get_markets = endless
    _, c = await run(db, setup=setup)
    assert len(c.gets("get_markets")) == 20


async def test_markets_fallback_stops_on_repeated_cursor(db, client):
    def setup(c):
        async def stuck(limit=100, cursor=None, **k):
            c.calls.append(("get_markets", cursor))
            if len(c.calls) > 60:
                raise RuntimeError("runaway")
            return {"markets": [mk(f"KXR{len(c.calls)}-1")], "cursor": "same"}
        c.get_markets = stuck
    q, c = await run(db, setup=setup)
    assert len(c.gets("get_markets")) == 2  # first page, then the repeat is detected
    assert len(drain(q)) == 2


async def test_events_paging_stops_at_20_but_keeps_all_20_pages(db, client):
    def setup(c):
        c.event_pages = [{"events": [{"markets": [mk(f"KXP{i:02d}-1")]}], "cursor": f"c{i}"}
                         for i in range(40)]
    q, c = await run(db, setup=setup)
    assert len(drain(q)) == 20
    assert len([x for x in c.calls if x[0] == "GET"]) == 20


# --------------------------------------------------------------------------- category (lives on the event)


def cat_row(ticker, **over):
    r = mk(ticker, **over)
    r.pop("category", None)
    return r


async def test_category_taken_from_parent_event(db, client):
    """Kalshi v2 GetEvents: `category` is an Event field; nested markets carry none."""
    def setup(c):
        c.event_pages = [{"events": [{"category": "Sports", "markets": [cat_row("KXA-1")]},
                                     {"category": "Politics", "markets": [cat_row("KXB-1")]},
                                     {"markets": [cat_row("KXC-1")]}]}]
    q, _ = await run(db, setup=setup)
    assert {m.market_id: m.category for m in drain(q)} == {
        "KXA-1": "Sports", "KXB-1": "Politics", "KXC-1": "unknown"}
    assert {r["market_id"]: r["category"] for r in await rows(db)} == {
        "KXA-1": "Sports", "KXB-1": "Politics", "KXC-1": "unknown"}


async def test_market_own_category_beats_event_category(db, client):
    def setup(c):
        c.event_pages = [{"events": [{"category": "Sports",
                                      "markets": [mk("KXA-1", category="Weather")]}]}]
    q, _ = await run(db, setup=setup)
    assert [m.category for m in drain(q)] == ["Weather"]


async def test_excluded_and_preferred_filters_work_on_events_path(db, client, monkeypatch):
    def setup(c):
        c.event_pages = [{"events": [{"category": "Sports", "markets": [cat_row("KXS-1")]},
                                     {"category": "Politics", "markets": [cat_row("KXP-1")]}]}]
    monkeypatch.setattr(settings.trading, "excluded_categories", ["Sports"])
    monkeypatch.setattr(settings.trading, "preferred_categories", [])
    q, _ = await run(db, setup=setup)
    assert [m.market_id for m in drain(q)] == ["KXP-1"]
    monkeypatch.setattr(settings.trading, "excluded_categories", [])
    monkeypatch.setattr(settings.trading, "preferred_categories", ["Sports"])
    q, _ = await run(db, setup=setup)
    assert [m.market_id for m in drain(q)] == ["KXS-1"]


async def test_null_category_becomes_unknown_not_none(db):
    q, _ = await process([mk(category=None)], db)
    assert [m.category for m in drain(q)] == ["unknown"]


async def test_a_non_dict_event_or_nested_market_is_skipped(db, client):
    def setup(c):
        c.event_pages = [{"events": ["junk", {"markets": ["junk", mk("KXA-1")]}]}]
        c.market_pages = []
    q, _ = await run(db, setup=setup)
    assert [m.market_id for m in drain(q)] == ["KXA-1"]
