"""Behaviour tests for src/clients/kalshi_ws.py over a fake websocket transport.

No network, no real credentials: a throwaway RSA key is generated per test and
``websockets.connect`` is replaced with a fake that hands out scripted sockets.
"""

import asyncio
import base64
import json

import pytest
import websockets.exceptions
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.clients import kalshi_ws
from src.clients.kalshi_ws import (
    CHANNEL_FILL,
    CHANNEL_ORDERBOOK_DELTA,
    CHANNEL_TICKER,
    CHANNEL_TRADE,
    WS_PATH,
    ConnectionState,
    KalshiWebSocket,
)
from src.events.event_bus import (
    EVENT_FILL_RECEIVED,
    EVENT_ORDERBOOK_UPDATE,
    EVENT_PRICE_UPDATE,
    EVENT_TRADE_EXECUTED,
    EventBus,
)

_REAL_SLEEP = asyncio.sleep


def _closed(code=1006, reason="gone"):
    # ConnectionClosed.code/.reason are read-only properties in websockets 16;
    # the client only reads them, so shadow them on a subclass.
    class _Closed(websockets.exceptions.ConnectionClosed):
        pass

    exc = _Closed(None, None)
    _Closed.code, _Closed.reason = code, reason
    return exc


class FakeWS:
    """Scripted socket: recv() pops from an inbox; exceptions in it are raised."""

    def __init__(self, inbox=()):
        self.inbox = asyncio.Queue()
        for item in inbox:
            self.inbox.put_nowait(item)
        self.sent = []
        self.pings = 0
        self.closed = False
        self.close_raises = None

    async def recv(self):
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def ping(self):
        self.pings += 1

    async def close(self):
        self.closed = True
        if self.close_raises:
            raise self.close_raises
        # a real socket's pending recv() fails once closed
        self.inbox.put_nowait(_closed(1000, "bye"))


@pytest.fixture
def key_path(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    p = tmp_path / "throwaway_test_key.pem"
    p.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    p.with_suffix(".pub")  # path only; public key is derived in-test
    return str(p), key.public_key()


@pytest.fixture
def transport(monkeypatch):
    """Replace websockets.connect; ``transport.sockets`` is the queue of fakes
    handed out in order. An Exception instance in the queue is raised instead."""

    class T:
        sockets = []
        calls = []  # (url, headers)

    async def fake_connect(url, **kw):
        T.calls.append((url, kw.get("additional_headers")))
        nxt = T.sockets.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    T.sockets, T.calls = [], []
    monkeypatch.setattr(kalshi_ws.websockets, "connect", fake_connect)
    return T


@pytest.fixture
def ws(key_path):
    return KalshiWebSocket(api_key="TESTKEY", private_key_path=key_path[0])


@pytest.fixture
def bus():
    EventBus.reset_instance()
    yield EventBus.get_instance()
    EventBus.reset_instance()


@pytest.fixture
def delays(monkeypatch):
    """Record backoff sleeps without waiting; park the keepalive loop."""
    rec = []

    async def fake_sleep(d, *a):
        if d == kalshi_ws._PING_INTERVAL_S:
            await _REAL_SLEEP(3600)  # cancelled on shutdown
        rec.append(d)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(kalshi_ws.asyncio, "sleep", fake_sleep)
    return rec


async def _connected(ws, transport, sock=None):
    sock = sock or FakeWS()
    transport.sockets.append(sock)
    await ws.connect()
    return sock


# ---------------------------------------------------------------- construction


def test_missing_key_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        KalshiWebSocket(api_key="k", private_key_path=str(tmp_path / "nope.pem"))


# ------------------------------------------------------------------------ auth


def test_auth_headers_have_the_three_kalshi_fields(ws):
    h = ws._build_auth_headers()
    assert set(h) == {"KALSHI-ACCESS-KEY", "KALSHI-ACCESS-TIMESTAMP", "KALSHI-ACCESS-SIGNATURE"}
    assert h["KALSHI-ACCESS-KEY"] == "TESTKEY"
    assert h["KALSHI-ACCESS-TIMESTAMP"].isdigit()
    assert len(h["KALSHI-ACCESS-TIMESTAMP"]) == 13  # epoch milliseconds


def test_signature_verifies_over_timestamp_get_and_ws_path(ws, key_path):
    h = ws._build_auth_headers()
    msg = (h["KALSHI-ACCESS-TIMESTAMP"] + "GET" + WS_PATH).encode()
    # raises InvalidSignature if the scheme (RSA-PSS/SHA256, digest-length salt) drifts
    key_path[1].verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]),
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_sign_uppercases_method(ws, key_path):
    sig = ws._sign("123", "get", "/x")
    key_path[1].verify(
        base64.b64decode(sig),
        b"123GET/x",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


# ------------------------------------------------------------------ connection


async def test_connect_uses_wss_url_and_sends_auth_headers(ws, transport):
    await _connected(ws, transport)
    url, headers = transport.calls[0]
    assert url.startswith("wss://") and url.endswith(WS_PATH)
    assert headers["KALSHI-ACCESS-KEY"] == "TESTKEY"
    assert ws.state is ConnectionState.CONNECTED and ws.is_connected


async def test_connect_failure_resets_state_and_propagates(ws, transport):
    transport.sockets.append(OSError("refused"))
    with pytest.raises(OSError):
        await ws.connect()
    assert ws.state is ConnectionState.DISCONNECTED


async def test_connect_when_already_connected_is_a_noop(ws, transport):
    await _connected(ws, transport)
    await ws.connect()
    assert len(transport.calls) == 1


async def test_run_requires_connect_first(ws):
    with pytest.raises(RuntimeError):
        await ws.run()


# ------------------------------------------------------------------- subscribe


async def test_subscribe_message_format_and_incrementing_ids(ws, transport):
    sock = await _connected(ws, transport)
    await ws.subscribe(["MKT-A", "MKT-B"], [CHANNEL_TRADE, CHANNEL_FILL])
    await ws.subscribe(["MKT-C"])
    assert sock.sent[0] == {
        "id": 1,
        "cmd": "subscribe",
        "params": {"channels": [CHANNEL_TRADE, CHANNEL_FILL], "market_tickers": ["MKT-A", "MKT-B"]},
    }
    assert sock.sent[1]["id"] == 2
    # default channels
    assert sock.sent[1]["params"]["channels"] == [CHANNEL_ORDERBOOK_DELTA, CHANNEL_TICKER]


async def test_subscribe_rejects_unknown_channel_without_sending(ws, transport):
    sock = await _connected(ws, transport)
    with pytest.raises(ValueError):
        await ws.subscribe(["MKT"], ["bogus"])
    assert sock.sent == []
    assert ws._sub_state.tickers == set()


async def test_subscribe_requires_connection(ws):
    with pytest.raises(RuntimeError):
        await ws.subscribe(["MKT"])


async def test_unsubscribe_message_and_tracking(ws, transport):
    sock = await _connected(ws, transport)
    await ws.subscribe(["A", "B"], [CHANNEL_TICKER])
    await ws.unsubscribe(["A"])
    msg = sock.sent[-1]
    assert msg["cmd"] == "unsubscribe"
    assert msg["params"] == {"channels": [CHANNEL_TICKER], "market_tickers": ["A"]}
    assert ws._sub_state.tickers == {"B"}


async def test_unsubscribe_when_disconnected_is_silent(ws):
    await ws.unsubscribe(["A"])  # must not raise


# -------------------------------------------------------------------- dispatch


@pytest.mark.parametrize(
    "msg_type,register,event",
    [
        ("ticker", "on_ticker", EVENT_PRICE_UPDATE),
        ("orderbook_delta", "on_orderbook", EVENT_ORDERBOOK_UPDATE),
        ("orderbook_snapshot", "on_orderbook", EVENT_ORDERBOOK_UPDATE),
        ("trade", "on_trade", EVENT_TRADE_EXECUTED),
        ("fill", "on_fill", EVENT_FILL_RECEIVED),
    ],
)
async def test_each_channel_reaches_its_callback_and_the_event_bus(ws, bus, msg_type, register, event):
    got, bus_got = [], []

    async def cb(m):
        got.append(m)

    async def on_event(e):
        bus_got.append(e)

    getattr(ws, register)(cb)
    bus.subscribe(event, on_event)
    payload = {"type": msg_type, "msg": {"market_ticker": "MKT"}}
    await ws._dispatch(json.dumps(payload))
    assert got == [payload]
    assert len(bus_got) == 1 and bus_got[0].event_type == event and bus_got[0].data == payload


async def test_callbacks_are_channel_isolated(ws, bus):
    ticks, trades = [], []

    async def t(m):
        ticks.append(m)

    async def tr(m):
        trades.append(m)

    ws.on_ticker(t)
    ws.on_trade(tr)
    await ws._dispatch(json.dumps({"type": "ticker"}))
    assert len(ticks) == 1 and trades == []


def test_on_decorators_return_the_callback(ws):
    async def cb(m):
        pass

    assert ws.on_ticker(cb) is cb and ws.on_fill(cb) is cb


async def test_failing_callback_does_not_block_later_callbacks_or_bus(ws, bus):
    order, bus_got = [], []

    async def bad(m):
        raise RuntimeError("boom")

    async def good(m):
        order.append(m["type"])

    async def on_event(e):
        bus_got.append(e)

    ws.on_ticker(bad)
    ws.on_ticker(good)
    bus.subscribe(EVENT_PRICE_UPDATE, on_event)
    await ws._dispatch(json.dumps({"type": "ticker"}))
    assert order == ["ticker"] and len(bus_got) == 1


async def test_publish_to_event_bus_false_skips_bus(key_path, bus):
    w = KalshiWebSocket(api_key="k", private_key_path=key_path[0], publish_to_event_bus=False)
    seen = []

    async def on_event(e):
        seen.append(e)

    bus.subscribe(EVENT_PRICE_UPDATE, on_event)
    await w._dispatch(json.dumps({"type": "ticker"}))
    assert seen == []


@pytest.mark.parametrize("raw", ["not json", "", '{"type": "subscribed"}', '{"type": "error"}', "{}"])
async def test_garbage_and_control_messages_are_ignored(ws, bus, raw):
    seen = []

    async def cb(m):
        seen.append(m)

    for reg in (ws.on_ticker, ws.on_orderbook, ws.on_trade, ws.on_fill):
        reg(cb)
    await ws._dispatch(raw)
    assert seen == []


# --------------------------------------------------------------------- run loop


async def test_run_dispatches_socket_messages_then_close_stops_it(ws, transport, bus, delays):
    sock = await _connected(ws, transport, FakeWS([json.dumps({"type": "trade", "n": 1}), json.dumps({"type": "fill", "n": 2})]))
    got = []

    async def cb(m):
        got.append(m["n"])
        if len(got) == 2:
            await ws.close()

    ws.on_trade(cb)
    ws.on_fill(cb)
    await asyncio.wait_for(ws.run(), 5)
    assert got == [1, 2]
    assert sock.closed and ws.state is ConnectionState.DISCONNECTED
    assert len(transport.calls) == 1  # no reconnect after a requested close


async def test_close_is_clean_when_socket_close_raises(ws, transport):
    sock = await _connected(ws, transport)
    sock.close_raises = OSError("already dead")
    await ws.close()
    assert ws.state is ConnectionState.DISCONNECTED


async def test_close_without_connect_is_safe(ws):
    await ws.close()
    assert ws.state is ConnectionState.DISCONNECTED


async def test_async_context_manager_connects_and_closes(ws, transport):
    sock = FakeWS()
    transport.sockets.append(sock)
    async with ws as w:
        assert w is ws and ws.is_connected
    assert sock.closed and ws.state is ConnectionState.DISCONNECTED


async def test_keepalive_pings_while_connected(ws, transport, monkeypatch):
    sock = await _connected(ws, transport)
    monkeypatch.setattr(kalshi_ws, "_PING_INTERVAL_S", 0.01)
    ws._should_run = True
    task = asyncio.create_task(ws._keepalive_loop())
    await _REAL_SLEEP(0.1)
    ws._should_run = False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert sock.pings >= 2


async def test_keepalive_survives_a_failed_ping(ws, transport, monkeypatch):
    sock = await _connected(ws, transport)
    monkeypatch.setattr(kalshi_ws, "_PING_INTERVAL_S", 0.01)
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("ping failed")

    sock.ping = flaky
    ws._should_run = True
    task = asyncio.create_task(ws._keepalive_loop())
    await _REAL_SLEEP(0.1)
    ws._should_run = False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert calls["n"] >= 2


# ------------------------------------------------------------------- reconnect


async def test_unexpected_disconnect_reconnects_and_resubscribes(ws, transport, bus, delays):
    first = await _connected(ws, transport, FakeWS())
    await ws.subscribe(["A", "B"], [CHANNEL_TICKER])
    second = FakeWS([json.dumps({"type": "ticker", "n": 7})])
    transport.sockets.append(second)
    got = []

    async def cb(m):
        got.append(m["n"])
        await ws.close()

    ws.on_ticker(cb)
    first.inbox.put_nowait(_closed())
    await asyncio.wait_for(ws.run(), 5)

    assert got == [7]  # message from the NEW socket was delivered
    assert len(transport.calls) == 2
    resub = second.sent[0]
    assert resub["cmd"] == "subscribe"
    assert resub["params"]["channels"] == [CHANNEL_TICKER]
    assert sorted(resub["params"]["market_tickers"]) == ["A", "B"]


async def test_reconnect_does_not_resubscribe_when_nothing_was_subscribed(ws, transport, delays):
    first = await _connected(ws, transport, FakeWS())
    second = FakeWS([json.dumps({"type": "ticker"})])
    transport.sockets.append(second)

    async def cb(m):
        await ws.close()

    ws.on_ticker(cb)
    first.inbox.put_nowait(_closed())
    await asyncio.wait_for(ws.run(), 5)
    assert second.sent == []


async def test_reconnect_backoff_doubles_and_caps(ws, transport, delays):
    first = await _connected(ws, transport, FakeWS())
    # 8 failures then success; backoff must go 1,2,4,...,capped at 60
    transport.sockets.extend([OSError("down")] * 8)
    final = FakeWS([json.dumps({"type": "ticker"})])
    transport.sockets.append(final)

    async def cb(m):
        await ws.close()

    ws.on_ticker(cb)
    first.inbox.put_nowait(_closed())
    await asyncio.wait_for(ws.run(), 5)
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


async def test_close_during_backoff_stops_reconnecting(ws, transport, delays, monkeypatch):
    first = await _connected(ws, transport, FakeWS())
    transport.sockets.extend([OSError("down")] * 50)

    real = kalshi_ws.asyncio.sleep

    async def sleep_then_close(d, *a):
        if d == 2.0:  # second backoff
            await ws.close()
        await real(d, *a)

    monkeypatch.setattr(kalshi_ws.asyncio, "sleep", sleep_then_close)
    first.inbox.put_nowait(_closed())
    await asyncio.wait_for(ws.run(), 5)
    assert len(transport.calls) <= 3
    assert ws.state is ConnectionState.DISCONNECTED


async def test_closed_by_us_does_not_reconnect(ws, transport, delays):
    sock = await _connected(ws, transport)
    task = asyncio.create_task(ws.run())
    await _REAL_SLEEP(0.01)
    await ws.close()
    await asyncio.wait_for(task, 5)
    assert sock.closed and len(transport.calls) == 1


async def test_unexpected_exception_in_loop_triggers_reconnect(ws, transport, delays):
    first = await _connected(ws, transport, FakeWS())
    second = FakeWS([json.dumps({"type": "ticker"})])
    transport.sockets.append(second)

    async def cb(m):
        await ws.close()

    ws.on_ticker(cb)
    first.inbox.put_nowait(RuntimeError("socket exploded"))
    await asyncio.wait_for(ws.run(), 5)
    assert len(transport.calls) == 2


# ------------------------------------------------- sequence gaps / resubscribe


def _ob(kind, sid, seq):
    return json.dumps({"type": kind, "sid": sid, "seq": seq, "msg": {}})


async def _collecting(ws):
    got = []

    async def cb(m):
        got.append((m["sid"], m["seq"], m["type"]))

    ws.on_orderbook(cb)
    return got


async def test_in_order_sequence_passes_through_without_resync(ws, transport):
    sock = await _connected(ws, transport)
    got = await _collecting(ws)
    for kind, seq in [("orderbook_snapshot", 1), ("orderbook_delta", 2), ("orderbook_delta", 3)]:
        await ws._dispatch(_ob(kind, 7, seq))
    assert [g[1] for g in got] == [1, 2, 3]
    assert sock.sent == []


async def test_sequence_gap_drops_message_and_requests_snapshot(ws, transport):
    sock = await _connected(ws, transport)
    got = await _collecting(ws)
    await ws._dispatch(_ob("orderbook_snapshot", 7, 1))
    await ws._dispatch(_ob("orderbook_delta", 7, 4))  # 2, 3 missing
    assert [g[1] for g in got] == [1]
    assert len(sock.sent) == 1
    assert sock.sent[0]["cmd"] == "update_subscription"
    assert sock.sent[0]["params"] == {"sid": 7, "action": "get_snapshot"}


async def test_deltas_stay_dropped_until_snapshot_then_resume(ws, transport):
    sock = await _connected(ws, transport)
    got = await _collecting(ws)
    await ws._dispatch(_ob("orderbook_snapshot", 7, 1))
    await ws._dispatch(_ob("orderbook_delta", 7, 4))
    await ws._dispatch(_ob("orderbook_delta", 7, 5))  # still stale: no 2nd request
    assert len(sock.sent) == 1
    await ws._dispatch(_ob("orderbook_snapshot", 7, 6))
    await ws._dispatch(_ob("orderbook_delta", 7, 7))
    assert [g[1] for g in got] == [1, 6, 7]


async def test_gap_warning_is_logged(ws, transport, monkeypatch):
    from unittest.mock import MagicMock

    await _connected(ws, transport)
    await ws._dispatch(_ob("orderbook_snapshot", 7, 1))
    log = MagicMock()
    monkeypatch.setattr(type(ws), "logger", log, raising=False)
    await ws._dispatch(_ob("orderbook_delta", 7, 9))
    assert any(c.args and "gap" in c.args[0].lower() for c in log.warning.call_args_list)


async def test_duplicate_and_old_seq_are_dropped(ws, transport):
    sock = await _connected(ws, transport)
    got = await _collecting(ws)
    for seq in (1, 2, 2, 1, 3):
        await ws._dispatch(_ob("orderbook_delta", 7, seq))
    assert [g[1] for g in got] == [1, 2, 3]
    assert sock.sent == []  # duplicates are not gaps


async def test_seq_is_tracked_per_sid(ws, transport):
    sock = await _connected(ws, transport)
    got = await _collecting(ws)
    for sid, seq in [(1, 1), (2, 5), (1, 2), (2, 6), (1, 3)]:
        await ws._dispatch(_ob("orderbook_delta", sid, seq))
    assert got == [(1, 1, "orderbook_delta"), (2, 5, "orderbook_delta"), (1, 2, "orderbook_delta"),
                   (2, 6, "orderbook_delta"), (1, 3, "orderbook_delta")]
    assert sock.sent == []
    await ws._dispatch(_ob("orderbook_delta", 2, 9))  # gap on sid 2 only
    assert sock.sent[-1]["params"]["sid"] == 2
    await ws._dispatch(_ob("orderbook_delta", 1, 4))  # sid 1 unaffected
    assert got[-1] == (1, 4, "orderbook_delta")


async def test_failed_snapshot_request_is_retried_on_next_message(ws, transport):
    sock = await _connected(ws, transport)
    got = await _collecting(ws)
    await ws._dispatch(_ob("orderbook_delta", 7, 1))
    real_send, calls = sock.send, {"n": 0}

    async def flaky(data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("send failed")
        await real_send(data)

    sock.send = flaky
    await ws._dispatch(_ob("orderbook_delta", 7, 5))
    await ws._dispatch(_ob("orderbook_delta", 7, 6))
    assert [g[1] for g in got] == [1]
    assert len(sock.sent) == 1 and sock.sent[0]["params"]["action"] == "get_snapshot"


async def test_messages_without_sid_or_seq_pass_through(ws, transport):
    await _connected(ws, transport)
    got = []

    async def cb(m):
        got.append(m)

    ws.on_ticker(cb)
    for m in ({"type": "ticker"}, {"type": "ticker", "sid": 1}, {"type": "ticker", "sid": 1, "seq": "x"},
              {"type": "ticker", "sid": 1, "seq": True}):
        await ws._dispatch(json.dumps(m))
    assert len(got) == 4


async def test_reconnect_resets_sequence_state(ws, transport, delays):
    first = await _connected(ws, transport, FakeWS())
    second = FakeWS([_ob("orderbook_delta", 1, 1)])  # new connection restarts seq at 1
    transport.sockets.append(second)
    got = await _collecting(ws)

    async def cb(m):
        await ws.close()

    await ws._dispatch(_ob("orderbook_delta", 1, 5))
    ws.on_orderbook(cb)
    first.inbox.put_nowait(_closed())
    await asyncio.wait_for(ws.run(), 5)
    assert [g[1] for g in got] == [5, 1]
    assert second.sent == []


async def test_reconnect_restores_exactly_the_prior_subscriptions(ws, transport, delays):
    first = await _connected(ws, transport, FakeWS())
    await ws.subscribe(["A"], [CHANNEL_TICKER])
    await ws.subscribe(["B"], [CHANNEL_TRADE, CHANNEL_ORDERBOOK_DELTA])
    second = FakeWS([json.dumps({"type": "ticker"})])
    transport.sockets.append(second)

    async def cb(m):
        await ws.close()

    ws.on_ticker(cb)
    first.inbox.put_nowait(_closed())
    await asyncio.wait_for(ws.run(), 5)

    restored = {(c, t) for m in second.sent for c in m["params"]["channels"] for t in m["params"]["market_tickers"]}
    assert restored == {(CHANNEL_TICKER, "A"), (CHANNEL_TRADE, "B"), (CHANNEL_ORDERBOOK_DELTA, "B")}
    assert all(m["cmd"] == "subscribe" for m in second.sent)


async def test_reconnect_does_not_restore_unsubscribed_tickers(ws, transport, delays):
    first = await _connected(ws, transport, FakeWS())
    await ws.subscribe(["A", "B"], [CHANNEL_TICKER])
    await ws.unsubscribe(["A"])
    second = FakeWS([json.dumps({"type": "ticker"})])
    transport.sockets.append(second)

    async def cb(m):
        await ws.close()

    ws.on_ticker(cb)
    first.inbox.put_nowait(_closed())
    await asyncio.wait_for(ws.run(), 5)
    assert [m["params"]["market_tickers"] for m in second.sent] == [["B"]]


def _ack(msg_id, channel, sid):
    return json.dumps({"id": msg_id, "type": "subscribed", "msg": {"channel": channel, "sid": sid}})


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(kalshi_ws, "_now", lambda: t["now"])
    return t


async def _gapped(ws, transport, sid=7):
    """Connected, subscribed to A on orderbook_delta (acked as ``sid``), then a gap."""
    sock = await _connected(ws, transport)
    await ws.subscribe(["A"], [CHANNEL_ORDERBOOK_DELTA])
    await ws._dispatch(_ack(sock.sent[-1]["id"], CHANNEL_ORDERBOOK_DELTA, sid))
    await ws._dispatch(_ob("orderbook_snapshot", sid, 1))
    await ws._dispatch(_ob("orderbook_delta", sid, 5))  # gap -> get_snapshot sent
    return sock


async def test_snapshot_in_time_means_no_fallback(ws, transport, clock):
    sock = await _gapped(ws, transport)
    clock["now"] += 4.9
    await ws._dispatch(_ob("orderbook_snapshot", 7, 6))
    clock["now"] += 60
    await ws._dispatch(_ob("orderbook_delta", 7, 7))
    await ws._check_snapshot_timeouts()
    assert [m["cmd"] for m in sock.sent[1:]] == ["update_subscription"]


async def test_snapshot_timeout_unsubscribes_then_resubscribes_and_tracks_new_sid(ws, transport, clock):
    sock = await _gapped(ws, transport)
    got = await _collecting(ws)
    clock["now"] += 5.0
    await ws._dispatch(_ob("orderbook_delta", 7, 6))  # message for the sid triggers the check
    assert sock.sent[-1] == {"id": sock.sent[-1]["id"], "cmd": "unsubscribe", "params": {"sids": [7]}}
    n = len(sock.sent)
    await ws._dispatch(_ob("orderbook_delta", 7, 7))  # old sid stays dropped meanwhile
    assert len(sock.sent) == n and got == []
    # docs: wait for 'unsubscribed', then subscribe again
    await ws._dispatch(json.dumps({"type": "unsubscribed", "sid": 7, "seq": 8}))
    resub = sock.sent[-1]
    assert resub["cmd"] == "subscribe"
    assert resub["params"] == {"channels": [CHANNEL_ORDERBOOK_DELTA], "market_tickers": ["A"]}
    # new sid 9 gets its own baseline
    await ws._dispatch(_ack(resub["id"], CHANNEL_ORDERBOOK_DELTA, 9))
    await ws._dispatch(_ob("orderbook_snapshot", 9, 1))
    await ws._dispatch(_ob("orderbook_delta", 9, 2))
    assert [(g[0], g[1]) for g in got] == [(9, 1), (9, 2)]
    await ws._dispatch(_ob("orderbook_delta", 9, 5))  # gap on the NEW sid is detected
    assert sock.sent[-1]["params"] == {"sid": 9, "action": "get_snapshot"}


async def test_fallback_fires_only_once_per_gap(ws, transport, clock):
    sock = await _gapped(ws, transport)
    clock["now"] += 6
    for seq in (6, 7, 8):
        await ws._dispatch(_ob("orderbook_delta", 7, seq))
    await ws._check_snapshot_timeouts()
    clock["now"] += 1
    await ws._check_snapshot_timeouts()
    assert [m["cmd"] for m in sock.sent].count("unsubscribe") == 1


async def test_no_repeat_fallback_after_resubscribe_completes(ws, transport, clock):
    sock = await _gapped(ws, transport)
    clock["now"] += 5
    await ws._check_snapshot_timeouts()
    await ws._dispatch(json.dumps({"type": "unsubscribed", "sid": 7, "seq": 8}))
    n = len(sock.sent)
    assert ws._awaiting_snapshot == {}
    clock["now"] += 30
    await ws._check_snapshot_timeouts()
    await ws._check_snapshot_timeouts()
    assert len(sock.sent) == n


async def test_missing_unsubscribed_ack_resubscribes_after_timeout(ws, transport, clock):
    sock = await _gapped(ws, transport)
    clock["now"] += 5
    await ws._check_snapshot_timeouts()
    assert sock.sent[-1]["cmd"] == "unsubscribe"
    clock["now"] += 5
    await ws._check_snapshot_timeouts()
    assert sock.sent[-1]["cmd"] == "subscribe"


async def test_run_loop_checks_timeout_without_traffic(ws, transport, clock, delays):
    sock = await _gapped(ws, transport)
    clock["now"] += 6
    task = asyncio.create_task(ws.run())
    await _REAL_SLEEP(0.05)
    assert sock.sent[-1]["cmd"] == "unsubscribe"
    await ws.close()
    await asyncio.wait_for(task, 5)


async def test_subscribed_acks_map_channels_to_sids(ws, transport):
    sock = await _connected(ws, transport)
    await ws.subscribe(["A", "B"], [CHANNEL_TICKER, CHANNEL_TRADE])
    mid = sock.sent[-1]["id"]
    await ws._dispatch(_ack(mid, CHANNEL_TICKER, 3))
    await ws._dispatch(_ack(mid, CHANNEL_TRADE, 4))
    assert ws._sid_info == {3: (CHANNEL_TICKER, {"A", "B"}), 4: (CHANNEL_TRADE, {"A", "B"})}
    assert ws._pending_acks == {}


async def test_unsubscribe_all_tickers_of_a_sid_sends_sids(ws, transport):
    sock = await _connected(ws, transport)
    await ws.subscribe(["A", "B"], [CHANNEL_TICKER, CHANNEL_TRADE])
    mid = sock.sent[-1]["id"]
    await ws._dispatch(_ack(mid, CHANNEL_TICKER, 3))
    await ws._dispatch(_ack(mid, CHANNEL_TRADE, 4))
    await ws.unsubscribe(["A", "B"])
    msgs = sock.sent[1:]
    assert len(msgs) == 1
    assert msgs[0]["cmd"] == "unsubscribe" and msgs[0]["params"] == {"sids": [3, 4]}
    assert ws._sid_info == {}


async def test_unsubscribe_some_tickers_trims_instead_of_dropping_the_sid(ws, transport):
    sock = await _connected(ws, transport)
    await ws.subscribe(["A", "B"], [CHANNEL_TICKER])
    await ws._dispatch(_ack(sock.sent[-1]["id"], CHANNEL_TICKER, 3))
    await ws.unsubscribe(["A"])
    msg = sock.sent[-1]
    assert msg["cmd"] == "update_subscription"
    assert msg["params"] == {"sid": 3, "market_tickers": ["A"], "action": "delete_markets"}
    assert ws._sid_info[3] == (CHANNEL_TICKER, {"B"})


async def test_unsubscribe_unknown_sid_keeps_legacy_message_and_logs(ws, transport, monkeypatch):
    from unittest.mock import MagicMock

    sock = await _connected(ws, transport)
    await ws.subscribe(["A", "B"], [CHANNEL_TICKER])  # never acked
    log = MagicMock()
    monkeypatch.setattr(type(ws), "logger", log, raising=False)
    await ws.unsubscribe(["A"])
    assert sock.sent[-1]["params"] == {"channels": [CHANNEL_TICKER], "market_tickers": ["A"]}
    assert log.warning.called


async def test_unsubscribe_mixes_known_and_unknown(ws, transport):
    sock = await _connected(ws, transport)
    await ws.subscribe(["A"], [CHANNEL_TICKER])
    await ws._dispatch(_ack(sock.sent[-1]["id"], CHANNEL_TICKER, 3))
    await ws.subscribe(["B"], [CHANNEL_TRADE])  # unacked
    await ws.unsubscribe(["A", "B"])
    cmds = [(m["cmd"], m["params"]) for m in sock.sent[2:]]
    assert cmds[0] == ("unsubscribe", {"sids": [3]})
    assert cmds[1][0] == "unsubscribe" and cmds[1][1]["market_tickers"] == ["B"]


async def test_valid_json_non_object_frame_does_not_drop_the_connection(ws, transport, delays):
    await _connected(ws, transport, FakeWS(["[1, 2, 3]", json.dumps({"type": "ticker"})]))
    transport.sockets.append(FakeWS([json.dumps({"type": "ticker"})]))  # only reached by a (wrong) reconnect

    async def cb(m):
        await ws.close()

    ws.on_ticker(cb)
    await asyncio.wait_for(ws.run(), 5)
    assert len(transport.calls) == 1


async def test_event_bus_failure_does_not_drop_the_connection(ws, transport, bus, delays, monkeypatch):
    await _connected(ws, transport, FakeWS([json.dumps({"type": "ticker"}), json.dumps({"type": "trade"})]))
    transport.sockets.append(FakeWS([json.dumps({"type": "trade"})]))  # only reached by a (wrong) reconnect
    published = []

    async def flaky_publish(event_type, data):
        published.append(event_type)
        if len(published) == 1:
            raise RuntimeError("bus down")

    monkeypatch.setattr(bus, "publish", flaky_publish)

    async def cb(m):
        await ws.close()

    ws.on_trade(cb)  # ticker does not close; only the second frame does
    await asyncio.wait_for(ws.run(), 5)
    assert len(transport.calls) == 1


async def test_cancellation_during_bus_publish_propagates(ws, bus, monkeypatch):
    """The bus guard must contain errors, not cancellation: a CancelledError
    raised inside publish has to escape _dispatch so task.cancel() still works."""

    async def cancelled(event_type, data):
        raise asyncio.CancelledError()

    monkeypatch.setattr(bus, "publish", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await ws._dispatch(json.dumps({"type": "ticker"}))
