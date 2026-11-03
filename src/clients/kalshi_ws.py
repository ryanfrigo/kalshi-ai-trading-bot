"""
Kalshi WebSocket client for real-time market data streaming.

Connects to the Kalshi WebSocket API with RSA PSS authentication, subscribes
to live channels (orderbook_delta, ticker, trade, fill), and dispatches
incoming messages to registered callbacks and/or the global EventBus.
"""

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import websockets
import websockets.exceptions
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.config.settings import settings
from src.events.event_bus import (
    EVENT_FILL_RECEIVED,
    EVENT_ORDERBOOK_UPDATE,
    EVENT_PRICE_UPDATE,
    EVENT_TRADE_EXECUTED,
    EventBus,
)
from src.utils.logging_setup import TradingLoggerMixin


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_PATH = "/trade-api/ws/v2"

# Channel names accepted by the Kalshi WebSocket API
CHANNEL_ORDERBOOK_DELTA = "orderbook_delta"
CHANNEL_TICKER = "ticker"
CHANNEL_TRADE = "trade"
CHANNEL_FILL = "fill"
ALL_CHANNELS = {CHANNEL_ORDERBOOK_DELTA, CHANNEL_TICKER, CHANNEL_TRADE, CHANNEL_FILL}

# Reconnect parameters
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 60.0
_BACKOFF_MULTIPLIER = 2.0

# A sid that has a sequence gap must receive an orderbook_snapshot within this
# long, else we unsubscribe and resubscribe it (and give the unsubscribe ack the
# same long to arrive).
_SNAPSHOT_TIMEOUT_S = 5.0
_RECV_TIMEOUT_S = 30.0


def _now() -> float:
    return time.monotonic()


# Keepalive interval
_PING_INTERVAL_S = 10.0


class ConnectionState(str, Enum):
    """WebSocket connection lifecycle states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


# Type alias for user-facing message callbacks
MessageCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class _SubscriptionState:
    """Tracks which tickers/channels are actively subscribed."""
    tickers: Set[str] = field(default_factory=set)
    channels: Set[str] = field(default_factory=set)
    # Exact (channel -> tickers) pairs, so a reconnect restores what was
    # subscribed rather than the tickers x channels cross product.
    by_channel: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class _GapState:
    """A sid whose book is stale after a sequence gap, until a snapshot lands."""
    since: float
    requested: bool = False                  # get_snapshot request was sent
    unsubscribed_at: Optional[float] = None  # fallback unsubscribe was sent


class KalshiWebSocket(TradingLoggerMixin):
    """
    Authenticated WebSocket client for Kalshi real-time market data.

    Features:
        * RSA PSS authentication identical to the REST client.
        * Subscribe / unsubscribe to orderbook_delta, ticker, trade, fill channels.
        * Per-channel callback registration (``on_ticker``, ``on_orderbook``, etc.).
        * Automatic reconnection with exponential backoff.
        * Periodic ping/pong keepalive (every 10 s).
        * Graceful shutdown via ``close()``.
        * Publishes events to the global :class:`EventBus` for system-wide consumption.

    Example::

        ws = KalshiWebSocket()

        @ws.on_ticker
        async def handle_ticker(msg):
            print(msg)

        await ws.connect()
        await ws.subscribe(["AAPL-24"], [CHANNEL_TICKER])
        await ws.run()  # blocks until close() is called
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_path: str = "kalshi_private_key",
        publish_to_event_bus: bool = True,
    ) -> None:
        """
        Initialize the WebSocket client.

        Args:
            api_key: Kalshi API key (defaults to ``settings.api.kalshi_api_key``).
            private_key_path: Path to the PEM-encoded RSA private key.
            publish_to_event_bus: If True, every incoming message is also
                published to the global :class:`EventBus`.
        """
        self.api_key: str = api_key or settings.api.kalshi_api_key
        self.private_key_path: str = private_key_path
        self.publish_to_event_bus: bool = publish_to_event_bus

        self._private_key: Any = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._state: ConnectionState = ConnectionState.DISCONNECTED
        self._should_run: bool = False
        self._msg_id_counter: int = 0

        # Subscription tracking (for resubscription after reconnect)
        self._sub_state = _SubscriptionState()

        # Sequence tracking, per subscription id (sid). seq is only comparable
        # within one sid on one connection, so both are cleared on reconnect.
        self._last_seq: Dict[int, int] = {}
        self._awaiting_snapshot: Dict[int, _GapState] = {}

        # sid bookkeeping from 'subscribed' acks: sid -> (channel, tickers), and
        # subscribe-command id -> (channels still awaiting their ack, tickers).
        self._sid_info: Dict[int, tuple] = {}
        self._pending_acks: Dict[int, tuple] = {}

        # Per-channel user callbacks
        self._callbacks: Dict[str, List[MessageCallback]] = {
            CHANNEL_TICKER: [],
            CHANNEL_ORDERBOOK_DELTA: [],
            CHANNEL_TRADE: [],
            CHANNEL_FILL: [],
        }

        # Load RSA private key on init
        self._load_private_key()

        self.logger.info(
            "KalshiWebSocket initialized",
            api_key_length=len(self.api_key) if self.api_key else 0,
            publish_to_event_bus=self.publish_to_event_bus,
        )

    # ------------------------------------------------------------------
    # Private key / signing (mirrors kalshi_client.py)
    # ------------------------------------------------------------------

    def _load_private_key(self) -> None:
        """Load the RSA private key from disk."""
        key_path = Path(self.private_key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key file not found: {self.private_key_path}")
        with open(key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.logger.info("Private key loaded for WebSocket auth")

    def _sign(self, timestamp: str, method: str, path: str) -> str:
        """
        Produce a base64-encoded RSA PSS signature.

        The message format is ``timestamp + method + path`` matching the Kalshi
        REST authentication scheme.
        """
        message = (timestamp + method.upper() + path).encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _build_auth_headers(self) -> Dict[str, str]:
        """Build the three authentication headers required by the WS endpoint."""
        timestamp = str(int(time.time() * 1000))
        signature = self._sign(timestamp, "GET", WS_PATH)
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    def _next_msg_id(self) -> int:
        self._msg_id_counter += 1
        return self._msg_id_counter

    async def connect(self) -> None:
        """
        Establish an authenticated WebSocket connection.

        Raises:
            Exception: If the connection or authentication handshake fails.
        """
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            self.logger.warning("connect() called but already connected/connecting")
            return

        self._state = ConnectionState.CONNECTING

        base_url = settings.api.kalshi_base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{base_url}{WS_PATH}"
        headers = self._build_auth_headers()

        self.logger.info("Connecting to Kalshi WebSocket", url=ws_url)
        try:
            self._ws = await websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=None,   # we handle pings ourselves
                ping_timeout=None,
                close_timeout=5,
            )
            self._state = ConnectionState.CONNECTED
            self.logger.info("WebSocket connected")
        except Exception as exc:
            self._state = ConnectionState.DISCONNECTED
            self.logger.error("WebSocket connection failed", error=str(exc))
            raise

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff and resubscribe."""
        backoff = _INITIAL_BACKOFF_S
        self._state = ConnectionState.RECONNECTING

        while self._should_run:
            self.logger.info("Attempting reconnect", backoff_s=backoff)
            try:
                await self.connect()

                # New connection => new sids and fresh snapshots.
                self._last_seq.clear()
                self._awaiting_snapshot.clear()
                self._sid_info.clear()
                self._pending_acks.clear()

                # Resubscribe to exactly the (channel, tickers) pairs that were
                # active, grouping channels that share the same ticker set.
                groups: Dict[tuple, List[str]] = {}
                for channel, tickers in self._sub_state.by_channel.items():
                    if tickers:
                        groups.setdefault(tuple(sorted(tickers)), []).append(channel)
                for tickers, channels in groups.items():
                    await self.subscribe(list(tickers), sorted(channels))
                self.logger.info("Reconnected and resubscribed successfully")
                return
            except Exception as exc:
                self.logger.warning(
                    "Reconnect attempt failed",
                    error=str(exc),
                    next_backoff_s=min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_S),
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_S)

        self.logger.info("Reconnect loop exited (should_run=False)")

    # ------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        tickers: List[str],
        channels: Optional[List[str]] = None,
    ) -> None:
        """
        Subscribe to real-time channels for the given market tickers.

        Args:
            tickers: List of market ticker strings.
            channels: Channels to subscribe to. Defaults to
                ``["orderbook_delta", "ticker"]``.
        """
        if not self.is_connected:
            raise RuntimeError("Cannot subscribe: WebSocket is not connected")

        channels = channels or [CHANNEL_ORDERBOOK_DELTA, CHANNEL_TICKER]
        invalid = set(channels) - ALL_CHANNELS
        if invalid:
            raise ValueError(f"Invalid channels: {invalid}. Must be one of {ALL_CHANNELS}")

        msg = {
            "id": self._next_msg_id(),
            "cmd": "subscribe",
            "params": {
                "channels": channels,
                "market_tickers": tickers,
            },
        }
        self._pending_acks[msg["id"]] = (set(channels), set(tickers))
        await self._ws.send(json.dumps(msg))

        # Track state for reconnection
        self._sub_state.tickers.update(tickers)
        self._sub_state.channels.update(channels)
        for channel in channels:
            self._sub_state.by_channel.setdefault(channel, set()).update(tickers)

        self.logger.info(
            "Subscribed",
            tickers=tickers,
            channels=channels,
        )

    async def unsubscribe(self, tickers: List[str]) -> None:
        """
        Unsubscribe from all channels for the given tickers.

        Args:
            tickers: Tickers to unsubscribe from.
        """
        if not self.is_connected:
            self.logger.warning("Cannot unsubscribe: not connected")
            return

        # https://docs.kalshi.com/websockets/websocket-connection : unsubscribe
        # takes {"sids": [...]} and drops the whole subscription, so a sid that
        # also carries other tickers is trimmed with update_subscription /
        # delete_markets instead (the docs show that action; the sid form follows
        # the same schema).
        wanted = set(tickers)
        whole: List[int] = []
        partial: List[tuple] = []
        known: Set[str] = set()
        for sid, (_channel, subscribed) in self._sid_info.items():
            known |= subscribed
            hit = subscribed & wanted
            if hit and hit == subscribed:
                whole.append(sid)
            elif hit:
                partial.append((sid, sorted(hit)))

        if whole:
            await self._ws.send(json.dumps({
                "id": self._next_msg_id(),
                "cmd": "unsubscribe",
                "params": {"sids": whole},
            }))
        for sid, hit in partial:
            await self._ws.send(json.dumps({
                "id": self._next_msg_id(),
                "cmd": "update_subscription",
                "params": {"sid": sid, "market_tickers": hit, "action": "delete_markets"},
            }))

        unknown = sorted(wanted - known)
        if unknown:
            # No subscribed ack seen for these: keep the legacy message.
            self.logger.warning("Unsubscribe for tickers with unknown sid; sending legacy message", tickers=unknown)
            await self._ws.send(json.dumps({
                "id": self._next_msg_id(),
                "cmd": "unsubscribe",
                "params": {
                    "channels": list(self._sub_state.channels),
                    "market_tickers": unknown,
                },
            }))

        for sid in whole:
            self._sid_info.pop(sid, None)
            self._last_seq.pop(sid, None)
            self._awaiting_snapshot.pop(sid, None)
        for sid, hit in partial:
            self._sid_info[sid][1].difference_update(hit)
        self._sub_state.tickers.difference_update(tickers)
        for subscribed in self._sub_state.by_channel.values():
            subscribed.difference_update(tickers)

        self.logger.info("Unsubscribed", tickers=tickers)

    # ------------------------------------------------------------------
    # Callback registration (decorator-friendly)
    # ------------------------------------------------------------------

    def on_ticker(self, callback: MessageCallback) -> MessageCallback:
        """Register (or use as decorator) a callback for ticker channel messages."""
        self._callbacks[CHANNEL_TICKER].append(callback)
        return callback

    def on_orderbook(self, callback: MessageCallback) -> MessageCallback:
        """Register (or use as decorator) a callback for orderbook_delta channel messages."""
        self._callbacks[CHANNEL_ORDERBOOK_DELTA].append(callback)
        return callback

    def on_trade(self, callback: MessageCallback) -> MessageCallback:
        """Register (or use as decorator) a callback for trade channel messages."""
        self._callbacks[CHANNEL_TRADE].append(callback)
        return callback

    def on_fill(self, callback: MessageCallback) -> MessageCallback:
        """Register (or use as decorator) a callback for fill channel messages."""
        self._callbacks[CHANNEL_FILL].append(callback)
        return callback

    # ------------------------------------------------------------------
    # Sequence-gap detection
    # ------------------------------------------------------------------

    async def _accept_sequence(self, msg: Dict[str, Any], msg_type: str) -> bool:
        """
        Return True if ``msg`` should be dispatched, False if it must be dropped.

        Per https://docs.kalshi.com/websockets/orderbook-updates every message
        carries ``sid`` and a per-subscription ``seq``; deltas must be applied
        in seq order, and on a gap the local book must be replaced by a fresh
        snapshot ("Always replace your local book with a received snapshot
        before applying subsequent deltas"). A repeated ``subscribe`` alone does
        not resync, so we request ``update_subscription`` / ``get_snapshot``
        (https://docs.kalshi.com/websockets/websocket-connection, which lists
        the action but gives no example; the payload below follows its schema).

        Duplicate or older seq -> dropped. Gap -> the out-of-order message is
        dropped, a snapshot is requested, and deltas for that sid stay dropped
        until a snapshot arrives. Messages without an integer sid and seq are
        passed through untouched.
        """
        sid, seq = msg.get("sid"), msg.get("seq")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (sid, seq)):
            return True

        state = self._awaiting_snapshot.get(sid)
        if state is not None:
            if state.unsubscribed_at is None and msg_type == "orderbook_snapshot":
                del self._awaiting_snapshot[sid]
                self._last_seq[sid] = seq
                return True
            await self._check_snapshot_timeouts()
            if state.unsubscribed_at is None and not state.requested:
                await self._request_snapshot(sid)  # earlier request failed; retry
            return False

        last = self._last_seq.get(sid)
        if last is not None and seq <= last:
            self.logger.debug("Dropping duplicate/old message", sid=sid, seq=seq, last_seq=last)
            return False
        if last is not None and seq != last + 1:
            self.logger.warning(
                "Sequence gap detected; requesting fresh snapshot",
                sid=sid,
                expected_seq=last + 1,
                received_seq=seq,
            )
            self._awaiting_snapshot[sid] = _GapState(since=_now())
            await self._request_snapshot(sid)
            return False

        self._last_seq[sid] = seq
        return True

    async def _request_snapshot(self, sid: int) -> None:
        """Ask the server to resend an orderbook_snapshot for every market on ``sid``."""
        try:
            await self._ws.send(json.dumps({
                "id": self._next_msg_id(),
                "cmd": "update_subscription",
                "params": {"sid": sid, "action": "get_snapshot"},
            }))
            self._awaiting_snapshot[sid].requested = True
        except Exception as exc:
            self.logger.warning("Snapshot request failed", sid=sid, error=str(exc))

    async def _check_snapshot_timeouts(self) -> None:
        """
        Fallback for a snapshot that never comes (e.g. the get_snapshot payload
        is rejected): after ``_SNAPSHOT_TIMEOUT_S`` unsubscribe the sid, and once
        the 'unsubscribed' ack arrives (or the same timeout passes) subscribe its
        (channel, tickers) again. A new subscription gets a new sid and an
        initial snapshot (https://docs.kalshi.com/websockets/orderbook-updates:
        "unsubscribe, wait for unsubscribed, then subscribe again"). Once per gap.
        """
        now = _now()
        for sid, state in list(self._awaiting_snapshot.items()):
            if state.unsubscribed_at is None:
                if now - state.since < _SNAPSHOT_TIMEOUT_S:
                    continue
                self.logger.warning(
                    "No snapshot after sequence gap; unsubscribing to resubscribe",
                    sid=sid,
                    waited_s=round(now - state.since, 1),
                )
                try:
                    await self._ws.send(json.dumps({
                        "id": self._next_msg_id(),
                        "cmd": "unsubscribe",
                        "params": {"sids": [sid]},
                    }))
                    state.unsubscribed_at = now
                except Exception as exc:
                    self.logger.warning("Fallback unsubscribe failed", sid=sid, error=str(exc))
            elif now - state.unsubscribed_at >= _SNAPSHOT_TIMEOUT_S:
                self.logger.warning("No unsubscribed ack; resubscribing anyway", sid=sid)
                await self._resubscribe_sid(sid)

    async def _resubscribe_sid(self, sid: int) -> None:
        """Drop all state for ``sid`` and subscribe its (channel, tickers) afresh."""
        channel, tickers = self._sid_info.pop(sid, (None, None))
        if channel is None:  # never saw its ack; only orderbook sids get gap-checked
            channel = CHANNEL_ORDERBOOK_DELTA
            tickers = self._sub_state.by_channel.get(channel, set())
        self._awaiting_snapshot.pop(sid, None)
        self._last_seq.pop(sid, None)
        if not tickers:
            return
        try:
            await self.subscribe(sorted(tickers), [channel])
        except Exception as exc:
            self.logger.error("Resubscribe after sequence gap failed", sid=sid, error=str(exc))

    def _record_subscribed(self, msg: Dict[str, Any]) -> None:
        """Map a 'subscribed' ack (msg.channel, msg.sid) to the tickers we asked for."""
        body = msg.get("msg")
        pending = self._pending_acks.get(msg.get("id"))
        if not isinstance(body, dict) or pending is None:
            return
        channel, sid = body.get("channel"), body.get("sid")
        channels, tickers = pending
        if channel not in channels or not isinstance(sid, int) or isinstance(sid, bool):
            return
        channels.discard(channel)
        if not channels:
            del self._pending_acks[msg["id"]]
        # a repeated subscribe on a channel returns the same sid: union the tickers
        self._sid_info.setdefault(sid, (channel, set()))[1].update(tickers)

    async def _on_unsubscribed(self, msg: Dict[str, Any]) -> None:
        sid = msg.get("sid")
        state = self._awaiting_snapshot.get(sid)
        if state is not None and state.unsubscribed_at is not None:
            await self._resubscribe_sid(sid)
        else:
            self._sid_info.pop(sid, None)
            self._last_seq.pop(sid, None)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, raw: str) -> None:
        """Parse an incoming JSON message and fan out to callbacks + EventBus."""
        try:
            msg: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("Received non-JSON message", raw=raw[:200])
            return

        if not isinstance(msg, dict):
            self.logger.debug("Dropping non-object message", raw=raw[:200])
            return

        msg_type = msg.get("type", "")

        # Map Kalshi message types to our internal channel keys and EventBus event types
        channel_map: Dict[str, str] = {
            "ticker": CHANNEL_TICKER,
            "orderbook_delta": CHANNEL_ORDERBOOK_DELTA,
            "orderbook_snapshot": CHANNEL_ORDERBOOK_DELTA,
            "trade": CHANNEL_TRADE,
            "fill": CHANNEL_FILL,
        }

        event_bus_map: Dict[str, str] = {
            "ticker": EVENT_PRICE_UPDATE,
            "orderbook_delta": EVENT_ORDERBOOK_UPDATE,
            "orderbook_snapshot": EVENT_ORDERBOOK_UPDATE,
            "trade": EVENT_TRADE_EXECUTED,
            "fill": EVENT_FILL_RECEIVED,
        }

        if msg_type == "subscribed":
            self._record_subscribed(msg)
            return
        if msg_type == "unsubscribed":
            await self._on_unsubscribed(msg)
            return

        channel = channel_map.get(msg_type)
        if channel is None:
            # Subscription confirmations, errors, etc. -- log and skip
            self.logger.debug("Unhandled message type", msg_type=msg_type)
            return

        if not await self._accept_sequence(msg, msg_type):
            return

        # Fan out to registered per-channel callbacks
        for cb in self._callbacks.get(channel, []):
            try:
                await cb(msg)
            except Exception:
                self.logger.error(
                    "User callback error",
                    channel=channel,
                    exc_info=True,
                )

        # Publish to global EventBus
        if self.publish_to_event_bus:
            event_type = event_bus_map.get(msg_type)
            if event_type:
                try:
                    bus = EventBus.get_instance()
                    await bus.publish(event_type, msg)
                except Exception as exc:
                    self.logger.warning(
                        "EventBus publish error",
                        event_type=event_type,
                        error=str(exc),
                    )

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main event loop: read messages and dispatch.

        Blocks until :meth:`close` is called. Automatically reconnects on
        unexpected disconnections.
        """
        if not self.is_connected:
            raise RuntimeError("Must call connect() before run()")

        self._should_run = True
        self.logger.info("WebSocket event loop started")

        # Start the keepalive ping task
        ping_task = asyncio.create_task(self._keepalive_loop())

        try:
            while self._should_run:
                try:
                    await self._check_snapshot_timeouts()
                    # poll fast while a stale book is waiting on its snapshot
                    wait = _SNAPSHOT_TIMEOUT_S if self._awaiting_snapshot else _RECV_TIMEOUT_S
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=wait)
                    await self._dispatch(raw)
                except asyncio.TimeoutError:
                    # No message received within 30s -- normal, the ping keeps
                    # the connection alive.
                    continue
                except websockets.exceptions.ConnectionClosed as exc:
                    if not self._should_run:
                        break
                    self.logger.warning(
                        "WebSocket connection closed unexpectedly",
                        code=exc.code,
                        reason=exc.reason,
                    )
                    self._state = ConnectionState.DISCONNECTED
                    await self._reconnect()
                except Exception as exc:
                    if not self._should_run:
                        break
                    self.logger.error(
                        "Unexpected error in event loop",
                        error=str(exc),
                        exc_info=True,
                    )
                    self._state = ConnectionState.DISCONNECTED
                    await self._reconnect()
        finally:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
            self.logger.info("WebSocket event loop stopped")

    async def _keepalive_loop(self) -> None:
        """Send periodic WebSocket pings to keep the connection alive."""
        while self._should_run:
            try:
                await asyncio.sleep(_PING_INTERVAL_S)
                if self._ws and self.is_connected:
                    await self._ws.ping()
                    self.logger.debug("Keepalive ping sent")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.warning("Keepalive ping failed", error=str(exc))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully shut down the WebSocket connection."""
        self.logger.info("Closing WebSocket connection")
        self._should_run = False
        self._state = ConnectionState.CLOSING

        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                self.logger.warning("Error closing WebSocket", error=str(exc))

        self._state = ConnectionState.DISCONNECTED
        self.logger.info("WebSocket connection closed")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "KalshiWebSocket":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
