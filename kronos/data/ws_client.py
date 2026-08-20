"""
WebSocket client for real-time market data from OpenAlgo.

Connects to the Unified WebSocket Proxy (port 8765), authenticates,
subscribes to symbols, and routes ticks to registered callbacks.

Protocol
--------
Authenticate:  {"action": "authenticate", "api_key": "..."}
Response:      {"type": "auth", "status": "success", "broker": "..."}

Subscribe:     {"action": "subscribe", "symbols": [{"symbol": "NIFTY",
                 "exchange": "NSE_INDEX"}], "mode": "LTP"}
Response:      {"type": "subscribe", "status": "success",
                 "subscriptions": [...]}

Tick:          {"type": "market_data", "symbol": "NIFTY",
                 "exchange": "NSE_INDEX", "mode": 1,
                 "data": {"ltp": 24500.0, "timestamp": "..."}}
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from kronos.utils.config import KronosConfig
from kronos.utils.helpers import get_logger

logger = get_logger(__name__)

try:
    import websocket
except ImportError:
    websocket = None  # type: ignore[assignment]


# -- Types ---------------------------------------------------------------

OnTickCallback = Callable[[str, str, float], None]  # (symbol, exchange, ltp)


@dataclass
class Subscription:
    """A subscribed symbol + exchange pair."""

    symbol: str
    exchange: str


# -- Client --------------------------------------------------------------


class KronosWebSocketClient:
    """Persistent WebSocket connection to the OpenAlgo data proxy.

    Runs in a background daemon thread.  Reconnects automatically on
    disconnect unless ``stop()`` has been called.

    Usage::

        client = KronosWebSocketClient(config)
        client.set_on_ltp(my_callback)
        client.subscribe("NIFTY", "NSE_INDEX")
        client.start()

        # … later …
        latest = client.latest_ltp("NIFTY", "NSE_INDEX")
        client.stop()
    """

    RECONNECT_DELAY = 3.0  # seconds between reconnection attempts

    def __init__(self, config: KronosConfig | None = None) -> None:
        self.config = config or KronosConfig.from_env()
        self._ws_url = (
            self.config.openalgo_host.replace("http://", "ws://")
            .replace("https://", "wss://")
            .replace("/api", "")
            .rstrip("/")
            + ":8765"
        )

        self._api_key = self.config.openalgo_api_key
        if not self._api_key:
            raise ValueError("OPENALGO_API_KEY is required for WebSocket auth")

        self._on_ltp: OnTickCallback | None = None
        self._subscriptions: list[Subscription] = []
        self._latest: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

        self._ws: websocket.WebSocketApp | None = None  # type: ignore[name-defined]
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False

    # -- Public API -------------------------------------------------------

    def set_on_ltp(self, callback: OnTickCallback | None) -> None:
        """Register or clear the LTP tick callback."""
        self._on_ltp = callback

    def subscribe(self, symbol: str, exchange: str) -> None:
        """Add a symbol to the subscription list.

        If already connected, the subscription is sent immediately.
        """
        sub = Subscription(symbol=symbol, exchange=exchange)
        with self._lock:
            # Avoid duplicates
            for existing in self._subscriptions:
                if existing.symbol == symbol and existing.exchange == exchange:
                    return
            self._subscriptions.append(sub)

        if self._connected and self._ws:
            self._send_subscribe(sub)

    def unsubscribe(self, symbol: str, exchange: str) -> None:
        """Remove a symbol from the subscription list."""
        with self._lock:
            self._subscriptions = [
                s
                for s in self._subscriptions
                if not (s.symbol == symbol and s.exchange == exchange)
            ]

    def latest_ltp(self, symbol: str, exchange: str) -> float | None:
        """Return the most recent LTP for *symbol*\@*exchange*, or ``None``."""
        with self._lock:
            return self._latest.get((symbol, exchange))

    def start(self) -> None:
        """Start the background WebSocket thread."""
        if self._running:
            return
        if websocket is None:
            raise RuntimeError("websocket-client required -- pip install websocket-client")

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="kronos-ws"
        )
        self._thread.start()
        logger.info("WebSocket client started (connecting to %s)", self._ws_url)

    def stop(self) -> None:
        """Gracefully stop the WebSocket connection."""
        self._running = False
        if self._ws:
            self._ws.close()
            self._ws = None
        self._connected = False
        logger.info("WebSocket client stopped")

    # -- Internal ---------------------------------------------------------

    def _run_loop(self) -> None:
        """Connection loop with auto-reconnect."""
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self._ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                logger.warning("WebSocket error: %s", exc)

            if not self._running:
                break

            logger.info(
                "WebSocket disconnected.  Reconnecting in %.0fs…",
                self.RECONNECT_DELAY,
            )
            time.sleep(self.RECONNECT_DELAY)

    def _on_open(self, ws: Any) -> None:
        """Send authentication on connect."""
        logger.info("WebSocket connected")
        self._connected = True
        # Authenticate
        ws.send(json.dumps({"action": "authenticate", "api_key": self._api_key}))

    def _on_message(self, _ws: Any, raw: str) -> None:
        """Dispatch incoming messages."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "auth":
            status = msg.get("status", "")
            if status == "success":
                logger.info("WebSocket authenticated (broker: %s)", msg.get("broker", "?"))
                # Subscribe to all pending subscriptions
                with self._lock:
                    subs = list(self._subscriptions)
                for sub in subs:
                    self._send_subscribe(sub)
            else:
                logger.error("WebSocket auth failed: %s", msg.get("message", "unknown"))

        elif msg_type == "subscribe":
            logger.info("Subscribed: %s", msg.get("subscriptions", []))

        elif msg_type == "market_data":
            self._handle_tick(msg)

    def _on_error(self, _ws: Any, error: Any) -> None:
        logger.warning("WebSocket error: %s", error)

    def _on_close(self, _ws: Any, close_status: Any, close_msg: Any) -> None:
        self._connected = False
        logger.info("WebSocket closed (status=%s msg=%s)", close_status, close_msg)

    def _send_subscribe(self, sub: Subscription) -> None:
        """Send a subscribe message for a single symbol."""
        if not self._ws or not self._connected:
            return
        payload = json.dumps({
            "action": "subscribe",
            "symbols": [{"symbol": sub.symbol, "exchange": sub.exchange}],
            "mode": "LTP",
        })
        self._ws.send(payload)

    def _handle_tick(self, msg: dict[str, Any]) -> None:
        """Process a market data tick."""
        symbol = msg.get("symbol", "")
        exchange = msg.get("exchange", "")
        data = msg.get("data", {})
        ltp = data.get("ltp")

        if ltp is None:
            return

        ltp = float(ltp)

        with self._lock:
            self._latest[(symbol, exchange)] = ltp

        # Fire callback
        cb = self._on_ltp
        if cb:
            cb(symbol, exchange, ltp)
