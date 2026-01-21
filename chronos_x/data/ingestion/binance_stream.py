"""Binance WebSocket stream handler for real-time trade data."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
import websockets
from websockets.exceptions import ConnectionClosed

logger = structlog.get_logger(__name__)


class StreamType(Enum):
    """Binance WebSocket stream types."""

    TRADE = "trade"
    AGG_TRADE = "aggTrade"
    KLINE = "kline"
    DEPTH = "depth"
    BOOK_TICKER = "bookTicker"


@dataclass
class StreamConfig:
    """Configuration for Binance WebSocket streams."""

    # WebSocket endpoints
    base_url: str = "wss://stream.binance.com:9443/ws"
    testnet_url: str = "wss://testnet.binance.vision/ws"

    # Connection settings
    use_testnet: bool = False
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    ping_interval: float = 30.0
    ping_timeout: float = 10.0

    # Buffer settings
    buffer_size: int = 1000
    flush_interval: float = 1.0

    @property
    def websocket_url(self) -> str:
        """Get the appropriate WebSocket URL."""
        return self.testnet_url if self.use_testnet else self.base_url


@dataclass
class Trade:
    """Parsed trade data from WebSocket."""

    symbol: str
    trade_id: int
    price: float
    quantity: float
    timestamp: datetime
    side: str
    is_maker: bool

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> Trade:
        """Create Trade from WebSocket message."""
        return cls(
            symbol=data["s"],
            trade_id=data["t"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            timestamp=datetime.fromtimestamp(data["T"] / 1000, tz=UTC),
            side="sell" if data["m"] else "buy",  # m=True means buyer is maker
            is_maker=data["m"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "symbol": self.symbol.replace("USDT", "/USDT"),  # Convert to CCXT format
            "trade_id": self.trade_id,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp,
            "side": self.side,
            "is_maker": self.is_maker,
        }


@dataclass
class OrderBookUpdate:
    """Parsed order book update from WebSocket."""

    symbol: str
    timestamp: datetime
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> OrderBookUpdate:
        """Create OrderBookUpdate from WebSocket message."""
        return cls(
            symbol=data["s"],
            timestamp=datetime.fromtimestamp(data["E"] / 1000, tz=UTC),
            bids=[(float(b[0]), float(b[1])) for b in data.get("b", [])],
            asks=[(float(a[0]), float(a[1])) for a in data.get("a", [])],
        )


@dataclass
class StreamStats:
    """Statistics for stream monitoring."""

    messages_received: int = 0
    trades_processed: int = 0
    reconnect_count: int = 0
    last_message_time: datetime | None = None
    errors: list[str] = field(default_factory=list)

    def record_message(self) -> None:
        """Record a received message."""
        self.messages_received += 1
        self.last_message_time = datetime.now(UTC)

    def record_trade(self) -> None:
        """Record a processed trade."""
        self.trades_processed += 1

    def record_reconnect(self) -> None:
        """Record a reconnection."""
        self.reconnect_count += 1

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(f"{datetime.now(UTC)}: {error}")
        # Keep only last 100 errors
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]


class BinanceStreamHandler:
    """Handler for Binance WebSocket streams."""

    def __init__(
        self,
        symbols: list[str],
        config: StreamConfig | None = None,
        on_trade: Callable[[Trade], None] | None = None,
        on_orderbook: Callable[[OrderBookUpdate], None] | None = None,
    ) -> None:
        self.symbols = [s.replace("/", "").lower() for s in symbols]
        self.config = config or StreamConfig()
        self.on_trade = on_trade
        self.on_orderbook = on_orderbook

        self._running = False
        self._websocket = None
        self._stats = StreamStats()
        self._trade_buffer: list[Trade] = []
        self._buffer_lock = asyncio.Lock()
        self._log = logger.bind(component="binance_stream")

    @property
    def stats(self) -> StreamStats:
        """Get stream statistics."""
        return self._stats

    def _build_stream_url(self, stream_types: list[StreamType]) -> str:
        """Build the combined stream URL."""
        streams = []

        for symbol in self.symbols:
            for stream_type in stream_types:
                if stream_type == StreamType.TRADE:
                    streams.append(f"{symbol}@trade")
                elif stream_type == StreamType.AGG_TRADE:
                    streams.append(f"{symbol}@aggTrade")
                elif stream_type == StreamType.DEPTH:
                    streams.append(f"{symbol}@depth20@100ms")
                elif stream_type == StreamType.BOOK_TICKER:
                    streams.append(f"{symbol}@bookTicker")

        stream_param = "/".join(streams)
        return f"{self.config.websocket_url}/{stream_param}"

    async def _handle_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            self._stats.record_message()

            # Determine message type
            event_type = data.get("e")

            if event_type == "trade":
                trade = Trade.from_ws_message(data)
                self._stats.record_trade()

                async with self._buffer_lock:
                    self._trade_buffer.append(trade)

                if self.on_trade:
                    self.on_trade(trade)

            elif event_type == "depthUpdate":
                update = OrderBookUpdate.from_ws_message(data)
                if self.on_orderbook:
                    self.on_orderbook(update)

        except Exception as e:
            self._log.error("message_parse_error", error=str(e), message=message[:100])
            self._stats.record_error(str(e))

    async def _connect(self, stream_types: list[StreamType]) -> None:
        """Connect to WebSocket with reconnection logic."""
        url = self._build_stream_url(stream_types)
        reconnect_attempts = 0

        while self._running and reconnect_attempts < self.config.max_reconnect_attempts:
            try:
                self._log.info("connecting", url=url)

                async with websockets.connect(
                    url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                ) as websocket:
                    self._websocket = websocket
                    reconnect_attempts = 0
                    self._log.info("connected", symbols=self.symbols)

                    async for message in websocket:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except ConnectionClosed as e:
                self._log.warning("connection_closed", code=e.code, reason=e.reason)
                self._stats.record_reconnect()

            except Exception as e:
                self._log.error("connection_error", error=str(e))
                self._stats.record_error(str(e))

            if self._running:
                reconnect_attempts += 1
                delay = self.config.reconnect_delay * reconnect_attempts
                self._log.info(
                    "reconnecting",
                    attempt=reconnect_attempts,
                    delay=delay,
                )
                await asyncio.sleep(delay)

        if reconnect_attempts >= self.config.max_reconnect_attempts:
            self._log.error("max_reconnect_attempts_reached")
            self._running = False

    async def get_buffered_trades(self) -> list[Trade]:
        """Get and clear buffered trades."""
        async with self._buffer_lock:
            trades = self._trade_buffer.copy()
            self._trade_buffer.clear()
            return trades

    async def start(
        self,
        stream_types: list[StreamType] | None = None,
    ) -> None:
        """Start the WebSocket stream."""
        if stream_types is None:
            stream_types = [StreamType.TRADE]

        self._running = True
        await self._connect(stream_types)

    async def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        self._log.info("stream_stopped", stats=self._stats.__dict__)


class BinanceStreamManager:
    """Manager for multiple Binance streams with automatic flushing."""

    def __init__(
        self,
        symbols: list[str],
        config: StreamConfig | None = None,
        flush_callback: Callable[[list[Trade]], None] | None = None,
    ) -> None:
        self.symbols = symbols
        self.config = config or StreamConfig()
        self.flush_callback = flush_callback

        self._handler = BinanceStreamHandler(symbols, config)
        self._flush_task: asyncio.Task | None = None
        self._running = False
        self._log = logger.bind(component="stream_manager")

    async def _flush_loop(self) -> None:
        """Periodically flush buffered trades."""
        while self._running:
            await asyncio.sleep(self.config.flush_interval)

            trades = await self._handler.get_buffered_trades()
            if trades and self.flush_callback:
                try:
                    self.flush_callback([t.to_dict() for t in trades])
                    self._log.debug("flushed_trades", count=len(trades))
                except Exception as e:
                    self._log.error("flush_error", error=str(e))

    async def start(self) -> None:
        """Start the stream manager."""
        self._running = True

        # Start flush task
        self._flush_task = asyncio.create_task(self._flush_loop())

        # Start WebSocket handler
        await self._handler.start([StreamType.TRADE])

    async def stop(self) -> None:
        """Stop the stream manager."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self._handler.stop()

    @property
    def stats(self) -> StreamStats:
        """Get stream statistics."""
        return self._handler.stats


async def run_stream(
    symbols: list[str],
    db_client: Any,
    config: StreamConfig | None = None,
) -> None:
    """Run the Binance stream with database persistence.

    This is a convenience function for running the stream with automatic
    database flushing.
    """

    def flush_trades(trades: list[dict[str, Any]]) -> None:
        """Flush trades to ClickHouse."""
        if trades:
            db_client.insert_trades(trades)

    manager = BinanceStreamManager(
        symbols=symbols,
        config=config,
        flush_callback=flush_trades,
    )

    try:
        await manager.start()
    except KeyboardInterrupt:
        pass
    finally:
        await manager.stop()
