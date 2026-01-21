"""ClickHouse client with connection pooling for Chronos-X."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterator

import structlog
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError

if TYPE_CHECKING:
    import polars as pl

logger = structlog.get_logger(__name__)


@dataclass
class ClickHouseConfig:
    """ClickHouse connection configuration."""

    host: str = "localhost"
    port: int = 9000
    database: str = "chronos"
    user: str = "chronos"
    password: str = "chronos_secret"
    pool_size: int = 10
    connect_timeout: int = 10
    send_receive_timeout: int = 300


class ConnectionPool:
    """Thread-safe connection pool for ClickHouse."""

    def __init__(self, config: ClickHouseConfig) -> None:
        self.config = config
        self._pool: Queue[Client] = Queue(maxsize=config.pool_size)
        self._lock = Lock()
        self._created = 0

    def _create_connection(self) -> Client:
        """Create a new ClickHouse connection."""
        return Client(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            connect_timeout=self.config.connect_timeout,
            send_receive_timeout=self.config.send_receive_timeout,
            settings={"use_numpy": True},
        )

    def get_connection(self, timeout: float = 5.0) -> Client:
        """Get a connection from the pool."""
        try:
            return self._pool.get(block=False)
        except Empty:
            with self._lock:
                if self._created < self.config.pool_size:
                    self._created += 1
                    return self._create_connection()

            # Pool exhausted, wait for a connection
            return self._pool.get(block=True, timeout=timeout)

    def return_connection(self, conn: Client) -> None:
        """Return a connection to the pool."""
        try:
            self._pool.put(conn, block=False)
        except Exception:
            # Pool is full, close the connection
            try:
                conn.disconnect()
            except Exception:
                pass

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while True:
            try:
                conn = self._pool.get(block=False)
                conn.disconnect()
            except Empty:
                break


class ClickHouseClient:
    """High-level ClickHouse client for Chronos-X operations."""

    def __init__(self, config: ClickHouseConfig | None = None) -> None:
        self.config = config or ClickHouseConfig()
        self._pool = ConnectionPool(self.config)
        self._log = logger.bind(component="clickhouse")

    @contextmanager
    def connection(self) -> Iterator[Client]:
        """Context manager for getting a pooled connection."""
        conn = self._pool.get_connection()
        try:
            yield conn
        finally:
            self._pool.return_connection(conn)

    def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a query and return results."""
        with self.connection() as conn:
            try:
                return conn.execute(query, params or {})
            except ClickHouseError as e:
                self._log.error("query_failed", query=query[:100], error=str(e))
                raise

    def execute_many(self, query: str, data: list[tuple[Any, ...]]) -> int:
        """Execute a query with multiple data rows (batch insert)."""
        with self.connection() as conn:
            try:
                return conn.execute(query, data)
            except ClickHouseError as e:
                self._log.error("batch_insert_failed", error=str(e))
                raise

    def insert_trades(
        self,
        trades: list[dict[str, Any]],
        exchange: str = "binance",
    ) -> int:
        """Insert trade records into the trades table."""
        if not trades:
            return 0

        query = """
        INSERT INTO chronos.trades
        (exchange, symbol, trade_id, timestamp, price, quantity, side, is_maker)
        VALUES
        """

        data = [
            (
                exchange,
                t["symbol"],
                str(t["trade_id"]),
                t["timestamp"],
                float(t["price"]),
                float(t["quantity"]),
                "buy" if t["side"] == "buy" else "sell",
                t.get("is_maker", False),
            )
            for t in trades
        ]

        return self.execute_many(query, data)

    def insert_orderbook_snapshot(
        self,
        exchange: str,
        symbol: str,
        timestamp: datetime,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Insert an orderbook snapshot."""
        query = """
        INSERT INTO chronos.orderbook_snapshots
        (exchange, symbol, timestamp, bid_prices, bid_quantities, ask_prices, ask_quantities)
        VALUES
        """

        bid_prices = [b[0] for b in bids[:20]]
        bid_quantities = [b[1] for b in bids[:20]]
        ask_prices = [a[0] for a in asks[:20]]
        ask_quantities = [a[1] for a in asks[:20]]

        self.execute_many(
            query,
            [(exchange, symbol, timestamp, bid_prices, bid_quantities, ask_prices, ask_quantities)],
        )

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        exchange: str = "binance",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV data from materialized views."""
        table_map = {
            "1m": "ohlcv_1m",
            "5m": "ohlcv_5m",
            "1h": "ohlcv_1h",
            "1d": "ohlcv_1d",
        }

        table = table_map.get(timeframe, "ohlcv_1m")

        query = f"""
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            notional_volume,
            trade_count,
            buy_volume,
            sell_volume
        FROM chronos.{table}
        WHERE exchange = %(exchange)s
          AND symbol = %(symbol)s
        """

        params: dict[str, Any] = {"exchange": exchange, "symbol": symbol}

        if start:
            query += " AND timestamp >= %(start)s"
            params["start"] = start
        if end:
            query += " AND timestamp <= %(end)s"
            params["end"] = end

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        rows = self.execute(query, params)

        return [
            {
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "notional_volume": row[6],
                "trade_count": row[7],
                "buy_volume": row[8],
                "sell_volume": row[9],
            }
            for row in rows
        ]

    def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        exchange: str = "binance",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw trades from the database."""
        query = """
        SELECT
            trade_id,
            timestamp,
            price,
            quantity,
            side,
            is_maker,
            notional
        FROM chronos.trades
        WHERE exchange = %(exchange)s
          AND symbol = %(symbol)s
          AND timestamp >= %(start)s
          AND timestamp <= %(end)s
        ORDER BY timestamp ASC
        """

        params: dict[str, Any] = {
            "exchange": exchange,
            "symbol": symbol,
            "start": start,
            "end": end,
        }

        if limit:
            query += f" LIMIT {limit}"

        rows = self.execute(query, params)

        return [
            {
                "trade_id": row[0],
                "timestamp": row[1],
                "price": row[2],
                "quantity": row[3],
                "side": row[4],
                "is_maker": row[5],
                "notional": row[6],
            }
            for row in rows
        ]

    def get_latest_timestamp(
        self,
        symbol: str,
        exchange: str = "binance",
    ) -> datetime | None:
        """Get the latest trade timestamp for a symbol."""
        query = """
        SELECT max(timestamp)
        FROM chronos.trades
        WHERE exchange = %(exchange)s
          AND symbol = %(symbol)s
        """

        result = self.execute(query, {"exchange": exchange, "symbol": symbol})
        if result and result[0][0]:
            return result[0][0]
        return None

    def get_trade_count(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        exchange: str = "binance",
    ) -> int:
        """Get the count of trades for a symbol."""
        query = """
        SELECT count()
        FROM chronos.trades
        WHERE exchange = %(exchange)s
          AND symbol = %(symbol)s
        """

        params: dict[str, Any] = {"exchange": exchange, "symbol": symbol}

        if start:
            query += " AND timestamp >= %(start)s"
            params["start"] = start
        if end:
            query += " AND timestamp <= %(end)s"
            params["end"] = end

        result = self.execute(query, params)
        return result[0][0] if result else 0

    def insert_microstructure(
        self,
        records: list[dict[str, Any]],
        exchange: str = "binance",
    ) -> int:
        """Insert microstructure metrics."""
        if not records:
            return 0

        query = """
        INSERT INTO chronos.microstructure
        (exchange, symbol, timestamp, window_minutes, ofi, ofi_normalized,
         vpin, volume_imbalance, effective_spread, realized_spread,
         realized_volatility, parkinson_volatility, trade_count, avg_trade_size)
        VALUES
        """

        data = [
            (
                exchange,
                r["symbol"],
                r["timestamp"],
                r["window_minutes"],
                r.get("ofi", 0.0),
                r.get("ofi_normalized", 0.0),
                r.get("vpin", 0.0),
                r.get("volume_imbalance", 0.0),
                r.get("effective_spread", 0.0),
                r.get("realized_spread", 0.0),
                r.get("realized_volatility", 0.0),
                r.get("parkinson_volatility", 0.0),
                r.get("trade_count", 0),
                r.get("avg_trade_size", 0.0),
            )
            for r in records
        ]

        return self.execute_many(query, data)

    def insert_labels(
        self,
        labels: list[dict[str, Any]],
        exchange: str = "binance",
    ) -> int:
        """Insert training labels."""
        if not labels:
            return 0

        query = """
        INSERT INTO chronos.labels
        (exchange, symbol, timestamp, label, return_pct, holding_period_minutes,
         exit_type, meta_label, tp_multiple, sl_multiple, max_holding_minutes,
         volatility_window)
        VALUES
        """

        data = [
            (
                exchange,
                lb["symbol"],
                lb["timestamp"],
                lb["label"],
                lb["return_pct"],
                lb["holding_period_minutes"],
                lb["exit_type"],
                lb.get("meta_label", 0.5),
                lb["tp_multiple"],
                lb["sl_multiple"],
                lb["max_holding_minutes"],
                lb["volatility_window"],
            )
            for lb in labels
        ]

        return self.execute_many(query, data)

    def run_schema(self, schema_path: Path | str) -> None:
        """Execute SQL schema file."""
        schema_path = Path(schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        sql = schema_path.read_text()

        # Split by semicolon and execute each statement
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        for stmt in statements:
            if stmt and not stmt.startswith("--"):
                try:
                    self.execute(stmt)
                    self._log.debug("executed_statement", statement=stmt[:50])
                except ClickHouseError as e:
                    self._log.warning("statement_failed", statement=stmt[:50], error=str(e))

    def health_check(self) -> bool:
        """Check if ClickHouse is responsive."""
        try:
            result = self.execute("SELECT 1")
            return result == [(1,)]
        except Exception as e:
            self._log.error("health_check_failed", error=str(e))
            return False

    def close(self) -> None:
        """Close all connections."""
        self._pool.close_all()


# Async wrapper for use with asyncio
class AsyncClickHouseClient:
    """Async wrapper around ClickHouseClient using thread pool."""

    def __init__(self, config: ClickHouseConfig | None = None) -> None:
        self._sync_client = ClickHouseClient(config)
        self._executor = None

    async def insert_trades(
        self,
        trades: list[dict[str, Any]],
        exchange: str = "binance",
    ) -> int:
        """Async insert trades."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_client.insert_trades,
            trades,
            exchange,
        )

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        exchange: str = "binance",
    ) -> list[dict[str, Any]]:
        """Async get OHLCV data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._sync_client.get_ohlcv(symbol, timeframe, start, end, exchange),
        )

    async def health_check(self) -> bool:
        """Async health check."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_client.health_check)

    async def close(self) -> None:
        """Close the client."""
        self._sync_client.close()
