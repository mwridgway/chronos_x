"""Historical data loader using CCXT for backfilling data."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import ccxt
import ccxt.async_support as ccxt_async
import structlog

if TYPE_CHECKING:
    from chronos_x.data.ingestion.clickhouse_client import ClickHouseClient

logger = structlog.get_logger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for historical data loading."""

    # Exchange settings
    exchange: str = "binance"
    sandbox: bool = True

    # API credentials (optional for public data)
    api_key: str = ""
    api_secret: str = ""

    # Rate limiting
    rate_limit: bool = True
    requests_per_second: float = 10.0

    # Batch settings
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0

    # Data settings
    default_timeframe: str = "1m"


class HistoricalLoader:
    """Load historical OHLCV and trade data from exchanges via CCXT."""

    def __init__(
        self,
        config: LoaderConfig | None = None,
        db_client: ClickHouseClient | None = None,
    ) -> None:
        self.config = config or LoaderConfig()
        self.db_client = db_client
        self._log = logger.bind(component="historical_loader")

        # Initialize sync exchange for simple operations
        exchange_class = getattr(ccxt, self.config.exchange)
        self._exchange = exchange_class(
            {
                "apiKey": self.config.api_key or None,
                "secret": self.config.api_secret or None,
                "sandbox": self.config.sandbox,
                "enableRateLimit": self.config.rate_limit,
            }
        )

    def _create_async_exchange(self) -> ccxt_async.Exchange:
        """Create async exchange instance."""
        exchange_class = getattr(ccxt_async, self.config.exchange)
        return exchange_class(
            {
                "apiKey": self.config.api_key or None,
                "secret": self.config.api_secret or None,
                "sandbox": self.config.sandbox,
                "enableRateLimit": self.config.rate_limit,
            }
        )

    def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange information and available symbols."""
        self._exchange.load_markets()
        return {
            "name": self._exchange.name,
            "symbols": list(self._exchange.symbols),
            "timeframes": list(self._exchange.timeframes.keys()),
            "has_fetch_trades": self._exchange.has.get("fetchTrades", False),
            "has_fetch_ohlcv": self._exchange.has.get("fetchOHLCV", False),
        }

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV data synchronously."""
        since_ms = int(since.timestamp() * 1000) if since else None

        try:
            ohlcv = self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit,
            )

            return [
                {
                    "timestamp": datetime.fromtimestamp(candle[0] / 1000, tz=UTC),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                }
                for candle in ohlcv
            ]
        except Exception as e:
            self._log.error("fetch_ohlcv_error", symbol=symbol, error=str(e))
            raise

    def fetch_trades(
        self,
        symbol: str,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch trade data synchronously."""
        since_ms = int(since.timestamp() * 1000) if since else None

        try:
            trades = self._exchange.fetch_trades(
                symbol,
                since=since_ms,
                limit=limit,
            )

            return [
                {
                    "symbol": symbol,
                    "trade_id": t["id"],
                    "timestamp": datetime.fromtimestamp(t["timestamp"] / 1000, tz=UTC),
                    "price": float(t["price"]),
                    "quantity": float(t["amount"]),
                    "side": t["side"],
                    "is_maker": t.get("takerOrMaker", "taker") == "maker",
                }
                for t in trades
            ]
        except Exception as e:
            self._log.error("fetch_trades_error", symbol=symbol, error=str(e))
            raise

    async def fetch_trades_async(
        self,
        symbol: str,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch trade data asynchronously."""
        exchange = self._create_async_exchange()
        since_ms = int(since.timestamp() * 1000) if since else None

        try:
            trades = await exchange.fetch_trades(
                symbol,
                since=since_ms,
                limit=limit,
            )

            return [
                {
                    "symbol": symbol,
                    "trade_id": t["id"],
                    "timestamp": datetime.fromtimestamp(t["timestamp"] / 1000, tz=UTC),
                    "price": float(t["price"]),
                    "quantity": float(t["amount"]),
                    "side": t["side"],
                    "is_maker": t.get("takerOrMaker", "taker") == "maker",
                }
                for t in trades
            ]
        finally:
            await exchange.close()

    async def backfill_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime | None = None,
        progress_callback: Any | None = None,
    ) -> int:
        """Backfill historical trades to database.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            start_date: Start date for backfill
            end_date: End date for backfill (default: now)
            progress_callback: Optional callback for progress updates

        Returns:
            Total number of trades loaded
        """
        if not self.db_client:
            raise ValueError("Database client required for backfill")

        end_date = end_date or datetime.now(UTC)
        total_loaded = 0
        current_time = start_date

        self._log.info(
            "starting_backfill",
            symbol=symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )

        exchange = self._create_async_exchange()

        try:
            while current_time < end_date:
                retries = 0
                trades = []

                while retries < self.config.max_retries:
                    try:
                        trades = await exchange.fetch_trades(
                            symbol,
                            since=int(current_time.timestamp() * 1000),
                            limit=self.config.batch_size,
                        )
                        break
                    except Exception as e:
                        retries += 1
                        self._log.warning(
                            "fetch_retry",
                            attempt=retries,
                            error=str(e),
                        )
                        if retries < self.config.max_retries:
                            await asyncio.sleep(self.config.retry_delay * retries)
                        else:
                            raise

                if not trades:
                    # No more trades, advance time window
                    current_time += timedelta(hours=1)
                    continue

                # Convert and insert
                trade_records = [
                    {
                        "symbol": symbol,
                        "trade_id": t["id"],
                        "timestamp": datetime.fromtimestamp(t["timestamp"] / 1000, tz=UTC),
                        "price": float(t["price"]),
                        "quantity": float(t["amount"]),
                        "side": t["side"],
                        "is_maker": t.get("takerOrMaker", "taker") == "maker",
                    }
                    for t in trades
                ]

                # Insert to database
                inserted = self.db_client.insert_trades(trade_records)
                total_loaded += inserted

                # Update progress
                if trades:
                    last_trade_time = datetime.fromtimestamp(
                        trades[-1]["timestamp"] / 1000, tz=UTC
                    )
                    current_time = last_trade_time + timedelta(milliseconds=1)

                if progress_callback:
                    progress_callback(
                        current_time=current_time,
                        end_time=end_date,
                        trades_loaded=total_loaded,
                    )

                self._log.debug(
                    "batch_loaded",
                    trades=len(trades),
                    total=total_loaded,
                    current_time=current_time.isoformat(),
                )

                # Rate limiting delay
                await asyncio.sleep(1.0 / self.config.requests_per_second)

        finally:
            await exchange.close()

        self._log.info("backfill_complete", symbol=symbol, total_trades=total_loaded)
        return total_loaded

    async def backfill_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> int:
        """Backfill historical OHLCV data.

        Note: OHLCV is typically computed from trades via materialized views,
        but this can be used for quick data availability.
        """
        if not self.db_client:
            raise ValueError("Database client required for backfill")

        end_date = end_date or datetime.now(UTC)
        total_loaded = 0
        current_time = start_date

        # Calculate timeframe duration for advancing
        tf_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        tf_delta = tf_map.get(timeframe, timedelta(minutes=1))

        exchange = self._create_async_exchange()

        try:
            while current_time < end_date:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=int(current_time.timestamp() * 1000),
                    limit=self.config.batch_size,
                )

                if not ohlcv:
                    break

                # Note: Direct OHLCV insertion would require a separate table
                # For now, we rely on materialized views from trade data
                total_loaded += len(ohlcv)

                # Advance time
                last_candle_time = datetime.fromtimestamp(ohlcv[-1][0] / 1000, tz=UTC)
                current_time = last_candle_time + tf_delta

                self._log.debug(
                    "ohlcv_batch_loaded",
                    candles=len(ohlcv),
                    total=total_loaded,
                )

                await asyncio.sleep(1.0 / self.config.requests_per_second)

        finally:
            await exchange.close()

        return total_loaded

    def get_latest_trade_time(self, symbol: str) -> datetime | None:
        """Get the latest trade time from the database."""
        if self.db_client:
            return self.db_client.get_latest_timestamp(symbol)
        return None

    async def sync_to_present(
        self,
        symbol: str,
        progress_callback: Any | None = None,
    ) -> int:
        """Sync trades from last known time to present.

        This is useful for incremental updates.
        """
        last_time = self.get_latest_trade_time(symbol)

        if last_time:
            # Start from slightly after last known trade
            start_time = last_time + timedelta(milliseconds=1)
        else:
            # Default to 7 days ago if no data exists
            start_time = datetime.now(UTC) - timedelta(days=7)

        return await self.backfill_trades(
            symbol=symbol,
            start_date=start_time,
            end_date=datetime.now(UTC),
            progress_callback=progress_callback,
        )

    def close(self) -> None:
        """Close the exchange connection."""
        pass  # CCXT sync client doesn't need explicit closing


class MultiSymbolLoader:
    """Load data for multiple symbols concurrently."""

    def __init__(
        self,
        symbols: list[str],
        config: LoaderConfig | None = None,
        db_client: ClickHouseClient | None = None,
        max_concurrent: int = 3,
    ) -> None:
        self.symbols = symbols
        self.config = config or LoaderConfig()
        self.db_client = db_client
        self.max_concurrent = max_concurrent
        self._log = logger.bind(component="multi_loader")

    async def backfill_all(
        self,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> dict[str, int]:
        """Backfill all symbols concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: dict[str, int] = {}

        async def load_symbol(symbol: str) -> tuple[str, int]:
            async with semaphore:
                loader = HistoricalLoader(self.config, self.db_client)
                try:
                    count = await loader.backfill_trades(symbol, start_date, end_date)
                    return symbol, count
                except Exception as e:
                    self._log.error("symbol_load_failed", symbol=symbol, error=str(e))
                    return symbol, 0

        tasks = [load_symbol(s) for s in self.symbols]
        completed = await asyncio.gather(*tasks)

        for symbol, count in completed:
            results[symbol] = count

        return results

    async def sync_all(self) -> dict[str, int]:
        """Sync all symbols to present."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: dict[str, int] = {}

        async def sync_symbol(symbol: str) -> tuple[str, int]:
            async with semaphore:
                loader = HistoricalLoader(self.config, self.db_client)
                try:
                    count = await loader.sync_to_present(symbol)
                    return symbol, count
                except Exception as e:
                    self._log.error("symbol_sync_failed", symbol=symbol, error=str(e))
                    return symbol, 0

        tasks = [sync_symbol(s) for s in self.symbols]
        completed = await asyncio.gather(*tasks)

        for symbol, count in completed:
            results[symbol] = count

        return results
