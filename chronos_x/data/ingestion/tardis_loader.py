"""
Tardis.dev Integration Module

Provides high-frequency tick data loading capabilities including:
- L2/L3 orderbook reconstruction
- Tick-by-tick trade data
- Event-driven orderbook updates
- Historical data replay with microsecond precision
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncIterator, Tuple
from dataclasses import dataclass
import asyncio
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TardisDataType(Enum):
    """Supported Tardis data types"""
    TRADES = "trades"
    BOOK_SNAPSHOT_5 = "book_snapshot_5"
    BOOK_SNAPSHOT_25 = "book_snapshot_25"
    BOOK_CHANGE = "book_change"
    BOOK_L3_SNAPSHOT = "book_l3_snapshot"
    DERIVATIVE_TICKER = "derivative_ticker"
    LIQUIDATION = "liquidation"


@dataclass
class TardisConfig:
    """Configuration for Tardis.dev client"""
    api_key: str
    exchanges: List[str] = None
    data_types: List[str] = None
    buffer_size: int = 10000
    cache_dir: Optional[str] = None
    use_cache: bool = True

    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = ["binance", "coinbase"]
        if self.data_types is None:
            self.data_types = ["trades", "book_snapshot_5", "book_change"]


class TardisLoader:
    """
    High-frequency market data loader using Tardis.dev

    Provides access to historical and real-time tick data with microsecond precision.
    Supports orderbook reconstruction and event-driven updates.
    """

    def __init__(self, config: TardisConfig):
        """
        Initialize Tardis loader

        Args:
            config: Tardis configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Lazy import tardis-client to avoid dependency if not used
        try:
            from tardis_dev import datasets
            self.datasets = datasets
        except ImportError:
            self.logger.error(
                "tardis-client not installed. Install with: pip install tardis-client"
            )
            raise

        self._client = None
        self._buffer = {}

    def _get_client(self):
        """Get or create Tardis client"""
        if self._client is None:
            from tardis_dev import datasets
            self._client = datasets
        return self._client

    async def load_historical_trades(
        self,
        exchange: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_microseconds: bool = True
    ) -> pd.DataFrame:
        """
        Load historical trade data

        Args:
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start_date: Start timestamp
            end_date: End timestamp
            include_microseconds: Whether to preserve microsecond precision

        Returns:
            DataFrame with columns: timestamp, price, amount, side, trade_id
        """
        self.logger.info(
            f"Loading trades for {exchange}:{symbol} from {start_date} to {end_date}"
        )

        try:
            client = self._get_client()

            # Tardis expects dates in YYYY-MM-DD format
            trades_data = []

            # Use tardis-client to fetch data
            messages = client.replay(
                exchange=exchange,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
                filters=[
                    {
                        "channel": "trades",
                        "symbols": [symbol]
                    }
                ],
                api_key=self.config.api_key
            )

            for message in messages:
                if message["type"] == "trade":
                    trades_data.append({
                        "timestamp": pd.to_datetime(message["timestamp"], unit="us") if include_microseconds
                                    else pd.to_datetime(message["timestamp"], unit="ms"),
                        "price": float(message["price"]),
                        "amount": float(message["amount"]),
                        "side": message["side"],
                        "trade_id": message.get("id", "")
                    })

                # Buffer management
                if len(trades_data) >= self.config.buffer_size:
                    yield pd.DataFrame(trades_data)
                    trades_data = []

            # Yield remaining data
            if trades_data:
                yield pd.DataFrame(trades_data)

        except Exception as e:
            self.logger.error(f"Error loading trades: {e}")
            raise

    async def load_orderbook_snapshots(
        self,
        exchange: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        depth: int = 5
    ) -> AsyncIterator[Dict]:
        """
        Load orderbook snapshot data

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            start_date: Start timestamp
            end_date: End timestamp
            depth: Orderbook depth (5 or 25)

        Yields:
            Orderbook snapshots with bids/asks
        """
        self.logger.info(
            f"Loading orderbook snapshots for {exchange}:{symbol} (depth={depth})"
        )

        try:
            client = self._get_client()

            channel = f"book_snapshot_{depth}"

            messages = client.replay(
                exchange=exchange,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
                filters=[
                    {
                        "channel": channel,
                        "symbols": [symbol]
                    }
                ],
                api_key=self.config.api_key
            )

            for message in messages:
                if message["type"] == "book_snapshot":
                    snapshot = {
                        "timestamp": pd.to_datetime(message["timestamp"], unit="us"),
                        "bids": [[float(p), float(q)] for p, q in message["bids"]],
                        "asks": [[float(p), float(q)] for p, q in message["asks"]],
                        "exchange": exchange,
                        "symbol": symbol
                    }
                    yield snapshot

        except Exception as e:
            self.logger.error(f"Error loading orderbook snapshots: {e}")
            raise

    async def load_l3_orderbook_events(
        self,
        exchange: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> AsyncIterator[Dict]:
        """
        Load L3 orderbook change events (order-by-order)

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            start_date: Start timestamp
            end_date: End timestamp

        Yields:
            L3 orderbook events (add, update, delete)
        """
        self.logger.info(
            f"Loading L3 orderbook events for {exchange}:{symbol}"
        )

        try:
            client = self._get_client()

            messages = client.replay(
                exchange=exchange,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
                filters=[
                    {
                        "channel": "book_change",
                        "symbols": [symbol]
                    }
                ],
                api_key=self.config.api_key
            )

            for message in messages:
                if message["type"] == "book_change":
                    event = {
                        "timestamp": pd.to_datetime(message["timestamp"], unit="us"),
                        "order_id": message.get("id", ""),
                        "side": message["side"],
                        "price": float(message["price"]),
                        "quantity": float(message.get("amount", 0)),
                        "event_type": message["event"],  # add, update, delete
                        "exchange": exchange,
                        "symbol": symbol
                    }
                    yield event

        except Exception as e:
            self.logger.error(f"Error loading L3 events: {e}")
            raise

    def reconstruct_orderbook(
        self,
        l3_events: List[Dict],
        initial_snapshot: Optional[Dict] = None
    ) -> Dict:
        """
        Reconstruct full orderbook from L3 events

        Args:
            l3_events: List of L3 orderbook events
            initial_snapshot: Optional initial orderbook snapshot

        Returns:
            Reconstructed orderbook state
        """
        orderbook = {
            "bids": {},  # price -> {order_id -> quantity}
            "asks": {}
        }

        # Initialize from snapshot if provided
        if initial_snapshot:
            for price, qty in initial_snapshot.get("bids", []):
                if price not in orderbook["bids"]:
                    orderbook["bids"][price] = {}
                orderbook["bids"][price]["_snapshot"] = qty

            for price, qty in initial_snapshot.get("asks", []):
                if price not in orderbook["asks"]:
                    orderbook["asks"][price] = {}
                orderbook["asks"][price]["_snapshot"] = qty

        # Apply L3 events
        for event in l3_events:
            side = "bids" if event["side"] == "buy" else "asks"
            price = event["price"]
            order_id = event["order_id"]
            qty = event["quantity"]
            event_type = event["event_type"]

            if price not in orderbook[side]:
                orderbook[side][price] = {}

            if event_type == "add":
                orderbook[side][price][order_id] = qty
            elif event_type == "update":
                orderbook[side][price][order_id] = qty
            elif event_type == "delete":
                orderbook[side][price].pop(order_id, None)
                # Clean up empty price levels
                if not orderbook[side][price]:
                    del orderbook[side][price]

        # Aggregate to L2
        l2_orderbook = {
            "bids": [[price, sum(orders.values())]
                     for price, orders in sorted(orderbook["bids"].items(), reverse=True)],
            "asks": [[price, sum(orders.values())]
                     for price, orders in sorted(orderbook["asks"].items())]
        }

        return l2_orderbook

    def compute_effective_spread(
        self,
        trades: pd.DataFrame,
        orderbook_snapshots: List[Dict],
        window_ms: int = 100
    ) -> pd.Series:
        """
        Compute effective spread from trades and orderbook

        Effective spread = 2 * |trade_price - mid_price|

        Args:
            trades: DataFrame with trade data
            orderbook_snapshots: List of orderbook snapshots
            window_ms: Time window for matching trades to orderbook (milliseconds)

        Returns:
            Series of effective spreads
        """
        # Convert orderbook to DataFrame for easier lookup
        ob_df = pd.DataFrame([
            {
                "timestamp": ob["timestamp"],
                "best_bid": ob["bids"][0][0] if ob["bids"] else np.nan,
                "best_ask": ob["asks"][0][0] if ob["asks"] else np.nan
            }
            for ob in orderbook_snapshots
        ])
        ob_df["mid_price"] = (ob_df["best_bid"] + ob_df["best_ask"]) / 2

        # Merge trades with nearest orderbook
        trades = trades.sort_values("timestamp")
        ob_df = ob_df.sort_values("timestamp")

        # Forward fill mid prices
        merged = pd.merge_asof(
            trades,
            ob_df[["timestamp", "mid_price"]],
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(milliseconds=window_ms)
        )

        # Compute effective spread
        effective_spread = 2 * np.abs(merged["price"] - merged["mid_price"])

        return effective_spread

    async def stream_realtime_data(
        self,
        exchange: str,
        symbols: List[str],
        data_types: List[str]
    ) -> AsyncIterator[Dict]:
        """
        Stream real-time market data

        Args:
            exchange: Exchange name
            symbols: List of trading symbols
            data_types: List of data types to stream

        Yields:
            Real-time market data messages
        """
        self.logger.info(
            f"Starting real-time stream for {exchange}: {symbols}"
        )

        try:
            from tardis_dev import datasets

            # Note: Real-time streaming requires tardis-machine
            # This is a placeholder for the streaming interface
            self.logger.warning(
                "Real-time streaming requires tardis-machine. "
                "Use replay() for historical data."
            )

            # Placeholder - actual implementation would use tardis-machine
            yield {"status": "not_implemented", "message": "Use replay for historical data"}

        except Exception as e:
            self.logger.error(f"Error in real-time stream: {e}")
            raise

    def get_available_symbols(self, exchange: str) -> List[str]:
        """
        Get list of available symbols for an exchange

        Args:
            exchange: Exchange name

        Returns:
            List of available trading symbols
        """
        # This would query Tardis API for available symbols
        # Placeholder implementation
        common_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
            "XRPUSDT", "DOTUSDT", "UNIUSDT", "LINKUSDT", "MATICUSDT"
        ]
        return common_symbols


def create_tardis_loader(api_key: str, **kwargs) -> TardisLoader:
    """
    Factory function to create Tardis loader

    Args:
        api_key: Tardis.dev API key
        **kwargs: Additional configuration options

    Returns:
        Configured TardisLoader instance
    """
    config = TardisConfig(api_key=api_key, **kwargs)
    return TardisLoader(config)


# Example usage
if __name__ == "__main__":
    import asyncio

    async def example():
        # Initialize loader
        loader = create_tardis_loader(
            api_key="your_api_key_here",
            exchanges=["binance"],
            data_types=["trades", "book_snapshot_5"]
        )

        # Load historical trades
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)

        async for trades_df in loader.load_historical_trades(
            "binance", "BTCUSDT", start, end
        ):
            print(f"Loaded {len(trades_df)} trades")
            print(trades_df.head())

        # Load orderbook snapshots
        async for snapshot in loader.load_orderbook_snapshots(
            "binance", "BTCUSDT", start, end, depth=5
        ):
            print(f"Snapshot at {snapshot['timestamp']}")
            print(f"Best bid: {snapshot['bids'][0]}")
            print(f"Best ask: {snapshot['asks'][0]}")
            break  # Just show first snapshot

    asyncio.run(example())
