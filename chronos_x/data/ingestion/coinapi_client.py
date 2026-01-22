"""
CoinAPI Integration Module

Alternative data source for cryptocurrency market data with:
- REST API for OHLCV and historical trades
- WebSocket for real-time data streaming
- Multi-exchange normalization
- Rate limiting and quota management
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncIterator
from dataclasses import dataclass
import asyncio
from enum import Enum
import time

import pandas as pd
import numpy as np
import aiohttp
from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger(__name__)


class CoinAPIDataType(Enum):
    """CoinAPI data types"""
    OHLCV = "ohlcv"
    TRADES = "trades"
    QUOTES = "quotes"
    ORDERBOOK = "orderbook"
    ORDERBOOK_L3 = "orderbook_l3"


@dataclass
class CoinAPIConfig:
    """Configuration for CoinAPI client"""
    api_key: str
    base_url: str = "https://rest.coinapi.io"
    ws_url: str = "wss://ws.coinapi.io/v1/"
    rate_limit: int = 100  # requests per second
    timeout: int = 30
    max_retries: int = 3
    exchanges: List[str] = None

    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = ["BINANCE", "COINBASE", "KRAKEN"]


class RateLimiter:
    """Simple rate limiter for API requests"""

    def __init__(self, max_calls: int, period: float = 1.0):
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum number of calls allowed per period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make API call"""
        async with self._lock:
            now = time.time()

            # Remove old calls outside the window
            self.calls = [call_time for call_time in self.calls
                         if now - call_time < self.period]

            if len(self.calls) >= self.max_calls:
                # Wait until oldest call expires
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.calls = self.calls[1:]

            self.calls.append(now)


class CoinAPIClient:
    """
    CoinAPI client for cryptocurrency market data

    Provides unified access to multiple exchanges with rate limiting
    and automatic retries.
    """

    def __init__(self, config: CoinAPIConfig):
        """
        Initialize CoinAPI client

        Args:
            config: CoinAPI configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = RateLimiter(config.rate_limit)
        self._session: Optional[ClientSession] = None
        self._ws_connection = None

        # CoinAPI uses API key in headers
        self.headers = {
            "X-CoinAPI-Key": config.api_key,
            "Accept": "application/json"
        }

    async def _get_session(self) -> ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.config.timeout)
            self._session = ClientSession(
                headers=self.headers,
                timeout=timeout
            )
        return self._session

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict:
        """
        Make HTTP request to CoinAPI

        Args:
            endpoint: API endpoint
            params: Query parameters
            retry_count: Current retry attempt

        Returns:
            JSON response data
        """
        await self.rate_limiter.acquire()

        session = await self._get_session()
        url = f"{self.config.base_url}{endpoint}"

        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limit
                    if retry_count < self.config.max_retries:
                        self.logger.warning("Rate limit hit, retrying...")
                        await asyncio.sleep(2 ** retry_count)
                        return await self._make_request(endpoint, params, retry_count + 1)
                    else:
                        raise Exception("Rate limit exceeded")
                elif response.status == 401:
                    raise Exception("Invalid API key")
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

        except asyncio.TimeoutError:
            if retry_count < self.config.max_retries:
                self.logger.warning(f"Request timeout, retrying... ({retry_count + 1})")
                await asyncio.sleep(1)
                return await self._make_request(endpoint, params, retry_count + 1)
            else:
                raise

        except Exception as e:
            self.logger.error(f"Request error: {e}")
            raise

    async def get_exchanges(self) -> List[Dict]:
        """
        Get list of supported exchanges

        Returns:
            List of exchange information
        """
        endpoint = "/v1/exchanges"
        return await self._make_request(endpoint)

    async def get_symbols(
        self,
        exchange_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get available trading symbols

        Args:
            exchange_id: Optional exchange filter (e.g., "BINANCE")

        Returns:
            List of symbol information
        """
        endpoint = "/v1/symbols"
        params = {}
        if exchange_id:
            params["filter_exchange_id"] = exchange_id

        return await self._make_request(endpoint, params)

    async def get_ohlcv(
        self,
        symbol_id: str,
        period: str = "1MIN",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get OHLCV (candlestick) data

        Args:
            symbol_id: Symbol identifier (e.g., "BINANCE_SPOT_BTC_USDT")
            period: Time period (1MIN, 5MIN, 1HRS, 1DAY, etc.)
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"/v1/ohlcv/{symbol_id}/history"
        params = {
            "period_id": period,
            "limit": limit
        }

        if start_time:
            params["time_start"] = start_time.isoformat()
        if end_time:
            params["time_end"] = end_time.isoformat()

        self.logger.info(f"Fetching OHLCV for {symbol_id} ({period})")

        data = await self._make_request(endpoint, params)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if not df.empty:
            df["time_period_start"] = pd.to_datetime(df["time_period_start"])
            df["time_period_end"] = pd.to_datetime(df["time_period_end"])
            df.rename(columns={
                "time_period_start": "timestamp",
                "price_open": "open",
                "price_high": "high",
                "price_low": "low",
                "price_close": "close",
                "volume_traded": "volume"
            }, inplace=True)

        return df

    async def get_trades(
        self,
        symbol_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical trades

        Args:
            symbol_id: Symbol identifier
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of trades

        Returns:
            DataFrame with trade data
        """
        endpoint = f"/v1/trades/{symbol_id}/history"
        params = {"limit": limit}

        if start_time:
            params["time_start"] = start_time.isoformat()
        if end_time:
            params["time_end"] = end_time.isoformat()

        self.logger.info(f"Fetching trades for {symbol_id}")

        data = await self._make_request(endpoint, params)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if not df.empty:
            df["time_exchange"] = pd.to_datetime(df["time_exchange"])
            df.rename(columns={
                "time_exchange": "timestamp",
                "taker_side": "side"
            }, inplace=True)

        return df

    async def get_quotes(
        self,
        symbol_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get quote data (best bid/ask)

        Args:
            symbol_id: Symbol identifier
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of quotes

        Returns:
            DataFrame with quote data
        """
        endpoint = f"/v1/quotes/{symbol_id}/history"
        params = {"limit": limit}

        if start_time:
            params["time_start"] = start_time.isoformat()
        if end_time:
            params["time_end"] = end_time.isoformat()

        self.logger.info(f"Fetching quotes for {symbol_id}")

        data = await self._make_request(endpoint, params)

        df = pd.DataFrame(data)
        if not df.empty:
            df["time_exchange"] = pd.to_datetime(df["time_exchange"])
            df.rename(columns={"time_exchange": "timestamp"}, inplace=True)

        return df

    async def get_orderbook(
        self,
        symbol_id: str,
        limit_levels: Optional[int] = None
    ) -> Dict:
        """
        Get current orderbook snapshot

        Args:
            symbol_id: Symbol identifier
            limit_levels: Optional limit on depth levels

        Returns:
            Orderbook data with bids and asks
        """
        endpoint = f"/v1/orderbooks/{symbol_id}/current"
        params = {}
        if limit_levels:
            params["limit_levels"] = limit_levels

        self.logger.info(f"Fetching orderbook for {symbol_id}")

        data = await self._make_request(endpoint, params)

        return {
            "timestamp": pd.to_datetime(data["time_exchange"]),
            "bids": [[level["price"], level["size"]] for level in data.get("bids", [])],
            "asks": [[level["price"], level["size"]] for level in data.get("asks", [])],
            "symbol_id": symbol_id
        }

    async def stream_websocket(
        self,
        symbols: List[str],
        data_type: str = "trade"
    ) -> AsyncIterator[Dict]:
        """
        Stream real-time data via WebSocket

        Args:
            symbols: List of symbol IDs to subscribe
            data_type: Data type to stream (trade, quote, book, etc.)

        Yields:
            Real-time market data messages
        """
        import websockets

        url = f"{self.config.ws_url}"

        try:
            async with websockets.connect(url, extra_headers=self.headers) as ws:
                self._ws_connection = ws

                # Subscribe to symbols
                subscribe_message = {
                    "type": "hello",
                    "apikey": self.config.api_key,
                    "heartbeat": False,
                    "subscribe_data_type": [data_type],
                    "subscribe_filter_symbol_id": symbols
                }

                await ws.send(str(subscribe_message))

                self.logger.info(f"WebSocket connected, subscribed to {len(symbols)} symbols")

                # Receive messages
                async for message in ws:
                    try:
                        import json
                        data = json.loads(message)
                        yield data
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON: {message}")
                        continue

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            raise

    def normalize_symbol(
        self,
        exchange: str,
        base: str,
        quote: str,
        market_type: str = "SPOT"
    ) -> str:
        """
        Create CoinAPI symbol identifier

        Args:
            exchange: Exchange name (e.g., "BINANCE")
            base: Base currency (e.g., "BTC")
            quote: Quote currency (e.g., "USDT")
            market_type: Market type (SPOT, FUTURES, etc.)

        Returns:
            CoinAPI symbol ID (e.g., "BINANCE_SPOT_BTC_USDT")
        """
        return f"{exchange.upper()}_{market_type}_{base.upper()}_{quote.upper()}"

    async def get_exchange_rate(
        self,
        base: str,
        quote: str,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Get exchange rate between two currencies

        Args:
            base: Base currency
            quote: Quote currency
            timestamp: Optional historical timestamp

        Returns:
            Exchange rate
        """
        endpoint = f"/v1/exchangerate/{base.upper()}/{quote.upper()}"
        params = {}
        if timestamp:
            params["time"] = timestamp.isoformat()

        data = await self._make_request(endpoint, params)
        return data["rate"]

    async def batch_fetch_ohlcv(
        self,
        symbols: List[str],
        period: str = "1MIN",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols in parallel

        Args:
            symbols: List of symbol IDs
            period: Time period
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        tasks = [
            self.get_ohlcv(symbol, period, start_time, end_time)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching {symbol}: {result}")
                data_dict[symbol] = pd.DataFrame()
            else:
                data_dict[symbol] = result

        return data_dict


def create_coinapi_client(api_key: str, **kwargs) -> CoinAPIClient:
    """
    Factory function to create CoinAPI client

    Args:
        api_key: CoinAPI API key
        **kwargs: Additional configuration options

    Returns:
        Configured CoinAPIClient instance
    """
    config = CoinAPIConfig(api_key=api_key, **kwargs)
    return CoinAPIClient(config)


# Example usage
if __name__ == "__main__":
    import asyncio

    async def example():
        # Initialize client
        client = create_coinapi_client(
            api_key="your_api_key_here",
            exchanges=["BINANCE", "COINBASE"]
        )

        try:
            # Get OHLCV data
            symbol = client.normalize_symbol("BINANCE", "BTC", "USDT")
            df = await client.get_ohlcv(
                symbol,
                period="1MIN",
                limit=100
            )
            print(f"Loaded {len(df)} candles")
            print(df.head())

            # Get current orderbook
            orderbook = await client.get_orderbook(symbol, limit_levels=10)
            print(f"\nOrderbook at {orderbook['timestamp']}")
            print(f"Best bid: {orderbook['bids'][0]}")
            print(f"Best ask: {orderbook['asks'][0]}")

            # Get exchange rate
            rate = await client.get_exchange_rate("BTC", "USD")
            print(f"\nBTC/USD rate: {rate}")

        finally:
            await client.close()

    asyncio.run(example())
