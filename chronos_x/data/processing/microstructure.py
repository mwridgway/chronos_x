"""Microstructure features: Order Flow Imbalance, VPIN, spreads, and more.

Implements key market microstructure metrics for high-frequency trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numba import jit, prange

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MicrostructureFeatures:
    """Container for computed microstructure features."""

    timestamps: NDArray[np.datetime64]
    ofi: NDArray[np.float64]
    ofi_normalized: NDArray[np.float64]
    vpin: NDArray[np.float64]
    volume_imbalance: NDArray[np.float64]
    effective_spread: NDArray[np.float64]
    realized_spread: NDArray[np.float64]
    realized_volatility: NDArray[np.float64]
    parkinson_volatility: NDArray[np.float64]
    trade_count: NDArray[np.int64]
    avg_trade_size: NDArray[np.float64]


@jit(nopython=True, cache=True)
def _compute_ofi_numba(
    bid_prices: NDArray[np.float64],
    bid_sizes: NDArray[np.float64],
    ask_prices: NDArray[np.float64],
    ask_sizes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Order Flow Imbalance using Numba.

    OFI = Σ(ΔBid_size * I{bid_price >= prev_bid} - ΔAsk_size * I{ask_price <= prev_ask})
    """
    n = len(bid_prices)
    ofi = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        # Bid side contribution
        if bid_prices[i] >= bid_prices[i - 1]:
            bid_contribution = bid_sizes[i] - bid_sizes[i - 1]
        elif bid_prices[i] < bid_prices[i - 1]:
            bid_contribution = -bid_sizes[i - 1]
        else:
            bid_contribution = 0.0

        # Ask side contribution
        if ask_prices[i] <= ask_prices[i - 1]:
            ask_contribution = ask_sizes[i] - ask_sizes[i - 1]
        elif ask_prices[i] > ask_prices[i - 1]:
            ask_contribution = -ask_sizes[i - 1]
        else:
            ask_contribution = 0.0

        ofi[i] = bid_contribution - ask_contribution

    return ofi


@jit(nopython=True, cache=True)
def _compute_vpin_numba(
    buy_volume: NDArray[np.float64],
    sell_volume: NDArray[np.float64],
    bucket_size: int,
) -> NDArray[np.float64]:
    """Compute Volume-Synchronized Probability of Informed Trading.

    VPIN = Σ|V_buy - V_sell| / (V_buy + V_sell) over rolling buckets
    """
    n = len(buy_volume)
    vpin = np.full(n, np.nan, dtype=np.float64)

    for i in range(bucket_size - 1, n):
        total_abs_imbalance = 0.0
        total_volume = 0.0

        for j in range(bucket_size):
            idx = i - j
            abs_imbalance = abs(buy_volume[idx] - sell_volume[idx])
            volume = buy_volume[idx] + sell_volume[idx]
            total_abs_imbalance += abs_imbalance
            total_volume += volume

        if total_volume > 0:
            vpin[i] = total_abs_imbalance / total_volume
        else:
            vpin[i] = 0.0

    return vpin


@jit(nopython=True, parallel=True, cache=True)
def _compute_realized_volatility_numba(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute realized volatility using Numba."""
    n = len(returns)
    rv = np.full(n, np.nan, dtype=np.float64)

    for i in prange(window - 1, n):
        sum_sq = 0.0
        for j in range(window):
            sum_sq += returns[i - j] ** 2
        rv[i] = np.sqrt(sum_sq)

    return rv


@jit(nopython=True, cache=True)
def _compute_parkinson_volatility_numba(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute Parkinson volatility estimator."""
    n = len(high)
    pv = np.full(n, np.nan, dtype=np.float64)
    factor = 1.0 / (4.0 * np.log(2.0))

    for i in range(window - 1, n):
        sum_sq = 0.0
        for j in range(window):
            idx = i - j
            if low[idx] > 0:
                log_hl = np.log(high[idx] / low[idx])
                sum_sq += log_hl ** 2

        pv[i] = np.sqrt(factor * sum_sq / window)

    return pv


def compute_order_flow_imbalance(
    df: pl.DataFrame,
    bid_price_col: str = "bid_price",
    bid_size_col: str = "bid_size",
    ask_price_col: str = "ask_price",
    ask_size_col: str = "ask_size",
    normalize_window: int = 100,
) -> pl.DataFrame:
    """Compute Order Flow Imbalance (OFI) from order book data.

    Args:
        df: DataFrame with order book data
        bid_price_col: Column name for best bid price
        bid_size_col: Column name for best bid size
        ask_price_col: Column name for best ask price
        ask_size_col: Column name for best ask size
        normalize_window: Window for normalizing OFI

    Returns:
        DataFrame with OFI columns added
    """
    bid_prices = df[bid_price_col].to_numpy()
    bid_sizes = df[bid_size_col].to_numpy()
    ask_prices = df[ask_price_col].to_numpy()
    ask_sizes = df[ask_size_col].to_numpy()

    ofi = _compute_ofi_numba(bid_prices, bid_sizes, ask_prices, ask_sizes)

    # Normalize OFI by rolling standard deviation
    ofi_series = pl.Series("ofi", ofi)
    ofi_std = ofi_series.rolling_std(window_size=normalize_window)

    result = df.with_columns([
        ofi_series,
        (ofi_series / ofi_std.fill_null(1.0)).alias("ofi_normalized"),
    ])

    return result


def compute_vpin(
    df: pl.DataFrame,
    buy_volume_col: str = "buy_volume",
    sell_volume_col: str = "sell_volume",
    bucket_size: int = 50,
) -> pl.DataFrame:
    """Compute Volume-Synchronized Probability of Informed Trading (VPIN).

    Args:
        df: DataFrame with volume data
        buy_volume_col: Column name for buy volume
        sell_volume_col: Column name for sell volume
        bucket_size: Number of periods per VPIN bucket

    Returns:
        DataFrame with VPIN column added
    """
    buy_volume = df[buy_volume_col].to_numpy()
    sell_volume = df[sell_volume_col].to_numpy()

    vpin = _compute_vpin_numba(buy_volume, sell_volume, bucket_size)

    return df.with_columns([
        pl.Series("vpin", vpin),
    ])


def compute_effective_spread(
    df: pl.DataFrame,
    trade_price_col: str = "price",
    mid_price_col: str = "mid_price",
    side_col: str = "side",
) -> pl.DataFrame:
    """Compute effective spread from trade data.

    Effective Spread = 2 * |Trade Price - Mid Price| * side_indicator
    where side_indicator is +1 for buys and -1 for sells.
    """
    result = df.with_columns([
        (
            2.0
            * (pl.col(trade_price_col) - pl.col(mid_price_col)).abs()
            / pl.col(mid_price_col)
            * 10000  # Convert to basis points
        ).alias("effective_spread_bps"),
    ])

    return result


def compute_realized_spread(
    df: pl.DataFrame,
    trade_price_col: str = "price",
    mid_price_col: str = "mid_price",
    side_col: str = "side",
    delay_periods: int = 5,
) -> pl.DataFrame:
    """Compute realized spread (spread after price impact dissipates).

    Realized Spread = 2 * (Trade Price - Mid Price at t+delay) * side_indicator
    """
    side_indicator = pl.when(pl.col(side_col) == "buy").then(1).otherwise(-1)

    future_mid = pl.col(mid_price_col).shift(-delay_periods)

    result = df.with_columns([
        (
            2.0
            * (pl.col(trade_price_col) - future_mid)
            * side_indicator
            / pl.col(mid_price_col)
            * 10000
        ).alias("realized_spread_bps"),
    ])

    return result


def compute_volatility_metrics(
    df: pl.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    window: int = 20,
) -> pl.DataFrame:
    """Compute various volatility metrics.

    Args:
        df: DataFrame with OHLC data
        close_col: Column name for close price
        high_col: Column name for high price
        low_col: Column name for low price
        window: Rolling window size

    Returns:
        DataFrame with volatility columns added
    """
    # Log returns
    returns = np.log(df[close_col].to_numpy()[1:] / df[close_col].to_numpy()[:-1])
    returns = np.concatenate([[np.nan], returns])

    # Realized volatility
    rv = _compute_realized_volatility_numba(returns, window)

    # Parkinson volatility
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    pv = _compute_parkinson_volatility_numba(high, low, window)

    result = df.with_columns([
        pl.Series("log_return", returns),
        pl.Series("realized_volatility", rv),
        pl.Series("parkinson_volatility", pv),
    ])

    return result


def compute_volume_imbalance(
    df: pl.DataFrame,
    buy_volume_col: str = "buy_volume",
    sell_volume_col: str = "sell_volume",
    window: int = 20,
) -> pl.DataFrame:
    """Compute volume imbalance metrics.

    Volume Imbalance = (Buy Volume - Sell Volume) / (Buy Volume + Sell Volume)
    """
    total_volume = pl.col(buy_volume_col) + pl.col(sell_volume_col)
    imbalance = (pl.col(buy_volume_col) - pl.col(sell_volume_col)) / total_volume

    result = df.with_columns([
        imbalance.alias("volume_imbalance"),
        imbalance.rolling_mean(window_size=window).alias("volume_imbalance_ma"),
    ])

    return result


def compute_trade_metrics(
    df: pl.DataFrame,
    volume_col: str = "volume",
    trade_count_col: str = "trade_count",
    window: int = 20,
) -> pl.DataFrame:
    """Compute trade-related metrics.

    Args:
        df: DataFrame with trade data
        volume_col: Column name for volume
        trade_count_col: Column name for trade count
        window: Rolling window size

    Returns:
        DataFrame with trade metrics added
    """
    avg_trade_size = pl.col(volume_col) / pl.col(trade_count_col).cast(pl.Float64)

    result = df.with_columns([
        avg_trade_size.alias("avg_trade_size"),
        avg_trade_size.rolling_mean(window_size=window).alias("avg_trade_size_ma"),
        pl.col(trade_count_col).rolling_mean(window_size=window).alias("trade_intensity"),
    ])

    return result


def compute_all_microstructure_features(
    ohlcv_df: pl.DataFrame,
    orderbook_df: pl.DataFrame | None = None,
    window: int = 20,
    vpin_bucket_size: int = 50,
) -> pl.DataFrame:
    """Compute all microstructure features from OHLCV and order book data.

    Args:
        ohlcv_df: DataFrame with OHLCV and buy/sell volume columns
        orderbook_df: Optional DataFrame with order book snapshots
        window: Rolling window size for metrics
        vpin_bucket_size: VPIN bucket size

    Returns:
        DataFrame with all microstructure features
    """
    result = ohlcv_df.clone()

    # Ensure required columns exist
    required_cols = ["close", "high", "low", "volume", "buy_volume", "sell_volume"]
    for col in required_cols:
        if col not in result.columns:
            if col == "buy_volume":
                result = result.with_columns((pl.col("volume") * 0.5).alias("buy_volume"))
            elif col == "sell_volume":
                result = result.with_columns((pl.col("volume") * 0.5).alias("sell_volume"))

    # Add trade_count if missing
    if "trade_count" not in result.columns:
        result = result.with_columns(pl.lit(100).alias("trade_count"))

    # Volatility metrics
    result = compute_volatility_metrics(result, window=window)

    # Volume imbalance
    result = compute_volume_imbalance(result, window=window)

    # VPIN
    result = compute_vpin(result, bucket_size=vpin_bucket_size)

    # Trade metrics
    result = compute_trade_metrics(result, window=window)

    # Mid price (for spread calculations)
    if "mid_price" not in result.columns:
        result = result.with_columns(
            ((pl.col("high") + pl.col("low")) / 2).alias("mid_price")
        )

    # If order book data is available, compute OFI
    if orderbook_df is not None:
        # Join and compute OFI
        # This would need timestamp alignment
        pass

    return result


class MicrostructureProcessor:
    """Processor for computing microstructure features in batches."""

    def __init__(
        self,
        window: int = 20,
        vpin_bucket_size: int = 50,
        ofi_normalize_window: int = 100,
    ) -> None:
        self.window = window
        self.vpin_bucket_size = vpin_bucket_size
        self.ofi_normalize_window = ofi_normalize_window

    def process_ohlcv(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process OHLCV data to compute microstructure features."""
        return compute_all_microstructure_features(
            df,
            window=self.window,
            vpin_bucket_size=self.vpin_bucket_size,
        )

    def process_trades(
        self,
        trades_df: pl.DataFrame,
        resample_freq: str = "1m",
    ) -> pl.DataFrame:
        """Aggregate trades and compute features.

        Args:
            trades_df: DataFrame with individual trades
            resample_freq: Resampling frequency

        Returns:
            Aggregated DataFrame with features
        """
        # Aggregate trades to OHLCV
        ohlcv = trades_df.group_by_dynamic(
            "timestamp",
            every=resample_freq,
        ).agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            pl.col("quantity")
            .filter(pl.col("side") == "buy")
            .sum()
            .alias("buy_volume"),
            pl.col("quantity")
            .filter(pl.col("side") == "sell")
            .sum()
            .alias("sell_volume"),
            pl.count().alias("trade_count"),
        ])

        # Fill null volumes
        ohlcv = ohlcv.with_columns([
            pl.col("buy_volume").fill_null(0),
            pl.col("sell_volume").fill_null(0),
        ])

        return self.process_ohlcv(ohlcv)

    def to_dict_records(
        self,
        df: pl.DataFrame,
        symbol: str,
    ) -> list[dict]:
        """Convert processed DataFrame to list of dicts for database insertion."""
        records = []

        feature_cols = [
            "ofi",
            "ofi_normalized",
            "vpin",
            "volume_imbalance",
            "effective_spread_bps",
            "realized_spread_bps",
            "realized_volatility",
            "parkinson_volatility",
            "trade_count",
            "avg_trade_size",
        ]

        # Filter to existing columns
        existing_cols = [c for c in feature_cols if c in df.columns]

        for row in df.iter_rows(named=True):
            record = {
                "symbol": symbol,
                "timestamp": row["timestamp"],
                "window_minutes": self.window,
            }

            for col in existing_cols:
                value = row.get(col)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    record[col.replace("_bps", "")] = value

            records.append(record)

        return records
