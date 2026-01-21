"""Triple Barrier Labeling and Meta-Labeling for ML targets.

Implements labeling methods from "Advances in Financial Machine Learning"
by Marcos López de Prado.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numba import jit, prange

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ExitType(IntEnum):
    """Exit type for triple barrier."""

    TAKE_PROFIT = 1
    STOP_LOSS = -1
    TIMEOUT = 0


@dataclass
class TripleBarrierLabel:
    """Single label from triple barrier method."""

    timestamp: np.datetime64
    label: int  # -1, 0, 1
    return_pct: float
    holding_period: int  # In bars
    exit_type: ExitType
    tp_barrier: float
    sl_barrier: float


@dataclass
class LabelingConfig:
    """Configuration for triple barrier labeling."""

    # Barrier settings
    tp_multiple: float = 2.0  # Take profit = tp_multiple * volatility
    sl_multiple: float = 2.0  # Stop loss = sl_multiple * volatility
    max_holding_periods: int = 60  # Maximum holding period in bars

    # Volatility settings
    volatility_window: int = 20  # Window for volatility calculation
    volatility_type: str = "returns"  # "returns" or "parkinson"

    # Label settings
    min_return_threshold: float = 0.0  # Minimum return for non-neutral label
    neutral_zone: float = 0.0  # Returns within this zone are labeled neutral

    # Sampling
    min_sample_weight: float = 0.1


@jit(nopython=True, cache=True)
def _compute_triple_barrier_numba(
    prices: NDArray[np.float64],
    volatilities: NDArray[np.float64],
    tp_multiple: float,
    sl_multiple: float,
    max_holding: int,
) -> tuple[
    NDArray[np.int8],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int8],
]:
    """Compute triple barrier labels using Numba.

    Returns:
        labels: Array of labels (-1, 0, 1)
        returns: Array of realized returns
        holding_periods: Array of holding periods
        exit_types: Array of exit types (1=TP, -1=SL, 0=Timeout)
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float64)
    holding_periods = np.zeros(n, dtype=np.int32)
    exit_types = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        entry_price = prices[i]
        vol = volatilities[i]

        if np.isnan(vol) or vol <= 0:
            labels[i] = 0
            continue

        # Set barriers
        tp_barrier = entry_price * (1 + tp_multiple * vol)
        sl_barrier = entry_price * (1 - sl_multiple * vol)

        # Find exit
        exit_idx = min(i + max_holding, n - 1)
        exit_type = 0  # Timeout by default
        exit_price = prices[exit_idx]

        for j in range(i + 1, min(i + max_holding + 1, n)):
            price = prices[j]

            # Check take profit
            if price >= tp_barrier:
                exit_idx = j
                exit_type = 1
                exit_price = price
                break

            # Check stop loss
            if price <= sl_barrier:
                exit_idx = j
                exit_type = -1
                exit_price = price
                break

        # Calculate return
        ret = (exit_price - entry_price) / entry_price
        holding = exit_idx - i

        # Assign label
        if exit_type == 1:
            labels[i] = 1
        elif exit_type == -1:
            labels[i] = -1
        else:
            # Timeout - label based on return sign
            if ret > 0:
                labels[i] = 1
            elif ret < 0:
                labels[i] = -1
            else:
                labels[i] = 0

        returns[i] = ret
        holding_periods[i] = holding
        exit_types[i] = exit_type

    return labels, returns, holding_periods, exit_types


def compute_ewm_volatility(
    prices: pl.Series | NDArray[np.float64],
    window: int = 20,
    min_periods: int = 5,
) -> NDArray[np.float64]:
    """Compute exponentially weighted volatility of returns."""
    if isinstance(prices, pl.Series):
        prices = prices.to_numpy()

    # Log returns
    returns = np.zeros_like(prices)
    returns[1:] = np.log(prices[1:] / prices[:-1])

    # EWM standard deviation
    alpha = 2 / (window + 1)
    ewm_var = np.zeros_like(returns)
    ewm_mean = np.zeros_like(returns)

    ewm_mean[0] = returns[0]
    ewm_var[0] = 0.0

    for i in range(1, len(returns)):
        ewm_mean[i] = alpha * returns[i] + (1 - alpha) * ewm_mean[i - 1]
        diff = returns[i] - ewm_mean[i - 1]
        ewm_var[i] = (1 - alpha) * (ewm_var[i - 1] + alpha * diff * diff)

    volatility = np.sqrt(ewm_var)

    # Set initial values to NaN
    volatility[:min_periods] = np.nan

    return volatility


def compute_parkinson_volatility(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    window: int = 20,
) -> NDArray[np.float64]:
    """Compute Parkinson volatility estimator."""
    n = len(high)
    vol = np.full(n, np.nan)

    factor = 1.0 / (4.0 * np.log(2.0))

    for i in range(window - 1, n):
        sum_sq = 0.0
        for j in range(window):
            idx = i - j
            if low[idx] > 0:
                log_hl = np.log(high[idx] / low[idx])
                sum_sq += log_hl ** 2
        vol[i] = np.sqrt(factor * sum_sq / window)

    return vol


def triple_barrier_labels(
    df: pl.DataFrame,
    price_col: str = "close",
    config: LabelingConfig | None = None,
) -> pl.DataFrame:
    """Apply triple barrier labeling to price data.

    Args:
        df: DataFrame with price data
        price_col: Column name for price
        config: Labeling configuration

    Returns:
        DataFrame with label columns added
    """
    config = config or LabelingConfig()

    prices = df[price_col].to_numpy()

    # Compute volatility
    volatility = compute_ewm_volatility(prices, config.volatility_window)

    # Compute labels
    labels, returns, holding_periods, exit_types = _compute_triple_barrier_numba(
        prices,
        volatility,
        config.tp_multiple,
        config.sl_multiple,
        config.max_holding_periods,
    )

    # Apply neutral zone
    if config.neutral_zone > 0:
        neutral_mask = np.abs(returns) < config.neutral_zone
        labels[neutral_mask] = 0

    # Add columns to DataFrame
    result = df.with_columns([
        pl.Series("label", labels),
        pl.Series("return_pct", returns * 100),  # Convert to percentage
        pl.Series("holding_period", holding_periods),
        pl.Series("exit_type", exit_types),
        pl.Series("volatility", volatility),
        pl.lit(config.tp_multiple).alias("tp_multiple"),
        pl.lit(config.sl_multiple).alias("sl_multiple"),
        pl.lit(config.max_holding_periods).alias("max_holding_periods"),
        pl.lit(config.volatility_window).alias("volatility_window"),
    ])

    return result


def compute_sample_weights(
    labels_df: pl.DataFrame,
    return_col: str = "return_pct",
    min_weight: float = 0.1,
) -> pl.DataFrame:
    """Compute sample weights based on return magnitude.

    Higher absolute returns get higher weights to focus on significant moves.
    """
    abs_returns = labels_df[return_col].abs()
    max_return = abs_returns.max()

    if max_return is None or max_return == 0:
        weights = pl.lit(1.0)
    else:
        weights = (abs_returns / max_return).clip(min_weight, 1.0)

    return labels_df.with_columns(weights.alias("sample_weight"))


def compute_concurrent_labels(
    labels_df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    holding_col: str = "holding_period",
) -> pl.DataFrame:
    """Compute number of concurrent labels (label overlap).

    This is used for adjusting sample weights to account for
    overlapping prediction periods.
    """
    n = len(labels_df)
    concurrent = np.ones(n, dtype=np.int32)

    holding = labels_df[holding_col].to_numpy()

    for i in range(n):
        for j in range(max(0, i - int(holding.max())), i):
            if j + holding[j] > i:
                concurrent[i] += 1

    return labels_df.with_columns(pl.Series("concurrent_count", concurrent))


def apply_metalabeling(
    primary_labels: pl.DataFrame,
    secondary_model_probs: NDArray[np.float64],
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Apply meta-labeling to filter primary model signals.

    Meta-labeling uses a secondary model to predict whether the
    primary model's prediction will be correct.

    Args:
        primary_labels: DataFrame with primary labels
        secondary_model_probs: Probability predictions from secondary model
        threshold: Probability threshold for accepting primary signal

    Returns:
        DataFrame with meta-label columns added
    """
    # Meta-label: probability that primary signal is correct
    meta_labels = secondary_model_probs

    # Filter signals below threshold
    filtered_labels = primary_labels["label"].to_numpy().copy()
    filtered_labels[secondary_model_probs < threshold] = 0

    result = primary_labels.with_columns([
        pl.Series("meta_label", meta_labels),
        pl.Series("filtered_label", filtered_labels),
        pl.Series("signal_strength", meta_labels * np.abs(filtered_labels)),
    ])

    return result


class TripleBarrierLabeler:
    """Class for applying triple barrier labeling with state."""

    def __init__(self, config: LabelingConfig | None = None) -> None:
        self.config = config or LabelingConfig()
        self._label_stats: dict = {}

    def fit_transform(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
    ) -> pl.DataFrame:
        """Apply labeling and track statistics."""
        result = triple_barrier_labels(df, price_col, self.config)

        # Compute statistics
        labels = result["label"].to_numpy()
        self._label_stats = {
            "total": len(labels),
            "positive": int((labels == 1).sum()),
            "negative": int((labels == -1).sum()),
            "neutral": int((labels == 0).sum()),
            "positive_pct": float((labels == 1).mean() * 100),
            "negative_pct": float((labels == -1).mean() * 100),
            "neutral_pct": float((labels == 0).mean() * 100),
            "avg_holding_period": float(result["holding_period"].mean()),
            "avg_return": float(result["return_pct"].mean()),
        }

        return result

    @property
    def label_stats(self) -> dict:
        """Get labeling statistics."""
        return self._label_stats

    def to_database_records(
        self,
        df: pl.DataFrame,
        symbol: str,
        exchange: str = "binance",
    ) -> list[dict]:
        """Convert labeled DataFrame to database records."""
        records = []

        exit_type_map = {1: "tp", -1: "sl", 0: "timeout"}

        for row in df.iter_rows(named=True):
            if row.get("label") is None:
                continue

            record = {
                "exchange": exchange,
                "symbol": symbol,
                "timestamp": row["timestamp"],
                "label": int(row["label"]),
                "return_pct": float(row["return_pct"]),
                "holding_period_minutes": int(row["holding_period"]),
                "exit_type": exit_type_map.get(row["exit_type"], "timeout"),
                "meta_label": float(row.get("meta_label", 0.5)),
                "tp_multiple": float(row["tp_multiple"]),
                "sl_multiple": float(row["sl_multiple"]),
                "max_holding_minutes": int(row["max_holding_periods"]),
                "volatility_window": int(row["volatility_window"]),
            }
            records.append(record)

        return records


def create_binary_labels(
    labels: NDArray[np.int8] | pl.Series,
) -> tuple[NDArray[np.int8], NDArray[np.bool_]]:
    """Convert tri-class labels to binary for meta-labeling.

    Returns:
        side: Direction of trade (1 for long, -1 for short)
        is_correct: Whether the trade was profitable
    """
    if isinstance(labels, pl.Series):
        labels = labels.to_numpy()

    # Side is the original label direction
    side = labels.copy()

    # For meta-labeling, we predict if the side is correct
    # A label of 1 or -1 means it was correct (hit TP or avoided SL)
    # This is simplified - in practice you'd use actual returns
    is_correct = labels != 0

    return side, is_correct


def balance_labels(
    df: pl.DataFrame,
    label_col: str = "label",
    method: str = "undersample",
    random_state: int = 42,
) -> pl.DataFrame:
    """Balance label distribution.

    Args:
        df: DataFrame with labels
        label_col: Column name for labels
        method: "undersample" or "oversample"
        random_state: Random seed

    Returns:
        Balanced DataFrame
    """
    np.random.seed(random_state)

    labels = df[label_col].to_numpy()
    unique_labels = np.unique(labels[~np.isnan(labels.astype(float))])

    # Count each class
    counts = {int(label): int((labels == label).sum()) for label in unique_labels}

    if method == "undersample":
        target_count = min(counts.values())
    else:
        target_count = max(counts.values())

    indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]

        if method == "undersample":
            selected = np.random.choice(label_indices, target_count, replace=False)
        else:
            if len(label_indices) < target_count:
                # Oversample with replacement
                additional = np.random.choice(
                    label_indices,
                    target_count - len(label_indices),
                    replace=True,
                )
                selected = np.concatenate([label_indices, additional])
            else:
                selected = label_indices

        indices.extend(selected.tolist())

    # Sort to maintain temporal order
    indices = sorted(indices)

    return df[indices]
