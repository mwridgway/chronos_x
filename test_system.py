"""Standalone test script for Chronos-X without ClickHouse.

This script:
1. Downloads historical data using CCXT
2. Prepares features
3. Trains a CryptoMamba model
4. Runs a backtest
"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ccxt
import numpy as np
import polars as pl
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from chronos_x.data.processing.labeling import LabelingConfig, TripleBarrierLabeler
from chronos_x.models.mamba.crypto_mamba import CryptoMamba, CryptoMambaConfig
from chronos_x.models.training.trainer import (
    CryptoDataset,
    CryptoMambaTrainer,
    TrainerConfig,
)
from chronos_x.validation.metrics import compute_all_metrics, compute_sharpe_ratio

console = Console()


def download_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    days: int = 30,
) -> list[dict]:
    """Download historical OHLCV data from Binance."""
    console.print(f"[blue]Downloading {days} days of {symbol} data...[/blue]")

    exchange = ccxt.binance()

    # Calculate start time
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    since = int(start.timestamp() * 1000)

    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=1000,
        )

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)

        # Update since for next batch
        since = ohlcv[-1][0] + 1

        # Stop if we've reached the end
        if since >= int(end.timestamp() * 1000):
            break

        console.print(f"  Downloaded {len(all_ohlcv)} candles...", end="\r")

    console.print(f"[green]Downloaded {len(all_ohlcv)} candles[/green]")

    # Convert to dict format
    data = []
    for candle in all_ohlcv:
        data.append({
            "timestamp": datetime.fromtimestamp(candle[0] / 1000, tz=UTC),
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5],
            "buy_volume": candle[5] * 0.5,  # Estimate
            "sell_volume": candle[5] * 0.5,  # Estimate
        })

    return data


def prepare_features(ohlcv_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Prepare features from OHLCV data."""
    df = pl.DataFrame(ohlcv_data)

    # Basic features
    df = df.with_columns([
        (pl.col("close").log() - pl.col("close").log().shift(1)).alias("log_return"),
        ((pl.col("close") - pl.col("open")) / pl.col("close")).alias("body"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range"),
        (pl.col("volume").log()).alias("log_volume"),
    ])

    # Moving averages - normalize by current price
    for window in [5, 10, 20, 50]:
        df = df.with_columns([
            (pl.col("close").rolling_mean(window_size=window) / pl.col("close")).alias(f"ma_{window}"),
            (pl.col("volume").rolling_mean(window_size=window) / (pl.col("volume") + 1)).alias(f"vol_ma_{window}"),
        ])

    # Volatility
    df = df.with_columns([
        pl.col("log_return").rolling_std(window_size=20).alias("volatility_20"),
    ])

    # Volume features
    df = df.with_columns([
        ((pl.col("buy_volume") - pl.col("sell_volume")) /
         (pl.col("buy_volume") + pl.col("sell_volume") + 1)).alias("volume_imbalance"),
    ])

    # Select feature columns
    feature_cols = [
        "log_return", "body", "range", "log_volume",
        "volatility_20", "volume_imbalance",
    ]
    feature_cols += [f"ma_{w}" for w in [5, 10, 20, 50]]

    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Extract arrays
    features = df.select(feature_cols).to_numpy()
    prices = df["close"].to_numpy()

    # Handle NaN - replace with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # CRITICAL FIX: Normalize features to prevent gradient explosion
    # Use robust scaling to handle outliers
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0) + 1e-8  # Add small epsilon
    features = (features - feature_mean) / feature_std

    # Clip extreme values
    features = np.clip(features, -10, 10)

    return features, prices


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
) -> CryptoMamba:
    """Train CryptoMamba model."""
    console.print("[blue]Preparing training data...[/blue]")

    # Create sequences
    seq_len = 256
    n_samples = len(features) - seq_len

    X = np.array([features[i:i + seq_len] for i in range(n_samples)])
    y = labels[seq_len:]
    r = returns[seq_len:]

    # Map labels to 0, 1, 2
    y = y + 1  # -1, 0, 1 -> 0, 1, 2

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    r_train, r_val = r[:split_idx], r[split_idx:]

    console.print(f"[blue]Train: {len(X_train)}, Val: {len(X_val)}[/blue]")

    # Create datasets
    train_dataset = CryptoDataset(X_train, y_train, r_train)
    val_dataset = CryptoDataset(X_val, y_val, r_val)

    # Create model
    model_config = CryptoMambaConfig(
        input_dim=X.shape[-1],
        seq_len=seq_len,
        hidden_dim=128,  # Smaller for faster training
        num_layers=2,
        num_classes=3,
    )
    model = CryptoMamba(model_config)

    console.print(f"[green]Model parameters: {model.num_parameters:,}[/green]")

    # Create trainer with safer settings
    trainer_config = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-4,  # Reduced from 1e-3 to prevent gradient explosion
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="./models",
        gradient_clip=1.0,  # Enable gradient clipping (already default)
        use_amp=False,  # Disable AMP to avoid numerical instability
    )
    trainer = CryptoMambaTrainer(model, trainer_config)

    # Train
    console.print("[blue]Training model...[/blue]")
    state = trainer.train(train_dataset, val_dataset)

    console.print(f"[green]Training complete![/green]")
    console.print(f"[blue]Best epoch: {state.best_epoch}, Best metric: {state.best_metric:.4f}[/blue]")

    return model


def run_backtest(
    model: CryptoMamba,
    features: np.ndarray,
    prices: np.ndarray,
) -> dict:
    """Run backtest with trained model."""
    console.print("[blue]Running backtest...[/blue]")

    seq_len = 256
    signals = np.zeros(len(features))

    # Get device from model
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for i in range(seq_len, len(features)):
            x = features[i - seq_len:i]
            # Features are already normalized in prepare_features, no need to normalize again
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

            output = model(x_tensor)
            pred = output["class_logits"].argmax(dim=-1).item()

            # Map 0, 1, 2 to -1, 0, 1
            signals[i] = pred - 1

    # Simulate trading
    position = 0
    cash = 10000.0
    equity = []
    trades = []

    for i in range(1, len(prices)):
        target_position = int(signals[i])

        if target_position != position:
            # Close existing
            if position != 0:
                pnl_pct = (prices[i] - prices[i-1]) / prices[i-1] * position
                cash *= (1 + pnl_pct - 0.001)  # 0.1% fee

                trades.append({
                    "pnl_pct": pnl_pct,
                    "return_pct": pnl_pct * 100,
                })

            position = target_position

        # Mark-to-market
        if position != 0:
            unrealized = (prices[i] - prices[i-1]) / prices[i-1] * position
            equity.append(cash * (1 + unrealized))
        else:
            equity.append(cash)

    # Compute metrics
    equity_array = np.array(equity)
    returns = np.diff(equity_array) / equity_array[:-1]

    trade_returns = np.array([t["return_pct"] / 100 for t in trades]) if trades else np.array([])

    metrics = compute_all_metrics(
        returns=returns,
        trade_returns=trade_returns if len(trade_returns) > 0 else None,
    )

    # Benchmark
    benchmark_return = (prices[-1] - prices[0]) / prices[0]
    benchmark_returns = np.diff(prices) / prices[:-1]
    benchmark_sharpe = compute_sharpe_ratio(benchmark_returns)

    console.print("\n[green]═══ Backtest Results ═══[/green]\n")
    console.print(f"Strategy Return: {metrics.total_return:.2%}")
    console.print(f"Benchmark Return: {benchmark_return:.2%}")
    console.print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    console.print(f"Benchmark Sharpe: {benchmark_sharpe:.2f}")
    console.print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    console.print(f"Total Trades: {metrics.num_trades}")
    console.print(f"Win Rate: {metrics.win_rate:.2%}")

    return {
        "metrics": metrics,
        "benchmark_return": benchmark_return,
        "benchmark_sharpe": benchmark_sharpe,
    }


def main():
    """Main entry point."""
    console.print("[bold blue]Chronos-X System Test[/bold blue]\n")

    # Download data
    ohlcv_data = download_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        days=7,  # Start with 7 days for faster testing
    )

    if len(ohlcv_data) < 1000:
        console.print("[red]Insufficient data downloaded![/red]")
        return

    # Prepare features
    console.print("[blue]Preparing features...[/blue]")
    features, prices = prepare_features(ohlcv_data)

    # Generate labels
    console.print("[blue]Generating labels...[/blue]")
    price_df = pl.DataFrame({
        "close": prices,
        "timestamp": [d["timestamp"] for d in ohlcv_data]
    })

    labeler = TripleBarrierLabeler(LabelingConfig(
        tp_multiple=2.0,
        sl_multiple=2.0,
        max_holding_periods=60,
    ))
    labeled_df = labeler.fit_transform(price_df)
    labels = labeled_df["label"].to_numpy()
    returns = labeled_df["return_pct"].to_numpy() / 100

    console.print(f"[green]Label distribution: {labeler.label_stats}[/green]")

    # Train model
    model = train_model(
        features=features,
        labels=labels,
        returns=returns,
        epochs=20,  # Fewer epochs for quick test
        batch_size=64,
    )

    # Save model
    model_path = Path("./models/crypto_mamba_test.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    console.print(f"[green]Model saved to {model_path}[/green]\n")

    # Run backtest
    results = run_backtest(model, features, prices)

    console.print("\n[bold green]System test complete![/bold green]")


if __name__ == "__main__":
    main()
