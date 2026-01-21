"""CLI tool for model training."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import hydra
import numpy as np
import structlog
import torch
import typer
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from chronos_x.data.ingestion.clickhouse_client import ClickHouseClient, ClickHouseConfig
from chronos_x.data.processing.frac_diff import frac_diff_dataframe
from chronos_x.data.processing.labeling import LabelingConfig, TripleBarrierLabeler
from chronos_x.models.mamba.crypto_mamba import CryptoMamba, CryptoMambaConfig
from chronos_x.models.training.trainer import (
    CryptoDataset,
    CryptoMambaTrainer,
    TrainerConfig,
)
from chronos_x.validation.cpcv import CombinatorialPurgedKFold

logger = structlog.get_logger(__name__)
console = Console()
app = typer.Typer(help="Chronos-X Model Training CLI")


def create_db_client(cfg: DictConfig) -> ClickHouseClient:
    """Create ClickHouse client from config."""
    db_cfg = ClickHouseConfig(
        host=cfg.database.clickhouse.host,
        port=cfg.database.clickhouse.port,
        database=cfg.database.clickhouse.database,
        user=cfg.database.clickhouse.user,
        password=cfg.database.clickhouse.password,
    )
    return ClickHouseClient(db_cfg)


def prepare_features(
    ohlcv_data: list[dict],
    frac_diff_d: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare features from OHLCV data.

    Returns:
        Tuple of (features, prices)
    """
    import polars as pl

    # Convert to DataFrame
    df = pl.DataFrame(ohlcv_data)

    # Basic features
    df = df.with_columns([
        (pl.col("close").log() - pl.col("close").log().shift(1)).alias("log_return"),
        ((pl.col("close") - pl.col("open")) / pl.col("close")).alias("body"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range"),
        (pl.col("volume").log()).alias("log_volume"),
    ])

    # Moving averages
    for window in [5, 10, 20, 50]:
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=window).alias(f"ma_{window}"),
            pl.col("volume").rolling_mean(window_size=window).alias(f"vol_ma_{window}"),
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

    # Apply fractional differentiation if specified
    if frac_diff_d is not None:
        df, _ = frac_diff_dataframe(df, ["close"], d=frac_diff_d)

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

    # Handle NaN
    features = np.nan_to_num(features, nan=0.0)

    return features, prices


@app.command()
def train(
    symbol: str = typer.Argument("BTC/USDT", help="Trading symbol"),
    start_date: str = typer.Option("2023-01-01", help="Training start date"),
    end_date: str = typer.Option(None, help="Training end date (default: now)"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(64, help="Batch size"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    hidden_dim: int = typer.Option(256, help="Model hidden dimension"),
    num_layers: int = typer.Option(4, help="Number of Mamba layers"),
    output_dir: str = typer.Option("./models", help="Output directory for model"),
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Train a CryptoMamba model."""
    console.print("[blue]Starting model training...[/blue]")

    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name="config")

        # Connect to database
        client = create_db_client(cfg)

        if not client.health_check():
            console.print("[red]Cannot connect to ClickHouse![/red]")
            raise typer.Exit(1)

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end = (
            datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
            if end_date
            else datetime.now(UTC)
        )

        console.print(f"[blue]Loading data for {symbol} from {start_date} to {end_date or 'now'}[/blue]")

        # Fetch OHLCV data
        ohlcv_data = client.get_ohlcv(
            symbol=symbol,
            timeframe="1m",
            start=start,
            end=end,
        )

        if len(ohlcv_data) < 1000:
            console.print(f"[red]Insufficient data: {len(ohlcv_data)} candles[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Loaded {len(ohlcv_data)} candles[/green]")

        # Prepare features
        console.print("[blue]Preparing features...[/blue]")
        features, prices = prepare_features(ohlcv_data)

        # Generate labels
        console.print("[blue]Generating labels...[/blue]")
        import polars as pl

        price_df = pl.DataFrame({"close": prices, "timestamp": [d["timestamp"] for d in ohlcv_data]})
        labeler = TripleBarrierLabeler(LabelingConfig(
            tp_multiple=2.0,
            sl_multiple=2.0,
            max_holding_periods=60,
        ))
        labeled_df = labeler.fit_transform(price_df)
        labels = labeled_df["label"].to_numpy()
        returns = labeled_df["return_pct"].to_numpy() / 100

        console.print(f"[green]Label distribution: {labeler.label_stats}[/green]")

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

        console.print(f"[blue]Train samples: {len(X_train)}, Val samples: {len(X_val)}[/blue]")

        # Create datasets
        train_dataset = CryptoDataset(X_train, y_train, r_train)
        val_dataset = CryptoDataset(X_val, y_val, r_val)

        # Create model
        model_config = CryptoMambaConfig(
            input_dim=X.shape[-1],
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=3,
        )
        model = CryptoMamba(model_config)

        console.print(f"[green]Model parameters: {model.num_parameters:,}[/green]")

        # Create trainer
        trainer_config = TrainerConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device="cuda" if torch.cuda.is_available() else "cpu",
            checkpoint_dir=output_dir,
        )
        trainer = CryptoMambaTrainer(model, trainer_config)

        # Train
        console.print("[blue]Starting training...[/blue]")
        state = trainer.train(train_dataset, val_dataset)

        # Save final model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / f"crypto_mamba_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        trainer.save_model(model_path)

        console.print(f"[green]Training complete![/green]")
        console.print(f"[green]Model saved to: {model_path}[/green]")
        console.print(f"[blue]Best epoch: {state.best_epoch}, Best metric: {state.best_metric:.4f}[/blue]")

        client.close()


@app.command()
def cross_validate(
    symbol: str = typer.Argument("BTC/USDT", help="Trading symbol"),
    n_splits: int = typer.Option(5, help="Number of CV splits"),
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Run cross-validation on model."""
    console.print("[blue]Starting cross-validation...[/blue]")

    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name="config")

        client = create_db_client(cfg)

        if not client.health_check():
            console.print("[red]Cannot connect to ClickHouse![/red]")
            raise typer.Exit(1)

        # Fetch data
        start = datetime.now(UTC) - timedelta(days=365)
        ohlcv_data = client.get_ohlcv(symbol=symbol, timeframe="1m", start=start)

        if len(ohlcv_data) < 10000:
            console.print("[red]Insufficient data for CV[/red]")
            raise typer.Exit(1)

        # Prepare data
        features, prices = prepare_features(ohlcv_data)

        import polars as pl
        price_df = pl.DataFrame({"close": prices, "timestamp": [d["timestamp"] for d in ohlcv_data]})
        labeler = TripleBarrierLabeler()
        labeled_df = labeler.fit_transform(price_df)
        labels = labeled_df["label"].to_numpy() + 1  # Map to 0, 1, 2

        # Create sequences
        seq_len = 256
        n_samples = len(features) - seq_len
        X = np.array([features[i:i + seq_len] for i in range(n_samples)])
        y = labels[seq_len:]

        # CPCV
        cpcv = CombinatorialPurgedKFold(
            n_splits=n_splits,
            n_test_splits=2,
            purge_gap=10,
            embargo_pct=0.01,
        )

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(cpcv.split(X)):
            console.print(f"[blue]Fold {fold_idx + 1}/{cpcv.get_n_splits()}[/blue]")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create model and train
            model = CryptoMamba(CryptoMambaConfig(
                input_dim=X.shape[-1],
                hidden_dim=128,
                num_layers=2,
            ))

            trainer = CryptoMambaTrainer(model, TrainerConfig(
                epochs=20,
                batch_size=64,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ))

            train_dataset = CryptoDataset(X_train, y_train)
            val_dataset = CryptoDataset(X_test, y_test)

            trainer.train(train_dataset, val_dataset)

            # Evaluate
            metrics = trainer.evaluate(
                torch.utils.data.DataLoader(val_dataset, batch_size=128)
            )
            fold_metrics.append(metrics)

            console.print(f"  Accuracy: {metrics['accuracy']:.4f}")

        # Summary
        console.print("\n[green]Cross-Validation Results:[/green]")
        avg_accuracy = np.mean([m["accuracy"] for m in fold_metrics])
        std_accuracy = np.std([m["accuracy"] for m in fold_metrics])
        console.print(f"  Mean Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

        client.close()


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
