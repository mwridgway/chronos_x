"""CLI tool for backtesting strategies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import hydra
import numpy as np
import structlog
import torch
import typer
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from chronos_x.data.ingestion.clickhouse_client import ClickHouseClient, ClickHouseConfig
from chronos_x.models.mamba.crypto_mamba import CryptoMamba, CryptoMambaConfig
from chronos_x.strategy.risk import RiskConfig, RiskManager, VolatilityTargeting
from chronos_x.validation.metrics import (
    PerformanceMetrics,
    compute_all_metrics,
    compute_deflated_sharpe_ratio,
    compute_sharpe_ratio,
)

logger = structlog.get_logger(__name__)
console = Console()
app = typer.Typer(help="Chronos-X Backtesting CLI")


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


def prepare_features(ohlcv_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Prepare features from OHLCV data."""
    import polars as pl

    df = pl.DataFrame(ohlcv_data)

    # Basic features
    df = df.with_columns([
        (pl.col("close").log() - pl.col("close").log().shift(1)).alias("log_return"),
        ((pl.col("close") - pl.col("open")) / pl.col("close")).alias("body"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range"),
        (pl.col("volume").log()).alias("log_volume"),
    ])

    for window in [5, 10, 20, 50]:
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=window).alias(f"ma_{window}"),
        ])

    df = df.with_columns([
        pl.col("log_return").rolling_std(window_size=20).alias("volatility_20"),
    ])

    df = df.with_columns([
        ((pl.col("buy_volume") - pl.col("sell_volume")) /
         (pl.col("buy_volume") + pl.col("sell_volume") + 1)).alias("volume_imbalance"),
    ])

    feature_cols = ["log_return", "body", "range", "log_volume", "volatility_20", "volume_imbalance"]
    feature_cols += [f"ma_{w}" for w in [5, 10, 20, 50]]
    feature_cols = [c for c in feature_cols if c in df.columns]

    features = df.select(feature_cols).to_numpy()
    prices = df["close"].to_numpy()
    timestamps = df["timestamp"].to_list()

    features = np.nan_to_num(features, nan=0.0)

    return features, prices, timestamps


def run_backtest_simulation(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    slippage_bps: float = 5.0,
) -> dict:
    """Run backtest simulation.

    Args:
        prices: Array of prices
        signals: Array of signals (-1, 0, 1)
        initial_capital: Starting capital
        fee_rate: Trading fee rate
        slippage_bps: Slippage in basis points

    Returns:
        Dictionary of results
    """
    n = len(prices)
    position = 0  # Current position: -1, 0, 1
    cash = initial_capital
    equity = np.zeros(n)
    returns = np.zeros(n)
    trades = []

    entry_price = 0.0
    entry_idx = 0

    for i in range(1, n):
        # Current price with slippage
        slippage = prices[i] * slippage_bps / 10000
        buy_price = prices[i] + slippage
        sell_price = prices[i] - slippage

        # Position change
        target_position = int(signals[i])

        if target_position != position:
            # Close existing position
            if position != 0:
                if position == 1:  # Long -> close
                    pnl = (sell_price - entry_price) / entry_price
                    cash *= (1 + pnl - fee_rate)
                else:  # Short -> close
                    pnl = (entry_price - buy_price) / entry_price
                    cash *= (1 + pnl - fee_rate)

                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": prices[i],
                    "side": "long" if position == 1 else "short",
                    "pnl": pnl,
                    "return_pct": pnl * 100,
                })

            # Open new position
            if target_position != 0:
                entry_price = buy_price if target_position == 1 else sell_price
                entry_idx = i
                cash *= (1 - fee_rate)  # Fee for opening

            position = target_position

        # Mark-to-market
        if position == 1:
            equity[i] = cash * (1 + (prices[i] - entry_price) / entry_price)
        elif position == -1:
            equity[i] = cash * (1 + (entry_price - prices[i]) / entry_price)
        else:
            equity[i] = cash

        # Calculate returns
        if i > 0 and equity[i - 1] > 0:
            returns[i] = equity[i] / equity[i - 1] - 1

    # Close final position
    if position != 0:
        if position == 1:
            pnl = (prices[-1] - entry_price) / entry_price
        else:
            pnl = (entry_price - prices[-1]) / entry_price
        cash *= (1 + pnl - fee_rate)

        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": len(prices) - 1,
            "entry_price": entry_price,
            "exit_price": prices[-1],
            "side": "long" if position == 1 else "short",
            "pnl": pnl,
            "return_pct": pnl * 100,
        })

    equity[-1] = cash

    return {
        "equity": equity,
        "returns": returns[1:],  # Skip first zero
        "trades": trades,
        "final_equity": cash,
        "total_return": (cash - initial_capital) / initial_capital,
    }


@app.command()
def backtest(
    symbol: str = typer.Argument("BTC/USDT", help="Trading symbol"),
    model_path: str = typer.Option(None, help="Path to trained model"),
    start_date: str = typer.Option("2024-01-01", help="Backtest start date"),
    end_date: str = typer.Option(None, help="Backtest end date"),
    initial_capital: float = typer.Option(10000.0, help="Initial capital"),
    fee_rate: float = typer.Option(0.001, help="Fee rate (0.001 = 0.1%)"),
    slippage_bps: float = typer.Option(5.0, help="Slippage in basis points"),
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Run backtest on historical data."""
    console.print("[blue]Starting backtest...[/blue]")

    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name="config")

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

        console.print(f"[blue]Loading data for {symbol}[/blue]")

        # Fetch data
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
        features, prices, timestamps = prepare_features(ohlcv_data)

        # Generate signals
        if model_path and Path(model_path).exists():
            console.print(f"[blue]Loading model from {model_path}[/blue]")

            # Load model
            model = CryptoMamba(CryptoMambaConfig(input_dim=features.shape[-1]))
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            # Generate predictions
            seq_len = 256
            signals = np.zeros(len(features))

            with torch.no_grad():
                for i in range(seq_len, len(features)):
                    x = features[i - seq_len:i]
                    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

                    output = model(x_tensor)
                    pred = output["class_logits"].argmax(dim=-1).item()

                    # Map 0, 1, 2 to -1, 0, 1
                    signals[i] = pred - 1
        else:
            console.print("[yellow]No model provided, using simple momentum strategy[/yellow]")

            # Simple momentum strategy
            lookback = 20
            signals = np.zeros(len(prices))

            for i in range(lookback, len(prices)):
                momentum = (prices[i] - prices[i - lookback]) / prices[i - lookback]

                if momentum > 0.02:
                    signals[i] = 1  # Long
                elif momentum < -0.02:
                    signals[i] = -1  # Short
                else:
                    signals[i] = 0  # Neutral

        # Run backtest
        console.print("[blue]Running simulation...[/blue]")

        results = run_backtest_simulation(
            prices=prices,
            signals=signals,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
        )

        # Compute metrics
        trade_returns = np.array([t["return_pct"] / 100 for t in results["trades"]])
        metrics = compute_all_metrics(
            returns=results["returns"],
            trade_returns=trade_returns if len(trade_returns) > 0 else None,
        )

        # Compute benchmark (buy and hold)
        benchmark_return = (prices[-1] - prices[0]) / prices[0]
        benchmark_returns = np.diff(prices) / prices[:-1]
        benchmark_sharpe = compute_sharpe_ratio(benchmark_returns)

        # Display results
        console.print("\n[green]═══ Backtest Results ═══[/green]\n")

        # Summary table
        summary_table = Table(title="Performance Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Strategy", style="green")
        summary_table.add_column("Benchmark", style="yellow")

        summary_table.add_row(
            "Total Return",
            f"{metrics.total_return:.2%}",
            f"{benchmark_return:.2%}",
        )
        summary_table.add_row(
            "Annualized Return",
            f"{metrics.annualized_return:.2%}",
            "-",
        )
        summary_table.add_row(
            "Sharpe Ratio",
            f"{metrics.sharpe_ratio:.2f}",
            f"{benchmark_sharpe:.2f}",
        )
        summary_table.add_row(
            "Sortino Ratio",
            f"{metrics.sortino_ratio:.2f}",
            "-",
        )
        summary_table.add_row(
            "Max Drawdown",
            f"{metrics.max_drawdown:.2%}",
            "-",
        )
        summary_table.add_row(
            "Calmar Ratio",
            f"{metrics.calmar_ratio:.2f}",
            "-",
        )

        console.print(summary_table)

        # Trade statistics
        trade_table = Table(title="Trade Statistics")
        trade_table.add_column("Metric", style="cyan")
        trade_table.add_column("Value", style="green")

        trade_table.add_row("Total Trades", str(metrics.num_trades))
        trade_table.add_row("Win Rate", f"{metrics.win_rate:.2%}")
        trade_table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
        trade_table.add_row("Avg Win", f"{metrics.avg_win:.2%}")
        trade_table.add_row("Avg Loss", f"{metrics.avg_loss:.2%}")
        trade_table.add_row("Avg Trade", f"{metrics.avg_trade:.2%}")

        console.print(trade_table)

        # Deflated Sharpe Ratio
        dsr = compute_deflated_sharpe_ratio(
            sharpe_ratio=metrics.sharpe_ratio,
            num_trials=10,  # Assume we tested 10 strategies
            num_returns=len(results["returns"]),
        )
        console.print(f"\n[blue]Deflated Sharpe Ratio (10 trials): {dsr:.4f}[/blue]")

        # Final equity
        console.print(f"\n[green]Final Equity: ${results['final_equity']:,.2f}[/green]")

        client.close()


@app.command()
def compare_strategies(
    symbol: str = typer.Argument("BTC/USDT", help="Trading symbol"),
    start_date: str = typer.Option("2024-01-01", help="Start date"),
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Compare multiple trading strategies."""
    console.print("[blue]Comparing strategies...[/blue]")

    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name="config")

        client = create_db_client(cfg)

        if not client.health_check():
            console.print("[red]Cannot connect to ClickHouse![/red]")
            raise typer.Exit(1)

        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        ohlcv_data = client.get_ohlcv(symbol=symbol, timeframe="1m", start=start)

        features, prices, _ = prepare_features(ohlcv_data)

        strategies = {}

        # Buy and Hold
        strategies["Buy & Hold"] = np.ones(len(prices))

        # Momentum (20-bar)
        signals = np.zeros(len(prices))
        for i in range(20, len(prices)):
            mom = (prices[i] - prices[i - 20]) / prices[i - 20]
            signals[i] = 1 if mom > 0.01 else (-1 if mom < -0.01 else 0)
        strategies["Momentum 20"] = signals

        # Mean Reversion
        signals = np.zeros(len(prices))
        for i in range(20, len(prices)):
            ma = np.mean(prices[i - 20:i])
            zscore = (prices[i] - ma) / (np.std(prices[i - 20:i]) + 1e-8)
            signals[i] = -1 if zscore > 2 else (1 if zscore < -2 else 0)
        strategies["Mean Reversion"] = signals

        # Compare
        results_table = Table(title="Strategy Comparison")
        results_table.add_column("Strategy", style="cyan")
        results_table.add_column("Return", style="green")
        results_table.add_column("Sharpe", style="yellow")
        results_table.add_column("MaxDD", style="red")
        results_table.add_column("Trades", style="blue")

        for name, signals in strategies.items():
            result = run_backtest_simulation(prices, signals)
            metrics = compute_all_metrics(result["returns"])

            results_table.add_row(
                name,
                f"{metrics.total_return:.2%}",
                f"{metrics.sharpe_ratio:.2f}",
                f"{metrics.max_drawdown:.2%}",
                str(metrics.num_trades),
            )

        console.print(results_table)
        client.close()


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
