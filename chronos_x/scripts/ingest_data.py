"""CLI tool for data ingestion."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import hydra
import structlog
import typer
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from chronos_x.data.ingestion.binance_stream import (
    BinanceStreamManager,
    StreamConfig,
)
from chronos_x.data.ingestion.clickhouse_client import ClickHouseClient, ClickHouseConfig
from chronos_x.data.ingestion.historical_loader import HistoricalLoader, LoaderConfig

logger = structlog.get_logger(__name__)
console = Console()
app = typer.Typer(help="Chronos-X Data Ingestion CLI")


def create_db_client(cfg: DictConfig) -> ClickHouseClient:
    """Create ClickHouse client from config."""
    db_cfg = ClickHouseConfig(
        host=cfg.database.clickhouse.host,
        port=cfg.database.clickhouse.port,
        database=cfg.database.clickhouse.database,
        user=cfg.database.clickhouse.user,
        password=cfg.database.clickhouse.password,
        pool_size=cfg.database.clickhouse.pool_size,
    )
    return ClickHouseClient(db_cfg)


@app.command()
def init_schema(
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Initialize ClickHouse schema."""
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name="config")

        client = create_db_client(cfg)

        schema_path = Path(__file__).parent.parent / "data" / "schema" / "market_trades.sql"

        if not schema_path.exists():
            # Try relative to project root
            schema_path = Path("data/schema/market_trades.sql")

        if schema_path.exists():
            console.print(f"[blue]Running schema from {schema_path}[/blue]")
            client.run_schema(schema_path)
            console.print("[green]Schema initialized successfully![/green]")
        else:
            console.print("[red]Schema file not found![/red]")
            raise typer.Exit(1)

        client.close()


@app.command()
def backfill(
    symbol: str = typer.Argument("BTC/USDT", help="Trading pair symbol"),
    start: str = typer.Option("2024-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, help="End date (YYYY-MM-DD), default: now"),
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Backfill historical trade data."""

    async def run_backfill() -> None:
        with hydra.initialize(config_path=config_path, version_base=None):
            cfg = hydra.compose(config_name="config")

            client = create_db_client(cfg)

            # Check connection
            if not client.health_check():
                console.print("[red]Cannot connect to ClickHouse![/red]")
                raise typer.Exit(1)

            loader_cfg = LoaderConfig(
                exchange=cfg.exchange.name,
                sandbox=cfg.exchange.sandbox,
                api_key=cfg.exchange.get("api_key", ""),
                api_secret=cfg.exchange.get("api_secret", ""),
            )

            loader = HistoricalLoader(loader_cfg, client)

            start_date = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
            end_date = (
                datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
                if end
                else datetime.now(UTC)
            )

            console.print(f"[blue]Backfilling {symbol} from {start} to {end or 'now'}[/blue]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading trades...", total=None)

                def update_progress(
                    current_time: datetime, end_time: datetime, trades_loaded: int
                ) -> None:
                    pct = (
                        (current_time - start_date).total_seconds()
                        / (end_time - start_date).total_seconds()
                        * 100
                    )
                    progress.update(
                        task,
                        description=f"Loaded {trades_loaded:,} trades ({pct:.1f}%)",
                    )

                total = await loader.backfill_trades(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    progress_callback=update_progress,
                )

            console.print(f"[green]Backfill complete! Loaded {total:,} trades[/green]")

            # Verify data
            count = client.get_trade_count(symbol, start_date, end_date)
            console.print(f"[blue]Total trades in database: {count:,}[/blue]")

            client.close()

    asyncio.run(run_backfill())


@app.command()
def stream(
    symbols: list[str] = typer.Argument(["BTC/USDT"], help="Trading pair symbols"),
    testnet: bool = typer.Option(True, help="Use testnet"),
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Start real-time trade stream."""

    async def run_stream() -> None:
        with hydra.initialize(config_path=config_path, version_base=None):
            cfg = hydra.compose(config_name="config")

            client = create_db_client(cfg)

            if not client.health_check():
                console.print("[red]Cannot connect to ClickHouse![/red]")
                raise typer.Exit(1)

            stream_cfg = StreamConfig(
                use_testnet=testnet,
                reconnect_delay=cfg.websocket.trades.reconnect_delay,
                max_reconnect_attempts=cfg.websocket.trades.max_reconnect_attempts,
            )

            def flush_trades(trades: list[dict]) -> None:
                if trades:
                    client.insert_trades(trades)
                    console.print(f"[dim]Flushed {len(trades)} trades[/dim]")

            manager = BinanceStreamManager(
                symbols=symbols,
                config=stream_cfg,
                flush_callback=flush_trades,
            )

            console.print(f"[blue]Starting stream for {symbols}[/blue]")
            console.print("[dim]Press Ctrl+C to stop[/dim]")

            try:
                await manager.start()
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping stream...[/yellow]")
            finally:
                await manager.stop()
                client.close()

            stats = manager.stats
            console.print(f"[green]Stream stopped. Stats:[/green]")
            console.print(f"  Messages received: {stats.messages_received:,}")
            console.print(f"  Trades processed: {stats.trades_processed:,}")
            console.print(f"  Reconnections: {stats.reconnect_count}")

    asyncio.run(run_stream())


@app.command()
def sync(
    symbol: str = typer.Argument("BTC/USDT", help="Trading pair symbol"),
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Sync data from last known point to present."""

    async def run_sync() -> None:
        with hydra.initialize(config_path=config_path, version_base=None):
            cfg = hydra.compose(config_name="config")

            client = create_db_client(cfg)

            if not client.health_check():
                console.print("[red]Cannot connect to ClickHouse![/red]")
                raise typer.Exit(1)

            loader_cfg = LoaderConfig(
                exchange=cfg.exchange.name,
                sandbox=cfg.exchange.sandbox,
            )

            loader = HistoricalLoader(loader_cfg, client)

            last_time = loader.get_latest_trade_time(symbol)
            if last_time:
                console.print(f"[blue]Last known trade: {last_time}[/blue]")
            else:
                console.print("[yellow]No existing data found, starting fresh sync[/yellow]")

            total = await loader.sync_to_present(symbol)
            console.print(f"[green]Sync complete! Loaded {total:,} new trades[/green]")

            client.close()

    asyncio.run(run_sync())


@app.command()
def status(
    symbol: str = typer.Argument("BTC/USDT", help="Trading pair symbol"),
    config_path: str = typer.Option("config", help="Path to Hydra config directory"),
) -> None:
    """Check data status for a symbol."""
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name="config")

        client = create_db_client(cfg)

        if not client.health_check():
            console.print("[red]Cannot connect to ClickHouse![/red]")
            raise typer.Exit(1)

        count = client.get_trade_count(symbol)
        latest = client.get_latest_timestamp(symbol)

        console.print(f"[blue]Symbol: {symbol}[/blue]")
        console.print(f"  Total trades: {count:,}")
        console.print(f"  Latest trade: {latest or 'N/A'}")

        if latest:
            age = datetime.now(UTC) - latest
            console.print(f"  Data age: {age}")

        client.close()


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
