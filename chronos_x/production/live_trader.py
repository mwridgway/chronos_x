"""Live trading system with async execution and paper trading support.

Main production trading loop that:
- Receives real-time data
- Generates predictions
- Manages positions
- Executes orders
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ccxt.async_support as ccxt_async
import numpy as np
import structlog
import torch

from chronos_x.strategy.execution import (
    ExecutionConfig,
    ExecutionResult,
    OrderSide,
    select_execution_algorithm,
)
from chronos_x.strategy.risk import RiskConfig, RiskManager
from chronos_x.strategy.signal_generator import (
    Signal,
    SignalGenerator,
    SignalGeneratorConfig,
    SignalType,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chronos_x.models.mamba.crypto_mamba import CryptoMamba

logger = structlog.get_logger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration."""

    PAPER = "paper"
    LIVE = "live"


@dataclass
class Position:
    """Current position information."""

    symbol: str
    side: str  # "long", "short", "flat"
    quantity: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class TradingConfig:
    """Configuration for live trading."""

    # Trading mode
    mode: TradingMode = TradingMode.PAPER
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])

    # Exchange settings
    exchange: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True

    # Capital
    initial_capital: float = 10000.0
    base_currency: str = "USDT"

    # Timing
    prediction_interval_seconds: int = 60
    data_lookback_minutes: int = 120

    # Model
    model_path: str = ""
    feature_config: dict[str, Any] = field(default_factory=dict)

    # Risk and execution
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    signal_config: SignalGeneratorConfig = field(default_factory=SignalGeneratorConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Monitoring
    enable_monitoring: bool = True
    alert_webhook: str = ""


@dataclass
class TradingState:
    """Current trading state."""

    is_running: bool = False
    positions: dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0
    cash: float = 0.0
    last_prediction_time: datetime | None = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0


class LiveTrader:
    """Main live trading system."""

    def __init__(
        self,
        model: CryptoMamba,
        config: TradingConfig,
    ) -> None:
        self.model = model
        self.config = config
        self._log = logger.bind(component="live_trader")

        # Initialize components
        self.risk_manager = RiskManager(config.risk_config)
        self.signal_generator = SignalGenerator(config.signal_config)

        # State
        self.state = TradingState(
            equity=config.initial_capital,
            cash=config.initial_capital,
        )

        # Exchange connection (lazy initialization)
        self._exchange: ccxt_async.Exchange | None = None

        # Data buffers
        self._price_buffers: dict[str, list[float]] = {
            s: [] for s in config.symbols
        }
        self._feature_buffers: dict[str, list[NDArray]] = {
            s: [] for s in config.symbols
        }

        # Event loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []

    async def _init_exchange(self) -> None:
        """Initialize exchange connection."""
        exchange_class = getattr(ccxt_async, self.config.exchange)

        exchange_config = {
            "apiKey": self.config.api_key or None,
            "secret": self.config.api_secret or None,
            "sandbox": self.config.testnet,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        }

        self._exchange = exchange_class(exchange_config)
        await self._exchange.load_markets()

        self._log.info(
            "exchange_connected",
            exchange=self.config.exchange,
            testnet=self.config.testnet,
        )

    async def _close_exchange(self) -> None:
        """Close exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    async def _fetch_recent_data(
        self,
        symbol: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Fetch recent OHLCV data for prediction.

        Returns:
            Tuple of (prices, features)
        """
        if self._exchange is None:
            raise RuntimeError("Exchange not initialized")

        # Fetch OHLCV
        limit = self.config.data_lookback_minutes
        ohlcv = await self._exchange.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=limit,
        )

        if not ohlcv:
            raise ValueError(f"No data received for {symbol}")

        # Convert to numpy arrays
        ohlcv_array = np.array(ohlcv)
        timestamps = ohlcv_array[:, 0]
        opens = ohlcv_array[:, 1]
        highs = ohlcv_array[:, 2]
        lows = ohlcv_array[:, 3]
        closes = ohlcv_array[:, 4]
        volumes = ohlcv_array[:, 5]

        prices = closes

        # Compute basic features (simplified - should match training)
        features = self._compute_features(
            opens, highs, lows, closes, volumes
        )

        return prices, features

    def _compute_features(
        self,
        opens: NDArray,
        highs: NDArray,
        lows: NDArray,
        closes: NDArray,
        volumes: NDArray,
    ) -> NDArray[np.float64]:
        """Compute features from OHLCV data.

        This should match the feature engineering used in training.
        """
        n = len(closes)

        # Price features
        returns = np.zeros(n)
        returns[1:] = np.log(closes[1:] / closes[:-1])

        log_volume = np.log(volumes + 1)
        normalized_volume = (log_volume - log_volume.mean()) / (log_volume.std() + 1e-8)

        # Moving averages
        ma_5 = np.convolve(closes, np.ones(5) / 5, mode="same")
        ma_20 = np.convolve(closes, np.ones(20) / 20, mode="same")

        # Volatility
        vol_20 = np.zeros(n)
        for i in range(20, n):
            vol_20[i] = np.std(returns[i - 20 : i])

        # Range
        high_low_range = (highs - lows) / closes

        # OHLC relationships
        body = (closes - opens) / (highs - lows + 1e-8)
        upper_shadow = (highs - np.maximum(opens, closes)) / (highs - lows + 1e-8)
        lower_shadow = (np.minimum(opens, closes) - lows) / (highs - lows + 1e-8)

        # Stack features
        features = np.column_stack([
            returns,
            normalized_volume,
            (closes - ma_5) / ma_5,
            (closes - ma_20) / ma_20,
            vol_20,
            high_low_range,
            body,
            upper_shadow,
            lower_shadow,
        ])

        # Pad to match expected input dimension
        expected_dim = 64  # Should match model config
        if features.shape[1] < expected_dim:
            padding = np.zeros((n, expected_dim - features.shape[1]))
            features = np.column_stack([features, padding])

        return features

    async def _generate_prediction(
        self,
        symbol: str,
        prices: NDArray[np.float64],
        features: NDArray[np.float64],
    ) -> Signal | None:
        """Generate trading signal from model prediction."""
        # Prepare input tensor
        seq_len = min(len(features), 256)  # Model sequence length
        x = features[-seq_len:]

        # Normalize features
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        # Model inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)

        # Get predictions
        probs = torch.softmax(output["class_logits"], dim=-1).numpy()[0]
        pred_class = int(output["class_logits"].argmax(dim=-1).item())

        # Map class to prediction (-1, 0, 1)
        class_mapping = {0: -1, 1: 0, 2: 1}
        prediction = class_mapping.get(pred_class, 0)

        # Meta-labeler probability (simplified - use primary prob)
        meta_prob = float(probs.max())

        # Generate signal
        signal = self.signal_generator.generate_signal(
            symbol=symbol,
            model_prediction=prediction,
            model_probabilities=probs,
            meta_probability=meta_prob,
            prices=prices,
            timestamp=datetime.now(UTC),
        )

        return signal

    async def _execute_signal(
        self,
        signal: Signal,
    ) -> ExecutionResult | None:
        """Execute a trading signal."""
        if self._exchange is None:
            return None

        symbol = signal.symbol
        current_position = self.state.positions.get(symbol)

        # Determine action
        if signal.signal_type == SignalType.LONG:
            if current_position and current_position.side == "short":
                # Close short first
                await self._close_position(symbol)

            if not current_position or current_position.side == "flat":
                # Open long
                return await self._open_position(
                    symbol, OrderSide.BUY, signal.position_size_suggestion
                )

        elif signal.signal_type == SignalType.SHORT:
            if current_position and current_position.side == "long":
                # Close long first
                await self._close_position(symbol)

            if not current_position or current_position.side == "flat":
                # Open short
                return await self._open_position(
                    symbol, OrderSide.SELL, signal.position_size_suggestion
                )

        elif signal.signal_type == SignalType.NEUTRAL:
            if current_position and current_position.side != "flat":
                # Close any open position
                return await self._close_position(symbol)

        return None

    async def _open_position(
        self,
        symbol: str,
        side: OrderSide,
        size_pct: float,
    ) -> ExecutionResult | None:
        """Open a new position."""
        # Calculate order quantity
        ticker = await self._exchange.fetch_ticker(symbol)
        current_price = ticker["last"]

        position_value = self.state.cash * size_pct
        quantity = position_value / current_price

        # Execute
        algorithm = select_execution_algorithm(
            quantity=quantity,
            urgency=0.5,
            config=self.config.execution_config,
        )

        # Create order callback
        async def place_order(order):
            if self.config.mode == TradingMode.PAPER:
                # Paper trading - simulate fill
                order.status = "filled"
                order.filled_quantity = order.quantity
                order.avg_fill_price = current_price
                return order
            else:
                # Live trading
                result = await self._exchange.create_order(
                    symbol=order.symbol,
                    type=order.order_type.value,
                    side=order.side.value,
                    amount=order.quantity,
                    price=order.price,
                )
                order.order_id = result["id"]
                order.status = result["status"]
                order.filled_quantity = result.get("filled", 0)
                order.avg_fill_price = result.get("average", order.price)
                return order

        algorithm.order_callback = place_order

        result = await algorithm.execute(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=current_price,
        )

        # Update state
        if result.filled_quantity > 0:
            self.state.positions[symbol] = Position(
                symbol=symbol,
                side="long" if side == OrderSide.BUY else "short",
                quantity=result.filled_quantity,
                entry_price=result.avg_fill_price,
                entry_time=datetime.now(UTC),
            )
            self.state.cash -= result.filled_quantity * result.avg_fill_price
            self.state.total_trades += 1

            self._log.info(
                "position_opened",
                symbol=symbol,
                side=side.value,
                quantity=result.filled_quantity,
                price=result.avg_fill_price,
            )

        return result

    async def _close_position(
        self,
        symbol: str,
    ) -> ExecutionResult | None:
        """Close an existing position."""
        position = self.state.positions.get(symbol)
        if not position or position.side == "flat":
            return None

        # Determine close side
        close_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY

        ticker = await self._exchange.fetch_ticker(symbol)
        current_price = ticker["last"]

        # Execute
        algorithm = select_execution_algorithm(
            quantity=position.quantity,
            urgency=0.7,  # More urgent for closing
            config=self.config.execution_config,
        )

        async def place_order(order):
            if self.config.mode == TradingMode.PAPER:
                order.status = "filled"
                order.filled_quantity = order.quantity
                order.avg_fill_price = current_price
                return order
            else:
                result = await self._exchange.create_order(
                    symbol=order.symbol,
                    type=order.order_type.value,
                    side=order.side.value,
                    amount=order.quantity,
                    price=order.price,
                )
                order.order_id = result["id"]
                order.status = result["status"]
                order.filled_quantity = result.get("filled", 0)
                order.avg_fill_price = result.get("average", order.price)
                return order

        algorithm.order_callback = place_order

        result = await algorithm.execute(
            symbol=symbol,
            side=close_side,
            quantity=position.quantity,
            reference_price=current_price,
        )

        # Update state
        if result.filled_quantity > 0:
            # Calculate PnL
            if position.side == "long":
                pnl = (result.avg_fill_price - position.entry_price) * result.filled_quantity
            else:
                pnl = (position.entry_price - result.avg_fill_price) * result.filled_quantity

            self.state.cash += result.filled_quantity * result.avg_fill_price + pnl
            self.state.total_pnl += pnl

            if pnl > 0:
                self.state.winning_trades += 1

            # Clear position
            self.state.positions[symbol] = Position(
                symbol=symbol,
                side="flat",
                quantity=0,
                entry_price=0,
                entry_time=datetime.now(UTC),
            )

            self._log.info(
                "position_closed",
                symbol=symbol,
                quantity=result.filled_quantity,
                price=result.avg_fill_price,
                pnl=pnl,
            )

        return result

    async def _update_equity(self) -> None:
        """Update equity calculation."""
        equity = self.state.cash

        for symbol, position in self.state.positions.items():
            if position.side != "flat" and position.quantity > 0:
                ticker = await self._exchange.fetch_ticker(symbol)
                current_price = ticker["last"]

                if position.side == "long":
                    unrealized = (current_price - position.entry_price) * position.quantity
                else:
                    unrealized = (position.entry_price - current_price) * position.quantity

                position.unrealized_pnl = unrealized
                equity += position.quantity * current_price

        self.state.equity = equity

        # Update max drawdown
        peak = max(self.config.initial_capital, equity)
        drawdown = (peak - equity) / peak
        self.state.max_drawdown = max(self.state.max_drawdown, drawdown)

    async def _trading_loop(self) -> None:
        """Main trading loop."""
        self._log.info("starting_trading_loop")

        while self.state.is_running:
            try:
                for symbol in self.config.symbols:
                    # Fetch data
                    prices, features = await self._fetch_recent_data(symbol)

                    # Generate prediction
                    signal = await self._generate_prediction(
                        symbol, prices, features
                    )

                    if signal is not None:
                        # Check risk limits
                        is_ok, reason = self.risk_manager.check_risk_limits(
                            self.state.equity,
                            self.state.total_pnl,
                        )

                        if not is_ok:
                            self._log.warning("risk_limit_breached", reason=reason)
                            continue

                        # Execute signal
                        await self._execute_signal(signal)

                # Update equity
                await self._update_equity()

                self.state.last_prediction_time = datetime.now(UTC)

                # Log status
                self._log.info(
                    "trading_status",
                    equity=self.state.equity,
                    cash=self.state.cash,
                    total_pnl=self.state.total_pnl,
                    trades=self.state.total_trades,
                    win_rate=(
                        self.state.winning_trades / self.state.total_trades
                        if self.state.total_trades > 0
                        else 0
                    ),
                )

                # Wait for next interval
                await asyncio.sleep(self.config.prediction_interval_seconds)

            except Exception as e:
                self._log.error("trading_loop_error", error=str(e))
                await asyncio.sleep(5)

    async def start(self) -> None:
        """Start the trading system."""
        self._log.info(
            "starting_trader",
            mode=self.config.mode.value,
            symbols=self.config.symbols,
        )

        # Initialize exchange
        await self._init_exchange()

        # Set running state
        self.state.is_running = True

        # Start trading loop
        self._tasks.append(asyncio.create_task(self._trading_loop()))

    async def stop(self) -> None:
        """Stop the trading system."""
        self._log.info("stopping_trader")

        self.state.is_running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close positions if configured
        for symbol in list(self.state.positions.keys()):
            position = self.state.positions[symbol]
            if position.side != "flat":
                await self._close_position(symbol)

        # Close exchange
        await self._close_exchange()

        self._log.info(
            "trader_stopped",
            final_equity=self.state.equity,
            total_pnl=self.state.total_pnl,
            total_trades=self.state.total_trades,
        )

    async def run(self) -> None:
        """Run the trading system until stopped."""
        await self.start()

        try:
            while self.state.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()


async def run_paper_trading(
    model: CryptoMamba,
    symbols: list[str],
    duration_hours: float = 24,
    initial_capital: float = 10000.0,
) -> TradingState:
    """Run paper trading for a specified duration.

    Args:
        model: Trained CryptoMamba model
        symbols: List of trading symbols
        duration_hours: How long to run
        initial_capital: Starting capital

    Returns:
        Final trading state
    """
    config = TradingConfig(
        mode=TradingMode.PAPER,
        symbols=symbols,
        testnet=True,
        initial_capital=initial_capital,
    )

    trader = LiveTrader(model, config)

    await trader.start()

    # Run for specified duration
    end_time = datetime.now(UTC) + timedelta(hours=duration_hours)

    try:
        while datetime.now(UTC) < end_time and trader.state.is_running:
            await asyncio.sleep(60)
    finally:
        await trader.stop()

    return trader.state
