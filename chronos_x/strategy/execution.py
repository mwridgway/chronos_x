"""Order execution algorithms: TWAP, VWAP, and adaptive execution.

Implements execution strategies to minimize market impact and slippage.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0


@dataclass
class ExecutionConfig:
    """Configuration for execution algorithms."""

    # TWAP settings
    twap_intervals: int = 10  # Number of intervals
    twap_duration_seconds: int = 60  # Total duration

    # VWAP settings
    vwap_participation_rate: float = 0.1  # Target % of volume
    vwap_max_slice_pct: float = 0.2  # Max slice as % of total

    # Adaptive settings
    urgency: float = 0.5  # 0 = passive, 1 = aggressive
    max_slippage_bps: float = 10.0  # Maximum allowed slippage

    # General settings
    min_order_size: float = 0.001  # Minimum order size
    retry_delay: float = 1.0  # Delay between retries
    max_retries: int = 3


@dataclass
class ExecutionResult:
    """Result of execution algorithm."""

    symbol: str
    side: OrderSide
    requested_quantity: float
    filled_quantity: float
    avg_fill_price: float
    vwap: float
    slippage_bps: float
    num_orders: int
    start_time: datetime
    end_time: datetime
    orders: list[Order] = field(default_factory=list)


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        order_callback: Callable[[Order], Any] | None = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        self.order_callback = order_callback
        self._log = logger.bind(algorithm=self.__class__.__name__)

    @abstractmethod
    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        **kwargs,
    ) -> ExecutionResult:
        """Execute the order using the algorithm.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Total quantity to execute

        Returns:
            ExecutionResult with details
        """
        pass

    async def _place_order(self, order: Order) -> Order:
        """Place an order through the callback.

        Args:
            order: Order to place

        Returns:
            Updated order with fill information
        """
        if self.order_callback is not None:
            result = await self.order_callback(order)
            if result is not None:
                return result

        # Simulate fill if no callback
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = order.price or 0.0

        return order


class TWAPExecution(ExecutionAlgorithm):
    """Time-Weighted Average Price execution.

    Splits order into equal-sized slices executed at regular intervals.
    """

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reference_price: float | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Execute order using TWAP algorithm.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Total quantity
            reference_price: Reference price for slippage calculation

        Returns:
            ExecutionResult
        """
        start_time = datetime.now(UTC)
        orders: list[Order] = []

        # Calculate slice parameters
        n_slices = self.config.twap_intervals
        slice_quantity = quantity / n_slices
        interval_seconds = self.config.twap_duration_seconds / n_slices

        total_filled = 0.0
        total_notional = 0.0

        self._log.info(
            "starting_twap",
            symbol=symbol,
            quantity=quantity,
            slices=n_slices,
            interval=interval_seconds,
        )

        for i in range(n_slices):
            # Adjust last slice for rounding
            if i == n_slices - 1:
                slice_qty = quantity - total_filled
            else:
                slice_qty = slice_quantity

            if slice_qty < self.config.min_order_size:
                continue

            # Create and place order
            order = Order(
                order_id=f"twap_{symbol}_{i}_{int(start_time.timestamp())}",
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=slice_qty,
            )

            order = await self._place_order(order)
            orders.append(order)

            if order.status == OrderStatus.FILLED:
                total_filled += order.filled_quantity
                total_notional += order.filled_quantity * order.avg_fill_price

            # Wait for next interval
            if i < n_slices - 1:
                await asyncio.sleep(interval_seconds)

        end_time = datetime.now(UTC)

        # Calculate metrics
        avg_fill_price = total_notional / total_filled if total_filled > 0 else 0
        vwap = avg_fill_price  # TWAP approximates VWAP
        slippage = (
            (avg_fill_price - reference_price) / reference_price * 10000
            if reference_price
            else 0
        )
        if side == OrderSide.SELL:
            slippage = -slippage

        return ExecutionResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            vwap=vwap,
            slippage_bps=slippage,
            num_orders=len(orders),
            start_time=start_time,
            end_time=end_time,
            orders=orders,
        )


class VWAPExecution(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution.

    Executes orders in proportion to expected volume profile.
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        order_callback: Callable | None = None,
        volume_profile: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(config, order_callback)
        self._volume_profile = volume_profile

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_volume: float | None = None,
        volume_profile: NDArray[np.float64] | None = None,
        reference_price: float | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Execute order using VWAP algorithm.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Total quantity
            current_volume: Current market volume
            volume_profile: Expected volume distribution
            reference_price: Reference price for slippage

        Returns:
            ExecutionResult
        """
        start_time = datetime.now(UTC)
        orders: list[Order] = []

        # Use provided or default volume profile
        profile = volume_profile or self._volume_profile
        if profile is None:
            # Default to uniform profile
            profile = np.ones(self.config.twap_intervals)
        profile = profile / profile.sum()

        total_filled = 0.0
        total_notional = 0.0
        interval_seconds = self.config.twap_duration_seconds / len(profile)

        self._log.info(
            "starting_vwap",
            symbol=symbol,
            quantity=quantity,
            intervals=len(profile),
        )

        for i, vol_weight in enumerate(profile):
            # Calculate slice based on volume weight
            slice_qty = quantity * vol_weight

            # Apply max slice constraint
            max_slice = quantity * self.config.vwap_max_slice_pct
            slice_qty = min(slice_qty, max_slice)

            if slice_qty < self.config.min_order_size:
                continue

            # Adjust for remaining quantity
            remaining = quantity - total_filled
            slice_qty = min(slice_qty, remaining)

            # Create and place order
            order = Order(
                order_id=f"vwap_{symbol}_{i}_{int(start_time.timestamp())}",
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=slice_qty,
            )

            order = await self._place_order(order)
            orders.append(order)

            if order.status == OrderStatus.FILLED:
                total_filled += order.filled_quantity
                total_notional += order.filled_quantity * order.avg_fill_price

            if total_filled >= quantity:
                break

            # Wait for next interval
            if i < len(profile) - 1:
                await asyncio.sleep(interval_seconds)

        end_time = datetime.now(UTC)

        avg_fill_price = total_notional / total_filled if total_filled > 0 else 0
        slippage = (
            (avg_fill_price - reference_price) / reference_price * 10000
            if reference_price
            else 0
        )
        if side == OrderSide.SELL:
            slippage = -slippage

        return ExecutionResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            vwap=avg_fill_price,
            slippage_bps=slippage,
            num_orders=len(orders),
            start_time=start_time,
            end_time=end_time,
            orders=orders,
        )


class AdaptiveExecution(ExecutionAlgorithm):
    """Adaptive execution that adjusts based on market conditions.

    Combines TWAP/VWAP with real-time adjustments based on:
    - Price movement
    - Volume changes
    - Urgency parameter
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        order_callback: Callable | None = None,
        price_callback: Callable | None = None,
    ) -> None:
        super().__init__(config, order_callback)
        self.price_callback = price_callback

    async def _get_current_price(self, symbol: str) -> float | None:
        """Get current market price."""
        if self.price_callback:
            return await self.price_callback(symbol)
        return None

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reference_price: float | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Execute order using adaptive algorithm.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Total quantity
            reference_price: Reference price

        Returns:
            ExecutionResult
        """
        start_time = datetime.now(UTC)
        orders: list[Order] = []

        # Initial parameters
        n_intervals = self.config.twap_intervals
        base_slice = quantity / n_intervals
        interval = self.config.twap_duration_seconds / n_intervals

        total_filled = 0.0
        total_notional = 0.0
        urgency = self.config.urgency

        if reference_price is None:
            reference_price = await self._get_current_price(symbol) or 0

        self._log.info(
            "starting_adaptive",
            symbol=symbol,
            quantity=quantity,
            urgency=urgency,
        )

        for i in range(n_intervals):
            # Get current price
            current_price = await self._get_current_price(symbol) or reference_price

            # Calculate price momentum
            if reference_price > 0:
                price_change = (current_price - reference_price) / reference_price
            else:
                price_change = 0

            # Adjust slice size based on conditions
            adjustment = 1.0

            if side == OrderSide.BUY:
                # More aggressive if price rising
                if price_change > 0.001:  # Price up 0.1%
                    adjustment = 1.0 + urgency * 0.5
                elif price_change < -0.001:  # Price down 0.1%
                    adjustment = 1.0 - (1 - urgency) * 0.3
            else:
                # More aggressive if price falling
                if price_change < -0.001:
                    adjustment = 1.0 + urgency * 0.5
                elif price_change > 0.001:
                    adjustment = 1.0 - (1 - urgency) * 0.3

            # Calculate adjusted slice
            remaining = quantity - total_filled
            slice_qty = min(base_slice * adjustment, remaining)

            if slice_qty < self.config.min_order_size:
                continue

            # Choose order type based on urgency
            if urgency > 0.7:
                order_type = OrderType.MARKET
                price = None
            else:
                order_type = OrderType.LIMIT
                # Set limit at favorable side of spread
                if side == OrderSide.BUY:
                    price = current_price * 0.9999
                else:
                    price = current_price * 1.0001

            # Create and place order
            order = Order(
                order_id=f"adaptive_{symbol}_{i}_{int(start_time.timestamp())}",
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=slice_qty,
                price=price,
            )

            order = await self._place_order(order)
            orders.append(order)

            if order.status == OrderStatus.FILLED:
                total_filled += order.filled_quantity
                total_notional += order.filled_quantity * order.avg_fill_price

                # Increase urgency as we fill more
                urgency = min(1.0, urgency + 0.05)

            if total_filled >= quantity:
                break

            # Dynamic interval based on fill rate
            expected_fill = (i + 1) / n_intervals * quantity
            fill_ratio = total_filled / expected_fill if expected_fill > 0 else 1

            if fill_ratio < 0.8:
                # Behind schedule, reduce wait time
                wait_time = interval * 0.7
            elif fill_ratio > 1.2:
                # Ahead of schedule, increase wait time
                wait_time = interval * 1.3
            else:
                wait_time = interval

            if i < n_intervals - 1:
                await asyncio.sleep(wait_time)

        end_time = datetime.now(UTC)

        avg_fill_price = total_notional / total_filled if total_filled > 0 else 0
        slippage = (
            (avg_fill_price - reference_price) / reference_price * 10000
            if reference_price
            else 0
        )
        if side == OrderSide.SELL:
            slippage = -slippage

        return ExecutionResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            vwap=avg_fill_price,
            slippage_bps=slippage,
            num_orders=len(orders),
            start_time=start_time,
            end_time=end_time,
            orders=orders,
        )


class SimpleExecution(ExecutionAlgorithm):
    """Simple immediate execution for small orders."""

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reference_price: float | None = None,
        limit_price: float | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Execute order immediately.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Quantity to execute
            reference_price: Reference price for slippage
            limit_price: Optional limit price

        Returns:
            ExecutionResult
        """
        start_time = datetime.now(UTC)

        order_type = OrderType.LIMIT if limit_price else OrderType.MARKET

        order = Order(
            order_id=f"simple_{symbol}_{int(start_time.timestamp())}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
        )

        order = await self._place_order(order)

        end_time = datetime.now(UTC)

        slippage = 0.0
        if reference_price and order.avg_fill_price:
            slippage = (
                (order.avg_fill_price - reference_price) / reference_price * 10000
            )
            if side == OrderSide.SELL:
                slippage = -slippage

        return ExecutionResult(
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=order.filled_quantity,
            avg_fill_price=order.avg_fill_price,
            vwap=order.avg_fill_price,
            slippage_bps=slippage,
            num_orders=1,
            start_time=start_time,
            end_time=end_time,
            orders=[order],
        )


def select_execution_algorithm(
    quantity: float,
    urgency: float,
    market_volume: float | None = None,
    config: ExecutionConfig | None = None,
) -> ExecutionAlgorithm:
    """Select appropriate execution algorithm based on order characteristics.

    Args:
        quantity: Order quantity
        urgency: Urgency level (0-1)
        market_volume: Estimated market volume
        config: Execution configuration

    Returns:
        Selected execution algorithm
    """
    config = config or ExecutionConfig()

    # For small orders or high urgency, use simple execution
    if urgency > 0.9:
        return SimpleExecution(config)

    # For medium urgency, use adaptive
    if urgency > 0.5:
        return AdaptiveExecution(config)

    # For low urgency with volume info, use VWAP
    if market_volume is not None:
        return VWAPExecution(config)

    # Default to TWAP
    return TWAPExecution(config)
