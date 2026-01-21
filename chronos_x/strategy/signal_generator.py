"""Signal generator combining model predictions with rules-based filters.

Combines ML model outputs with technical indicators and risk filters
to generate actionable trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


class SignalType(Enum):
    """Signal type enumeration."""

    LONG = 1
    NEUTRAL = 0
    SHORT = -1


@dataclass
class Signal:
    """Trading signal with metadata."""

    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # 0 to 1
    confidence: float  # 0 to 1

    # Model outputs
    model_prediction: int
    model_probability: float
    meta_label_probability: float

    # Technical indicators
    trend_score: float
    momentum_score: float
    volatility_regime: str

    # Risk metrics
    position_size_suggestion: float
    stop_loss_price: float | None = None
    take_profit_price: float | None = None

    # Metadata
    features_used: list[str] = field(default_factory=list)
    model_version: str = ""


@dataclass
class SignalGeneratorConfig:
    """Configuration for signal generation."""

    # Probability thresholds
    min_model_probability: float = 0.55
    min_meta_probability: float = 0.5
    min_combined_confidence: float = 0.3

    # Signal strength weighting
    model_weight: float = 0.6
    meta_weight: float = 0.3
    technical_weight: float = 0.1

    # Technical filters
    use_trend_filter: bool = True
    use_momentum_filter: bool = True
    use_volatility_filter: bool = True

    # Trend settings
    trend_ma_short: int = 20
    trend_ma_long: int = 50

    # Momentum settings
    momentum_period: int = 14

    # Volatility settings
    volatility_lookback: int = 20
    high_volatility_threshold: float = 0.25
    low_volatility_threshold: float = 0.10

    # Position sizing
    base_position_size: float = 0.1
    confidence_scaling: bool = True


class TechnicalIndicators:
    """Calculate technical indicators for filtering."""

    @staticmethod
    def compute_sma(
        prices: NDArray[np.float64],
        period: int,
    ) -> NDArray[np.float64]:
        """Compute Simple Moving Average."""
        sma = np.full_like(prices, np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1 : i + 1])
        return sma

    @staticmethod
    def compute_ema(
        prices: NDArray[np.float64],
        period: int,
    ) -> NDArray[np.float64]:
        """Compute Exponential Moving Average."""
        ema = np.full_like(prices, np.nan)
        alpha = 2 / (period + 1)

        ema[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    @staticmethod
    def compute_rsi(
        prices: NDArray[np.float64],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """Compute Relative Strength Index."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        rsi = np.full(len(prices), np.nan)

        # Initial average
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            rsi[period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))

        # Rolling calculation
        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def compute_macd(
        prices: NDArray[np.float64],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute MACD indicator."""
        ema_fast = TechnicalIndicators.compute_ema(prices, fast)
        ema_slow = TechnicalIndicators.compute_ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.compute_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def compute_volatility(
        prices: NDArray[np.float64],
        period: int = 20,
    ) -> NDArray[np.float64]:
        """Compute annualized volatility."""
        returns = np.diff(np.log(prices))
        vol = np.full(len(prices), np.nan)

        for i in range(period, len(prices)):
            vol[i] = np.std(returns[i - period : i]) * np.sqrt(252)

        return vol


class SignalGenerator:
    """Generate trading signals from model predictions and filters."""

    def __init__(
        self,
        config: SignalGeneratorConfig | None = None,
    ) -> None:
        self.config = config or SignalGeneratorConfig()
        self._log = logger.bind(component="signal_generator")

    def generate_signal(
        self,
        symbol: str,
        model_prediction: int,
        model_probabilities: NDArray[np.float64],
        meta_probability: float,
        prices: NDArray[np.float64],
        timestamp: datetime | None = None,
        features: list[str] | None = None,
        model_version: str = "",
    ) -> Signal | None:
        """Generate a trading signal from model output and market data.

        Args:
            symbol: Trading symbol
            model_prediction: Model's predicted class (-1, 0, 1)
            model_probabilities: Class probabilities [down, neutral, up]
            meta_probability: Meta-labeler probability
            prices: Recent price data
            timestamp: Signal timestamp
            features: List of features used
            model_version: Model version string

        Returns:
            Signal object or None if filtered out
        """
        timestamp = timestamp or datetime.now(UTC)

        # Get model probability for predicted class
        if model_prediction == -1:
            model_prob = model_probabilities[0]
        elif model_prediction == 1:
            model_prob = model_probabilities[2]
        else:
            model_prob = model_probabilities[1]

        # Apply probability thresholds
        if model_prob < self.config.min_model_probability:
            self._log.debug("signal_filtered", reason="low_model_probability")
            return None

        if meta_probability < self.config.min_meta_probability:
            self._log.debug("signal_filtered", reason="low_meta_probability")
            return None

        # Compute technical indicators
        trend_score = self._compute_trend_score(prices)
        momentum_score = self._compute_momentum_score(prices)
        volatility, volatility_regime = self._compute_volatility_regime(prices)

        # Apply technical filters
        if not self._apply_technical_filters(
            model_prediction, trend_score, momentum_score, volatility_regime
        ):
            self._log.debug("signal_filtered", reason="technical_filter")
            return None

        # Compute signal strength and confidence
        strength = self._compute_signal_strength(
            model_prob, meta_probability, trend_score, momentum_score
        )

        confidence = self._compute_confidence(
            model_prob, meta_probability, model_probabilities
        )

        if confidence < self.config.min_combined_confidence:
            self._log.debug("signal_filtered", reason="low_confidence")
            return None

        # Compute position size
        position_size = self._compute_position_size(strength, confidence, volatility)

        # Compute stop loss and take profit
        current_price = prices[-1]
        stop_loss, take_profit = self._compute_exit_levels(
            current_price, model_prediction, volatility
        )

        # Create signal
        signal_type = SignalType(model_prediction)

        return Signal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            model_prediction=model_prediction,
            model_probability=model_prob,
            meta_label_probability=meta_probability,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_regime=volatility_regime,
            position_size_suggestion=position_size,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            features_used=features or [],
            model_version=model_version,
        )

    def _compute_trend_score(self, prices: NDArray[np.float64]) -> float:
        """Compute trend score (-1 to 1)."""
        if len(prices) < self.config.trend_ma_long:
            return 0.0

        ma_short = TechnicalIndicators.compute_sma(prices, self.config.trend_ma_short)
        ma_long = TechnicalIndicators.compute_sma(prices, self.config.trend_ma_long)

        # Current MA relationship
        current_short = ma_short[-1]
        current_long = ma_long[-1]

        if np.isnan(current_short) or np.isnan(current_long):
            return 0.0

        # Trend score based on MA distance
        diff_pct = (current_short - current_long) / current_long
        trend_score = np.clip(diff_pct * 10, -1, 1)

        return float(trend_score)

    def _compute_momentum_score(self, prices: NDArray[np.float64]) -> float:
        """Compute momentum score (-1 to 1)."""
        if len(prices) < self.config.momentum_period + 1:
            return 0.0

        rsi = TechnicalIndicators.compute_rsi(prices, self.config.momentum_period)
        current_rsi = rsi[-1]

        if np.isnan(current_rsi):
            return 0.0

        # Normalize RSI to -1 to 1 scale
        momentum_score = (current_rsi - 50) / 50

        return float(momentum_score)

    def _compute_volatility_regime(
        self, prices: NDArray[np.float64]
    ) -> tuple[float, str]:
        """Compute current volatility and regime."""
        if len(prices) < self.config.volatility_lookback + 1:
            return 0.15, "normal"

        vol = TechnicalIndicators.compute_volatility(
            prices, self.config.volatility_lookback
        )
        current_vol = vol[-1]

        if np.isnan(current_vol):
            return 0.15, "normal"

        if current_vol > self.config.high_volatility_threshold:
            regime = "high"
        elif current_vol < self.config.low_volatility_threshold:
            regime = "low"
        else:
            regime = "normal"

        return float(current_vol), regime

    def _apply_technical_filters(
        self,
        prediction: int,
        trend_score: float,
        momentum_score: float,
        volatility_regime: str,
    ) -> bool:
        """Apply technical filters to signal."""
        # Trend filter: signal should align with trend
        if self.config.use_trend_filter:
            if prediction == 1 and trend_score < -0.3:
                return False  # Don't go long in strong downtrend
            if prediction == -1 and trend_score > 0.3:
                return False  # Don't go short in strong uptrend

        # Momentum filter: avoid extreme overbought/oversold
        if self.config.use_momentum_filter:
            if prediction == 1 and momentum_score > 0.8:
                return False  # Don't go long when overbought
            if prediction == -1 and momentum_score < -0.8:
                return False  # Don't go short when oversold

        # Volatility filter: reduce activity in extreme volatility
        if self.config.use_volatility_filter:
            if volatility_regime == "high":
                # More conservative in high volatility
                pass  # Could add additional filtering

        return True

    def _compute_signal_strength(
        self,
        model_prob: float,
        meta_prob: float,
        trend_score: float,
        momentum_score: float,
    ) -> float:
        """Compute overall signal strength."""
        # Normalize probabilities to 0-1 scale for the positive class
        model_component = (model_prob - 0.5) * 2  # 0.5 -> 0, 1.0 -> 1
        meta_component = meta_prob

        # Technical component
        technical_component = (abs(trend_score) + abs(momentum_score)) / 2

        # Weighted combination
        strength = (
            self.config.model_weight * model_component
            + self.config.meta_weight * meta_component
            + self.config.technical_weight * technical_component
        )

        return float(np.clip(strength, 0, 1))

    def _compute_confidence(
        self,
        model_prob: float,
        meta_prob: float,
        all_probs: NDArray[np.float64],
    ) -> float:
        """Compute signal confidence."""
        # Higher confidence when model is decisive
        prob_spread = np.max(all_probs) - np.min(all_probs)

        # Combined confidence
        confidence = (model_prob + meta_prob) / 2 * prob_spread

        return float(np.clip(confidence, 0, 1))

    def _compute_position_size(
        self,
        strength: float,
        confidence: float,
        volatility: float,
    ) -> float:
        """Compute suggested position size."""
        base_size = self.config.base_position_size

        if self.config.confidence_scaling:
            # Scale by confidence
            size = base_size * confidence

            # Adjust for volatility
            vol_adjustment = 0.15 / max(volatility, 0.05)  # Target 15% vol
            size *= min(vol_adjustment, 2.0)

            # Scale by strength
            size *= strength
        else:
            size = base_size

        return float(np.clip(size, 0.01, 0.5))

    def _compute_exit_levels(
        self,
        current_price: float,
        prediction: int,
        volatility: float,
    ) -> tuple[float, float]:
        """Compute stop loss and take profit levels."""
        # Use ATR-like approach based on volatility
        atr_multiple_sl = 2.0
        atr_multiple_tp = 3.0

        daily_range = current_price * volatility / np.sqrt(252)

        if prediction == 1:  # Long
            stop_loss = current_price - atr_multiple_sl * daily_range
            take_profit = current_price + atr_multiple_tp * daily_range
        elif prediction == -1:  # Short
            stop_loss = current_price + atr_multiple_sl * daily_range
            take_profit = current_price - atr_multiple_tp * daily_range
        else:
            stop_loss = current_price
            take_profit = current_price

        return stop_loss, take_profit

    def batch_generate_signals(
        self,
        symbol: str,
        model_predictions: NDArray[np.int8],
        model_probabilities: NDArray[np.float64],
        meta_probabilities: NDArray[np.float64],
        prices: NDArray[np.float64],
        timestamps: list[datetime],
    ) -> list[Signal]:
        """Generate signals for a batch of predictions.

        Args:
            symbol: Trading symbol
            model_predictions: Array of predictions
            model_probabilities: Array of probabilities (n, 3)
            meta_probabilities: Array of meta-label probabilities
            prices: Price array
            timestamps: List of timestamps

        Returns:
            List of generated signals
        """
        signals = []
        lookback = max(
            self.config.trend_ma_long,
            self.config.volatility_lookback,
            self.config.momentum_period,
        )

        for i in range(lookback, len(model_predictions)):
            signal = self.generate_signal(
                symbol=symbol,
                model_prediction=int(model_predictions[i]),
                model_probabilities=model_probabilities[i],
                meta_probability=float(meta_probabilities[i]),
                prices=prices[: i + 1],
                timestamp=timestamps[i],
            )

            if signal is not None:
                signals.append(signal)

        return signals
