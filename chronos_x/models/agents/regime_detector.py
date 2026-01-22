"""
Market Regime Detection using LLMs

Detects current market regimes using:
- LLM-based analysis of market conditions
- On-chain metrics
- Sentiment signals
- Technical indicators

Regimes: High Volatility, Trending Up, Sideways, Bearish Crash, Bull Market, etc.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Defined market regimes"""
    HIGH_VOLATILITY = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    BEARISH_CRASH = "bearish_crash"
    BULL_MARKET = "bull_market"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    UNKNOWN = "unknown"


@dataclass
class RegimeDetectionResult:
    """Container for regime detection results"""
    regime: MarketRegime
    confidence: float  # 0 to 1
    regime_embedding: np.ndarray  # High-dimensional embedding
    indicators: Dict[str, float]
    reasoning: str
    timestamp: datetime
    metadata: Dict


class RegimeIndicators:
    """Compute technical indicators for regime detection"""

    @staticmethod
    def compute_volatility_regime(returns: np.ndarray, window: int = 20) -> float:
        """
        Compute volatility regime indicator

        Args:
            returns: Array of returns
            window: Rolling window size

        Returns:
            Volatility percentile (0-1)
        """
        if len(returns) < window:
            return 0.5

        # Rolling volatility
        volatility = pd.Series(returns).rolling(window).std()

        # Current volatility percentile
        current_vol = volatility.iloc[-1]
        percentile = (volatility < current_vol).sum() / len(volatility)

        return percentile

    @staticmethod
    def compute_trend_strength(prices: np.ndarray, window: int = 50) -> Tuple[float, str]:
        """
        Compute trend strength and direction

        Args:
            prices: Array of prices
            window: Window for trend analysis

        Returns:
            Tuple of (strength, direction)
            - strength: 0 (no trend) to 1 (strong trend)
            - direction: 'up', 'down', or 'sideways'
        """
        if len(prices) < window:
            return 0.0, 'sideways'

        # Linear regression slope
        x = np.arange(window)
        y = prices[-window:]
        slope, _ = np.polyfit(x, y, 1)

        # Normalize slope by price level
        normalized_slope = slope / np.mean(y)

        # R-squared for trend strength
        y_pred = slope * x + np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        strength = abs(r_squared)

        # Direction based on slope
        if normalized_slope > 0.001:
            direction = 'up'
        elif normalized_slope < -0.001:
            direction = 'down'
        else:
            direction = 'sideways'

        return strength, direction

    @staticmethod
    def compute_drawdown(prices: np.ndarray) -> float:
        """
        Compute current drawdown from peak

        Args:
            prices: Array of prices

        Returns:
            Drawdown percentage (negative value)
        """
        if len(prices) == 0:
            return 0.0

        cummax = np.maximum.accumulate(prices)
        drawdown = (prices[-1] - cummax[-1]) / cummax[-1]

        return drawdown

    @staticmethod
    def detect_regime_from_indicators(
        volatility_percentile: float,
        trend_strength: float,
        trend_direction: str,
        drawdown: float,
        sentiment: Optional[float] = None
    ) -> Tuple[MarketRegime, float]:
        """
        Detect regime from indicators

        Args:
            volatility_percentile: Volatility percentile (0-1)
            trend_strength: Trend strength (0-1)
            trend_direction: Trend direction
            drawdown: Current drawdown (negative)
            sentiment: Optional sentiment score (-1 to 1)

        Returns:
            Tuple of (regime, confidence)
        """
        confidence = 0.5

        # Bearish crash: High vol + strong downtrend + large drawdown
        if volatility_percentile > 0.8 and trend_direction == 'down' and drawdown < -0.15:
            confidence = min(0.9, 0.6 + volatility_percentile * 0.3)
            return MarketRegime.BEARISH_CRASH, confidence

        # Bull market: Low vol + strong uptrend + positive sentiment
        if (volatility_percentile < 0.4 and trend_direction == 'up' and
                trend_strength > 0.6):
            confidence = 0.7 + trend_strength * 0.2
            if sentiment and sentiment > 0.3:
                confidence = min(0.95, confidence + 0.1)
            return MarketRegime.BULL_MARKET, confidence

        # High volatility: High vol + no clear trend
        if volatility_percentile > 0.7 and trend_strength < 0.4:
            confidence = 0.6 + volatility_percentile * 0.2
            return MarketRegime.HIGH_VOLATILITY, confidence

        # Trending up: Moderate vol + uptrend
        if trend_direction == 'up' and trend_strength > 0.4:
            confidence = 0.6 + trend_strength * 0.3
            return MarketRegime.TRENDING_UP, confidence

        # Trending down: Moderate vol + downtrend
        if trend_direction == 'down' and trend_strength > 0.4:
            confidence = 0.6 + trend_strength * 0.3
            return MarketRegime.TRENDING_DOWN, confidence

        # Sideways/Accumulation: Low vol + weak trend
        if volatility_percentile < 0.5 and trend_strength < 0.3:
            confidence = 0.5
            if drawdown < -0.05:  # Some drawdown suggests accumulation
                return MarketRegime.ACCUMULATION, confidence
            return MarketRegime.SIDEWAYS, confidence

        # Default to unknown
        return MarketRegime.UNKNOWN, 0.3


class LLMRegimeDetector:
    """
    LLM-based regime detection

    Uses language models to analyze market conditions and detect regimes
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        use_local: bool = False,
        embedding_dim: int = 128
    ):
        self.model_name = model_name
        self.use_local = use_local
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)

        self._client = None
        self._embedding_model = None

    def _get_openai_client(self):
        """Initialize OpenAI client"""
        if self._client is None:
            try:
                import openai
                import os

                self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}")
                raise

        return self._client

    def detect_regime_llm(
        self,
        market_summary: str,
        indicators: Dict[str, float],
        news_headlines: Optional[List[str]] = None
    ) -> Tuple[MarketRegime, float, str]:
        """
        Detect market regime using LLM

        Args:
            market_summary: Text summary of market conditions
            indicators: Dictionary of technical indicators
            news_headlines: Optional list of recent news headlines

        Returns:
            Tuple of (regime, confidence, reasoning)
        """
        # Construct prompt
        prompt = self._construct_regime_prompt(
            market_summary, indicators, news_headlines
        )

        try:
            client = self._get_openai_client()

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency market analyst specializing in regime detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            result_text = response.choices[0].message.content

            # Parse LLM response
            regime, confidence, reasoning = self._parse_llm_response(result_text)

            return regime, confidence, reasoning

        except Exception as e:
            self.logger.error(f"LLM regime detection error: {e}")
            return MarketRegime.UNKNOWN, 0.0, f"Error: {str(e)}"

    def _construct_regime_prompt(
        self,
        market_summary: str,
        indicators: Dict[str, float],
        news_headlines: Optional[List[str]] = None
    ) -> str:
        """Construct prompt for LLM regime detection"""
        prompt = f"""Analyze the following cryptocurrency market conditions and determine the current market regime.

Market Summary:
{market_summary}

Technical Indicators:
"""
        for key, value in indicators.items():
            prompt += f"- {key}: {value:.4f}\n"

        if news_headlines:
            prompt += "\nRecent News Headlines:\n"
            for headline in news_headlines[:5]:
                prompt += f"- {headline}\n"

        prompt += """
Based on this information, classify the market regime as ONE of the following:
- HIGH_VOLATILITY: High uncertainty, large price swings
- TRENDING_UP: Clear upward trend with moderate volatility
- TRENDING_DOWN: Clear downward trend
- SIDEWAYS: Range-bound, no clear direction
- BEARISH_CRASH: Severe downtrend with panic selling
- BULL_MARKET: Strong uptrend with optimism
- ACCUMULATION: Consolidation after decline, building positions
- DISTRIBUTION: Consolidation before decline, smart money exiting

Respond in the following format:
REGIME: <regime_name>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""
        return prompt

    def _parse_llm_response(self, response: str) -> Tuple[MarketRegime, float, str]:
        """Parse LLM response to extract regime, confidence, and reasoning"""
        try:
            lines = response.strip().split('\n')

            regime_line = [l for l in lines if l.startswith('REGIME:')]
            confidence_line = [l for l in lines if l.startswith('CONFIDENCE:')]
            reasoning_line = [l for l in lines if l.startswith('REASONING:')]

            # Extract regime
            if regime_line:
                regime_str = regime_line[0].split(':', 1)[1].strip().upper()
                regime = MarketRegime[regime_str]
            else:
                regime = MarketRegime.UNKNOWN

            # Extract confidence
            if confidence_line:
                confidence_str = confidence_line[0].split(':', 1)[1].strip()
                confidence = float(confidence_str)
            else:
                confidence = 0.5

            # Extract reasoning
            if reasoning_line:
                reasoning = reasoning_line[0].split(':', 1)[1].strip()
            else:
                reasoning = "No reasoning provided"

            return regime, confidence, reasoning

        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return MarketRegime.UNKNOWN, 0.0, str(e)

    def get_regime_embedding(
        self,
        regime: MarketRegime,
        indicators: Dict[str, float],
        confidence: float
    ) -> np.ndarray:
        """
        Generate regime embedding vector

        Args:
            regime: Detected regime
            indicators: Dictionary of indicators
            confidence: Detection confidence

        Returns:
            Embedding vector
        """
        # One-hot encode regime
        regime_values = list(MarketRegime)
        regime_onehot = np.zeros(len(regime_values))
        if regime in regime_values:
            regime_onehot[regime_values.index(regime)] = 1.0

        # Indicator values
        indicator_values = np.array(list(indicators.values())[:10])  # Use first 10
        if len(indicator_values) < 10:
            indicator_values = np.pad(indicator_values, (0, 10 - len(indicator_values)))

        # Combine into embedding
        embedding = np.concatenate([
            regime_onehot,
            indicator_values,
            [confidence]
        ])

        # Pad or truncate to desired dimension
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]

        return embedding


class RegimeDetector:
    """
    Main regime detector combining indicators and LLM analysis
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_model: str = "gpt-3.5-turbo",
        embedding_dim: int = 128,
        update_frequency_hours: int = 1
    ):
        self.use_llm = use_llm
        self.update_frequency_hours = update_frequency_hours
        self.logger = logging.getLogger(__name__)

        self.indicator_computer = RegimeIndicators()

        if use_llm:
            self.llm_detector = LLMRegimeDetector(
                model_name=llm_model,
                embedding_dim=embedding_dim
            )
        else:
            self.llm_detector = None

        # Cache for regime
        self._cached_regime: Optional[RegimeDetectionResult] = None
        self._cache_timestamp: Optional[datetime] = None

    def detect(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        sentiment: Optional[float] = None,
        news_headlines: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> RegimeDetectionResult:
        """
        Detect current market regime

        Args:
            prices: Array of historical prices
            returns: Array of returns
            sentiment: Optional sentiment score
            news_headlines: Optional news headlines
            use_cache: Whether to use cached regime

        Returns:
            RegimeDetectionResult object
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            return self._cached_regime

        # Compute indicators
        indicators = self._compute_all_indicators(prices, returns, sentiment)

        # Indicator-based detection
        vol_percentile = indicators['volatility_percentile']
        trend_strength = indicators['trend_strength']
        trend_direction = indicators['trend_direction']
        drawdown = indicators['drawdown']

        regime_indicator, confidence_indicator = (
            RegimeIndicators.detect_regime_from_indicators(
                vol_percentile, trend_strength, trend_direction, drawdown, sentiment
            )
        )

        # LLM-based detection (if enabled)
        if self.use_llm and self.llm_detector:
            market_summary = self._create_market_summary(prices, returns, indicators)

            regime_llm, confidence_llm, reasoning = self.llm_detector.detect_regime_llm(
                market_summary, indicators, news_headlines
            )

            # Combine indicator and LLM results
            if confidence_llm > confidence_indicator:
                regime = regime_llm
                confidence = confidence_llm * 0.7 + confidence_indicator * 0.3
            else:
                regime = regime_indicator
                confidence = confidence_indicator * 0.7 + confidence_llm * 0.3
                reasoning = f"Indicator-based: {regime.value}"
        else:
            regime = regime_indicator
            confidence = confidence_indicator
            reasoning = f"Indicator-based detection: {regime.value}"

        # Generate embedding
        if self.llm_detector:
            embedding = self.llm_detector.get_regime_embedding(regime, indicators, confidence)
        else:
            embedding = self._simple_embedding(regime, indicators, confidence)

        result = RegimeDetectionResult(
            regime=regime,
            confidence=confidence,
            regime_embedding=embedding,
            indicators=indicators,
            reasoning=reasoning,
            timestamp=datetime.utcnow(),
            metadata={
                'llm_used': self.use_llm,
                'prices_count': len(prices)
            }
        )

        # Update cache
        self._cached_regime = result
        self._cache_timestamp = datetime.utcnow()

        return result

    def _compute_all_indicators(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        sentiment: Optional[float]
    ) -> Dict[str, float]:
        """Compute all regime indicators"""
        vol_percentile = self.indicator_computer.compute_volatility_regime(returns)
        trend_strength, trend_direction = self.indicator_computer.compute_trend_strength(prices)
        drawdown = self.indicator_computer.compute_drawdown(prices)

        indicators = {
            'volatility_percentile': vol_percentile,
            'trend_strength': trend_strength,
            'trend_direction': 1.0 if trend_direction == 'up' else (-1.0 if trend_direction == 'down' else 0.0),
            'drawdown': drawdown,
            'current_volatility': np.std(returns[-20:]) if len(returns) >= 20 else 0.0,
            'price_change_24h': (prices[-1] - prices[-24]) / prices[-24] if len(prices) >= 24 else 0.0,
            'price_change_7d': (prices[-1] - prices[-168]) / prices[-168] if len(prices) >= 168 else 0.0,
        }

        if sentiment is not None:
            indicators['sentiment'] = sentiment

        return indicators

    def _create_market_summary(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        indicators: Dict[str, float]
    ) -> str:
        """Create market summary text for LLM"""
        current_price = prices[-1]
        price_change_24h = indicators.get('price_change_24h', 0) * 100
        volatility = indicators.get('current_volatility', 0) * 100
        drawdown = indicators.get('drawdown', 0) * 100

        summary = f"""Current Price: ${current_price:.2f}
24h Change: {price_change_24h:+.2f}%
Current Volatility: {volatility:.2f}%
Drawdown from Peak: {drawdown:.2f}%
Trend: {indicators.get('trend_direction', 'unknown')} (strength: {indicators.get('trend_strength', 0):.2f})
"""

        return summary

    def _simple_embedding(
        self,
        regime: MarketRegime,
        indicators: Dict[str, float],
        confidence: float,
        dim: int = 128
    ) -> np.ndarray:
        """Create simple embedding without LLM"""
        values = list(indicators.values())[:dim-1]
        values.append(confidence)

        embedding = np.array(values)
        if len(embedding) < dim:
            embedding = np.pad(embedding, (0, dim - len(embedding)))
        else:
            embedding = embedding[:dim]

        return embedding

    def _is_cache_valid(self) -> bool:
        """Check if cached regime is still valid"""
        if self._cached_regime is None or self._cache_timestamp is None:
            return False

        age = datetime.utcnow() - self._cache_timestamp
        return age < timedelta(hours=self.update_frequency_hours)


def create_regime_detector(
    use_llm: bool = False,
    **kwargs
) -> RegimeDetector:
    """
    Factory function to create regime detector

    Args:
        use_llm: Whether to use LLM for regime detection
        **kwargs: Additional configuration options

    Returns:
        Configured RegimeDetector instance
    """
    return RegimeDetector(use_llm=use_llm, **kwargs)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000)) + 100
    returns = np.diff(prices) / prices[:-1]

    # Create detector
    detector = create_regime_detector(use_llm=False)  # Set to True to use LLM

    # Detect regime
    result = detector.detect(prices, returns, sentiment=0.3)

    print(f"Detected Regime: {result.regime.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Embedding shape: {result.regime_embedding.shape}")
    print(f"\nIndicators:")
    for key, value in result.indicators.items():
        print(f"  {key}: {value:.4f}")
