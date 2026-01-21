"""Performance metrics for trading strategy evaluation.

Includes:
- Deflated Sharpe Ratio (DSR)
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown and recovery
- Win rate, profit factor
- Classification metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PerformanceMetrics:
    """Container for strategy performance metrics."""

    # Returns
    total_return: float
    annualized_return: float
    volatility: float
    annualized_volatility: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    deflated_sharpe_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float

    # Trade statistics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    num_trades: int

    # Time metrics
    avg_holding_period: float
    trades_per_day: float


def compute_returns(
    prices: NDArray[np.float64],
    log_returns: bool = True,
) -> NDArray[np.float64]:
    """Compute returns from price series.

    Args:
        prices: Array of prices
        log_returns: If True, compute log returns; otherwise simple returns

    Returns:
        Array of returns
    """
    if log_returns:
        return np.log(prices[1:] / prices[:-1])
    return prices[1:] / prices[:-1] - 1


def compute_sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return == 0:
        return 0.0

    return mean_return / std_return * np.sqrt(periods_per_year)


def compute_sortino_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sortino ratio.

    Uses downside deviation instead of total volatility.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        target_return: Minimum acceptable return
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)

    # Downside deviation
    downside = np.minimum(returns - target_return / periods_per_year, 0)
    downside_std = np.sqrt(np.mean(downside ** 2))

    if downside_std == 0:
        return 0.0

    return mean_return / downside_std * np.sqrt(periods_per_year)


def compute_max_drawdown(
    returns: NDArray[np.float64],
) -> tuple[float, int, int, int]:
    """Compute maximum drawdown and related metrics.

    Args:
        returns: Array of returns

    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx, recovery_idx)
    """
    if len(returns) == 0:
        return 0.0, 0, 0, 0

    # Cumulative returns
    cum_returns = np.cumprod(1 + returns)

    # Running maximum
    running_max = np.maximum.accumulate(cum_returns)

    # Drawdown series
    drawdowns = cum_returns / running_max - 1

    # Maximum drawdown
    max_dd = np.min(drawdowns)
    trough_idx = np.argmin(drawdowns)

    # Find peak (before trough)
    peak_idx = np.argmax(cum_returns[:trough_idx + 1])

    # Find recovery (after trough)
    recovery_idx = len(returns) - 1
    for i in range(trough_idx, len(cum_returns)):
        if cum_returns[i] >= cum_returns[peak_idx]:
            recovery_idx = i
            break

    return abs(max_dd), peak_idx, trough_idx, recovery_idx


def compute_calmar_ratio(
    returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Compute Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    annualized_return = np.mean(returns) * periods_per_year
    max_dd, _, _, _ = compute_max_drawdown(returns)

    if max_dd == 0:
        return 0.0

    return annualized_return / max_dd


def compute_deflated_sharpe_ratio(
    sharpe_ratio: float,
    num_trials: int,
    variance_sharpe: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    num_returns: int = 252,
) -> float:
    """Compute Deflated Sharpe Ratio for multiple testing correction.

    DSR adjusts for the probability of a false positive when
    many strategies are tested (selection bias).

    Args:
        sharpe_ratio: Observed Sharpe ratio
        num_trials: Number of independent trials/backtests
        variance_sharpe: Variance of Sharpe ratios under null
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns
        num_returns: Number of return observations

    Returns:
        Deflated Sharpe Ratio (probability that Sharpe > 0)
    """
    if num_trials <= 1:
        return sharpe_ratio

    # Expected maximum Sharpe ratio under null hypothesis
    # E[max(SR)] ≈ sqrt(2 * log(N)) for N trials
    expected_max_sharpe = np.sqrt(2 * np.log(num_trials)) * np.sqrt(variance_sharpe)

    # Adjust for non-normality of returns
    # Variance of Sharpe ratio estimator
    sr_std = np.sqrt(
        (1 + 0.25 * (skewness ** 2) + (kurtosis - 3) / 4 * sharpe_ratio ** 2)
        / (num_returns - 1)
    )

    if sr_std == 0:
        return 0.0

    # Probability that observed Sharpe > expected max under null
    # Using normal CDF approximation
    z_score = (sharpe_ratio - expected_max_sharpe) / sr_std
    dsr = stats.norm.cdf(z_score)

    return dsr


def compute_probabilistic_sharpe_ratio(
    sharpe_ratio: float,
    benchmark_sharpe: float = 0.0,
    num_returns: int = 252,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Probabilistic Sharpe Ratio.

    PSR is the probability that the true Sharpe ratio exceeds a benchmark.

    Args:
        sharpe_ratio: Observed Sharpe ratio
        benchmark_sharpe: Benchmark Sharpe ratio to beat
        num_returns: Number of return observations
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns

    Returns:
        Probability that true Sharpe > benchmark
    """
    # Standard error of Sharpe ratio
    se_sharpe = np.sqrt(
        (1 + 0.5 * sharpe_ratio ** 2 - skewness * sharpe_ratio
         + (kurtosis - 3) / 4 * sharpe_ratio ** 2)
        / (num_returns - 1)
    )

    if se_sharpe == 0:
        return 0.5

    z = (sharpe_ratio - benchmark_sharpe) / se_sharpe
    return stats.norm.cdf(z)


def compute_win_rate(
    returns: NDArray[np.float64],
) -> float:
    """Compute win rate (percentage of positive returns).

    Args:
        returns: Array of trade returns

    Returns:
        Win rate as decimal
    """
    if len(returns) == 0:
        return 0.0

    return (returns > 0).mean()


def compute_profit_factor(
    returns: NDArray[np.float64],
) -> float:
    """Compute profit factor (gross profit / gross loss).

    Args:
        returns: Array of trade returns

    Returns:
        Profit factor
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def compute_information_ratio(
    returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Compute Information Ratio (active return / tracking error).

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Information Ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark must have same length")

    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        return 0.0

    return np.mean(active_returns) / tracking_error * np.sqrt(periods_per_year)


def compute_tail_ratio(
    returns: NDArray[np.float64],
    percentile: float = 5.0,
) -> float:
    """Compute tail ratio (right tail / left tail).

    Measures asymmetry in extreme returns.

    Args:
        returns: Array of returns
        percentile: Percentile for tail calculation

    Returns:
        Tail ratio
    """
    right_tail = np.percentile(returns, 100 - percentile)
    left_tail = abs(np.percentile(returns, percentile))

    if left_tail == 0:
        return float("inf") if right_tail > 0 else 0.0

    return right_tail / left_tail


def compute_all_metrics(
    returns: NDArray[np.float64],
    trade_returns: NDArray[np.float64] | None = None,
    holding_periods: NDArray[np.int64] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    num_trials: int = 1,
) -> PerformanceMetrics:
    """Compute all performance metrics.

    Args:
        returns: Array of period returns
        trade_returns: Array of individual trade returns (optional)
        holding_periods: Array of trade holding periods (optional)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        num_trials: Number of backtests for DSR

    Returns:
        PerformanceMetrics dataclass
    """
    if len(returns) == 0:
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            deflated_sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            avg_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_trade=0.0,
            num_trades=0,
            avg_holding_period=0.0,
            trades_per_day=0.0,
        )

    # Return metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    volatility = np.std(returns, ddof=1)
    annualized_volatility = volatility * np.sqrt(periods_per_year)

    # Risk-adjusted returns
    sharpe = compute_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = compute_sortino_ratio(returns, risk_free_rate, 0.0, periods_per_year)
    calmar = compute_calmar_ratio(returns, periods_per_year)

    # DSR
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns) + 3  # scipy returns excess kurtosis
    dsr = compute_deflated_sharpe_ratio(
        sharpe, num_trials, 1.0, skewness, kurtosis, len(returns)
    )

    # Drawdown
    max_dd, peak_idx, trough_idx, recovery_idx = compute_max_drawdown(returns)
    dd_duration = recovery_idx - peak_idx

    # Average drawdown
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = 1 - cum_returns / running_max
    avg_dd = np.mean(drawdowns)

    # Trade metrics
    if trade_returns is not None:
        trades = trade_returns
    else:
        trades = returns

    num_trades = len(trades)
    win_rate = compute_win_rate(trades)
    profit_factor = compute_profit_factor(trades)

    wins = trades[trades > 0]
    losses = trades[trades < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    avg_trade = np.mean(trades)

    # Time metrics
    if holding_periods is not None:
        avg_holding = np.mean(holding_periods)
    else:
        avg_holding = 1.0

    trades_per_day = num_trades / (len(returns) / periods_per_year * 365)

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        deflated_sharpe_ratio=dsr,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        avg_drawdown=avg_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_trade=avg_trade,
        num_trades=num_trades,
        avg_holding_period=avg_holding,
        trades_per_day=trades_per_day,
    )


def compute_classification_metrics(
    y_true: NDArray[np.int8],
    y_pred: NDArray[np.int8],
    y_proba: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Compute classification metrics for trading signals.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Direction accuracy (ignoring neutral)
    direction_mask = (y_true != 0) & (y_pred != 0)
    if direction_mask.sum() > 0:
        metrics["direction_accuracy"] = (
            y_true[direction_mask] == y_pred[direction_mask]
        ).mean()
    else:
        metrics["direction_accuracy"] = 0.0

    # Log loss if probabilities available
    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except Exception:
            metrics["log_loss"] = float("inf")

    return metrics
