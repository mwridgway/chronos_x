"""Fractional Differentiation for stationarity transformation.

Implements Fixed-Width Window Fractional Differentiation (FFD) from
"Advances in Financial Machine Learning" by Marcos López de Prado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numba import jit, prange
from scipy.stats import adfuller

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class FracDiffResult:
    """Result of fractional differentiation."""

    series: pl.Series | NDArray[np.float64]
    d: float
    adf_stat: float
    adf_pvalue: float
    is_stationary: bool
    weights: NDArray[np.float64]


@jit(nopython=True, cache=True)
def _compute_weights(d: float, threshold: float, max_size: int) -> NDArray[np.float64]:
    """Compute FFD weights using binomial expansion.

    w_k = -w_{k-1} * (d - k + 1) / k for k >= 1, w_0 = 1
    """
    weights = np.zeros(max_size, dtype=np.float64)
    weights[0] = 1.0

    k = 1
    while k < max_size:
        weights[k] = -weights[k - 1] * (d - k + 1) / k
        if abs(weights[k]) < threshold:
            return weights[:k]
        k += 1

    return weights


@jit(nopython=True, parallel=True, cache=True)
def _apply_ffd_numba(
    series: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply FFD transformation using Numba for performance."""
    n = len(series)
    w_len = len(weights)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in prange(w_len - 1, n):
        acc = 0.0
        for j in range(w_len):
            acc += weights[j] * series[i - j]
        result[i] = acc

    return result


def compute_ffd_weights(
    d: float,
    threshold: float = 1e-5,
    max_size: int = 10000,
) -> NDArray[np.float64]:
    """Compute Fixed-Width Window Fractional Differentiation weights.

    Args:
        d: Fractional differentiation order (0 < d < 1)
        threshold: Weight threshold for truncation
        max_size: Maximum number of weights to compute

    Returns:
        Array of FFD weights
    """
    return _compute_weights(d, threshold, max_size)


def apply_ffd(
    series: pl.Series | NDArray[np.float64],
    d: float,
    threshold: float = 1e-5,
) -> pl.Series | NDArray[np.float64]:
    """Apply Fixed-Width Window Fractional Differentiation.

    Args:
        series: Input price series
        d: Differentiation order (0 < d < 1)
        threshold: Weight threshold for truncation

    Returns:
        Fractionally differentiated series
    """
    is_polars = isinstance(series, pl.Series)
    if is_polars:
        values = series.to_numpy()
        name = series.name
    else:
        values = series
        name = None

    weights = compute_ffd_weights(d, threshold)
    result = _apply_ffd_numba(values, weights)

    if is_polars:
        return pl.Series(name=name or "ffd", values=result)
    return result


def adf_test(
    series: pl.Series | NDArray[np.float64],
    significance: float = 0.05,
) -> tuple[float, float, bool]:
    """Perform Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Input series
        significance: Significance level for stationarity test

    Returns:
        Tuple of (ADF statistic, p-value, is_stationary)
    """
    if isinstance(series, pl.Series):
        values = series.drop_nulls().to_numpy()
    else:
        values = series[~np.isnan(series)]

    if len(values) < 20:
        return np.nan, 1.0, False

    try:
        result = adfuller(values, maxlag=None, autolag="AIC")
        adf_stat = result[0]
        p_value = result[1]
        is_stationary = p_value < significance
        return adf_stat, p_value, is_stationary
    except Exception:
        return np.nan, 1.0, False


def find_optimal_d(
    series: pl.Series | NDArray[np.float64],
    d_range: tuple[float, float] = (0.0, 1.0),
    d_step: float = 0.05,
    significance: float = 0.05,
    threshold: float = 1e-5,
) -> FracDiffResult:
    """Find minimum d that achieves stationarity while preserving memory.

    Uses binary search to find the smallest d value that produces
    a stationary series according to the ADF test.

    Args:
        series: Input price series
        d_range: Range of d values to search
        d_step: Step size for initial grid search
        significance: Significance level for ADF test
        threshold: Weight threshold for FFD

    Returns:
        FracDiffResult with optimal d and transformed series
    """
    if isinstance(series, pl.Series):
        values = series.to_numpy()
        name = series.name
    else:
        values = series
        name = None

    # Remove NaN values for testing
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 50:
        raise ValueError("Series too short for fractional differentiation")

    d_min, d_max = d_range
    optimal_d = d_max
    best_result = None

    # Grid search to find approximate optimal d
    d_values = np.arange(d_min, d_max + d_step, d_step)

    for d in d_values:
        if d == 0:
            continue

        ffd_series = apply_ffd(values, d, threshold)
        adf_stat, p_value, is_stationary = adf_test(ffd_series, significance)

        if is_stationary:
            optimal_d = d
            weights = compute_ffd_weights(d, threshold)
            best_result = FracDiffResult(
                series=ffd_series,
                d=d,
                adf_stat=adf_stat,
                adf_pvalue=p_value,
                is_stationary=True,
                weights=weights,
            )
            break

    # If no stationary d found, use d=1 (regular differencing)
    if best_result is None:
        d = 1.0
        ffd_series = apply_ffd(values, d, threshold)
        adf_stat, p_value, is_stationary = adf_test(ffd_series, significance)
        weights = compute_ffd_weights(d, threshold)
        best_result = FracDiffResult(
            series=ffd_series,
            d=d,
            adf_stat=adf_stat,
            adf_pvalue=p_value,
            is_stationary=is_stationary,
            weights=weights,
        )

    # Convert back to Polars if input was Polars
    if name is not None:
        best_result.series = pl.Series(name=f"{name}_ffd", values=best_result.series)

    return best_result


def frac_diff_dataframe(
    df: pl.DataFrame,
    columns: list[str],
    d: float | dict[str, float] | None = None,
    find_optimal: bool = True,
    significance: float = 0.05,
    threshold: float = 1e-5,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Apply fractional differentiation to multiple columns.

    Args:
        df: Input DataFrame
        columns: Columns to transform
        d: Fixed d value(s) or None for optimal search
        find_optimal: If True and d is None, find optimal d for each column
        significance: Significance level for ADF test
        threshold: Weight threshold for FFD

    Returns:
        Tuple of (transformed DataFrame, dict of d values used)
    """
    result_df = df.clone()
    d_values: dict[str, float] = {}

    for col in columns:
        series = df[col]

        if d is not None:
            # Use provided d value
            d_val = d[col] if isinstance(d, dict) else d
            ffd_series = apply_ffd(series, d_val, threshold)
            d_values[col] = d_val
        elif find_optimal:
            # Find optimal d
            ffd_result = find_optimal_d(series, significance=significance, threshold=threshold)
            ffd_series = ffd_result.series
            d_values[col] = ffd_result.d
        else:
            # Default to d=1 (regular differencing)
            ffd_series = apply_ffd(series, 1.0, threshold)
            d_values[col] = 1.0

        # Add transformed column
        if isinstance(ffd_series, pl.Series):
            result_df = result_df.with_columns(ffd_series.alias(f"{col}_ffd"))
        else:
            result_df = result_df.with_columns(
                pl.Series(name=f"{col}_ffd", values=ffd_series)
            )

    return result_df, d_values


class FracDiffTransformer:
    """Scikit-learn compatible transformer for fractional differentiation."""

    def __init__(
        self,
        d: float | None = None,
        find_optimal: bool = True,
        significance: float = 0.05,
        threshold: float = 1e-5,
    ) -> None:
        self.d = d
        self.find_optimal = find_optimal
        self.significance = significance
        self.threshold = threshold
        self._fitted_d: float | None = None
        self._weights: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[np.float64]) -> FracDiffTransformer:
        """Fit the transformer to find optimal d."""
        if X.ndim > 1:
            X = X.flatten()

        if self.d is not None:
            self._fitted_d = self.d
        elif self.find_optimal:
            result = find_optimal_d(
                X,
                significance=self.significance,
                threshold=self.threshold,
            )
            self._fitted_d = result.d
        else:
            self._fitted_d = 1.0

        self._weights = compute_ffd_weights(self._fitted_d, self.threshold)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform the series using fitted d."""
        if self._fitted_d is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        if X.ndim > 1:
            X = X.flatten()

        return apply_ffd(X, self._fitted_d, self.threshold)

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    @property
    def fitted_d(self) -> float | None:
        """Get the fitted d value."""
        return self._fitted_d

    @property
    def weights(self) -> NDArray[np.float64] | None:
        """Get the computed weights."""
        return self._weights
