"""Risk management module with HRP and volatility targeting.

Implements:
- Hierarchical Risk Parity (HRP)
- Volatility targeting position sizing
- Kelly criterion
- Risk limits and drawdown controls
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class RiskConfig:
    """Configuration for risk management."""

    # Position sizing method
    sizing_method: str = "volatility_target"  # "fixed", "volatility_target", "kelly"

    # Fixed sizing
    fixed_size_pct: float = 0.1  # 10% per trade

    # Volatility targeting
    target_volatility: float = 0.15  # 15% annualized
    volatility_lookback: int = 20
    max_leverage: float = 1.0

    # Kelly criterion
    kelly_fraction: float = 0.25  # Quarter Kelly
    max_kelly_size: float = 0.2

    # Risk limits
    max_position_size: float = 0.2
    max_drawdown: float = 0.15
    max_daily_loss: float = 0.05
    max_correlation: float = 0.7

    # HRP settings
    hrp_linkage: str = "ward"  # "ward", "single", "complete", "average"


class VolatilityTargeting:
    """Position sizing based on volatility targeting."""

    def __init__(
        self,
        target_volatility: float = 0.15,
        lookback: int = 20,
        max_leverage: float = 1.0,
        annualization_factor: int = 252,
    ) -> None:
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.annualization_factor = annualization_factor

    def compute_position_size(
        self,
        returns: NDArray[np.float64],
        current_volatility: float | None = None,
    ) -> float:
        """Compute position size based on volatility targeting.

        Args:
            returns: Historical returns array
            current_volatility: Pre-computed volatility (optional)

        Returns:
            Position size as fraction of capital
        """
        if current_volatility is None:
            if len(returns) < self.lookback:
                return 0.0

            # Compute realized volatility
            recent_returns = returns[-self.lookback:]
            current_volatility = np.std(recent_returns) * np.sqrt(self.annualization_factor)

        if current_volatility == 0:
            return 0.0

        # Position size = target_vol / current_vol
        position_size = self.target_volatility / current_volatility

        # Apply leverage constraint
        position_size = min(position_size, self.max_leverage)

        return position_size

    def compute_ewm_volatility(
        self,
        returns: NDArray[np.float64],
        span: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute exponentially weighted volatility.

        Args:
            returns: Historical returns
            span: EWM span (default: lookback)

        Returns:
            Array of volatility estimates
        """
        span = span or self.lookback
        alpha = 2.0 / (span + 1)

        variance = np.zeros_like(returns)
        variance[0] = returns[0] ** 2

        for i in range(1, len(returns)):
            variance[i] = alpha * returns[i] ** 2 + (1 - alpha) * variance[i - 1]

        return np.sqrt(variance) * np.sqrt(self.annualization_factor)


class KellyCriterion:
    """Position sizing using Kelly criterion."""

    def __init__(
        self,
        fraction: float = 0.25,
        max_size: float = 0.2,
    ) -> None:
        self.fraction = fraction
        self.max_size = max_size

    def compute_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Compute Kelly optimal position size.

        Kelly = W - (1 - W) / R
        where W is win rate and R is win/loss ratio

        Args:
            win_rate: Probability of winning
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)

        Returns:
            Optimal position size (fractional Kelly)
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # Apply fractional Kelly and max size
        kelly = max(0, kelly) * self.fraction
        kelly = min(kelly, self.max_size)

        return kelly

    def compute_from_returns(
        self,
        returns: NDArray[np.float64],
    ) -> float:
        """Compute Kelly size from historical returns.

        Args:
            returns: Array of trade returns

        Returns:
            Kelly position size
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        return self.compute_kelly_size(win_rate, avg_win, avg_loss)


class HierarchicalRiskParity:
    """Hierarchical Risk Parity for portfolio construction.

    Based on López de Prado's HRP algorithm.
    """

    def __init__(
        self,
        linkage_method: str = "ward",
    ) -> None:
        self.linkage_method = linkage_method

    def compute_weights(
        self,
        returns: NDArray[np.float64],
        asset_names: list[str] | None = None,
    ) -> NDArray[np.float64]:
        """Compute HRP portfolio weights.

        Args:
            returns: Matrix of asset returns (n_samples, n_assets)
            asset_names: Optional list of asset names

        Returns:
            Array of portfolio weights
        """
        n_assets = returns.shape[1]

        if n_assets == 1:
            return np.array([1.0])

        # Step 1: Compute correlation and covariance matrices
        corr_matrix = np.corrcoef(returns.T)
        cov_matrix = np.cov(returns.T)

        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # Step 2: Compute distance matrix
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # Step 3: Hierarchical clustering
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method=self.linkage_method)

        # Step 4: Quasi-diagonalization
        sorted_indices = self._get_quasi_diag(linkage_matrix, n_assets)

        # Step 5: Recursive bisection
        weights = self._recursive_bisection(cov_matrix, sorted_indices)

        return weights

    def _get_quasi_diag(
        self,
        linkage_matrix: NDArray[np.float64],
        n_assets: int,
    ) -> list[int]:
        """Quasi-diagonalize the covariance matrix.

        Args:
            linkage_matrix: Hierarchical clustering linkage matrix
            n_assets: Number of assets

        Returns:
            Sorted list of asset indices
        """
        # Get flat clusters at maximum distance
        clusters = fcluster(linkage_matrix, t=n_assets, criterion="maxclust")

        # Sort by cluster order from dendrogram
        sorted_indices = []
        for cluster_id in range(1, n_assets + 1):
            cluster_assets = np.where(clusters == cluster_id)[0]
            sorted_indices.extend(cluster_assets.tolist())

        return sorted_indices

    def _recursive_bisection(
        self,
        cov_matrix: NDArray[np.float64],
        sorted_indices: list[int],
    ) -> NDArray[np.float64]:
        """Compute weights through recursive bisection.

        Args:
            cov_matrix: Covariance matrix
            sorted_indices: Quasi-diagonalized asset indices

        Returns:
            Array of weights
        """
        n = len(sorted_indices)
        weights = np.ones(n)

        items = [sorted_indices]

        while len(items) > 0:
            # Pop first cluster
            cluster = items.pop(0)

            if len(cluster) == 1:
                continue

            # Split cluster in half
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Compute cluster variances
            left_var = self._cluster_variance(cov_matrix, left)
            right_var = self._cluster_variance(cov_matrix, right)

            # Allocate inversely proportional to variance
            total_var = left_var + right_var
            if total_var > 0:
                left_weight = 1 - left_var / total_var
                right_weight = 1 - right_var / total_var
            else:
                left_weight = 0.5
                right_weight = 0.5

            # Update weights
            for idx in left:
                weights[sorted_indices.index(idx)] *= left_weight
            for idx in right:
                weights[sorted_indices.index(idx)] *= right_weight

            # Add sub-clusters to process
            items.extend([left, right])

        # Reorder weights to original asset order
        final_weights = np.zeros(n)
        for i, idx in enumerate(sorted_indices):
            final_weights[idx] = weights[i]

        return final_weights

    def _cluster_variance(
        self,
        cov_matrix: NDArray[np.float64],
        cluster_indices: list[int],
    ) -> float:
        """Compute variance of an inverse-variance weighted cluster.

        Args:
            cov_matrix: Covariance matrix
            cluster_indices: Indices of assets in cluster

        Returns:
            Cluster variance
        """
        cluster_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]

        # Inverse variance weights within cluster
        variances = np.diag(cluster_cov)
        variances = np.where(variances > 0, variances, 1e-10)
        inv_var_weights = 1.0 / variances
        inv_var_weights = inv_var_weights / inv_var_weights.sum()

        # Portfolio variance
        cluster_var = inv_var_weights @ cluster_cov @ inv_var_weights

        return cluster_var


class RiskManager:
    """Main risk management class combining all components."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

        # Initialize components
        self.volatility_targeter = VolatilityTargeting(
            target_volatility=self.config.target_volatility,
            lookback=self.config.volatility_lookback,
            max_leverage=self.config.max_leverage,
        )

        self.kelly = KellyCriterion(
            fraction=self.config.kelly_fraction,
            max_size=self.config.max_kelly_size,
        )

        self.hrp = HierarchicalRiskParity(linkage_method=self.config.hrp_linkage)

        # State tracking
        self._peak_equity = 1.0
        self._daily_pnl = 0.0
        self._current_drawdown = 0.0

    def compute_position_size(
        self,
        signal_strength: float,
        returns: NDArray[np.float64],
        current_volatility: float | None = None,
    ) -> float:
        """Compute position size based on configured method.

        Args:
            signal_strength: Model signal strength (0 to 1)
            returns: Historical returns
            current_volatility: Pre-computed volatility

        Returns:
            Position size as fraction of capital
        """
        if self.config.sizing_method == "fixed":
            base_size = self.config.fixed_size_pct

        elif self.config.sizing_method == "volatility_target":
            base_size = self.volatility_targeter.compute_position_size(
                returns, current_volatility
            )

        elif self.config.sizing_method == "kelly":
            base_size = self.kelly.compute_from_returns(returns)

        else:
            base_size = self.config.fixed_size_pct

        # Scale by signal strength
        position_size = base_size * signal_strength

        # Apply maximum position limit
        position_size = min(position_size, self.config.max_position_size)

        return position_size

    def check_risk_limits(
        self,
        current_equity: float,
        daily_pnl: float,
    ) -> tuple[bool, str]:
        """Check if risk limits are breached.

        Args:
            current_equity: Current portfolio equity
            daily_pnl: Today's profit/loss

        Returns:
            Tuple of (is_ok, reason_if_breached)
        """
        # Update peak equity
        self._peak_equity = max(self._peak_equity, current_equity)

        # Check drawdown
        self._current_drawdown = 1 - current_equity / self._peak_equity
        if self._current_drawdown > self.config.max_drawdown:
            return False, f"Max drawdown breached: {self._current_drawdown:.2%}"

        # Check daily loss
        self._daily_pnl = daily_pnl
        if daily_pnl < -self.config.max_daily_loss:
            return False, f"Max daily loss breached: {daily_pnl:.2%}"

        return True, ""

    def reset_daily(self) -> None:
        """Reset daily tracking."""
        self._daily_pnl = 0.0

    @property
    def current_drawdown(self) -> float:
        """Get current drawdown."""
        return self._current_drawdown

    @property
    def daily_pnl(self) -> float:
        """Get daily P&L."""
        return self._daily_pnl


def compute_correlation_filter(
    returns: NDArray[np.float64],
    threshold: float = 0.7,
) -> NDArray[np.bool_]:
    """Filter highly correlated assets.

    Args:
        returns: Matrix of asset returns (n_samples, n_assets)
        threshold: Maximum correlation threshold

    Returns:
        Boolean mask of assets to keep
    """
    n_assets = returns.shape[1]

    if n_assets <= 1:
        return np.ones(n_assets, dtype=bool)

    corr_matrix = np.corrcoef(returns.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Find pairs above threshold
    mask = np.ones(n_assets, dtype=bool)

    for i in range(n_assets):
        if not mask[i]:
            continue

        for j in range(i + 1, n_assets):
            if not mask[j]:
                continue

            if abs(corr_matrix[i, j]) > threshold:
                # Remove asset with higher volatility
                vol_i = np.std(returns[:, i])
                vol_j = np.std(returns[:, j])

                if vol_i > vol_j:
                    mask[i] = False
                else:
                    mask[j] = False

    return mask
