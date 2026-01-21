"""Monitoring and drift detection for production systems.

Includes:
- Feature drift detection
- Model performance monitoring
- Alerting system
- Health checks
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import structlog
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert message."""

    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""

    # Drift detection
    drift_check_interval_minutes: int = 60
    drift_threshold_psi: float = 0.2  # Population Stability Index threshold
    drift_threshold_ks: float = 0.05  # KS test p-value threshold

    # Performance monitoring
    performance_window_trades: int = 100
    min_sharpe_threshold: float = 0.5
    max_drawdown_threshold: float = 0.15
    min_win_rate_threshold: float = 0.45

    # Reference windows
    reference_window_days: int = 30
    comparison_window_days: int = 7

    # Health checks
    max_latency_ms: float = 1000.0
    max_data_gap_minutes: int = 5

    # Alerting
    enable_alerts: bool = True
    alert_webhook: str = ""
    alert_cooldown_minutes: int = 15


@dataclass
class DriftReport:
    """Report on feature drift detection."""

    timestamp: datetime
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    is_drifted: bool
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass
class PerformanceReport:
    """Report on model performance."""

    timestamp: datetime
    window_trades: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    is_degraded: bool
    degradation_reasons: list[str] = field(default_factory=list)


class PopulationStabilityIndex:
    """Calculate Population Stability Index for drift detection."""

    @staticmethod
    def compute(
        reference: NDArray[np.float64],
        current: NDArray[np.float64],
        n_bins: int = 10,
    ) -> float:
        """Compute PSI between reference and current distributions.

        PSI = Σ (current_pct - reference_pct) * ln(current_pct / reference_pct)

        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins for discretization

        Returns:
            PSI score
        """
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Count frequencies
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to percentages (with small epsilon to avoid log(0))
        eps = 1e-6
        ref_pct = (ref_counts + eps) / (len(reference) + n_bins * eps)
        cur_pct = (cur_counts + eps) / (len(current) + n_bins * eps)

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    @staticmethod
    def interpret(psi: float) -> str:
        """Interpret PSI score.

        Args:
            psi: PSI score

        Returns:
            Interpretation string
        """
        if psi < 0.1:
            return "no_significant_change"
        elif psi < 0.2:
            return "moderate_change"
        else:
            return "significant_change"


class DriftDetector:
    """Detect drift in features and predictions."""

    def __init__(
        self,
        config: MonitoringConfig | None = None,
    ) -> None:
        self.config = config or MonitoringConfig()
        self._reference_data: dict[str, NDArray] = {}
        self._current_data: dict[str, deque] = {}
        self._log = logger.bind(component="drift_detector")

    def set_reference(
        self,
        feature_name: str,
        data: NDArray[np.float64],
    ) -> None:
        """Set reference distribution for a feature.

        Args:
            feature_name: Name of the feature
            data: Reference data array
        """
        self._reference_data[feature_name] = data.copy()
        self._current_data[feature_name] = deque(
            maxlen=len(data) * 2  # Keep enough for comparison
        )

    def add_observation(
        self,
        feature_name: str,
        value: float,
    ) -> None:
        """Add a new observation for drift monitoring.

        Args:
            feature_name: Name of the feature
            value: New observation value
        """
        if feature_name not in self._current_data:
            self._current_data[feature_name] = deque(maxlen=10000)

        self._current_data[feature_name].append(value)

    def check_drift(
        self,
        feature_name: str,
    ) -> DriftReport | None:
        """Check for drift in a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            DriftReport or None if insufficient data
        """
        if feature_name not in self._reference_data:
            return None

        reference = self._reference_data[feature_name]
        current = np.array(self._current_data.get(feature_name, []))

        if len(current) < 100:  # Minimum samples needed
            return None

        # Calculate PSI
        psi = PopulationStabilityIndex.compute(reference, current)

        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)

        # Determine if drifted
        is_drifted = psi > self.config.drift_threshold_psi or ks_pvalue < self.config.drift_threshold_ks

        report = DriftReport(
            timestamp=datetime.now(UTC),
            feature_name=feature_name,
            psi_score=psi,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            is_drifted=is_drifted,
            reference_mean=float(reference.mean()),
            current_mean=float(current.mean()),
            reference_std=float(reference.std()),
            current_std=float(current.std()),
        )

        if is_drifted:
            self._log.warning(
                "drift_detected",
                feature=feature_name,
                psi=psi,
                ks_pvalue=ks_pvalue,
            )

        return report

    def check_all_features(self) -> list[DriftReport]:
        """Check drift for all monitored features.

        Returns:
            List of drift reports
        """
        reports = []
        for feature_name in self._reference_data:
            report = self.check_drift(feature_name)
            if report is not None:
                reports.append(report)
        return reports


class PerformanceMonitor:
    """Monitor model and strategy performance."""

    def __init__(
        self,
        config: MonitoringConfig | None = None,
    ) -> None:
        self.config = config or MonitoringConfig()
        self._trades: deque = deque(maxlen=self.config.performance_window_trades * 2)
        self._predictions: deque = deque(maxlen=10000)
        self._log = logger.bind(component="performance_monitor")

    def record_trade(
        self,
        pnl: float,
        return_pct: float,
        prediction: int,
        actual: int,
    ) -> None:
        """Record a completed trade.

        Args:
            pnl: Trade profit/loss
            return_pct: Trade return percentage
            prediction: Model prediction
            actual: Actual outcome
        """
        self._trades.append({
            "timestamp": datetime.now(UTC),
            "pnl": pnl,
            "return_pct": return_pct,
            "prediction": prediction,
            "actual": actual,
            "correct": prediction == actual,
        })

    def record_prediction(
        self,
        prediction: int,
        probability: float,
        actual: int | None = None,
    ) -> None:
        """Record a model prediction.

        Args:
            prediction: Predicted class
            probability: Prediction probability
            actual: Actual outcome (if known)
        """
        self._predictions.append({
            "timestamp": datetime.now(UTC),
            "prediction": prediction,
            "probability": probability,
            "actual": actual,
        })

    def compute_metrics(
        self,
        window: int | None = None,
    ) -> dict[str, float]:
        """Compute performance metrics over recent trades.

        Args:
            window: Number of trades to consider

        Returns:
            Dictionary of metrics
        """
        window = window or self.config.performance_window_trades
        trades = list(self._trades)[-window:]

        if len(trades) < 10:
            return {}

        returns = np.array([t["return_pct"] for t in trades])
        pnls = np.array([t["pnl"] for t in trades])
        correct = np.array([t["correct"] for t in trades])

        # Win rate
        win_rate = (returns > 0).mean()

        # Average return
        avg_return = returns.mean()

        # Sharpe ratio (simplified)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = drawdown.max()

        # Prediction accuracy
        accuracy = correct.mean()

        return {
            "win_rate": float(win_rate),
            "avg_return": float(avg_return),
            "sharpe_ratio": float(sharpe),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(max_drawdown),
            "accuracy": float(accuracy),
            "num_trades": len(trades),
        }

    def check_performance(self) -> PerformanceReport:
        """Check if performance has degraded.

        Returns:
            PerformanceReport with degradation status
        """
        metrics = self.compute_metrics()

        if not metrics:
            return PerformanceReport(
                timestamp=datetime.now(UTC),
                window_trades=0,
                win_rate=0.0,
                avg_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                is_degraded=False,
            )

        degradation_reasons = []

        # Check thresholds
        if metrics["sharpe_ratio"] < self.config.min_sharpe_threshold:
            degradation_reasons.append(
                f"Low Sharpe ratio: {metrics['sharpe_ratio']:.2f}"
            )

        if metrics["max_drawdown"] > self.config.max_drawdown_threshold:
            degradation_reasons.append(
                f"High drawdown: {metrics['max_drawdown']:.2%}"
            )

        if metrics["win_rate"] < self.config.min_win_rate_threshold:
            degradation_reasons.append(
                f"Low win rate: {metrics['win_rate']:.2%}"
            )

        is_degraded = len(degradation_reasons) > 0

        report = PerformanceReport(
            timestamp=datetime.now(UTC),
            window_trades=metrics["num_trades"],
            win_rate=metrics["win_rate"],
            avg_return=metrics["avg_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            profit_factor=metrics["profit_factor"],
            is_degraded=is_degraded,
            degradation_reasons=degradation_reasons,
        )

        if is_degraded:
            self._log.warning(
                "performance_degraded",
                reasons=degradation_reasons,
                metrics=metrics,
            )

        return report


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(
        self,
        config: MonitoringConfig | None = None,
        alert_callback: Callable[[Alert], None] | None = None,
    ) -> None:
        self.config = config or MonitoringConfig()
        self.alert_callback = alert_callback
        self._recent_alerts: deque = deque(maxlen=1000)
        self._last_alert_time: dict[str, datetime] = {}
        self._log = logger.bind(component="alert_manager")

    def send_alert(
        self,
        severity: AlertSeverity,
        category: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Send an alert if not in cooldown.

        Args:
            severity: Alert severity
            category: Alert category
            message: Alert message
            details: Additional details

        Returns:
            True if alert was sent
        """
        if not self.config.enable_alerts:
            return False

        # Check cooldown
        cooldown_key = f"{category}:{message}"
        now = datetime.now(UTC)

        if cooldown_key in self._last_alert_time:
            elapsed = now - self._last_alert_time[cooldown_key]
            if elapsed < timedelta(minutes=self.config.alert_cooldown_minutes):
                return False

        # Create alert
        alert = Alert(
            timestamp=now,
            severity=severity,
            category=category,
            message=message,
            details=details or {},
        )

        self._recent_alerts.append(alert)
        self._last_alert_time[cooldown_key] = now

        self._log.info(
            "alert_sent",
            severity=severity.value,
            category=category,
            message=message,
        )

        # Call callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                self._log.error("alert_callback_error", error=str(e))

        return True

    def get_recent_alerts(
        self,
        severity: AlertSeverity | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Get recent alerts with optional filtering.

        Args:
            severity: Filter by severity
            category: Filter by category
            limit: Maximum number of alerts

        Returns:
            List of alerts
        """
        alerts = list(self._recent_alerts)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if category:
            alerts = [a for a in alerts if a.category == category]

        return alerts[-limit:]


class SystemMonitor:
    """Overall system health monitoring."""

    def __init__(
        self,
        config: MonitoringConfig | None = None,
    ) -> None:
        self.config = config or MonitoringConfig()
        self.drift_detector = DriftDetector(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.alert_manager = AlertManager(config)

        self._last_data_time: datetime | None = None
        self._latency_samples: deque = deque(maxlen=100)
        self._log = logger.bind(component="system_monitor")

    def record_data_received(self) -> None:
        """Record that data was received."""
        self._last_data_time = datetime.now(UTC)

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._latency_samples.append(latency_ms)

        if latency_ms > self.config.max_latency_ms:
            self.alert_manager.send_alert(
                AlertSeverity.WARNING,
                "latency",
                f"High latency detected: {latency_ms:.0f}ms",
                {"latency_ms": latency_ms},
            )

    def check_data_freshness(self) -> bool:
        """Check if data is fresh.

        Returns:
            True if data is fresh
        """
        if self._last_data_time is None:
            return False

        elapsed = datetime.now(UTC) - self._last_data_time
        max_gap = timedelta(minutes=self.config.max_data_gap_minutes)

        is_fresh = elapsed < max_gap

        if not is_fresh:
            self.alert_manager.send_alert(
                AlertSeverity.ERROR,
                "data_gap",
                f"Data gap detected: {elapsed.total_seconds():.0f}s since last update",
                {"gap_seconds": elapsed.total_seconds()},
            )

        return is_fresh

    def run_health_check(self) -> dict[str, Any]:
        """Run full health check.

        Returns:
            Health check results
        """
        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "data_fresh": self.check_data_freshness(),
            "avg_latency_ms": (
                np.mean(self._latency_samples) if self._latency_samples else 0
            ),
            "performance": self.performance_monitor.compute_metrics(),
            "drift_reports": [],
            "alerts": [],
        }

        # Check drift
        drift_reports = self.drift_detector.check_all_features()
        for report in drift_reports:
            if report.is_drifted:
                results["drift_reports"].append({
                    "feature": report.feature_name,
                    "psi": report.psi_score,
                    "ks_pvalue": report.ks_pvalue,
                })

                self.alert_manager.send_alert(
                    AlertSeverity.WARNING,
                    "drift",
                    f"Feature drift detected: {report.feature_name}",
                    {"psi": report.psi_score, "ks_pvalue": report.ks_pvalue},
                )

        # Check performance
        perf_report = self.performance_monitor.check_performance()
        if perf_report.is_degraded:
            self.alert_manager.send_alert(
                AlertSeverity.ERROR,
                "performance",
                "Model performance degradation detected",
                {"reasons": perf_report.degradation_reasons},
            )

        results["is_healthy"] = (
            results["data_fresh"]
            and not perf_report.is_degraded
            and len([r for r in drift_reports if r.is_drifted]) == 0
        )

        return results
