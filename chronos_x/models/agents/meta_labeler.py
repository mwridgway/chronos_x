"""Meta-labeler using gradient boosting for signal filtering.

Implements meta-labeling from "Advances in Financial Machine Learning"
to filter primary model signals based on probability of correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


@dataclass
class MetaLabelerConfig:
    """Configuration for meta-labeler."""

    model_type: Literal["lightgbm", "xgboost"] = "lightgbm"
    probability_threshold: float = 0.5

    # LightGBM parameters
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 100
    lgb_min_child_samples: int = 20
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 0.1

    # XGBoost parameters
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 100
    xgb_min_child_weight: int = 1
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 0.1

    # Training settings
    early_stopping_rounds: int = 10
    eval_metric: str = "auc"
    random_state: int = 42


class MetaLabeler:
    """Meta-labeler for filtering trading signals.

    Uses a secondary model (XGBoost or LightGBM) to predict
    whether the primary model's signals will be correct.
    """

    def __init__(self, config: MetaLabelerConfig | None = None) -> None:
        self.config = config or MetaLabelerConfig()
        self._model = None
        self._feature_importance: dict[str, float] = {}
        self._is_fitted = False
        self._log = logger.bind(component="meta_labeler")

    def _create_model(self) -> Any:
        """Create the gradient boosting model."""
        if self.config.model_type == "lightgbm":
            try:
                import lightgbm as lgb

                return lgb.LGBMClassifier(
                    num_leaves=self.config.lgb_num_leaves,
                    learning_rate=self.config.lgb_learning_rate,
                    n_estimators=self.config.lgb_n_estimators,
                    min_child_samples=self.config.lgb_min_child_samples,
                    reg_alpha=self.config.lgb_reg_alpha,
                    reg_lambda=self.config.lgb_reg_lambda,
                    random_state=self.config.random_state,
                    verbosity=-1,
                )
            except ImportError:
                self._log.warning("LightGBM not available, falling back to XGBoost")
                self.config.model_type = "xgboost"

        if self.config.model_type == "xgboost":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    max_depth=self.config.xgb_max_depth,
                    learning_rate=self.config.xgb_learning_rate,
                    n_estimators=self.config.xgb_n_estimators,
                    min_child_weight=self.config.xgb_min_child_weight,
                    reg_alpha=self.config.xgb_reg_alpha,
                    reg_lambda=self.config.xgb_reg_lambda,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            except ImportError:
                raise ImportError("Neither LightGBM nor XGBoost is available")

        raise ValueError(f"Unknown model type: {self.config.model_type}")

    def fit(
        self,
        X: NDArray[np.float64],
        y_primary: NDArray[np.int8],
        y_actual: NDArray[np.int8],
        sample_weight: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        eval_set: tuple[NDArray, NDArray] | None = None,
    ) -> MetaLabeler:
        """Fit the meta-labeler.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_primary: Primary model predictions
            y_actual: Actual labels (ground truth)
            sample_weight: Optional sample weights
            feature_names: Optional feature names for importance
            eval_set: Optional validation set (X_val, y_val)

        Returns:
            Self for chaining
        """
        # Create binary target: 1 if primary prediction was correct
        y_meta = (y_primary == y_actual).astype(np.int8)

        # Filter to only samples where primary model made a prediction
        mask = y_primary != 0
        X_filtered = X[mask]
        y_meta_filtered = y_meta[mask]

        if sample_weight is not None:
            sample_weight = sample_weight[mask]

        self._log.info(
            "fitting_meta_labeler",
            samples=len(X_filtered),
            positive_rate=float(y_meta_filtered.mean()),
        )

        self._model = self._create_model()

        # Prepare fit parameters
        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        # Handle early stopping
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_meta = (y_val == y_actual[: len(y_val)]).astype(np.int8)

            if self.config.model_type == "lightgbm":
                fit_params["eval_set"] = [(X_val, y_val_meta)]
                fit_params["callbacks"] = [
                    self._get_lgb_callback()
                ]
            else:
                fit_params["eval_set"] = [(X_val, y_val_meta)]
                fit_params["early_stopping_rounds"] = self.config.early_stopping_rounds
                fit_params["verbose"] = False

        self._model.fit(X_filtered, y_meta_filtered, **fit_params)

        # Store feature importance
        if feature_names is not None:
            importance = self._model.feature_importances_
            self._feature_importance = dict(zip(feature_names, importance))

        self._is_fitted = True
        return self

    def _get_lgb_callback(self) -> Any:
        """Get LightGBM early stopping callback."""
        import lightgbm as lgb

        return lgb.early_stopping(
            stopping_rounds=self.config.early_stopping_rounds,
            verbose=False,
        )

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probability that primary signal is correct.

        Args:
            X: Feature matrix

        Returns:
            Probability of correctness for each sample
        """
        if not self._is_fitted:
            raise ValueError("Meta-labeler not fitted. Call fit() first.")

        proba = self._model.predict_proba(X)
        return proba[:, 1]  # Probability of correct prediction

    def filter_signals(
        self,
        X: NDArray[np.float64],
        y_primary: NDArray[np.int8],
        threshold: float | None = None,
    ) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
        """Filter primary signals based on meta-label probability.

        Args:
            X: Feature matrix
            y_primary: Primary model predictions
            threshold: Probability threshold (uses config if None)

        Returns:
            Tuple of (filtered_signals, probabilities)
        """
        threshold = threshold or self.config.probability_threshold

        proba = self.predict_proba(X)
        filtered = y_primary.copy()
        filtered[proba < threshold] = 0

        return filtered, proba

    def get_signal_strength(
        self,
        X: NDArray[np.float64],
        y_primary: NDArray[np.int8],
    ) -> NDArray[np.float64]:
        """Compute signal strength as |signal| * probability.

        Args:
            X: Feature matrix
            y_primary: Primary model predictions

        Returns:
            Signal strength array
        """
        proba = self.predict_proba(X)
        return np.abs(y_primary) * proba

    @property
    def feature_importance(self) -> dict[str, float]:
        """Get feature importance dictionary."""
        return self._feature_importance

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def save(self, path: str) -> None:
        """Save the meta-labeler model."""
        import joblib

        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")

        joblib.dump(
            {
                "model": self._model,
                "config": self.config,
                "feature_importance": self._feature_importance,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> MetaLabeler:
        """Load a saved meta-labeler."""
        import joblib

        data = joblib.load(path)

        labeler = cls(data["config"])
        labeler._model = data["model"]
        labeler._feature_importance = data["feature_importance"]
        labeler._is_fitted = True

        return labeler


class StackedMetaLabeler:
    """Stacked ensemble of meta-labelers for improved robustness."""

    def __init__(
        self,
        configs: list[MetaLabelerConfig] | None = None,
        n_models: int = 3,
    ) -> None:
        if configs is None:
            # Create diverse configurations
            configs = [
                MetaLabelerConfig(
                    model_type="lightgbm",
                    lgb_num_leaves=31,
                    lgb_learning_rate=0.05,
                ),
                MetaLabelerConfig(
                    model_type="lightgbm",
                    lgb_num_leaves=63,
                    lgb_learning_rate=0.03,
                ),
                MetaLabelerConfig(
                    model_type="xgboost",
                    xgb_max_depth=6,
                    xgb_learning_rate=0.05,
                ),
            ][:n_models]

        self.labelers = [MetaLabeler(cfg) for cfg in configs]
        self._is_fitted = False

    def fit(
        self,
        X: NDArray[np.float64],
        y_primary: NDArray[np.int8],
        y_actual: NDArray[np.int8],
        **kwargs,
    ) -> StackedMetaLabeler:
        """Fit all meta-labelers."""
        for labeler in self.labelers:
            labeler.fit(X, y_primary, y_actual, **kwargs)

        self._is_fitted = True
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict averaged probability across ensemble."""
        probas = [labeler.predict_proba(X) for labeler in self.labelers]
        return np.mean(probas, axis=0)

    def filter_signals(
        self,
        X: NDArray[np.float64],
        y_primary: NDArray[np.int8],
        threshold: float = 0.5,
    ) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
        """Filter signals using ensemble average."""
        proba = self.predict_proba(X)
        filtered = y_primary.copy()
        filtered[proba < threshold] = 0
        return filtered, proba
