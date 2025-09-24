"""
XGBoost component for DML ensemble.
"""

import numpy as np
import xgboost as xgb
from typing import Dict
import logging

from ..config.model_config import XGBoostConfig

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost model with confidence estimation.
    """

    def __init__(self, config: XGBoostConfig, random_state: int = 42):
        """
        Initialize XGBoost model.

        Args:
            config: XGBoost configuration
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.model = None
        self.n_features = None
        self._is_trained = False

        logger.info(f"Initializing XGBoost with config: {config}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the XGBoost model.

        Args:
            X: Training features
            y: Training targets
        """
        logger.info("Training XGBoost model...")

        self.n_features = X.shape[1]
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(X, y)
        self._is_trained = True

        logger.info("XGBoost training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate confidence using variance across trees.

        Args:
            X: Input features

        Returns:
            Confidence estimates (lower variance = higher confidence)
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before computing confidence")

        logger.debug(f"Calculating XGBoost confidence for {X.shape[0]} samples")

        # Get predictions from individual trees
        tree_predictions = []
        for booster in range(self.model.get_booster().num_boosted_rounds()):
            pred = self.model.predict(X, iteration_range=(booster, booster + 1))
            tree_predictions.append(pred)

        tree_predictions = np.array(tree_predictions)
        # Calculate variance across trees as confidence metric
        confidence = np.var(tree_predictions, axis=0)

        logger.debug(
            f"XGBoost confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]"
        )

        return confidence

    def get_feature_importance(self) -> np.ndarray:
        """
        Get XGBoost feature importance.

        Returns:
            Feature importance scores
        """
        if not self._is_trained:
            raise ValueError(
                "Model must be trained before computing feature importance"
            )

        logger.debug("Computing XGBoost feature importance")

        importance_dict = self.model.get_booster().get_score(importance_type="gain")
        importance = np.zeros(self.n_features)

        for i in range(self.n_features):
            key = f"f{i}"
            if key in importance_dict:
                importance[i] = importance_dict[key]

        # Normalize importance scores
        return importance / np.sum(importance) if np.sum(importance) > 0 else importance

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
