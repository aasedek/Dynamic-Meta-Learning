"""
Feature engineering for meta-learner in DML ensemble.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class MetaFeatureExtractor:
    """
    Extracts meta-features for the meta-learner component.
    """

    def __init__(self, lambda_importance: float = 0.5):
        """
        Initialize meta-feature extractor.

        Args:
            lambda_importance: Weight for combining XGBoost and NN feature importances
        """
        self.lambda_importance = lambda_importance
        logger.info(
            f"Initialized MetaFeatureExtractor with lambda_importance={lambda_importance}"
        )

    def extract_features(
        self,
        X: np.ndarray,
        xgb_predictions: np.ndarray,
        nn_predictions: np.ndarray,
        xgb_confidence: np.ndarray,
        nn_confidence: np.ndarray,
        xgb_importance: np.ndarray,
        nn_importance: np.ndarray,
    ) -> np.ndarray:
        """
        Extract comprehensive meta-features.

        Args:
            X: Original scaled features
            xgb_predictions: XGBoost predictions
            nn_predictions: Neural network predictions
            xgb_confidence: XGBoost confidence scores
            nn_confidence: Neural network confidence scores
            xgb_importance: XGBoost feature importance
            nn_importance: Neural network feature importance

        Returns:
            Meta-features array
        """
        logger.debug(f"Extracting meta-features for {X.shape[0]} samples")

        # Combined feature importance
        combined_importance = (
            self.lambda_importance * xgb_importance
            + (1 - self.lambda_importance) * nn_importance
        )

        # Raw input characteristics
        input_statistics = self._compute_input_statistics(X)

        # Model disagreement
        prediction_disagreement = np.abs(xgb_predictions - nn_predictions).reshape(
            -1, 1
        )

        # Combine all meta-features
        meta_features = np.column_stack(
            [
                X,  # Original scaled features
                xgb_confidence.reshape(-1, 1),  # XGBoost confidence
                nn_confidence.reshape(-1, 1),  # NN confidence
                np.tile(
                    combined_importance, (X.shape[0], 1)
                ),  # Combined feature importance
                input_statistics,  # Input statistics
                prediction_disagreement,  # Model disagreement
            ]
        )

        logger.debug(f"Meta-features shape: {meta_features.shape}")
        return meta_features

    def _compute_input_statistics(self, X: np.ndarray) -> np.ndarray:
        """
        Compute statistical characteristics of input features.

        Args:
            X: Input features

        Returns:
            Input statistics array
        """
        statistics = np.column_stack(
            [
                np.mean(X, axis=1),  # Mean of features per sample
                np.std(X, axis=1),  # Std of features per sample
                np.max(X, axis=1),  # Max of features per sample
                np.min(X, axis=1),  # Min of features per sample
            ]
        )

        return statistics
