"""
Main DML (Dynamic Meta-Learning for Adaptive XGBoost-Neural Ensembles) implementation.
"""

import numpy as np
import time
from typing import Dict, Tuple
import logging

from sklearn.model_selection import train_test_split

from ..config.model_config import DMLConfig
from ..utils.data_utils import FeatureProcessor
from .neural_network import MonteCarloDropoutNetwork
from .xgboost_model import XGBoostModel
from .meta_learner import MetaLearner
from .feature_extractor import MetaFeatureExtractor

logger = logging.getLogger(__name__)


class AdaptiveXGBoostNeuralEnsemble:
    """
    Dynamic Meta-Learning for Adaptive XGBoost-Neural Ensembles.

    This class implements the DML methodology that combines XGBoost and
    Neural Networks with a meta-learner for adaptive model selection.
    """

    def __init__(self, config: DMLConfig = None):
        """
        Initialize the DML ensemble.

        Args:
            config: DML configuration object
        """
        if config is None:
            config = DMLConfig.get_default_config()

        self.config = config

        # Initialize components
        self.feature_processor = FeatureProcessor()
        self.neural_network = MonteCarloDropoutNetwork(config.neural_network)
        self.xgboost_model = XGBoostModel(
            config.xgboost, random_state=config.ensemble.random_seed
        )
        self.meta_learner = MetaLearner(config.meta_learner)
        self.meta_feature_extractor = MetaFeatureExtractor(
            lambda_importance=config.ensemble.lambda_importance
        )

        self._is_fitted = False

        logger.info("DML ensemble initialized successfully")
        logger.info(f"Configuration: {config}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the DML ensemble.

        Args:
            X: Training features
            y: Training targets
        """
        logger.info("=== Starting DML Training Process ===")
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target data shape: {y.shape}")

        start_time = time.time()

        # Split data for base models and meta-learner
        X_base, X_meta, y_base, y_meta = self._split_training_data(X, y)

        # Scale features
        X_base_scaled = self.feature_processor.fit_transform(X_base)
        X_meta_scaled = self.feature_processor.transform(X_meta)

        # Phase 1: Train base models
        self._train_base_models(X_base_scaled, y_base)

        # Phase 2: Generate meta-features and train meta-learner
        self._train_meta_learner(X_meta_scaled, y_meta)

        self._is_fitted = True

        total_time = time.time() - start_time
        logger.info(f"=== DML Training Completed in {total_time:.2f} seconds ===")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the DML ensemble.

        Args:
            X: Input features

        Returns:
            Final ensemble predictions
        """
        if not self._is_fitted:
            raise ValueError("DML must be fitted before making predictions")

        logger.info(f"Making predictions for {X.shape[0]} samples")

        X_scaled = self.feature_processor.transform(X)

        # Get base model predictions and confidences
        xgb_pred = self.xgboost_model.predict(X_scaled)
        nn_pred, nn_confidence = self.neural_network.predict_with_uncertainty(X_scaled)
        xgb_confidence = self.xgboost_model.get_confidence(X_scaled)

        # Get feature importances
        xgb_importance = self.xgboost_model.get_feature_importance()
        nn_importance = self.neural_network.get_feature_importance(X_scaled)

        # Extract meta-features
        meta_features = self.meta_feature_extractor.extract_features(
            X_scaled,
            xgb_pred,
            nn_pred,
            xgb_confidence,
            nn_confidence,
            xgb_importance,
            nn_importance,
        )

        # Get model selection probabilities
        probs = self.meta_learner.predict_probabilities(meta_features)

        # Compute final predictions
        hybrid_pred = (xgb_pred + nn_pred) / 2
        final_pred = (
            probs[:, 0] * xgb_pred + probs[:, 1] * nn_pred + probs[:, 2] * hybrid_pred
        )

        self._log_prediction_statistics(probs, final_pred)

        return final_pred

    def get_model_weights(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get model selection probabilities for interpretability.

        Args:
            X: Input features

        Returns:
            Dictionary containing model probabilities
        """
        if not self._is_fitted:
            raise ValueError("DML must be fitted before getting model weights")

        X_scaled = self.feature_processor.transform(X)

        # Get all necessary components for meta-features
        xgb_pred = self.xgboost_model.predict(X_scaled)
        nn_pred, nn_confidence = self.neural_network.predict_with_uncertainty(X_scaled)
        xgb_confidence = self.xgboost_model.get_confidence(X_scaled)
        xgb_importance = self.xgboost_model.get_feature_importance()
        nn_importance = self.neural_network.get_feature_importance(X_scaled)

        meta_features = self.meta_feature_extractor.extract_features(
            X_scaled,
            xgb_pred,
            nn_pred,
            xgb_confidence,
            nn_confidence,
            xgb_importance,
            nn_importance,
        )

        probs = self.meta_learner.predict_probabilities(meta_features)

        return {
            "xgb_probs": probs[:, 0],
            "nn_probs": probs[:, 1],
            "hybrid_probs": probs[:, 2],
        }

    def _split_training_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data for base models and meta-learner training."""
        logger.info("Splitting data for base models and meta-learner...")

        X_base, X_meta, y_base, y_meta = train_test_split(
            X,
            y,
            test_size=self.config.ensemble.validation_split,
            random_state=self.config.ensemble.random_seed,
        )

        logger.info(f"Base training set: {X_base.shape[0]} samples")
        logger.info(f"Meta training set: {X_meta.shape[0]} samples")

        return X_base, X_meta, y_base, y_meta

    def _train_base_models(self, X_base: np.ndarray, y_base: np.ndarray) -> None:
        """Train XGBoost and Neural Network models."""
        logger.info("=== Phase 1: Training base models ===")

        # Train XGBoost
        xgb_start = time.time()
        self.xgboost_model.fit(X_base, y_base)
        xgb_time = time.time() - xgb_start
        logger.info(f"XGBoost training completed in {xgb_time:.2f} seconds")

        # Train Neural Network
        nn_start = time.time()
        self.neural_network.fit(X_base, y_base, validation_split=0.2)
        nn_time = time.time() - nn_start
        logger.info(f"Neural Network training completed in {nn_time:.2f} seconds")

    def _train_meta_learner(self, X_meta: np.ndarray, y_meta: np.ndarray) -> None:
        """Generate meta-features and train the meta-learner."""
        logger.info("=== Phase 2: Training meta-learner ===")

        meta_start = time.time()

        # Get base model predictions and confidences on meta set
        xgb_pred_meta = self.xgboost_model.predict(X_meta)
        nn_pred_meta, nn_confidence_meta = self.neural_network.predict_with_uncertainty(
            X_meta
        )
        xgb_confidence_meta = self.xgboost_model.get_confidence(X_meta)

        # Get feature importances
        xgb_importance = self.xgboost_model.get_feature_importance()
        nn_importance = self.neural_network.get_feature_importance(X_meta)

        # Extract meta-features
        meta_features = self.meta_feature_extractor.extract_features(
            X_meta,
            xgb_pred_meta,
            nn_pred_meta,
            xgb_confidence_meta,
            nn_confidence_meta,
            xgb_importance,
            nn_importance,
        )

        meta_gen_time = time.time() - meta_start
        logger.info(
            f"Meta-features generation completed in {meta_gen_time:.2f} seconds"
        )

        # Train meta-learner
        meta_train_start = time.time()
        self.meta_learner.fit(
            meta_features,
            xgb_pred_meta,
            nn_pred_meta,
            y_meta,
            alpha_regularization=self.config.ensemble.alpha_regularization,
        )
        meta_train_time = time.time() - meta_train_start
        logger.info(f"Meta-learner training completed in {meta_train_time:.2f} seconds")

    def _log_prediction_statistics(
        self, probs: np.ndarray, final_pred: np.ndarray
    ) -> None:
        """Log prediction statistics for monitoring."""
        avg_xgb_prob = np.mean(probs[:, 0])
        avg_nn_prob = np.mean(probs[:, 1])
        avg_hybrid_prob = np.mean(probs[:, 2])

        logger.info(
            f"Average model probabilities - XGBoost: {avg_xgb_prob:.3f}, "
            f"NN: {avg_nn_prob:.3f}, Hybrid: {avg_hybrid_prob:.3f}"
        )
        logger.info(
            f"Prediction range: [{final_pred.min():.4f}, {final_pred.max():.4f}]"
        )

    @property
    def is_fitted(self) -> bool:
        """Check if the ensemble is fitted."""
        return self._is_fitted
