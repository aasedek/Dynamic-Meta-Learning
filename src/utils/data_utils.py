"""
Data preprocessing and evaluation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preparation."""

    @staticmethod
    def load_california_housing() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load and prepare the California housing dataset.

        Returns:
            Tuple of (features, targets, feature_names)
        """
        logger.info("Loading California Housing dataset...")

        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        feature_names = list(housing.feature_names)

        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Features: {feature_names}")
        logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

        return X, y, feature_names

    @staticmethod
    def split_data(
        X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            X: Features
            y: Targets
            test_size: Fraction of data to use for testing
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")

        return X_train, X_test, y_train, y_test


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""

    @staticmethod
    def evaluate_regression_model(
        y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate regression model performance.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model for logging

        Returns:
            Dictionary containing evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results = {"rmse": rmse, "mae": mae, "r2": r2, "mse": mse}

        logger.info(
            f"{model_name} Performance: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
        )

        return results

    @staticmethod
    def create_results_summary(
        results_dict: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame from evaluation results.

        Args:
            results_dict: Dictionary mapping model names to their results

        Returns:
            DataFrame containing comparison results
        """
        data = []
        for model_name, metrics in results_dict.items():
            row = {"Model": model_name}
            row.update(metrics)
            data.append(row)

        return pd.DataFrame(data)


class FeatureProcessor:
    """Handles feature preprocessing and scaling."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform features."""
        logger.info("Fitting and transforming features...")
        X_scaled = self.scaler.fit_transform(X)
        self._is_fitted = True
        logger.info("Feature scaling completed")
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(X)

    @property
    def is_fitted(self) -> bool:
        """Check if scaler is fitted."""
        return self._is_fitted
