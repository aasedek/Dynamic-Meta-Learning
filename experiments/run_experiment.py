"""
Experiment runner for DML model comparison and evaluation.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers
import time
from typing import Dict, Any, Tuple
import logging

from src.config.model_config import DMLConfig
from src.models.dml_ensemble import AdaptiveXGBoostNeuralEnsemble
from src.utils.data_utils import DataLoader, ModelEvaluator, FeatureProcessor
from src.utils.visualization import DMLVisualizer, PerformanceVisualizer
from src.utils.logging_utils import get_project_logger

logger = get_project_logger(__name__)


class DMLExperiment:
    """
    Comprehensive experiment runner for DML evaluation.
    """

    def __init__(self, config: DMLConfig = None):
        """
        Initialize experiment runner.

        Args:
            config: DML configuration
        """
        self.config = config if config else DMLConfig.get_default_config()
        self.results = {}
        self.models = {}

        logger.info("DML Experiment initialized")

    def run_comprehensive_experiment(self) -> Dict[str, Any]:
        """
        Run comprehensive DML experiment with baseline comparisons.

        Returns:
            Dictionary containing all experiment results
        """
        logger.info("=== Starting Comprehensive DML Experiment ===")

        # Load and prepare data
        X, y, feature_names = self._load_and_prepare_data()
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        # Train and evaluate models
        self._train_dml_model(X_train, y_train)
        self._train_baseline_models(X_train, y_train)

        # Make predictions and evaluate
        predictions = self._make_predictions(X_test)
        results = self._evaluate_all_models(y_test, predictions)

        # Generate visualizations and analysis
        self._generate_analysis(X_test, y_test, predictions)

        # Compile final results
        experiment_results = {
            "config": self.config,
            "data_info": {
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                "feature_names": feature_names,
            },
            "model_performance": results,
            "train_test_split": {
                "train_size": X_train.shape[0],
                "test_size": X_test.shape[0],
            },
        }

        logger.info("=== Comprehensive DML Experiment Completed ===")
        return experiment_results

    def _load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """Load and prepare dataset."""
        logger.info("Loading and preparing data...")
        X, y, feature_names = DataLoader.load_california_housing()

        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Features: {feature_names}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

        return X, y, feature_names

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into training and testing sets."""
        return DataLoader.split_data(X, y, test_size=0.2, random_state=42)

    def _train_dml_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the DML model."""
        logger.info("Training DML model...")
        print("Training DML model...")

        start_time = time.time()

        self.models["DML"] = AdaptiveXGBoostNeuralEnsemble(self.config)
        self.models["DML"].fit(X_train, y_train)

        training_time = time.time() - start_time
        logger.info(f"DML training completed in {training_time:.2f} seconds")
        print(f"DML training completed in {training_time:.2f} seconds")

    def _train_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train baseline models for comparison."""
        logger.info("Training baseline models...")
        print("Training baseline models...")

        # Scale features for baseline models
        scaler = FeatureProcessor()
        X_train_scaled = scaler.fit_transform(X_train)

        # Store scaler for later use
        self._baseline_scaler = scaler

        # XGBoost
        logger.info("Training standalone XGBoost...")
        self.models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.models["XGBoost"].fit(X_train_scaled, y_train)

        # Neural Network
        logger.info("Training standalone Neural Network...")
        self.models["Neural Network"] = keras.Sequential(
            [
                layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1),
            ]
        )
        self.models["Neural Network"].compile(optimizer="adam", loss="mse")
        history = self.models["Neural Network"].fit(
            X_train_scaled, y_train, epochs=100, verbose=0, validation_split=0.2
        )

        # Random Forest
        logger.info("Training Random Forest...")
        self.models["Random Forest"] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.models["Random Forest"].fit(X_train_scaled, y_train)

        logger.info("Baseline model training completed")
        print("Baseline model training completed")

    def _make_predictions(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using all models."""
        logger.info("Making predictions with all models...")

        predictions = {}

        # DML predictions
        predictions["DML"] = self.models["DML"].predict(X_test)

        # Baseline model predictions (need scaled features)
        X_test_scaled = self._baseline_scaler.transform(X_test)

        predictions["XGBoost"] = self.models["XGBoost"].predict(X_test_scaled)
        predictions["Neural Network"] = (
            self.models["Neural Network"].predict(X_test_scaled).flatten()
        )
        predictions["Random Forest"] = self.models["Random Forest"].predict(
            X_test_scaled
        )

        # Simple average ensemble
        predictions["Simple Average"] = (
            predictions["XGBoost"] + predictions["Neural Network"]
        ) / 2

        logger.info("Predictions completed for all models")
        return predictions

    def _evaluate_all_models(
        self, y_test: np.ndarray, predictions: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all models and return results."""
        logger.info("Evaluating all models...")

        results = {}
        for model_name, y_pred in predictions.items():
            results[model_name] = ModelEvaluator.evaluate_regression_model(
                y_test, y_pred, model_name
            )

        # Print comparison table
        results_df = ModelEvaluator.create_results_summary(results)
        print("\n=== PERFORMANCE COMPARISON ===")
        print(results_df.round(4).to_string(index=False))

        return results

    def _generate_analysis(
        self, X_test: np.ndarray, y_test: np.ndarray, predictions: Dict[str, np.ndarray]
    ) -> None:
        """Generate comprehensive analysis and visualizations."""
        logger.info("Generating analysis and visualizations...")

        # Model weight analysis for DML
        model_probs = self.models["DML"].get_model_weights(X_test)
        DMLVisualizer.print_model_selection_analysis(model_probs)
        DMLVisualizer.plot_model_weight_analysis(
            model_probs, save_path="dml_weight_analysis.png"
        )

        # Prediction comparison plots
        PerformanceVisualizer.plot_prediction_comparison(
            y_test,
            {
                "DML": predictions["DML"],
                "XGBoost": predictions["XGBoost"],
                "Neural Network": predictions["Neural Network"],
                "Simple Average": predictions["Simple Average"],
            },
            save_path="prediction_comparison.png",
        )

        logger.info("Analysis and visualizations completed")
        print("\nAnalysis complete! Visualizations saved.")


def run_dml_experiment(config: DMLConfig = None) -> Dict[str, Any]:
    """
    Run a complete DML experiment.

    Args:
        config: Optional DML configuration

    Returns:
        Experiment results dictionary
    """
    experiment = DMLExperiment(config)
    return experiment.run_comprehensive_experiment()


if __name__ == "__main__":
    # Set up random seeds for reproducibility
    import numpy as np
    import tensorflow as tf

    np.random.seed(42)
    tf.random.set_seed(42)

    # Run experiment with default configuration
    results = run_dml_experiment()

    print("\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
