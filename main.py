"""
Main entry point for DML demonstration.

This script demonstrates the usage of the refactored DML implementation
with clean, modular architecture.
"""

import warnings

warnings.filterwarnings("ignore")

# Set up environment
import numpy as np
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

from src.config.model_config import DMLConfig
from src.models.dml_ensemble import AdaptiveXGBoostNeuralEnsemble
from src.utils.data_utils import DataLoader, ModelEvaluator
from src.utils.visualization import DMLVisualizer
from src.utils.logging_utils import get_project_logger

logger = get_project_logger(__name__)


def main():
    """Main execution function demonstrating DML usage."""
    print(
        "=== Dynamic Meta-Learning for Adaptive XGBoost-Neural Ensembles (Refactored) ===\n"
    )
    logger.info("Starting DML demonstration with refactored architecture")

    # Load and prepare data
    logger.info("Loading California housing dataset...")
    X, y, feature_names = DataLoader.load_california_housing()
    X_train, X_test, y_train, y_test = DataLoader.split_data(X, y, test_size=0.2)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples\n")

    # Create optimized configuration
    config = DMLConfig.get_default_config()
    logger.info("Using optimized DML configuration")

    # Initialize and train DML
    print("Training DML model...")
    logger.info("Initializing DML ensemble")

    dml = AdaptiveXGBoostNeuralEnsemble(config)
    dml.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    logger.info("Generating predictions on test set")
    y_pred = dml.predict(X_test)

    # Evaluate performance
    print("Evaluating performance...")
    results = ModelEvaluator.evaluate_regression_model(y_test, y_pred, "DML")

    print(f"\nDML Performance:")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  R²:   {results['r2']:.4f}")

    # Analyze model selection behavior
    print("\n=== Model Selection Analysis ===")
    model_probs = dml.get_model_weights(X_test)
    DMLVisualizer.print_model_selection_analysis(model_probs)

    # Generate visualizations
    print("\nGenerating visualizations...")
    DMLVisualizer.plot_model_weight_analysis(
        model_probs, save_path="dml_weight_analysis.png"
    )

    print("\n=== DML Demonstration Completed Successfully ===")
    print("Weight analysis plot saved as 'dml_weight_analysis.png'")

    logger.info("DML demonstration completed successfully")


if __name__ == "__main__":
    main()
