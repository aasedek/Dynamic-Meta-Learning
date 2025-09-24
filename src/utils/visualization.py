"""
Visualization utilities for DML analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DMLVisualizer:
    """
    Handles visualization and analysis for DML ensemble.
    """

    @staticmethod
    def plot_model_weight_analysis(
        model_probs: Dict[str, np.ndarray], save_path: str = "dml_weight_analysis.png"
    ) -> None:
        """
        Plot comprehensive model weight analysis.

        Args:
            model_probs: Dictionary containing model probabilities
            save_path: Path to save the plot
        """
        logger.info("Creating model weight analysis plots...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Probability distributions
        axes[0].hist(
            model_probs["xgb_probs"], bins=30, alpha=0.7, label="XGBoost", color="blue"
        )
        axes[0].hist(
            model_probs["nn_probs"],
            bins=30,
            alpha=0.7,
            label="Neural Network",
            color="orange",
        )
        axes[0].hist(
            model_probs["hybrid_probs"],
            bins=30,
            alpha=0.7,
            label="Hybrid",
            color="green",
        )
        axes[0].set_xlabel("Probability")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution of Model Selection Probabilities")
        axes[0].legend()

        # Plot 2: Probability space
        scatter = axes[1].scatter(
            model_probs["xgb_probs"],
            model_probs["nn_probs"],
            c=model_probs["hybrid_probs"],
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, ax=axes[1], label="Hybrid Probability")
        axes[1].set_xlabel("XGBoost Probability")
        axes[1].set_ylabel("Neural Network Probability")
        axes[1].set_title("Model Selection Probability Space")

        # Plot 3: Dominant model counts
        dominant_model = np.argmax(
            [
                model_probs["xgb_probs"],
                model_probs["nn_probs"],
                model_probs["hybrid_probs"],
            ],
            axis=0,
        )

        model_names = ["XGBoost", "Neural Network", "Hybrid"]
        unique, counts = np.unique(dominant_model, return_counts=True)

        axes[2].bar([model_names[i] for i in unique], counts)
        axes[2].set_title("Dominant Model Selection Count")
        axes[2].set_ylabel("Number of Predictions")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"Model weight analysis plot saved as '{save_path}'")

    @staticmethod
    def print_model_selection_analysis(model_probs: Dict[str, np.ndarray]) -> None:
        """
        Print detailed model selection analysis.

        Args:
            model_probs: Dictionary containing model probabilities
        """
        print("\n=== MODEL SELECTION ANALYSIS ===")

        # Average probabilities
        avg_xgb = np.mean(model_probs["xgb_probs"])
        avg_nn = np.mean(model_probs["nn_probs"])
        avg_hybrid = np.mean(model_probs["hybrid_probs"])

        print(f"Average XGBoost probability: {avg_xgb:.3f}")
        print(f"Average Neural Network probability: {avg_nn:.3f}")
        print(f"Average Hybrid probability: {avg_hybrid:.3f}")

        # Standard deviations
        std_xgb = np.std(model_probs["xgb_probs"])
        std_nn = np.std(model_probs["nn_probs"])
        std_hybrid = np.std(model_probs["hybrid_probs"])

        print(f"XGBoost probability std: {std_xgb:.3f}")
        print(f"Neural Network probability std: {std_nn:.3f}")
        print(f"Hybrid probability std: {std_hybrid:.3f}")

        # Dominant model analysis
        dominant_model = np.argmax(
            [
                model_probs["xgb_probs"],
                model_probs["nn_probs"],
                model_probs["hybrid_probs"],
            ],
            axis=0,
        )

        model_names = ["XGBoost", "Neural Network", "Hybrid"]
        unique, counts = np.unique(dominant_model, return_counts=True)
        total_predictions = len(model_probs["xgb_probs"])

        print(f"\nDominant model distribution:")
        for i, count in zip(unique, counts):
            percentage = (count / total_predictions) * 100
            print(f"  {model_names[i]}: {count} predictions ({percentage:.1f}%)")

        logger.info("Model selection analysis completed")


class PerformanceVisualizer:
    """
    Handles performance visualization and comparison.
    """

    @staticmethod
    def plot_training_history(
        history: Any, save_path: str = "training_history.png"
    ) -> None:
        """
        Plot neural network training history.

        Args:
            history: Keras training history
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(history.history["loss"], label="Training Loss")
        if "val_loss" in history.history:
            ax1.plot(history.history["val_loss"], label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # MAE plot (if available)
        if "mae" in history.history:
            ax2.plot(history.history["mae"], label="Training MAE")
            if "val_mae" in history.history:
                ax2.plot(history.history["val_mae"], label="Validation MAE")
            ax2.set_title("Model MAE")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("MAE")
            ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"Training history plot saved as '{save_path}'")

    @staticmethod
    def plot_prediction_comparison(
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        save_path: str = "prediction_comparison.png",
    ) -> None:
        """
        Plot prediction vs actual comparison for multiple models.

        Args:
            y_true: True target values
            predictions_dict: Dictionary mapping model names to predictions
            save_path: Path to save the plot
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
            axes[idx].scatter(y_true, y_pred, alpha=0.6)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

            axes[idx].set_xlabel("Actual Values")
            axes[idx].set_ylabel("Predicted Values")
            axes[idx].set_title(f"{model_name} Predictions")
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"Prediction comparison plot saved as '{save_path}'")
