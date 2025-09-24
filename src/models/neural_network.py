"""
Neural Network component for DML ensemble.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List
import logging

from ..config.model_config import NeuralNetworkConfig

logger = logging.getLogger(__name__)


class MonteCarloDropoutNetwork:
    """
    Neural Network with Monte Carlo Dropout for uncertainty estimation.
    """

    def __init__(self, config: NeuralNetworkConfig):
        """
        Initialize the neural network.

        Args:
            config: Neural network configuration
        """
        self.config = config
        self.model = None
        self._is_trained = False

        logger.info(f"Initializing Neural Network with config: {config}")

    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network with Monte Carlo Dropout.

        Args:
            input_dim: Input dimension

        Returns:
            Compiled Keras model
        """
        logger.info(f"Building neural network with input dimension: {input_dim}")
        logger.info(f"Architecture: {self.config.hidden_units} + output layer")

        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        for i, units in enumerate(self.config.hidden_units):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.config.dropout_rate)(
                x, training=True
            )  # Always enabled for MC Dropout
            logger.debug(
                f"Added dense layer {i+1}: {units} units with {self.config.dropout_rate} dropout"
            )

        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        logger.info("Neural network built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")

        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        verbose: int = 0,
    ) -> keras.callbacks.History:
        """
        Train the neural network.

        Args:
            X: Training features
            y: Training targets
            validation_split: Validation split ratio
            verbose: Verbosity level

        Returns:
            Training history
        """
        if self.model is None:
            self.model = self.build_model(X.shape[1])

        logger.info("Training Neural Network...")

        history = self.model.fit(
            X,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )

        self._is_trained = True
        final_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]

        logger.info(f"Neural Network training completed")
        logger.info(
            f"Final training loss: {final_loss:.4f}, validation loss: {final_val_loss:.4f}"
        )

        return history

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and uncertainty estimates using Monte Carlo Dropout.

        Args:
            X: Input features

        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")

        logger.debug(
            f"Calculating NN predictions with MC Dropout for {X.shape[0]} samples"
        )
        logger.debug(f"Using {self.config.mc_dropout_samples} MC dropout samples")

        predictions = []
        for i in range(self.config.mc_dropout_samples):
            pred = self.model(X, training=True)  # Keep dropout enabled
            predictions.append(pred.numpy().flatten())

            if (i + 1) % 20 == 0:
                logger.debug(
                    f"Completed {i + 1}/{self.config.mc_dropout_samples} MC dropout samples"
                )

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.var(predictions, axis=0)  # Variance as uncertainty metric

        logger.debug(
            f"NN prediction range: [{mean_pred.min():.4f}, {mean_pred.max():.4f}]"
        )
        logger.debug(
            f"NN uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]"
        )

        return mean_pred, uncertainty

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get mean predictions (without uncertainty).

        Args:
            X: Input features

        Returns:
            Mean predictions
        """
        mean_pred, _ = self.predict_with_uncertainty(X)
        return mean_pred

    def get_feature_importance(self, X: np.ndarray, steps: int = 50) -> np.ndarray:
        """
        Get feature importance using Integrated Gradients.

        Args:
            X: Input features
            steps: Number of integration steps

        Returns:
            Feature importance scores
        """
        if not self._is_trained:
            raise ValueError(
                "Model must be trained before computing feature importance"
            )

        logger.debug(
            "Computing neural network feature importance using Integrated Gradients"
        )

        # Baseline: zeros (could also use dataset mean)
        baseline = np.zeros_like(X)

        # Compute Integrated Gradients for each sample
        integrated_grads = []

        for i in range(X.shape[0]):
            x = X[i : i + 1]  # Single sample
            baseline_x = baseline[i : i + 1]

            # Path from baseline to input
            alphas = np.linspace(0, 1, steps)
            path_inputs = []

            for alpha in alphas:
                path_input = baseline_x + alpha * (x - baseline_x)
                path_inputs.append(path_input)

            path_inputs = np.concatenate(path_inputs, axis=0)

            # Calculate gradients along the path
            with tf.GradientTape() as tape:
                inputs_tensor = tf.Variable(path_inputs.astype(np.float32))
                tape.watch(inputs_tensor)
                predictions = self.model(inputs_tensor)
                loss = tf.reduce_mean(predictions)

            gradients = tape.gradient(loss, inputs_tensor).numpy()

            # Integrate gradients (trapezoidal rule approximation)
            avg_gradients = np.mean(gradients, axis=0)

            # Multiply by (x - baseline) to get integrated gradients
            integrated_grad = avg_gradients * (x - baseline_x).flatten()
            integrated_grads.append(integrated_grad)

        # Average across all samples and take absolute values
        integrated_grads = np.array(integrated_grads)
        importance = np.mean(np.abs(integrated_grads), axis=0)

        # Normalize importance scores
        return importance / np.sum(importance) if np.sum(importance) > 0 else importance

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
