"""
Meta-learner component for DML ensemble.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List
import logging

from ..config.model_config import MetaLearnerConfig

logger = logging.getLogger(__name__)


class MetaLearner:
    """
    Meta-learner for adaptive model selection in DML ensemble.
    """

    def __init__(self, config: MetaLearnerConfig):
        """
        Initialize meta-learner.

        Args:
            config: Meta-learner configuration
        """
        self.config = config
        self.model = None
        self._is_trained = False

        logger.info(f"Initializing Meta-learner with config: {config}")

    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build meta-learner neural network.

        Args:
            input_dim: Input dimension

        Returns:
            Compiled Keras model
        """
        logger.info(f"Building meta-learner with input dimension: {input_dim}")
        logger.info(
            f"Meta-learner architecture: {self.config.hidden_units} + 3 output units"
        )

        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        for i, units in enumerate(self.config.hidden_units):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            logger.debug(
                f"Added meta-learner layer {i+1}: {units} units with 0.2 dropout"
            )

        # Output 3 probabilities: XGBoost only, NN only, or Hybrid (average)
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        logger.info("Meta-learner built successfully")
        logger.info(f"Meta-learner parameters: {model.count_params():,}")

        return model

    def fit(
        self,
        meta_features: np.ndarray,
        xgb_predictions: np.ndarray,
        nn_predictions: np.ndarray,
        y_true: np.ndarray,
        alpha_regularization: float = 0.1,
    ) -> None:
        """
        Train the meta-learner.

        Args:
            meta_features: Meta-features for training
            xgb_predictions: XGBoost predictions
            nn_predictions: Neural network predictions
            y_true: True target values
            alpha_regularization: KL divergence regularization weight
        """
        logger.info("Training meta-learner...")

        if self.model is None:
            self.model = self.build_model(meta_features.shape[1])

        # Create optimal labels based on individual model performance
        optimal_labels = self._create_optimal_labels(
            xgb_predictions, nn_predictions, y_true
        )

        # Custom training loop with KL regularization
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        hybrid_pred = (xgb_predictions + nn_predictions) / 2

        logger.info("Starting meta-learner training with custom loop...")

        for epoch in range(self.config.epochs):
            with tf.GradientTape() as tape:
                logits = self.model(meta_features)

                # Final predictions using meta-learner probabilities
                final_pred = (
                    logits[:, 0:1] * xgb_predictions.reshape(-1, 1)
                    + logits[:, 1:2] * nn_predictions.reshape(-1, 1)
                    + logits[:, 2:3] * hybrid_pred.reshape(-1, 1)
                )
                final_pred = tf.squeeze(final_pred)

                # Main loss (MSE for regression)
                main_loss = tf.reduce_mean(tf.square(y_true - final_pred))

                # Classification loss for optimal model selection
                classification_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        optimal_labels, logits, from_logits=False
                    )
                )

                # KL divergence regularization
                uniform_dist = tf.ones_like(logits) / logits.shape[1]
                kl_loss = tf.reduce_mean(
                    tf.keras.losses.kullback_leibler_divergence(uniform_dist, logits)
                )

                total_loss = (
                    main_loss
                    + 0.1 * classification_loss
                    + alpha_regularization * kl_loss
                )

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            if epoch % 10 == 0:
                logger.info(
                    f"Meta-learner Epoch {epoch}: Total Loss = {total_loss:.4f} "
                    f"(Main: {main_loss:.4f}, Classification: {classification_loss:.4f}, "
                    f"KL: {kl_loss:.4f})"
                )

        self._is_trained = True
        logger.info("Meta-learner training completed")

    def predict_probabilities(self, meta_features: np.ndarray) -> np.ndarray:
        """
        Predict model selection probabilities.

        Args:
            meta_features: Meta-features for prediction

        Returns:
            Model selection probabilities [XGBoost, NN, Hybrid]
        """
        if not self._is_trained:
            raise ValueError("Meta-learner must be trained before making predictions")

        return self.model(meta_features).numpy()

    def _create_optimal_labels(
        self,
        xgb_predictions: np.ndarray,
        nn_predictions: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """
        Create optimal labels for meta-learner training.

        Args:
            xgb_predictions: XGBoost predictions
            nn_predictions: Neural network predictions
            y_true: True target values

        Returns:
            Optimal model labels (0=XGBoost, 1=NN, 2=Hybrid)
        """
        optimal_labels = []
        hybrid_pred = (xgb_predictions + nn_predictions) / 2

        for i in range(len(y_true)):
            xgb_error = abs(y_true[i] - xgb_predictions[i])
            nn_error = abs(y_true[i] - nn_predictions[i])
            hybrid_error = abs(y_true[i] - hybrid_pred[i])

            best_model = np.argmin([xgb_error, nn_error, hybrid_error])
            optimal_labels.append(best_model)

        return np.array(optimal_labels)

    @property
    def is_trained(self) -> bool:
        """Check if meta-learner is trained."""
        return self._is_trained
