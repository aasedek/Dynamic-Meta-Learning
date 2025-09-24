"""
Example configurations for different use cases.
"""

from src.config.model_config import (
    DMLConfig,
    NeuralNetworkConfig,
    XGBoostConfig,
    MetaLearnerConfig,
    EnsembleConfig,
)


class ConfigPresets:
    """Predefined configurations for different scenarios."""

    @staticmethod
    def get_quick_experiment_config() -> DMLConfig:
        """Configuration for quick experimentation (reduced training time)."""
        return DMLConfig(
            neural_network=NeuralNetworkConfig(
                hidden_units=[64, 32],
                dropout_rate=0.3,
                epochs=50,
                batch_size=32,
                mc_dropout_samples=50,
            ),
            xgboost=XGBoostConfig(n_estimators=50, learning_rate=0.1, max_depth=6),
            meta_learner=MetaLearnerConfig(hidden_units=[32, 16], epochs=30),
            ensemble=EnsembleConfig(
                lambda_importance=0.5,
                alpha_regularization=0.1,
                validation_split=0.2,
                random_seed=42,
            ),
        )

    @staticmethod
    def get_high_performance_config() -> DMLConfig:
        """Configuration optimized for maximum performance."""
        return DMLConfig(
            neural_network=NeuralNetworkConfig(
                hidden_units=[256, 128, 64, 32],
                dropout_rate=0.2,
                epochs=200,
                batch_size=128,
                mc_dropout_samples=150,
            ),
            xgboost=XGBoostConfig(n_estimators=200, learning_rate=0.05, max_depth=10),
            meta_learner=MetaLearnerConfig(
                hidden_units=[256, 128, 64], epochs=150, learning_rate=0.0005
            ),
            ensemble=EnsembleConfig(
                lambda_importance=0.4,
                alpha_regularization=0.02,
                validation_split=0.15,
                random_seed=42,
            ),
        )

    @staticmethod
    def get_balanced_config() -> DMLConfig:
        """Balanced configuration (default)."""
        return DMLConfig.get_default_config()
