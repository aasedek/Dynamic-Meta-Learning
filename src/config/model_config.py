"""
Configuration classes for DML model parameters.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class NeuralNetworkConfig:
    """Configuration for Neural Network component."""

    hidden_units: List[int] = None
    dropout_rate: float = 0.3
    epochs: int = 100
    batch_size: int = 32
    mc_dropout_samples: int = 100

    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [128, 64, 32]


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost component."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6


@dataclass
class MetaLearnerConfig:
    """Configuration for Meta-learner component."""

    hidden_units: List[int] = None
    epochs: int = 50
    learning_rate: float = 0.001

    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [64, 32]


@dataclass
class EnsembleConfig:
    """Configuration for the overall DML ensemble."""

    lambda_importance: float = 0.5
    alpha_regularization: float = 0.1
    use_cross_validation: bool = False
    cv_folds: int = 3
    validation_split: float = 0.2
    random_seed: int = 42


@dataclass
class DMLConfig:
    """Complete configuration for DML model."""

    neural_network: NeuralNetworkConfig = None
    xgboost: XGBoostConfig = None
    meta_learner: MetaLearnerConfig = None
    ensemble: EnsembleConfig = None

    def __post_init__(self):
        if self.neural_network is None:
            self.neural_network = NeuralNetworkConfig()
        if self.xgboost is None:
            self.xgboost = XGBoostConfig()
        if self.meta_learner is None:
            self.meta_learner = MetaLearnerConfig()
        if self.ensemble is None:
            self.ensemble = EnsembleConfig()

    @classmethod
    def get_default_config(cls) -> "DMLConfig":
        """Get default optimized configuration."""
        return cls(
            neural_network=NeuralNetworkConfig(
                hidden_units=[128, 64, 32],
                dropout_rate=0.3,
                epochs=150,
                batch_size=64,
                mc_dropout_samples=100,
            ),
            xgboost=XGBoostConfig(n_estimators=150, learning_rate=0.08, max_depth=8),
            meta_learner=MetaLearnerConfig(
                hidden_units=[128, 64], epochs=100, learning_rate=0.001
            ),
            ensemble=EnsembleConfig(
                lambda_importance=0.5,
                alpha_regularization=0.05,
                use_cross_validation=False,
                cv_folds=5,
                validation_split=0.2,
                random_seed=42,
            ),
        )
