"""
DML (Dynamic Meta-Learning for Adaptive XGBoost-Neural Ensembles) Package

A modular implementation of the Dynamic Meta-Learning for Adaptive XGBoost-Neural Ensembles
framework for enhanced predictive modeling.
"""

from .models.dml_ensemble import AdaptiveXGBoostNeuralEnsemble
from .config.model_config import DMLConfig
from .utils.data_utils import DataLoader, ModelEvaluator
from .utils.visualization import DMLVisualizer, PerformanceVisualizer

__version__ = "1.0.0"
__author__ = "Implementation based on Arthur Sedek's paper"

__all__ = [
    "AdaptiveXGBoostNeuralEnsemble",
    "DMLConfig",
    "DataLoader",
    "ModelEvaluator",
    "DMLVisualizer",
    "PerformanceVisualizer",
]
