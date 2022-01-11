from .exact import ExactTrainer
from .variational import VariationalTrainer
from .preestimator import PreEstimator, PartialPreEstimator
from .operator import NeuralOperatorTrainer
try:  # fenics may not be present
    from .partial import PDETrainer
except ImportError:
    pass


__all__ = [
    'ExactTrainer',
    'VariationalTrainer',
    'PDETrainer',
    'PreEstimator',
    'PartialPreEstimator',
    'NeuralOperatorTrainer',
]
