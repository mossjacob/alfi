from .trainer import Trainer
from .exact import ExactTrainer
from .variational import VariationalTrainer
from .preestimator import ParameterPreEstimator
try:  # fenics may not be present
    from .partial import PDETrainer
except ImportError:
    pass


__all__ = [
    'Trainer',
    'ExactTrainer',
    'VariationalTrainer',
    'PDETrainer',
    'ParameterPreEstimator'
]
