from .trainer import Trainer
from .exact import ExactTrainer
from .variational import VariationalTrainer
try:  # fenics may not be present
    from .partial import PDETrainer
except ImportError:
    pass


__all__ = [
    'Trainer',
    'ExactTrainer',
    'VariationalTrainer',
    'PDETrainer'
]
