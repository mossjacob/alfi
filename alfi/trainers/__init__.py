import gpytorch

from torte import Trainer as TorteTrainer
from .exact import ExactTrainer
from .variational import VariationalTrainer
from .preestimator import PreEstimator, PartialPreEstimator
from .operator import NeuralOperatorTrainer
try:  # fenics may not be present
    from .partial import PDETrainer
except ImportError:
    pass


class Trainer(TorteTrainer):
    def print_extra(self):
        if isinstance(self.model, gpytorch.models.GP):
            kernel = self.model.covar_module
            print(f') λ: {str(kernel.lengthscale.view(-1).detach().numpy())}', end='')
        elif hasattr(self.model, 'gp_model'):
            print(f') kernel: {self.model.summarise_gp_hyp()}', end='')
        else:
            print(')', end='')


__all__ = [
    'Trainer',
    'ExactTrainer',
    'VariationalTrainer',
    'PDETrainer',
    'PreEstimator',
    'PartialPreEstimator',
    'NeuralOperatorTrainer',
]
