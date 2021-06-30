from .lfm import LFM
from .exact_lfm import ExactLFM
from .variational_lfm import VariationalLFM, TrainMode
from .ordinary_lfm import OrdinaryLFM
from .approximate_gp import MultiOutputGP, generate_multioutput_gp
from .operator import NeuralOperator
from .recurrent_operator import RecurrentNeuralOperator
from .neural_lfm import NeuralLFM


modules = [
    'LFM',
    'NeuralLFM',
    'ExactLFM',
    'VariationalLFM',
    'OrdinaryLFM',
    'NeuralOperator',
    'RecurrentNeuralOperator',
    'MultiOutputGP',
    'generate_multioutput_rbf_gp',
    'TrainMode',
]

try:
    from fenics import *
    from fenics_adjoint import *
    import torch_fenics
    fenics_present = True
except ImportError:
    fenics_present = False


if fenics_present:
    from .partial_lfm import PartialLFM
    modules.extend([
        'PartialLFM',
    ])


__all__ = modules
