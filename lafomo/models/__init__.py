from .lfm import LFM
from .exact_lfm import ExactLFM
from .variational_lfm import VariationalLFM
from .ordinary_lfm import OrdinaryLFM
from .approximate_gp import MultiOutputGP


modules = [
    'LFM',
    'ExactLFM',
    'VariationalLFM',
    'OrdinaryLFM',
    'MultiOutputGP',
]

try:
    from fenics import *
    from fenics_adjoint import *
    import torch_fenics
    fenics_present = True
except ImportError:
    fenics_present = False


if fenics_present:
    from .partial_lfm import PartialLFM, ReactionDiffusion
    modules.extend([
        'PartialLFM',
        'ReactionDiffusion'
    ])


__all__ = modules
