from numpy import float64
from dataclasses import dataclass
import lafomo
import enum

@dataclass
class Params:
    latents:       object
    weights:       object
    kinetics:      object
    Δ:             object = None
    kernel_params: object = None
    σ2_m:          object = None
    σ2_f:          object = None

class ParamType(enum.Enum):
    HMC = 1
    MH = 2

class Parameter():
    def __init__(self,
                 name,
                 prior,
                 initial_value,
                 transform=None,
                 fixed=False):
        self.name = name
        self.prior = prior
        self.transform = (lambda x: x) if transform is None else transform
        self.value = initial_value
        self.fixed = fixed

    def propose(self, *args):
        if self.fixed:
            return self.value
        assert self.proposal_dist is not None, 'proposal_dist must not be None if you use propose()'
        return self.proposal_dist(*args).sample().numpy()


class MHParameter(Parameter):
    def __init__(self, name, prior, initial_value, proposal_dist=None, **kwargs):
        super().__init__(name, prior, initial_value, **kwargs)
        self.proposal_dist = proposal_dist
        self.param_type = ParamType.MH


class HMCParameter(Parameter):
    def __init__(self, name, prior, initial_value, **kwargs):
        super().__init__(name, prior, initial_value, **kwargs)
        self.param_type = ParamType.HMC
