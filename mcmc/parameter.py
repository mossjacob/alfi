from numpy import float64
from dataclasses import dataclass
import reggae

@dataclass
class Params:
    latents:       object
    weights:       object
    kinetics:      object
    Δ:             object = None
    kernel_params: object = None
    σ2_m:          object = None
    σ2_f:          object = None


class Parameter():
    def __init__(self, 
                 name, 
                 prior, 
                 initial_value, 
                 step_size=1., 
                 proposal_dist=None, 
                 constraint=None, 
                 fixed=False):
        self.name = name
        self.prior = prior
        self.step_size = step_size
        self.proposal_dist = proposal_dist
        if constraint is None:
            self.constrained = lambda x:x
        else:
            self.constrained = constraint
        self.value = initial_value
        self.fixed = fixed

    def constrain(self, *args):
        return self.constrained(*args)

    def propose(self, *args):
        if self.fixed:
            return self.value
        assert self.proposal_dist is not None, 'proposal_dist must not be None if you use propose()'
        return self.proposal_dist(*args).sample().numpy()

class KernelParameter(Parameter):
    def __init__(self,
                 name,
                 prior,
                 initial_value,
                 step_size=0.001,
                 requires_all_states=False, 
                 hmc_log_prob=None,
                 kernel=None):

        self.hmc_log_prob = hmc_log_prob
        self.requires_all_states = requires_all_states
        if hmc_log_prob is not None:
            self.kernel = reggae.mcmc.kernels.wrappers.NUTSWrapperKernel(hmc_log_prob, step_size=step_size)
        if kernel is not None:
            self.kernel = kernel
        super().__init__(name, prior, initial_value, step_size)