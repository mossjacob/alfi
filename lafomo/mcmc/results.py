import collections
from numpy import float64 as f64
from dataclasses import dataclass
from lafomo.utilities.tf import inverse_positivity, logit
import numpy as np

GenericResults = collections.namedtuple('GenericResults', [
    'target_log_prob',
    'is_accepted',
    'acc_iter',
], defaults=[(f64(0), f64(0))])

MixedKernelResults = collections.namedtuple('MixedKernelResults', [
    'inner_results',
#     'grads_target_log_prob',
#     'step_size',
#     'log_accept_ratio',
    'is_accepted',
    'iteration',
])

@dataclass
class SampleResults:
    options: object
    fbar: object
    kbar: object
    k_fbar: object
    Δ: object
    kernel_params: object
    wbar: object
    w_0bar: object
    σ2_m: object
    σ2_f: object = None
    @property
    def f(self):
        return inverse_positivity(self.fbar).numpy()
    @property
    def k(self):
        ret = logit(self.kbar).numpy()
        if self.options.kinetic_exponential:
            return np.exp(ret)
        return ret
    @property
    def k_f(self):
        if self.k_fbar is None:
            return None
        return logit(self.k_fbar).numpy()
    @property
    def weights(self):
        return [(self.wbar), (self.w_0bar)]

@dataclass
class SampleResultsMH(SampleResults):
    @property
    def k(self):
        return np.exp(self.kbar)
    @property
    def k_f(self):
        return np.exp(self.k_fbar)
    @property
    def weights(self):
        return [self.wbar, self.w_0bar]
