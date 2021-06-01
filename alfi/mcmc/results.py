import collections
from numpy import float64 as f64

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
