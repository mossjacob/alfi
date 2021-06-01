import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from alfi.mcmc.results import GenericResults
from .mixins import ParamGroupMixin

class GibbsSampler(tfp.mcmc.TransitionKernel, ParamGroupMixin):
    
    def __init__(self, param, sq_diff_fn, N):
        super().__init__()
        self.param_group = [param]
        self.prior = param.prior
        self.sq_diff_fn = sq_diff_fn
        self.N = N

    @tf.function
    def one_step(self, current_state, previous_kernel_results):
        # Prior parameters
        α = self.prior.concentration
        β = self.prior.scale
        # Conditional posterior of inv gamma parameters:
        sq_diff = self.sq_diff_fn()
        α_post = α + 0.5*self.N
        β_post = β + 0.5*tf.reduce_sum(sq_diff, axis=1)
        # print(α.shape, sq_diff.shape)
        # print('val', β_post.shape, params.σ2_m.value)
        new_state = tfd.InverseGamma(α_post, β_post).sample()
        new_state = tf.reshape(new_state, (sq_diff.shape[0], 1))
        return [new_state], GenericResults(list(), True)

    def bootstrap_results(self, init_state):
        return GenericResults(list(), True) 
    
    def is_calibrated(self):
        return True

