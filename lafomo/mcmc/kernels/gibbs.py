import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from lafomo.mcmc.results import GenericResults


class GibbsKernel(tfp.mcmc.TransitionKernel):
    
    def __init__(self, data, options, likelihood, prior, state_indices, sq_diff_fn):
        self.data = data
        self.options = options
        self.likelihood = likelihood
        self.prior = prior
        self.state_indices = state_indices
        self.sq_diff_fn = sq_diff_fn
        self.N_p = data.t_discretised.shape[0]

    @tf.function
    def one_step(self, current_state, previous_kernel_results, all_states):
        # Prior parameters
        α = self.prior.concentration
        β = self.prior.scale
        # Conditional posterior of inv gamma parameters:
        sq_diff = self.sq_diff_fn(all_states)
        α_post = α + 0.5*self.N_p
        β_post = β + 0.5*tf.reduce_sum(sq_diff, axis=1)
        # print(α.shape, sq_diff.shape)
        # print('val', β_post.shape, params.σ2_m.value)
        new_state = tfd.InverseGamma(α_post, β_post).sample()
        new_state = tf.reshape(new_state, (sq_diff.shape[0], 1))
        return new_state, GenericResults(list(), True)

    def bootstrap_results(self, init_state, all_states):
        return GenericResults(list(), True) 
    
    def is_calibrated(self):
        return True

