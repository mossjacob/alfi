import tensorflow as tf
from tensorflow import math as tfm
import tensorflow_probability as tfp

from alfi.mcmc.results import GenericResults
from .mixins import ParamGroupMixin
import numpy as np


class DelaySampler(tfp.mcmc.TransitionKernel, ParamGroupMixin):
    """
    Parameters:
        param: the associated Parameter (only one, not a list)
    """
    def __init__(self, likelihood, param, lower, upper, start_iteration=1):
        super().__init__()
        self.param_group = [param]
        self.likelihood = likelihood
        self.lower = lower
        self.upper = upper
        self.prior = param.prior
        self.start_iteration = start_iteration
        
    def one_step(self, current_state, previous_kernel_results, all_states):
        iteration_number = previous_kernel_results.target_log_prob[0] #just roll with it

        def proceed():
            num_tfs = current_state.shape[0]
            new_state = current_state
            Δrange = np.arange(self.lower, self.upper+1, dtype='float64')
            Δrange_tf = tf.range(self.lower, self.upper+1, dtype='float64')
            for i in range(num_tfs):
                # Generate normalised cumulative distribution
                probs = list()
                mask = np.zeros((num_tfs, ), dtype='float64')
                mask[i] = 1
                
                for Δ in Δrange:
                    test_state = (1-mask) * new_state + mask * Δ

                    probs.append(tf.reduce_sum(self.likelihood(
                        Δ=test_state,
                    )) + tf.reduce_sum(self.prior.log_prob(Δ)))
                # curri = tf.cast(current_state[i], 'int64')
                # start_index = tf.reduce_max([self.lower, curri-2])
                # probs = tf.gather(probs, tf.range(start_index, 
                #                                   tf.reduce_min([self.upper+1, curri+3])))

                probs =  tf.stack(probs) - tfm.reduce_max(probs)
                probs = tfm.exp(probs)
                probs = probs / tfm.reduce_sum(probs)
                cumsum = tfm.cumsum(probs)
                u = tf.random.uniform([], dtype='float64')
                index = tf.where(cumsum == tf.reduce_min(cumsum[(cumsum - u) > 0]))
                chosen = Δrange_tf[index[0][0]]
                new_state = (1-mask) * new_state + mask * chosen
            return new_state
#         tf.print('final chosen state', new_state)
        new_state = tf.cond(iteration_number < self.start_iteration, lambda: current_state, lambda: proceed())
        return new_state, GenericResults([iteration_number+1], True)

    def bootstrap_results(self, init_state, all_states):

        return GenericResults([0], True)
    
    def is_calibrated(self):
        return True

