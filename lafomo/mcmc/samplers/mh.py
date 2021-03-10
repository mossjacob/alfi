from abc import abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import math as tfm

from lafomo.mcmc.results import GenericResults

import numpy as np
f64 = np.float64


class MetropolisKernel(tfp.mcmc.TransitionKernel):
    def __init__(self, step_size, tune_every=20):
        self.step_size = tf.Variable(step_size)
        self.tune_every = tune_every

    def metropolis_is_accepted(self, new_log_prob, old_log_prob):
        alpha = tf.math.exp(new_log_prob - old_log_prob)
        return tf.random.uniform((1,), dtype='float64') < tf.math.minimum(f64(1), alpha)
    #     if is_tensor(alpha):
    #         alpha = alpha.numpy()
    #     return not np.isnan(alpha) and random.random() < min(1, alpha)

    def one_step(self, current_state, previous_kernel_results):
        new_state, prob, is_accepted = self._one_step(current_state, previous_kernel_results, all_states)

        acc_rate, iteration = previous_kernel_results.acc_iter
        acc = acc_rate*iteration
        iteration += f64(1)
        acc_rate = tf.cond(tf.equal(is_accepted, tf.constant(True)), 
                        lambda: (acc+1)/iteration, lambda: acc/iteration)
        # tf.print(acc_rate, iteration)
        tf.cond(tf.equal(tfm.floormod(iteration, self.tune_every), 0), lambda: self.tune(acc_rate), lambda:None)

        return new_state, GenericResults(prob, is_accepted, (acc_rate, iteration)) # TODO for multiple TFs

    @abstractmethod
    def _one_step(self, current_state, previous_kernel_results):
        pass
    
    def tune(self, acc_rate):
        self.step_size.assign(tf.case([
                (acc_rate < 0.01, lambda: self.step_size * 0.1),
                (acc_rate < 0.1, lambda: self.step_size * 0.5),
                (acc_rate < 0.3, lambda: self.step_size * 0.9),
                (acc_rate > 0.95, lambda: self.step_size * 10.0),
                (acc_rate > 0.75, lambda: self.step_size * 2.0),
                (acc_rate > 0.5, lambda: self.step_size * 1.1),
            ],
            default=lambda:self.step_size
        ))
        tf.print('Updating step_size', self.step_size[0], 'due to acc rate', acc_rate)
    
    def is_calibrated(self):
        return True

