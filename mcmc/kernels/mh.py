from abc import abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import math as tfm

from reggae.mcmc.results import GenericResults

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

    def one_step(self, current_state, previous_kernel_results, all_states):
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
    def _one_step(self, current_state, previous_kernel_results, all_states):
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



class KbarKernel(MetropolisKernel):
    def __init__(self, likelihood, prop_dist, prior_dist, num_genes, state_indices):
        self.prop_dist = prop_dist
        self.prior_dist = prior_dist
        self.num_genes = num_genes
        self.likelihood = likelihood
        self.state_indices = state_indices
        
    def one_step(self, current_state, previous_kernel_results, all_states):

        kbar = current_state
        kstar = tf.identity(kbar)
        old_probs = list()
        is_accepteds = list()
        for j in range(self.num_genes):
            sample = self.prop_dist(kstar[j]).sample()
#             sample = params.kbar.constrain(sample, j)
            kstar = tf.concat([kstar[:j], [sample], kstar[j+1:]], axis=0)
            
            new_prob = self.likelihood.genes(
                all_states,
                self.state_indices,
                kbar=kstar, 
            )[j] + tf.reduce_sum(self.prior_dist.log_prob(sample))
            
            old_prob = previous_kernel_results.target_log_prob[j] #old_m_likelihood[j] + sum(params.kbar.prior.log_prob(kbar[j]))

            is_accepted = self.metropolis_is_accepted(new_prob, old_prob)
            is_accepteds.append(is_accepted)
            
            prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)
            kstar = tf.cond(tf.equal(is_accepted, tf.constant(False)), 
                                     lambda:tf.concat([kstar[:j], [current_state[j]], kstar[j+1:]], axis=0), lambda:kstar)
            old_probs.append(prob)

        return kstar, GenericResults(old_probs, True) #TODO not just return true

    def bootstrap_results(self, init_state, all_states):
        probs = list()
        for j in range(self.num_genes):
            prob = self.likelihood.genes(
                all_states,
                self.state_indices,
                kbar=init_state, 
            )[j] + tf.reduce_sum(self.prior_dist.log_prob(init_state[j]))
            probs.append(prob)

        return GenericResults(probs, True) #TODO automatically adjust
    
    def is_calibrated(self):
        return True
