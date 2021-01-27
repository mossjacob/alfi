import numpy as np
from reggae.tf_utilities import exp, ArrayList, discretise
import tensorflow_probability as tfp

from ipywidgets import IntProgress
from tensorflow import is_tensor
import tensorflow as tf
import random
from IPython.display import display
f64 = np.float64


class MetropolisHastings():
    def __init__(self, params):
        self.params = params
        self.clear_samples()

    def clear_samples(self):
        self.samples = {param.name: ArrayList(param.value.shape) for param in self.params}
        self.samples['acc_rates'] = {param.name: ArrayList((1,)) for param in self.params}

    def sample(self, T=20000, store_every=10, burn_in=1000, report_every=100, tune_every=50):
        print('----- Metropolis Begins -----')
        
        self.acceptance_rates = {param.name: 0. for param in self.params} # Reset acceptance rates
        f = IntProgress(description='Running', min=0, max=T) # instantiate the bar
        display(f)
        for iteration_number in range(T):
            if iteration_number % report_every == 0:
                f.value = iteration_number 
            if iteration_number >= 1 and iteration_number % tune_every == 0:
                for param in self.params:
                    acc = self.acceptance_rates[param.name]/iteration_number
                    param.step_size = self.tune(param.step_size, acc)
                    #print(f'Updating {param.name} to {param.step_size} due to acceptance rate {acc}')

            self.iterate()
    
            if iteration_number >= burn_in and iteration_number % store_every == 0:
                # for j in range(num_genes):
                for param in self.params:
                    if param.value.ndim > 1:
                        if is_tensor(param.value):
                            self.samples[param.name].add(param.value.numpy().copy())
                        else:
                            self.samples[param.name].add(param.value.copy())
                    else:
                        self.samples[param.name].add(param.value)
                    acc = self.acceptance_rates[param.name]/(iteration_number if iteration_number > 0 else 1)
                    self.samples['acc_rates'][param.name].add(acc)

        # for key in self.acceptance_rates:
        #     self.acceptance_rates[key] /= T
        # rates = np.array(self.samples['acc_rates']).T/np.arange(1, T-burn_in+1, store_every)
        # self.samples['acc_rates'] = rates
        f.value = T
        print('----- Finished -----')

    def iterate(self):
        raise NotImplementedError('iterate() must be implemented')
        
    '''MH accept function'''
    def is_accepted(self, new_log_prob, old_log_prob):
        alpha = exp(new_log_prob - old_log_prob)
        if is_tensor(alpha):
            alpha = alpha.numpy()
        return not np.isnan(alpha) and random.random() < min(1, alpha)

    def tune(self, scale, acc_rate):
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate over the last tune_interval:
        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        """
        if acc_rate < 0.001:
            return scale * 0.1
        elif acc_rate < 0.05:
            return scale * 0.5
        elif acc_rate < 0.2:
            return scale * 0.9
        elif acc_rate > 0.95:
            return scale * 10.0
        elif acc_rate > 0.75:
            return scale * 2.0
        elif acc_rate > 0.5:
            return scale * 1.1

        return scale
