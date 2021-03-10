import tensorflow_probability as tfp
import tensorflow as tf
from .mixins import ParamGroupMixin


class HMCSampler(tfp.mcmc.NoUTurnSampler, ParamGroupMixin):

    def __init__(self, likelihood_fn, params_list, step_size):
        self.likelihood_fn = likelihood_fn
        self.param_group = params_list
        self.transforms = [param.transform for param in params_list]
        self.priors = [param.prior for param in params_list]
        super().__init__(self.log_prob, step_size)

    def log_prob(self, *args):
        print('log_prob hmc sampler args', *args)
        new_prob = 0

        param_kwargs = {}
        for i, param in enumerate(self.param_group):
            val = self.transforms[i](args[i])

            # Prepare likelihood params
            param_kwargs[param.name] = val

            # Add prior:
            new_prob += tf.reduce_sum(self.priors[i].log_prob(val))

        # Add likelihood:
        new_prob += tf.reduce_sum(self.likelihood_fn(**param_kwargs))
        return new_prob