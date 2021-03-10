import tensorflow as tf
import tensorflow_probability as tfp

from lafomo.mcmc.results import MixedKernelResults
from lafomo.utilities.tf import prog
from lafomo.mcmc.parameter import ParamType

from inspect import signature
from datetime import datetime


class MixedSampler(tfp.mcmc.TransitionKernel):
    def __init__(self, subsamplers, T, skip=None):
        """

        Parameters:
            send_all_states: a boolean array of size |samplers| indicating which components of the state
                            have samplers whose log probability depends on the state of others, in which case MixedKernel
                            will recompute the previous target_log_prob before handing it over in the `one_step` call.
            T: number of samples for logging purposes only
            skip: a boolean array of size |samplers| indicating whether a kernel should be ignored. For debugging.
                        @param data: a tuple (m, f) of shapes (reps, num, time), where time is a tuple (t, τ, common_indices)
            @param options: configuration of model

        """

        self.T = T
        self.subsamplers = subsamplers
        self.num_samplers = len(subsamplers)
        # self.one_step_receives_state = [len(signature(k.one_step).parameters) > 2 for k in samplers]
        self.skip = skip
        self.samples = None

        super().__init__()


    def sample(self, T=2000, store_every=10, burn_in=1000, report_every=100, skip=None, num_chains=4,
               profile=False):
        print('----- Sampling Begins -----')
        current_state = [param.value for param in self.active_params]

        def trace_fn(a, previous_kernel_results):
            return previous_kernel_results.is_accepted

        # Run the chain (with burn-in).
        @tf.function
        def run_chain():
            # Run the chain (with burn-in).
            samples, is_accepted = tfp.mcmc.sample_chain(
                num_results=T,
                num_burnin_steps=burn_in,
                current_state=current_state,
                kernel=self,
                trace_fn=trace_fn)

            return samples, is_accepted

        if profile:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = '.\\logs\\lafomo\\%s' % stamp
            tf.profiler.experimental.start(logdir)
        samples, is_accepted = run_chain()
        if profile:
            tf.profiler.experimental.stop()

        add_to_previous = (self.samples is not None)
        for param in self.active_params:
            index = self.state_indices[param.name]
            param_samples = samples[index]
            if type(param_samples) is list:
                if add_to_previous:
                    for i in range(len(param_samples)):
                        self.samples[index][i] = tf.concat([self.samples[index][i], samples[index][i]], axis=0)
                param_samples = [[param_samples[i][-1] for i in range(len(param_samples))]]
            else:
                if add_to_previous:
                    self.samples[index] = tf.concat([self.samples[index], samples[index]], axis=0)
            param.value = param_samples[-1]

        if not add_to_previous:
            self.samples = samples
        self.is_accepted = is_accepted
        print()
        print('----- Finished -----')
        return samples, is_accepted

    def one_step(self, current_state, previous_kernel_results):
        # tf.print('running iteration')
        # if previous_kernel_results.iteration % 10:
        prog(self.T, previous_kernel_results.iteration)
        # tf.print('running', current_state)
        new_state = list()
        is_accepted = list()
        inner_results = list()

        for i in range(self.num_samplers):
            if self.skip is not None and self.skip[i]:
                # print(previous_kernel_results)
                is_accepted.append(previous_kernel_results.inner_results[i].is_accepted)
                new_state.append(current_state[i])
                inner_results.append(previous_kernel_results.inner_results[i])
                continue

            # tgt_prob = self.subsamplers[i].target_log_prob_fn_fn(current_state)(*wrapped_state_i)
            # if hasattr(previous_kernel_results.inner_results[i], 'accepted_results'):
            #
            #     propres = previous_kernel_results.inner_results[i].accepted_results._replace(
            #         target_log_prob=tgt_prob)
            #     previous_kernel_results.inner_results[i] = previous_kernel_results.inner_results[i]._replace(
            #         proposed_results=propres)
            # else:
            #     previous_kernel_results.inner_results[i] = previous_kernel_results.inner_results[i]._replace(
            #         target_log_prob=tgt_prob)

            try:
                # state_chained = tf.expand_dims(current_state[i], 0)
                # print(state_chained)
                result_state, kernel_results = self.subsamplers[i].one_step(
                    current_state[i], previous_kernel_results.inner_results[i])
            except Exception as e:
                tf.print('Failed at ', i, self.kernels[i], current_state)
                raise e

            if type(result_state) is list:
                new_state.append([tf.identity(res) for res in result_state])
            else:
                new_state.append(result_state)

            is_accepted.append(kernel_results.is_accepted)
            inner_results.append(kernel_results)

        return new_state, MixedKernelResults(inner_results, is_accepted, previous_kernel_results.iteration + 1)

    def bootstrap_results(self, init_state):
        """
        """
        inner_kernels_bootstraps = list()
        is_accepted = list()
        for i in range(self.num_kernels):
            self.kernels[i].all_states_hack = init_state

            if self.one_step_receives_state[i]:
                results = self.kernels[i].bootstrap_results(init_state[i], init_state)
                inner_kernels_bootstraps.append(results)

            else:
                results = self.kernels[i].bootstrap_results(init_state[i])
                inner_kernels_bootstraps.append(results)

            is_accepted.append(True)

        return MixedKernelResults(inner_kernels_bootstraps, is_accepted, 0)

    def is_calibrated(self):
        return True
