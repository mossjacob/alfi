import tensorflow as tf
import tensorflow_probability as tfp

from lafomo.mcmc.results import MixedKernelResults
from lafomo.utilities.tf import prog

from inspect import signature


class MixedKernel(tfp.mcmc.TransitionKernel):
    def __init__(self, kernels, send_all_states, T, skip=None):
        """

        @param send_all_states: a boolean array of size |kernels| indicating which components of the state
        have kernels whose log probability depends on the state of others, in which case MixedKernel
        will recompute the previous target_log_prob before handing it over in the `one_step` call.
        @param T: number of samples for logging purposes only
        @param skip: a boolean array of size |kernels| indicating whether a kernel should be ignored. For debugging.
        """
        self.T = T
        self.kernels = kernels
        self.send_all_states = send_all_states
        self.num_kernels = len(kernels)
        self.last_m_log_lik = tf.Variable(tf.zeros((self.num_kernels)))
        self.one_step_receives_state = [len(signature(k.one_step).parameters) > 2 for k in kernels]
        self.skip = skip
        super().__init__()

    def one_step(self, current_state, previous_kernel_results):
        # tf.print('running iteration')
        # if previous_kernel_results.iteration % 10:
        prog(self.T, previous_kernel_results.iteration)
        # tf.print('running', current_state)
        new_state = list()
        is_accepted = list()
        inner_results = list()
        for i in range(self.num_kernels):
            if self.skip is not None and self.skip[i]:
                # print(previous_kernel_results)
                is_accepted.append(previous_kernel_results.inner_results[i].is_accepted)
                new_state.append(current_state[i])
                inner_results.append(previous_kernel_results.inner_results[i])
                continue

            if self.send_all_states[i]:
                wrapped_state_i = current_state[i]
                if type(wrapped_state_i) is not list:
                    wrapped_state_i = [wrapped_state_i]

                tgt_prob = self.kernels[i].target_log_prob_fn_fn(current_state)(*wrapped_state_i)
                if hasattr(previous_kernel_results.inner_results[i], 'accepted_results'):

                    propres = previous_kernel_results.inner_results[i].accepted_results._replace(
                        target_log_prob=tgt_prob)
                    previous_kernel_results.inner_results[i] = previous_kernel_results.inner_results[i]._replace(
                        proposed_results=propres)
                else:
                    previous_kernel_results.inner_results[i] = previous_kernel_results.inner_results[i]._replace(
                        target_log_prob=tgt_prob)

            args = []
            try:
                if self.one_step_receives_state[i]:
                    args = [current_state]

                # state_chained = tf.expand_dims(current_state[i], 0)
                # print(state_chained)
                result_state, kernel_results = self.kernels[i].one_step(
                    current_state[i], previous_kernel_results.inner_results[i], *args)
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
