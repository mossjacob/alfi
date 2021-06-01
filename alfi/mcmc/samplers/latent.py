import tensorflow as tf
from tensorflow_probability import distributions as tfd

from alfi.mcmc.gp.gp_kernels import GPKernelSelector
from alfi.mcmc.samplers import MetropolisKernel
from alfi.utilities.tf import jitter_cholesky, logit, add_diag
from alfi.mcmc.results import GenericResults
from .mixins import ParamGroupMixin

import numpy as np
f64 = np.float64


class LatentGPSampler(MetropolisKernel, ParamGroupMixin):
    """
    Latent Gaussian Process sampler.
    Parameters:
        likelihood_fn:
        latent_likelihood_fn:
        param: the associated Parameter (only one, not a list)
    """
    def __init__(self, likelihood_fn, latent_likelihood_fn,
                 param,
                 kernel_selector: GPKernelSelector, 
                 step_size, kernel_exponential=False):
        self.param_group = [param]
        self.fbar_prior_params = kernel_selector()
        self.kernel_priors = kernel_selector.priors()
        self.kernel_selector = kernel_selector
        self.likelihood_fn = likelihood_fn
        self.latent_likelihood_fn = latent_likelihood_fn
        self.kernel_exponential = kernel_exponential
        self.data_present = latent_likelihood_fn is not None
        self.step_fn = self.joint_one_step
        self.calc_prob_fn = self.joint_calc_prob

        super().__init__(step_size, tune_every=3000)

    def _one_step(self, current_state, previous_kernel_results):
        return self.step_fn(current_state, previous_kernel_results)

    @tf.function
    def joint_one_step(self, current_state, previous_kernel_results):
        current_state = current_state[0]
        new_state = tf.identity(current_state[0])
        num_replicates = new_state.shape[0]
        num_tfs = new_state.shape[1]
        new_params = []
        S = tf.linalg.diag(self.step_size)
        # MH
        m, K = self.fbar_prior_params(current_state[1], current_state[2])
        N_p = K.shape[-1]
        K = K+tf.linalg.diag(1e-7*tf.ones(N_p, dtype='float64'))

        # Propose new params
        v = self.kernel_selector.proposal(0, current_state[1]).sample()
        l2 = self.kernel_selector.proposal(1, current_state[2]).sample()
        m_, K_ = self.fbar_prior_params(v, l2)
        K_ = K_+tf.linalg.diag(1e-7*tf.ones(N_p, dtype='float64'))
        iK = tf.linalg.inv(K)
        iK_ = tf.linalg.inv(K_)
        U_invR = tf.linalg.cholesky(add_diag(iK, 1/S))
        U_invR = tf.transpose(U_invR, [0, 2, 1])
        U_invR_ = jitter_cholesky(add_diag(iK_, 1/S))
        U_invR_ = tf.transpose(U_invR_, [0, 2, 1])
        new_hyp = [v, l2]
        old_hyp = [current_state[1], current_state[2]]

        # Gibbs step
        fbar = new_state
        for r in range(num_replicates):
            fbar = new_state[r]

            gg = tfd.MultivariateNormalDiag(fbar, self.step_size).sample()

            Sinv_g = gg / self.step_size

            nu = tf.linalg.matvec(U_invR, fbar) - tf.squeeze(tf.linalg.solve(tf.transpose(U_invR, [0, 2, 1]), tf.expand_dims(Sinv_g, -1)), -1)
            f = tf.linalg.solve(U_invR_, tf.expand_dims(nu, -1)) + tf.linalg.cholesky_solve(tf.transpose(U_invR_, [0, 2, 1]), tf.expand_dims(Sinv_g, -1))
            f = tf.squeeze(f, -1)

            mask = np.zeros((num_replicates, 1, 1), dtype='float64')
            mask[r] = 1
            new_state = (1-mask) * new_state + mask * f
        
        test_state = tf.zeros((num_replicates, num_tfs, N_p), dtype='float64')

        for i in range(num_tfs): # Test each tf individually
            mask = np.zeros((num_replicates, num_tfs, 1), dtype='float64')
            hyp_mask = np.zeros((num_tfs,), dtype='float64')
            hyp_mask[i] = 1
            mask[:, i] = 1
            test_state = (1-mask) * current_state[0] + mask * new_state

            new_prob = self.calc_prob_fn(test_state, new_hyp, old_hyp)
            old_prob = self.calc_prob_fn(current_state[0], old_hyp, new_hyp) #previous_kernel_results.target_log_prob
            is_accepted = self.metropolis_is_accepted(new_prob, old_prob)
            # tf.print(new_prob, old_prob, is_accepted)
            if not is_accepted[0]:
                new_state = (1-mask) * new_state + mask * current_state[0]
                new_hyp[0] = (1-hyp_mask) * new_hyp[0] + hyp_mask * current_state[1]
                new_hyp[1] = (1-hyp_mask) * new_hyp[1] + hyp_mask * current_state[2]
            # prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)

            # new_state = tf.cond(tf.equal(is_accepted, tf.constant(False)),
            #                     lambda:current_state[0], lambda:new_state)
        # new_params = tf.cond(tf.equal(is_accepted, tf.constant(False)),
        #                         lambda:[current_state[1], current_state[2]], lambda:[v, l2])
        return [[new_state, *new_hyp]], f64(0), is_accepted[0]

    def _joint_one_step(self, current_state, previous_kernel_results):
        # Untransformed tf mRNA vectors F (Step 1)
        current_state = current_state[0]

        new_state = tf.identity(current_state[0])
        new_params = []
        S = tf.linalg.diag(self.step_size)
        # MH
        m, K = self.fbar_prior_params(current_state[1], current_state[2])
        # Propose new params
        v = self.kernel_selector.proposal(0, current_state[1]).sample()
        l2 = self.kernel_selector.proposal(1, current_state[2]).sample()
        m_, K_ = self.fbar_prior_params(v, l2)

        # Gibbs step
        fbar = new_state
        z_i = tfd.MultivariateNormalDiag(fbar, self.step_size).sample()

        # Compute K_i(K_i + S)^-1 
        Ksuminv = tf.matmul(K, tf.linalg.inv(K+S))
        # Compute chol(K-K(K+S)^-1 K)
        L = jitter_cholesky(K-tf.matmul(Ksuminv, K))
        c_mu = tf.linalg.matvec(Ksuminv, z_i)
        # Compute nu = L^-1 (f-mu)
        invL = tf.linalg.inv(L)
        nu = tf.linalg.matvec(invL, fbar-c_mu)

        Ksuminv = tf.matmul(K_, tf.linalg.inv(K_+S)) 
        L = jitter_cholesky(K_-tf.matmul(K_, Ksuminv))
        c_mu = tf.linalg.matvec(Ksuminv, z_i)
        fstar = tf.linalg.matvec(L, nu) + c_mu

        new_hyp = [v, l2]
        old_hyp = [current_state[1], current_state[2]]
        new_prob = self.calc_prob_fn(fstar, new_hyp, old_hyp)
        old_prob = self.calc_prob_fn(new_state, old_hyp, new_hyp) #previous_kernel_results.target_log_prob

        is_accepted = self.metropolis_is_accepted(new_prob, old_prob)

        prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)


        new_state = tf.cond(tf.equal(is_accepted, tf.constant(False)),
                            lambda:new_state, lambda:fstar)
        new_params = tf.cond(tf.equal(is_accepted, tf.constant(False)),
                                lambda:[current_state[1], current_state[2]], lambda:[v, l2])

        return [[new_state, *new_params]], prob, is_accepted[0]

    def joint_calc_prob(self, fstar, new_hyp, old_hyp):
        new_m_likelihood = self.likelihood_fn(
            latent=[fstar],
        )
        # σ2_f = 1e-6 * tf.ones(fstar.shape[1], dtype='float64')
        # if 'σ2_f' in self.state_indices:
        #     σ2_f = all_states[self.state_indices['σ2_f']]
        # tf.print(self.data_present, tf.equal(self.data_present, True))
        # print(self.data_present, tf.equal(self.data_present, True))
        if self.data_present:
            new_f_likelihood = tf.reduce_sum(self.latent_likelihood_fn(
                                       latent=[fstar]
                                   ))
        else:
            new_f_likelihood = f64(0)

        new_prob = tf.reduce_sum(new_m_likelihood) + new_f_likelihood

        new_prob += tf.reduce_sum(
            self.kernel_selector.proposal(0, new_hyp[0]).log_prob(old_hyp[0]) +
            self.kernel_selector.proposal(1, new_hyp[1]).log_prob(old_hyp[1])
        )
        if self.kernel_exponential:
            new_hyp = [tf.exp(new_hyp[0]), tf.exp(new_hyp[1])]

        new_prob += tf.reduce_sum(
            self.kernel_priors[0].log_prob(new_hyp[0]) +
            self.kernel_priors[1].log_prob(new_hyp[1])
        )
        return new_prob

    def bootstrap_results(self, init_state):
        init_state = init_state[0]
        prob = self.calc_prob_fn(
            init_state[0], [init_state[1], init_state[2]],
            [init_state[1], init_state[2]])

        return GenericResults(prob, True)
    
    def is_calibrated(self):
        return True
