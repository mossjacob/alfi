import tensorflow as tf
from tensorflow_probability import distributions as tfd

from reggae.mcmc.kernels.mh import MetropolisKernel
from reggae.mcmc.kernels.wrappers import ESSWrapper
from reggae.gp.std_kernels import GPKernelSelector
from reggae.tf_utilities import jitter_cholesky, logit, add_diag
from reggae.mcmc.results import GenericResults

import numpy as np
f64 = np.float64


class LatentKernel(MetropolisKernel):
    def __init__(self, data, options,
                 likelihood, 
                 kernel_selector: GPKernelSelector, 
                 state_indices, 
                 step_size):
        self.fbar_prior_params = kernel_selector()
        self.kernel_priors = kernel_selector.priors()
        self.kernel_selector = kernel_selector
        self.num_tfs = data.f_obs.shape[1]
        self.num_genes = data.m_obs.shape[1]
        self.likelihood = likelihood
        self.options = options
        self.tf_mrna_present = options.tf_mrna_present
        self.state_indices = state_indices
        self.num_replicates = data.f_obs.shape[0]
        self.step_fn = self.f_one_step
        self.calc_prob_fn = self.f_calc_prob
        if options.joint_latent:
            self.step_fn = self.joint_one_step
            self.calc_prob_fn = self.joint_calc_prob
            
        super().__init__(step_size, tune_every=100)

    def _one_step(self, current_state, previous_kernel_results, all_states):
        return self.step_fn(current_state, previous_kernel_results, all_states)

    def f_one_step(self, current_state, previous_kernel_results, all_states):
        old_probs = list()
        new_state = tf.identity(current_state)

        # MH
        kernel_params = (all_states[self.state_indices['kernel_params']][0], all_states[self.state_indices['kernel_params']][1])
        m, K = self.fbar_prior_params(*kernel_params)
        for r in range(self.num_replicates):
            # Gibbs step
            fbar = current_state[r]
            z_i = tfd.MultivariateNormalDiag(fbar, self.step_size).sample()
            fstar = tf.zeros_like(fbar)

            for i in range(self.num_tfs):
                invKsigmaK = tf.matmul(tf.linalg.inv(K[i]+tf.linalg.diag(self.step_size)), K[i]) # (C_i + hI)C_i
                L = jitter_cholesky(K[i]-tf.matmul(K[i], invKsigmaK))
                c_mu = tf.matmul(z_i[i, None], invKsigmaK)
                fstar_i = tf.matmul(tf.random.normal((1, L.shape[0]), dtype='float64'), L) + c_mu
                mask = np.zeros((self.num_tfs, 1), dtype='float64')
                mask[i] = 1
                fstar = (1-mask) * fstar + mask * fstar_i

            mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
            mask[r] = 1
            test_state = (1-mask) * new_state + mask * fstar

            new_prob = self.calc_prob_fn(test_state, all_states)
            old_prob = self.calc_prob_fn(new_state, all_states)
            #previous_kernel_results.target_log_prob #tf.reduce_sum(old_m_likelihood) + old_f_likelihood

            is_accepted = self.metropolis_is_accepted(new_prob, old_prob)
            
            prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)


            new_state = tf.cond(tf.equal(is_accepted, tf.constant(False)),
                                lambda:new_state, lambda:test_state)
        return new_state, prob, is_accepted[0]


    @tf.function
    def joint_one_step(self, current_state, previous_kernel_results, all_states):
        new_state = tf.identity(current_state[0])
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
        for r in range(self.num_replicates):
            fbar = new_state[r]

            gg = tfd.MultivariateNormalDiag(fbar, self.step_size).sample()

            Sinv_g = gg / self.step_size

            nu = tf.linalg.matvec(U_invR, fbar) - tf.squeeze(tf.linalg.solve(tf.transpose(U_invR, [0, 2, 1]), tf.expand_dims(Sinv_g, -1)), -1)
            f = tf.linalg.solve(U_invR_, tf.expand_dims(nu, -1)) + tf.linalg.cholesky_solve(tf.transpose(U_invR_, [0, 2, 1]), tf.expand_dims(Sinv_g, -1))
            f = tf.squeeze(f, -1)

            mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
            mask[r] = 1
            new_state = (1-mask) * new_state + mask * f
        
        test_state = tf.zeros((self.num_replicates, self.num_tfs, N_p), dtype='float64')

        for i in range(self.num_tfs): # Test each tf individually
            mask = np.zeros((self.num_replicates, self.num_tfs, 1), dtype='float64')
            hyp_mask = np.zeros((self.num_tfs,), dtype='float64')
            hyp_mask[i] = 1
            mask[:, i] = 1
            test_state = (1-mask) * current_state[0] + mask * new_state

            new_prob = self.calc_prob_fn(test_state, new_hyp, old_hyp, all_states)
            old_prob = self.calc_prob_fn(current_state[0], old_hyp, new_hyp, all_states) #previous_kernel_results.target_log_prob 
            is_accepted = self.metropolis_is_accepted(new_prob, old_prob)
            if not is_accepted[0]:
                new_state = (1-mask) * new_state + mask * current_state[0]
                new_hyp[0] = (1-hyp_mask) * new_hyp[0] + hyp_mask * current_state[1]
                new_hyp[1] = (1-hyp_mask) * new_hyp[1] + hyp_mask * current_state[2]
            # prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)

            # new_state = tf.cond(tf.equal(is_accepted, tf.constant(False)),
            #                     lambda:current_state[0], lambda:new_state)
        # new_params = tf.cond(tf.equal(is_accepted, tf.constant(False)),
        #                         lambda:[current_state[1], current_state[2]], lambda:[v, l2])
        return [new_state, *new_hyp], f64(0), is_accepted[0]

    def _joint_one_step(self, current_state, previous_kernel_results, all_states):
        # Untransformed tf mRNA vectors F (Step 1)
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
        new_prob = self.calc_prob_fn(fstar, new_hyp, old_hyp, all_states)
        old_prob = self.calc_prob_fn(new_state, old_hyp, new_hyp, all_states) #previous_kernel_results.target_log_prob 

        is_accepted = self.metropolis_is_accepted(new_prob, old_prob)

        prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)


        new_state = tf.cond(tf.equal(is_accepted, tf.constant(False)),
                            lambda:new_state, lambda:fstar)
        new_params = tf.cond(tf.equal(is_accepted, tf.constant(False)),
                                lambda:[current_state[1], current_state[2]], lambda:[v, l2])

        return [new_state, *new_params], prob, is_accepted[0]
    
    def f_calc_prob(self, fstar, all_states):
        new_m_likelihood = self.likelihood.genes(
            all_states,
            self.state_indices,
            fbar=fstar,
        )
        new_f_likelihood = tf.cond(tf.equal(self.tf_mrna_present, tf.constant(True)), 
                                   lambda:tf.reduce_sum(self.likelihood.tfs(
                                       1e-6*tf.ones(self.num_tfs, dtype='float64'), # TODO
                                       fstar
                                   )), lambda:f64(0))
        new_prob = tf.reduce_sum(new_m_likelihood) + new_f_likelihood
        return new_prob

    def joint_calc_prob(self, fstar, new_hyp, old_hyp, all_states):
        new_m_likelihood = self.likelihood.genes(
            all_states,
            self.state_indices,
            fbar=fstar,
        )
        σ2_f = 1e-6*tf.ones(self.num_tfs, dtype='float64')
        if 'σ2_f' in self.state_indices:
            σ2_f = all_states[self.state_indices['σ2_f']]

        new_f_likelihood = tf.cond(tf.equal(self.tf_mrna_present, tf.constant(True)), 
                                   lambda:tf.reduce_sum(self.likelihood.tfs(
                                       σ2_f,
                                       fstar
                                   )), lambda:f64(0))
        new_prob = tf.reduce_sum(new_m_likelihood) + new_f_likelihood

        new_prob += tf.reduce_sum(
            self.kernel_selector.proposal(0, new_hyp[0]).log_prob(old_hyp[0]) + \
            self.kernel_selector.proposal(1, new_hyp[1]).log_prob(old_hyp[1])
        )
        if self.options.kernel_exponential:
            new_hyp = [tf.exp(new_hyp[0]), tf.exp(new_hyp[1])]

        new_prob += tf.reduce_sum(
            self.kernel_priors[0].log_prob(new_hyp[0]) + \
            self.kernel_priors[1].log_prob(new_hyp[1])
        )
        return new_prob

    def bootstrap_results(self, init_state, all_states):
        prob = self.calc_prob_fn(init_state[0], [init_state[1], init_state[2]], 
                                 [init_state[1], init_state[2]], all_states)

        return GenericResults(prob, True)
    
    def is_calibrated(self):
        return True


class ESSBuilder:
    def __init__(self, data, state_indices, kernel_selector):
        self.state_indices = state_indices
        self.kernel_selector = kernel_selector
        self.num_replicates = data.f_obs.shape[0]
        self.num_tfs = data.f_obs.shape[1]

    def normal_sampler_fn_fn(self, all_states):
        def normal_sampler_fn(seed):
            p1, p2 = all_states[self.state_indices['kernel_params']]
            m, K = self.kernel_selector()(logit(p1), logit(p2))
            m = tf.zeros((self.num_replicates, self.num_tfs, self.N_p), dtype='float64')
            K = tf.stack([K for _ in range(3)], axis=0)
            jitter = tf.linalg.diag(1e-8 *tf.ones(self.N_p, dtype='float64'))
            z = tfd.MultivariateNormalTriL(loc=m, 
                                scale_tril=tf.linalg.cholesky(K+jitter)).sample(seed=seed)
            # tf.print(z)
            return z
        return normal_sampler_fn

    def f_log_prob_fn(self, all_states):
        def f_log_prob(fstar):
            # print(all_states)
            new_m_likelihood = self.likelihood.genes(
                all_states,
                self.state_indices,
                fbar=fstar,
            )
            σ2_f = 1e-6*tf.ones(self.num_tfs, dtype='float64')
            # if 'σ2_f' in self.state_indices:
            #     σ2_f = all_states[self.state_indices['σ2_f']]

            new_f_likelihood = tf.cond(tf.equal(self.options.tf_mrna_present, tf.constant(True)), 
                                    lambda:tf.reduce_sum(self.likelihood.tfs(
                                        σ2_f,
                                        fstar
                                    )), lambda:f64(0))
            return tf.reduce_sum(new_m_likelihood) + new_f_likelihood
        return f_log_prob
    latents_kernel = ESSWrapper(normal_sampler_fn_fn, f_log_prob_fn)
