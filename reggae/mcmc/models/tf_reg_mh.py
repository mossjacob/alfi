from collections import namedtuple
import pickle

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from reggae.mcmc import MetropolisHastings, Parameter
from reggae.gp import GPKernelSelector
from reggae.utilities import exp, mult, jitter_cholesky, save_object
from reggae.mcmc.results import GenericResults
import reggae

import numpy as np
from scipy.special import expit

f64 = np.float64


class TranscriptionLikelihood():
    """
    Implementation of [1], a Metropolis-Hastings sampler
    [1] Titsias, M. K., A. Honkela, N. D. Lawrence, and M. Rattray (2012).
        Identifying targets of multiple co-regulating transcription factors from expression time-series by
        bayesian model comparison. BMC systems biology 6(1), 53
    """
    def __init__(self, data, options):
        self.options = options
        self.data = data
        self.num_tfs = data.f_obs.shape[1]
        self.preprocessing_variance = options.preprocessing_variance
        self.num_genes = data.m_obs.shape[1]
        self.num_replicates = data.m_obs.shape[0]

    def calculate_protein(self, fbar, δbar): # Calculate p_i vector
        τ = self.data.τ
        N_p = self.data.τ.shape[0]
        f_i = np.log(1+np.exp(fbar))
        δ_i = np.exp(δbar)

        p_i = np.zeros_like(f_i)
        for r in range(self.num_replicates):
            # Approximate integral (trapezoid rule)
            resolution = τ[1]-τ[0]
            sum_term = mult(exp(δ_i*τ), f_i[r])
            integrals = tf.concat([np.zeros((self.num_tfs, 1), dtype='float64'), 
                                   0.5*resolution*np.cumsum(sum_term[:, :-1] + sum_term[:, 1:], axis=1)], axis=1) 
            exp_δt = exp(-δ_i*τ)
            mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
            mask[r] = 1
            p_ir = exp_δt * integrals
            p_i = (1-mask)*p_i + mask * p_ir

        return p_i

    def predict_m(self, kbar, δbar, w, fbar, w_0):
        # Take relevant parameters out of log-space
        kin = (np.exp(kbar[:, i]).reshape(-1, 1) for i in range(kbar.shape[1]))
        if self.options.initial_conditions:
            a_j, b_j, d_j, s_j = kin
        else:
            b_j, d_j, s_j = kin

        τ = self.data.τ
        N_p = self.data.τ.shape[0]

        # Calculate p_i vector
        p_i = np.log(1+np.exp(fbar))
        if self.options.tf_mrna_present:
            p_i = self.calculate_protein(fbar, δbar)

        # Calculate m_pred
        m_pred = np.zeros((self.num_replicates, self.num_genes, N_p), dtype='float64')
        for r in range(self.num_replicates):
            resolution = τ[1]-τ[0]
            integrals = np.zeros((self.num_genes, N_p))
            interactions = w[:, 0][:, None]*np.log(p_i[r]+1e-100) + w_0[:, None]# tf.matmul(w, tfm.log(p_i[r]+1e-100)) + w_0[:, None]
            G = expit(interactions) # TF Activation Function (sigmoid)
            sum_term = G * exp(d_j*τ)
            integrals = np.concatenate([np.zeros((self.num_genes, 1), dtype='float64'), # Trapezoid rule
                                0.5*resolution*np.cumsum(sum_term[:, :-1] + sum_term[:, 1:], axis=1)], axis=1) 
            exp_dt = exp(-d_j*τ)
            integrals = mult(exp_dt, integrals)

            mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
            mask[r] = 1
            m_pred_r = b_j/d_j + s_j*integrals
            if self.options.initial_conditions:
                m_pred_r += mult((a_j-b_j/d_j), exp_dt)

            m_pred = (1-mask)*m_pred + mask * m_pred_r
        return m_pred

    def genes(self, params, δbar=None,
                     fbar=None, 
                     kbar=None, 
                     w=None,
                     w_0=None,
                     σ2_m=None, return_sq_diff=False):
        '''
        Computes likelihood of the genes.
        If any of the optional args are None, they are replaced by their current value in params.
        '''
        if δbar is None:
            δbar = params.δbar.value
        if fbar is None:
            fbar = params.fbar.value
        if kbar is None:
            kbar = params.kbar.value
        w = params.w.value if w is None else w
        σ2_m = params.σ2_m.value if σ2_m is None else σ2_m

        w_0 = params.w_0.value if w_0 is None else w_0 # TODO no hardcode this!
        m_pred = self.predict_m(kbar, δbar, w, fbar, w_0)

        log_lik = np.zeros(self.num_genes)
        sq_diff = np.square(self.data.m_obs - m_pred[:, :, self.data.common_indices])
        sq_diff = np.sum(sq_diff, axis=0)
        variance = σ2_m.reshape(-1, 1)
        if self.preprocessing_variance:
            variance = variance + self.data.σ2_m_pre # add PUMA variance
#         print(variance.shape, sq_diff.shape)
        log_lik = -0.5*np.log(2*np.pi*(variance)) - 0.5*sq_diff/variance
        log_lik = np.sum(log_lik, axis=1)
        if return_sq_diff:
            return log_lik, sq_diff
        return log_lik

    def tfs(self, params, fbar, return_sq_diff=False): 
        '''
        Computes log-likelihood of the transcription factors.
        TODO this should be for the i-th TF
        '''
        assert self.options.tf_mrna_present
        if not self.preprocessing_variance:
            σ2_f = params.σ2_f.value
            variance = σ2_f.reshape(-1, 1)
        else:
            variance = self.data.σ2_f_pre
        f_pred = np.log(1+np.exp(fbar))
        f_pred = np.atleast_2d(f_pred)
        sq_diff = np.square(self.data.f_obs - f_pred[:, :, self.data.common_indices])
        sq_diff = np.sum(sq_diff, axis=0)
        log_lik = -0.5*np.log(2*np.pi*variance) - 0.5*sq_diff/variance
        log_lik = np.sum(log_lik, axis=1)
        if return_sq_diff:
            return log_lik, sq_diff
        return log_lik

TupleParams_pre = namedtuple('TupleParams_pre', ['fbar','δbar','kbar','σ2_m','w','w_0','L','V','σ2_f'])
TupleParams = namedtuple('TupleParams', ['fbar','δbar','kbar','σ2_m','w','w_0','L','V'])

class TranscriptionMCMC(MetropolisHastings):
    '''
    Data is a tuple (m, f) of shapes (num, time)
    time is a tuple (t, τ, common_indices)
    '''
    def __init__(self, data, options):
        self.data = data
        min_dist = min(data.t[1:]-data.t[:-1])
        self.N_p = data.τ.shape[0]
        self.N_m = data.t.shape[0]      # Number of observations
        self.num_replicates = data.f_obs.shape[0]
        self.num_tfs = data.f_obs.shape[1]
        self.num_genes = data.m_obs.shape[1]
        
        self.kernel_selector = GPKernelSelector(data, options)
        self.likelihood = TranscriptionLikelihood(data, options)
        self.options = options
        # Adaptable variances
        a = tf.constant(-0.5, dtype='float64')
        b2 = tf.constant(2., dtype='float64')
        self.h_f = 0.15*tf.ones(self.N_p, dtype='float64')

        # Interaction weights
        w_0 = Parameter('w_0', tfd.Normal(0, 2), np.zeros(self.num_genes), step_size=0.2*tf.ones(self.num_genes, dtype='float64'))
        w_0.proposal_dist=lambda mu, j:tfd.Normal(mu, w_0.step_size[j])
        w = Parameter('w', tfd.Normal(0, 2), 1*np.ones((self.num_genes, self.num_tfs)), step_size=0.2*tf.ones(self.num_genes, dtype='float64'))
        w.proposal_dist=lambda mu, j:tfd.Normal(mu, w.step_size[j]) #) w_j) # At the moment this is the same as w_j0 (see pg.8)
        # Latent function
        fbar = Parameter('fbar', self.fbar_prior, 0.5*np.ones((self.num_replicates, self.num_tfs, self.N_p)))

        # GP hyperparameters
        V = Parameter('V', tfd.InverseGamma(f64(0.01), f64(0.01)), f64(1), step_size=0.05, fixed=not options.tf_mrna_present)
        V.proposal_dist=lambda v: tfd.TruncatedNormal(v, V.step_size, low=0, high=100) #v_i Fix to 1 if translation model is not used (pg.8)
        L = Parameter('L', tfd.Uniform(f64(min_dist**2-0.5), f64(data.t[-1]**2)), f64(4), step_size=0.05) # TODO auto set
        L.proposal_dist=lambda l2: tfd.TruncatedNormal(l2, L.step_size, low=0, high=100) #l2_i


        # Translation kinetic parameters
        δbar = Parameter('δbar', tfd.Normal(a, b2), f64(-0.3), step_size=0.05)
        δbar.proposal_dist=lambda mu:tfd.Normal(mu, δbar.step_size)
        # White noise for genes
        σ2_m = Parameter('σ2_m', tfd.InverseGamma(f64(0.01), f64(0.01)), 1e-4*np.ones(self.num_genes), step_size=0.01)
        σ2_m.proposal_dist=lambda mu: tfd.TruncatedNormal(mu, σ2_m.step_size, low=0, high=0.1)
        # Transcription kinetic parameters
        constraint_index = 2 if self.options.initial_conditions else 1
        def constrain_kbar(kbar, gene):
            '''Constrains a given row in kbar'''
            # if gene == 3:
            #     kbar[constraint_index] = np.log(0.8)
            #     kbar[constraint_index+1] = np.log(1.0)
            kbar[kbar < -10] = -10
            kbar[kbar > 3] = 3
            return kbar
        num_var = 4 if self.options.initial_conditions else 3
        kbar_initial = -0.1*np.ones((self.num_genes, num_var), dtype='float64')
        for j, k in enumerate(kbar_initial):
            kbar_initial[j] = constrain_kbar(k, j)
        kbar = Parameter('kbar',
            tfd.Normal(a, b2), 
            kbar_initial,
            constraint=constrain_kbar, step_size=0.05*tf.ones(num_var, dtype='float64'))
        kbar.proposal_dist=lambda mu: tfd.MultivariateNormalDiag(mu, kbar.step_size)
        
        if not options.preprocessing_variance:
            σ2_f = Parameter('σ2_f', tfd.InverseGamma(f64(0.01), f64(0.01)), 1e-4*np.ones(self.num_tfs), step_size=tf.constant(0.5, dtype='float64'))
            super().__init__(TupleParams_pre(fbar, δbar, kbar, σ2_m, w, w_0, L, V, σ2_f))
        else:
            super().__init__(TupleParams(fbar, δbar, kbar, σ2_m, w, w_0, L, V))

    def fbar_prior_params(self, v, l2):
    #     print('vl2', v, l2)
        jitter = tf.linalg.diag(1e-5 * np.ones(self.N_p))
        K = self.kernel_selector()(v, l2)[1].numpy()[0]+jitter

        # K = mult(v, exp(-np.square(self.t_dist)/(2*l2))) + jitter
        m = np.zeros(self.N_p)
        return m, K

    def fbar_prior(self, fbar, v, l2):
        m, K = self.fbar_prior_params(v, l2)
        prob = 0
        for r in range(self.num_replicates):
            prob += tfd.MultivariateNormalTriL(loc=m, scale_tril=tf.linalg.cholesky(K)).log_prob(fbar[r, 0])
        return prob

    def iterate(self):
        params = self.params
        # Compute likelihood for comparison
        old_m_likelihood, sq_diff_m  = self.likelihood.genes(params, return_sq_diff=True)
        old_f_likelihood = 0
        if self.options.tf_mrna_present:
            old_f_likelihood, sq_diff_f  = self.likelihood.tfs(params, params.fbar.value, return_sq_diff=True)
        
        # Untransformed tf mRNA vectors F (Step 1)
        fbar = params.fbar.value
        # Gibbs step
        z_i = tfd.MultivariateNormalDiag(fbar, self.h_f).sample()
        # MH
        m, K = self.fbar_prior_params(params.V.value, params.L.value)
        for r in range(self.num_replicates):

            for i in range(self.num_tfs): # TODO does not really work for multiple TFs
                invKsigmaK = tf.matmul(tf.linalg.inv(K+tf.linalg.diag(self.h_f)), K) # (C_i + hI)C_i
                L = jitter_cholesky(K-tf.matmul(K, invKsigmaK))
                c_mu = tf.linalg.matvec(invKsigmaK, z_i[r][i])
                fstar_i = tf.matmul(tf.random.normal((1, L.shape[0]), dtype='float64'), L) + c_mu
                # fstar_i = tf.reshape(fstar, (-1, ))
                mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
                mask[r] = 1
                fstar = (1-mask) * fbar + mask * fstar_i
                new_m_likelihood = self.likelihood.genes(params, fbar=fstar)
                new_f_likelihood = 0 
                if self.options.tf_mrna_present:
                    new_f_likelihood = self.likelihood.tfs(params, fstar)
                new_prob = np.sum(new_m_likelihood) + np.sum(new_f_likelihood)
                old_prob = np.sum(old_m_likelihood) + np.sum(old_f_likelihood)
                if self.is_accepted(new_prob, old_prob):
                    params.fbar.value[r] = fstar_i
                    old_m_likelihood = new_m_likelihood
                    old_f_likelihood = new_f_likelihood
                    self.acceptance_rates['fbar'] += 1/(self.num_tfs*self.num_replicates)


        if self.options.tf_mrna_present: # (Step 2)
            # Log of translation ODE degradation rates
            δbar = params.δbar.value
            for i in range(self.num_tfs):# TODO make for self.num_tfs > 1
                # Proposal distribution
                δstar = params.δbar.propose(δbar) # δstar is in log-space, i.e. δstar = δbar*
                new_prob = np.sum(self.likelihood.genes(params, δbar=δstar)) + params.δbar.prior.log_prob(δstar)
                old_prob = np.sum(old_m_likelihood) + params.δbar.prior.log_prob(δbar)
    #             print(δstar, params.δbar.prior.log_prob(δstar))
    #             print(new_prob, old_prob)
                if self.is_accepted(new_prob, old_prob):
                    params.δbar.value = δstar
                    self.acceptance_rates['δbar'] += 1/self.num_tfs

        # Log of transcription ODE kinetic params (Step 3)
        kbar = params.kbar.value
        kstar = kbar.copy()
        for j in range(self.num_genes):
            sample = params.kbar.propose(kstar[j])
            sample = params.kbar.constrain(sample, j)
            kstar[j] = sample
            new_prob = self.likelihood.genes(params, kbar=kstar)[j] + sum(params.kbar.prior.log_prob(sample))
            old_prob = old_m_likelihood[j] + sum(params.kbar.prior.log_prob(kbar[j]))
            if self.is_accepted(new_prob, old_prob):
                test = params.kbar.value
                test[j]=sample
                params.kbar.value = test
                self.acceptance_rates['kbar'] += 1/self.num_genes
            else:
                kstar[j] = params.kbar.value[j]


        # Interaction weights and biases (note: should work for self.num_tfs > 1) (Step 4)
        if self.options.weights:
            w = params.w.value
            w_0 = params.w_0.value
            wstar = w.copy()
            w_0star = w_0.copy()
            for j in range(self.num_genes):
                sample_0 = params.w_0.propose(w_0[j], j)
                sample = params.w.propose(wstar[j], j)
                wstar[j] = sample
                w_0star[j] = sample_0
                new_prob = self.likelihood.genes(params, w=wstar, w_0=w_0star)[j] + sum(params.w.prior.log_prob(sample)) + params.w_0.prior.log_prob(sample_0)
                old_prob = old_m_likelihood[j] + sum(params.w.prior.log_prob(w[j,:])) + params.w_0.prior.log_prob(w_0[j])
                if self.is_accepted(new_prob, old_prob):
                    params.w.value[j] = sample
                    params.w_0.value[j] = sample_0
                    self.acceptance_rates['w'] += 1/self.num_genes
                    self.acceptance_rates['w_0'] += 1/self.num_genes
                else:
                    wstar[j] = params.w.value[j]

        # Noise variances
        if self.options.preprocessing_variance:
            σ2_m = params.σ2_m.value
            σ2_mstar = σ2_m.copy()
            for j in range(self.num_genes):
                sample = params.σ2_m.propose(σ2_m[j])
                σ2_mstar[j] = sample
                old_q = params.σ2_m.proposal_dist(σ2_mstar[j]).log_prob(σ2_m[j])
                new_prob = self.likelihood.genes(params, σ2_m=σ2_mstar)[j] +params.σ2_m.prior.log_prob(σ2_mstar[j])
                
                new_q = params.σ2_m.proposal_dist(σ2_m[j]).log_prob(σ2_mstar[j])
                old_prob = self.likelihood.genes(params, σ2_m=σ2_m)[j] + params.σ2_m.prior.log_prob(σ2_m[j])
                    
                if self.is_accepted(new_prob + old_q, old_prob + new_q):
                    params.σ2_m.value[j] = sample
                    self.acceptance_rates['σ2_m'] += 1/self.num_genes
                else:
                    σ2_mstar[j] = σ2_m[j]
        else: # Use Gibbs sampling
            # Prior parameters
            α = params.σ2_m.prior.concentration
            β = params.σ2_m.prior.scale
            # Conditional posterior of inv gamma parameters:
            α_post = α + 0.5*self.N_m
            β_post = β + 0.5*np.sum(sq_diff_m)
            # print(α.shape, sq_diff.shape)
            # print('val', β_post.shape, params.σ2_m.value)
            params.σ2_m.value = np.repeat(tfd.InverseGamma(α_post, β_post).sample(), self.num_genes)
            self.acceptance_rates['σ2_m'] += 1
            
            if self.options.tf_mrna_present: # (Step 5)
                # Prior parameters
                α = params.σ2_f.prior.concentration
                β = params.σ2_f.prior.scale
                # Conditional posterior of inv gamma parameters:
                α_post = α + 0.5*self.N_m
                β_post = β + 0.5*np.sum(sq_diff_f)
                # print(α.shape, sq_diff.shape)
                # print('val', β_post.shape, params.σ2_m.value)
                params.σ2_f.value = np.repeat(tfd.InverseGamma(α_post, β_post).sample(), self.num_tfs)
                self.acceptance_rates['σ2_f'] += 1

            # print('val', params.σ2_m.value)
        # Length scales and variances of GP kernels
        l2 = params.L.value
        v = params.V.value
        for i in range(self.num_tfs):
            # Proposal distributions
            Q_v = params.V.proposal_dist
            Q_l = params.L.proposal_dist
            vstar = params.V.propose(v)
            l2star = params.L.propose(l2)
            # Acceptance probabilities
            new_fbar_prior = params.fbar.prior(params.fbar.value, vstar, l2star)
            old_q = Q_v(vstar).log_prob(v) + Q_l(l2star).log_prob(l2) # Q(old|new)
            new_prob = new_fbar_prior + params.V.prior.log_prob(vstar) + params.L.prior.log_prob(l2star)
            
            new_q = Q_v(v).log_prob(vstar) + Q_l(l2).log_prob(l2star) # Q(new|old)
            old_prob = params.fbar.prior(params.fbar.value, v, l2) + params.V.prior.log_prob(v) + params.L.prior.log_prob(l2)
            accepted = self.is_accepted(new_prob + old_q, old_prob + new_q)
            if accepted:
                params.V.value = vstar
                params.L.value = l2star
                self.acceptance_rates['V'] += 1/self.num_tfs
                self.acceptance_rates['L'] += 1/self.num_tfs

    def save(self, name):
        save_object({'samples':self.samples, 'acc_rates': self.acceptance_rates}, f'mh-{name}')

    @staticmethod
    def load(args, name):
        model = TranscriptionMCMC(*args)

        import os
        path = os.path.join(os.getcwd(), 'saved_models')
        fs = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f'mh-{name}')]
        files = sorted(fs, key=os.path.getmtime)
        with open(files[-1], 'rb') as f:
            saved_model = pickle.load(f)
            model.samples = saved_model['samples']
            model.acceptance_rates = saved_model['acc_rates']            
        return model

    @staticmethod
    def initialise_from_state(args, state):
        model = TranscriptionMCMC(*args)
        model.acceptance_rates = state.acceptance_rates
        model.samples = state.samples
        return model

    def predict_m(self, kbar, δbar, w, fbar, w_0):
        return self.likelihood.predict_m(kbar, δbar, w, fbar, w_0)

    def predict_m_with_current(self):
        return self.likelihood.predict_m(self.params.kbar.value, 
                                         self.params.δbar.value, 
                                         self.params.w.value, 
                                         self.params.fbar.value,
                                         self.params.w_0.value)

