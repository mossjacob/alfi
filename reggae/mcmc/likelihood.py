import tensorflow as tf
from tensorflow import math as tfm

from reggae.data_loaders import DataHolder
from reggae.tf_utilities import rotate, jitter_cholesky, logit, logistic, LogisticNormal, inverse_positivity, save_object
from reggae.mcmc import Options

import numpy as np
PI = tf.constant(np.pi, dtype='float64')


class TranscriptionLikelihood:
    """
    Likelihood of the form:
    N(m(t), s(t))
    where m(t) = b/d + (a - b/d) exp(-dt) + s int^t_0 G(p(u); w) exp(-d(t-u)) du
    """
    def __init__(self, data: DataHolder, options: Options):
        self.options = options
        self.data = data
        self.preprocessing_variance = options.preprocessing_variance
        self.num_genes = data.m_obs.shape[1]
        self.num_tfs = data.f_obs.shape[1]
        self.num_replicates = data.f_obs.shape[0]

    @tf.function
    def calculate_protein(self, fbar, k_fbar, Δ): # Calculate p_i vector
        τ = self.data.τ
        f_i = inverse_positivity(fbar)
        δ_i = tf.reshape(logit(k_fbar), (-1, 1))
        if self.options.delays:
            # Add delay 
            Δ = tf.cast(Δ, 'int32')

            for r in range(self.num_replicates):
                f_ir = rotate(f_i[r], -Δ)
                mask = ~tf.sequence_mask(Δ, f_i.shape[2])
                f_ir = tf.where(mask, f_ir, 0)
                mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
                mask[r] = 1
                f_i = (1-mask) * f_i + mask * f_ir

        # Approximate integral (trapezoid rule)
        resolution = τ[1]-τ[0]
        sum_term = tfm.multiply(tfm.exp(δ_i*τ), f_i)
        cumsum = 0.5*resolution*tfm.cumsum(sum_term[:, :, :-1] + sum_term[:, :, 1:], axis=2)
        integrals = tf.concat([tf.zeros((self.num_replicates, self.num_tfs, 1), dtype='float64'), cumsum], axis=2) 
        exp_δt = tfm.exp(-δ_i*τ)
        p_i = exp_δt * integrals
        return p_i

    @tf.function
    def predict_m(self, kbar, k_fbar, wbar, fbar, w_0bar, Δ):
        # Take relevant parameters out of log-space
        if self.options.kinetic_exponential:
            kin = (tf.reshape(tf.exp(logit(kbar[:, i])), (-1, 1)) for i in range(kbar.shape[1]))
        else:
            kin = (tf.reshape(logit(kbar[:, i]), (-1, 1)) for i in range(kbar.shape[1]))
        if self.options.initial_conditions:
            a_j, b_j, d_j, s_j = kin
        else:
            b_j, d_j, s_j = kin
        w = (wbar)
        w_0 = tf.reshape((w_0bar), (-1, 1))
        τ = self.data.τ
        N_p = self.data.τ.shape[0]

        p_i = inverse_positivity(fbar)
        if self.options.translation:
            p_i = self.calculate_protein(fbar, k_fbar, Δ)

        # Calculate m_pred
        resolution = τ[1]-τ[0]
        interactions =  tf.matmul(w, tfm.log(p_i+1e-100)) + w_0
        G = tfm.sigmoid(interactions) # TF Activation Function (sigmoid)
        sum_term = G * tfm.exp(d_j*τ)
        integrals = tf.concat([tf.zeros((self.num_replicates, self.num_genes, 1), dtype='float64'), # Trapezoid rule
                            0.5*resolution*tfm.cumsum(sum_term[:, :, :-1] + sum_term[:, :, 1:], axis=2)], axis=2) 
        exp_dt = tfm.exp(-d_j*τ)
        integrals = tfm.multiply(exp_dt, integrals)

        m_pred = b_j/d_j + s_j*integrals
        if self.options.initial_conditions:
            m_pred += tfm.multiply((a_j-b_j/d_j), exp_dt)
        return m_pred

    def get_parameters_from_state(self, all_states, state_indices,
                                  fbar=None, kbar=None, k_fbar=None,
                                  wbar=None, w_0bar=None, σ2_m=None, Δ=None):
        nuts_index = 0
        kbar = all_states[state_indices['kinetics']][nuts_index] if kbar is None else kbar
        if self.options.translation:
            nuts_index+=1
            k_fbar = all_states[state_indices['kinetics']][nuts_index] if k_fbar is None else k_fbar
        else:
            k_fbar = None

        if self.options.weights:
            nuts_index+=1
            wbar = all_states[state_indices['kinetics']][nuts_index] if wbar is None else wbar
            w_0bar = all_states[state_indices['kinetics']][nuts_index+1] if w_0bar is None else w_0bar
        else:
            wbar = logistic(1*tf.ones((self.num_genes, self.num_tfs), dtype='float64'))
            w_0bar = 0.5*tf.ones(self.num_genes, dtype='float64')

        σ2_m = all_states[state_indices['σ2_m']] if σ2_m is None else σ2_m

        if fbar is None:
            fbar = all_states[state_indices['latents']]
            if self.options.joint_latent:
                fbar = fbar[0]
        if self.options.delays:
            Δ = all_states[state_indices['Δ']] if Δ is None else Δ
        else:
            Δ = tf.zeros((self.num_tfs,), dtype='float64')
        return fbar, kbar, k_fbar, wbar, w_0bar, σ2_m, Δ

    @tf.function
    def _genes(self, fbar, kbar, k_fbar, wbar, w_0bar, σ2_m, Δ):
        m_pred = self.predict_m(kbar, k_fbar, wbar, fbar, w_0bar, Δ)
        sq_diff = tfm.square(self.data.m_obs - tf.transpose(tf.gather(tf.transpose(m_pred),self.data.common_indices)))

        variance = tf.reshape(σ2_m, (-1, 1))
        if self.preprocessing_variance:
            variance = logit(variance) + self.data.σ2_m_pre # add PUMA variance
        log_lik = -0.5*tfm.log(2*PI*variance) - 0.5*sq_diff/variance
        log_lik = tf.reduce_sum(log_lik)
        return log_lik

    @tf.function#(experimental_compile=True)
    def genes(self, all_states=None, state_indices=None,
              kbar=None, 
              k_fbar=None,
              fbar=None, 
              wbar=None,
              w_0bar=None,
              σ2_m=None, 
              Δ=None):
        """
        Computes likelihood of the genes.
        If any of the optional args are None, they are replaced by their
        current value in all_states.
        """
        params = self.get_parameters_from_state(
            all_states, state_indices, fbar, kbar, k_fbar, wbar, w_0bar, σ2_m, Δ)
        return self._genes(*params)

    @tf.function#(experimental_compile=True)
    def tfs(self, σ2_f, fbar):
        """
        Computes log-likelihood of the transcription factors.
        """
        # assert self.options.tf_mrna_present
        if not self.preprocessing_variance:
            variance = tf.reshape(σ2_f, (-1, 1))
        else:
            variance = self.data.σ2_f_pre
        f_pred = inverse_positivity(fbar)
        sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred),self.data.common_indices)))
        log_lik = -0.5*tfm.log(2*PI*variance) - 0.5*sq_diff/variance
        log_lik = tf.reduce_sum(log_lik)

        return log_lik
