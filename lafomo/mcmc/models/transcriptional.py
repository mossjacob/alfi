import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd

from lafomo.datasets import DataHolder
from lafomo.utilities.tf import rotate, logit, logistic, LogisticNormal, inverse_positivity, \
    save_object
from lafomo.options import MCMCOptions
from lafomo.mcmc.parameter import Parameter, MHParameter, HMCParameter, ParamType
from lafomo.mcmc.samplers.hmc import HMCSampler
from lafomo.mcmc.samplers.latent import LatentGPSampler
from lafomo.mcmc.samplers.delay import DelaySampler
from lafomo.mcmc.samplers.gibbs import GibbsSampler
from lafomo.mcmc.samplers.mixed import MixedSampler
from lafomo.mcmc.gp.gp_kernels import GPKernelSelector
from lafomo.mcmc.results import SampleResults
from lafomo.mcmc.models import MCMCLFM

import numpy as np
import pickle

PI = tf.constant(np.pi, dtype='float64')
f64 = np.float64


class TranscriptionRegulationLFM(MCMCLFM):
    """
    An updated version of the Metropolis-Hastings model from Titsias et al. (2012) using a mixed sampler
    """
    def __init__(self, data: DataHolder, options: MCMCOptions):
        super().__init__(data, options)
        self.N_p = data.t_discretised.shape[0]
        step_sizes = self.options.initial_step_sizes
        logistic_step_size = step_sizes['nuts'] if 'nuts' in step_sizes else 0.00001
        self.subsamplers = list()
        # Kinetics
        if options.kinetic_exponential:
            kinetic_transform = lambda x: tf.exp(logit(x))
        else:
            kinetic_transform = logit

        basal_rate = HMCParameter(
            'basal',
            LogisticNormal(0.01, 30),
            0.8 * tf.ones((self.num_genes, 1), dtype='float64'),
            transform=kinetic_transform
        )
        sensitivity = HMCParameter(
            'sensitivity',
            LogisticNormal(0.01, 30),
            0.8 * tf.ones((self.num_genes, 1), dtype='float64'),
            transform=kinetic_transform
        )
        decay_rate = HMCParameter(
            'decay',
            LogisticNormal(0.01, 30),
            0.8 * tf.ones((self.num_genes, 1), dtype='float64'),
            transform=kinetic_transform
        )
        kinetics = [basal_rate, decay_rate, sensitivity]

        # Add optional kinetic parameters
        if options.initial_conditions:
            self.initial_conditions = HMCParameter(
                'initial',
                LogisticNormal(0.01, 30),
                0.8 * tf.ones((self.num_genes, 1), dtype='float64'),
                transform=kinetic_transform
            )
            kinetics.append(self.initial_conditions)
        if options.translation:
            self.protein_decay = HMCParameter(
                'protein_decay',
                LogisticNormal(0.1, 7),
                0.8 * tf.ones((self.num_tfs,), dtype='float64'),
                transform=logit
            )
            kinetics.append(self.protein_decay)

        kinetics_subsampler = HMCSampler(self.likelihood, kinetics, logistic_step_size)
        self.subsamplers.append(kinetics_subsampler)

        # Weights
        if options.weights:
            self.weights = HMCParameter(
                'w',
                LogisticNormal(f64(-2), f64(2)),
                logistic(1 * tf.ones((self.num_genes, self.num_tfs), dtype='float64')),
                param_type=ParamType.HMC
            )
            self.weights_biases = HMCParameter(
                'w_0',
                LogisticNormal(f64(-0.8), f64(0.8)),
                logistic(0 * tf.ones(self.num_genes, dtype='float64'))
            )
            weights = [self.weights, self.weights_biases]
            weights_subsampler = HMCSampler(self.likelihood, weights, logistic_step_size)
            self.subsamplers.append(weights_subsampler)

        # Latent function & GP hyperparameters
        self.kernel_selector = GPKernelSelector(data, options)
        kernel_initial = self.kernel_selector.initial_params()

        f_step_size = step_sizes['latents'] if 'latents' in step_sizes else 20
        latent_likelihood = self.tfs_likelihood if options.latent_data_present else None
        latents_initial = 0.3 * tf.ones((self.num_replicates, self.num_tfs, self.N_p), dtype='float64')

        if self.options.joint_latent:
            latents_initial = [latents_initial, *kernel_initial]
        else:
            # GP kernel
            kernel_initial = self.kernel_selector.initial_params()
            kernel_params = list()
            for i, k in enumerate(kernel_initial):
                range = self.kernel_selector.ranges()[i]
                transform = lambda x: logit(x, nan_replace=range[1])
                kernel_params.append(HMCParameter(
                    'kernel_{}' % i,
                    LogisticNormal(*range),
                    logistic(k),
                    transform=transform
                ))
            kernel_params_sampler = HMCSampler(self.likelihood, kernel_params, step_size=0.1 * logistic_step_size)
            self.subsamplers.append(kernel_params_sampler)

        latents = Parameter('latent', None, latents_initial)
        latents_sampler = LatentGPSampler(self.likelihood, latent_likelihood,
                                          latents, self.kernel_selector,
                                          f_step_size,
                                          joint=options.joint_latent,
                                          kernel_exponential=options.kernel_exponential)

        self.subsamplers.append(latents_sampler)

        if options.delays:
            delay_prior = tfd.InverseGamma(f64(0.01), f64(0.01)) #TODO choose between these two
            delay_prior = tfd.Exponential(f64(0.3))
            delay = Parameter('Δ', delay_prior,
                              0.6 * tf.ones(self.num_tfs, dtype='float64'))
            delay_sampler = DelaySampler(self.likelihood, delay, 0, 10)
            self.subsamplers.append(delay_sampler)
        σ2_f = None
        if not options.preprocessing_variance:
            def f_sq_diff_fn(all_states):
                f_pred = inverse_positivity(self.parameter_state['latent'][0])
                sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred),self.data.common_indices)))
                return tf.reduce_sum(sq_diff, axis=0)
            σ2_f = Parameter('σ2_f',
                             tfd.InverseGamma(f64(0.01), f64(0.01)),
                             1e-4*tf.ones((self.num_tfs,1), dtype='float64'))
            σ2_f_sampler = GibbsSampler(self.likelihood, σ2_f, f_sq_diff_fn, self.N_p)
            self.subsamplers.append(σ2_f_sampler)
        # White noise for genes
        if not options.preprocessing_variance:
            def m_sq_diff_fn():
                m_pred = self.likelihood.predict_m(**self.parameter_state)
                sq_diff = tfm.square(self.data.m_obs - tf.transpose(tf.gather(tf.transpose(m_pred), self.data.common_indices)))
                return tf.reduce_sum(sq_diff, axis=0)
            σ2_m = Parameter('σ2_m',
                                  tfd.InverseGamma(f64(0.01), f64(0.01)),
                                  1e-3*tf.ones((self.num_genes, 1), dtype='float64'))
            σ2_m_sampler = GibbsSampler(self.likelihood, σ2_m, m_sq_diff_fn, self.N_p)
        else:
            σ2_m = HMCParameter('σ2_m', LogisticNormal(f64(1e-5), f64(1e-2)), # f64(max(np.var(data.f_obs, axis=1)))                                logistic(f64(5e-3))*tf.ones(self.num_genes, dtype='float64'),
                                transform=logit)
            σ2_m_sampler = HMCSampler(self.likelihood, [self.σ2_m], logistic_step_size)

        self.subsamplers.append(σ2_m_sampler)

        def iteration_callback(current_state):
            print('iteration_callback()', current_state)
            self.parameter_state = current_state

        self.sampler = MixedSampler(self.subsamplers, iteration_callback=iteration_callback)

    def sample(self, T=2000, **kwargs):
        return self.sampler.sample(T, **kwargs)

    @tf.function
    def calculate_protein(self, fbar, protein_decay, Δ):  # Calculate p_i vector
        τ = self.data.t_discretised
        f_i = inverse_positivity(fbar)
        δ_i = tf.reshape(protein_decay, (-1, 1))
        if self.options.delays:
            # Add delay
            Δ = tf.cast(Δ, 'int32')

            for r in range(self.num_replicates):
                f_ir = rotate(f_i[r], -Δ)
                mask = ~tf.sequence_mask(Δ, f_i.shape[2])
                f_ir = tf.where(mask, f_ir, 0)
                mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
                mask[r] = 1
                f_i = (1 - mask) * f_i + mask * f_ir

        # Approximate integral (trapezoid rule)
        resolution = τ[1] - τ[0]
        sum_term = tfm.multiply(tfm.exp(δ_i * τ), f_i)
        cumsum = 0.5 * resolution * tfm.cumsum(sum_term[:, :, :-1] + sum_term[:, :, 1:], axis=2)
        integrals = tf.concat([tf.zeros((self.num_replicates, self.num_tfs, 1), dtype='float64'), cumsum], axis=2)
        exp_δt = tfm.exp(-δ_i * τ)
        p_i = exp_δt * integrals
        return p_i

    @tf.function
    def predict_m(self,
                  initial, basal, decay, sensitivity,
                  protein_decay, latent, **optional_parameters):

        τ = self.data.t_discretised
        fbar = latent[0]
        p_i = inverse_positivity(fbar)
        if self.options.translation:
            Δ = optional_parameters['Δ'] if self.options.delays else None
            p_i = self.calculate_protein(fbar, protein_decay, Δ)

        # Calculate m_pred
        resolution = τ[1] - τ[0]
        if self.options.weights:
            w = optional_parameters['w']
            w_0 = optional_parameters['w_0']
            interactions = tf.matmul(w, tfm.log(p_i + 1e-100)) + w_0
            G = tfm.sigmoid(interactions)  # TF Activation Function (sigmoid)
        else:
            G = tf.tile(p_i, (1, self.num_genes, 1))

        sum_term = G * tfm.exp(decay * τ)
        integrals = tf.concat([tf.zeros((self.num_replicates, self.num_genes, 1), dtype='float64'),  # Trapezoid rule
                               0.5 * resolution * tfm.cumsum(sum_term[:, :, :-1] + sum_term[:, :, 1:], axis=2)], axis=2)
        exp_dt = tfm.exp(-decay * τ)
        integrals = tfm.multiply(exp_dt, integrals)

        m_pred = basal / decay + sensitivity * integrals
        if self.options.initial_conditions:
            m_pred += tfm.multiply((initial - basal / decay), exp_dt)
        return m_pred

    @tf.function
    def _genes(self, σ2_m=None, **parameter_state):
        # print('_genes', parameter_state)
        m_pred = self.predict_m(σ2_m=σ2_m, **parameter_state)
        sq_diff = tfm.square(self.data.m_obs - tf.transpose(tf.gather(tf.transpose(m_pred), self.data.common_indices)))

        variance = tf.reshape(σ2_m, (-1, 1))
        if self.preprocessing_variance:
            variance = logit(variance) + self.data.σ2_m_pre  # add PUMA variance
        log_lik = -0.5 * tfm.log(2 * PI * variance) - 0.5 * sq_diff / variance
        log_lik = tf.reduce_sum(log_lik)
        return log_lik

    @tf.function  # (experimental_compile=True)
    def likelihood(self, **parameters):
        """
        Likelihood of the form:
        N(m(t), s(t))
        where m(t) = b/d + (a - b/d) exp(-dt) + s int^t_0 G(p(u); w) exp(-d(t-u)) du
        """
        # print(self.parameter_state)
        # print('likelihood', parameters)
        parameter_state = {**self.parameter_state, **parameters}
        return self._genes(**parameter_state)

    @tf.function  # (experimental_compile=True)
    def tfs_likelihood(self, **parameters):
        """
        Computes log-likelihood of the transcription factors.
        """
        parameter_state = {**self.parameter_state, **parameters}

        σ2_f = parameter_state['σ2_f']
        latent = parameter_state['latent']

        # assert self.options.tf_mrna_present
        if not self.preprocessing_variance:
            variance = tf.reshape(σ2_f, (-1, 1))
        else:
            variance = self.data.σ2_f_pre
        f_pred = inverse_positivity(latent[0])
        sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred), self.data.common_indices)))
        log_lik = -0.5 * tfm.log(2 * PI * variance) - 0.5 * sq_diff / variance
        log_lik = tf.reduce_sum(log_lik)

        return log_lik

    def sample_proteins(self, results, num_results):
        p_samples = list()
        for i in range(1, num_results + 1):
            delta = results.Δ[i] if results.Δ is not None else None
            p_samples.append(self.likelihood.calculate_protein(results.fbar[-i],
                                                               results.k_fbar[-i], delta))
        return np.array(p_samples)

    def sample_latents(self, results, num_results, step=1):
        m_preds = list()
        for i in range(1, num_results, step):
            m_preds.append(self.predict_m_with_results(results, i))
        return np.array(m_preds)

    def results(self, burnin=0):
        Δ = σ2_f = k_fbar = None
        σ2_m = self.samples[self.state_indices['σ2_m']][burnin:]
        if self.options.preprocessing_variance:
            σ2_m = logit(σ2_m)
        else:
            σ2_f = self.samples[self.state_indices['σ2_f']][burnin:]

        nuts_index = 0
        kbar = self.samples[self.state_indices['kinetics']][nuts_index].numpy()[burnin:]
        fbar = self.samples[self.state_indices['latents']]
        if self.options.translation:
            nuts_index += 1
            k_fbar = self.samples[self.state_indices['kinetics']][nuts_index].numpy()[burnin:]
            if k_fbar.ndim < 3:
                k_fbar = np.expand_dims(k_fbar, 2)
        if not self.options.joint_latent:
            kernel_params = self.samples[self.state_indices['kernel_params']][burnin:]
        else:
            kernel_params = [fbar[1][burnin:], fbar[2][burnin:]]
            fbar = fbar[0][burnin:]
        wbar = tf.stack(
            [logistic(1 * tf.ones((self.num_genes, self.num_tfs), dtype='float64')) for _ in range(fbar.shape[0])],
            axis=0)
        w_0bar = tf.stack([0.5 * tf.ones(self.num_genes, dtype='float64') for _ in range(fbar.shape[0])], axis=0)
        if self.options.weights:
            nuts_index += 1
            wbar = self.samples[self.state_indices['kinetics']][nuts_index][burnin:]
            w_0bar = self.samples[self.state_indices['kinetics']][nuts_index + 1][burnin:]
        if self.options.delays:
            Δ = self.samples[self.state_indices['Δ']][burnin:]
        return SampleResults(self.options, fbar, kbar, k_fbar, Δ, kernel_params, wbar, w_0bar, σ2_m, σ2_f)


    def predict_m_with_results(self, results, i=1):
        delay = results.Δ[-i] if self.options.delays else None
        k_fbar = results.k_fbar[-i] if self.options.translation else None
        return self.likelihood.predict_m(results.kbar[-i], k_fbar, results.wbar[-i],
                                         results.fbar[-i], results.w_0bar[-i], delay)

    def predict_m_with_current(self):
        return self.likelihood.predict_m(self.params.kinetics.value[0],
                                         self.params.kinetics.value[1],
                                         self.params.weights.value[0],
                                         self.params.latents.value,
                                         self.params.weights.value[1])

    def save(self, name):
        save_object({'samples': self.samples, 'is_accepted': self.is_accepted}, f'custom-{name}')

    @staticmethod
    def load(name, args):
        model = TranscriptionRegulationLFM(*args)

        import os
        path = os.path.join(os.getcwd(), 'saved_models')
        fs = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f'custom-{name}')]
        files = sorted(fs, key=os.path.getmtime)
        with open(files[-1], 'rb') as f:
            saved_model = pickle.load(f)
            model.samples = saved_model['samples']
            model.is_accepted = saved_model['is_accepted']
        for param in model.active_params:
            index = model.state_indices[param.name]
            param_samples = model.samples[index]
            if type(param_samples) is list:
                param_samples = [[param_samples[i][-1] for i in range(len(param_samples))]]

            param.value = param_samples[-1]

        return model

    @staticmethod
    def initialise_from_state(args, state):
        model = TranscriptionRegulationLFM(*args)
        model.is_accepted = state.is_accepted
        model.samples = state.samples
        return model
