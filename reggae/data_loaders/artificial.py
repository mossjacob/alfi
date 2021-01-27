import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from scipy.interpolate import interp1d
from tensorflow import math as tfm

from reggae.data_loaders import DataHolder
from reggae.tf_utilities import discretise, logistic
from reggae.mcmc import Options

f64 = np.float64

def artificial_dataset(opt, likelihood_class, t_end=12, num_genes=13, num_tfs=3, weights=None, delays=[1, 4, 10], true_kbar=None):
    t = np.arange(t_end)
    τ, common_indices = discretise(t)
    time = (t, τ, tf.constant(common_indices))
    N_p = τ.shape[0]
    N_m = t.shape[0]

    # Transcription factors
    C = np.array([0.01, 0.2, 0.47, 0.51, 0.35, 0.19, 0.17, 0.24, 0.37, 0.47, 0.39, 0.2, 0.05, 0.025, 0.025, 0.01, 0.005, 0.005])
    A = np.array([0.01, 0.1, 0.22, 0.44, 0.53, 0.41, 0.23, 0.13, 0.09, 0.035, 0.022, 0.02, 0.015, 0.01, 0.005, 0.005])
    B = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.16, 0.4, 0.36, 0.23, 0.12, 0.05, 0.025, 0.025, 0.01, 0.005, 0.005])
    interp = interp1d(np.arange(A.shape[0]), A, kind='cubic')
    A = interp(np.linspace(0,14, τ.shape[0]))
    interp = interp1d(np.arange(B.shape[0]), B, kind='cubic')
    B = interp(np.linspace(0,14, τ.shape[0]))
    interp = interp1d(np.arange(C.shape[0]), C, kind='quadratic')
    C = interp(np.linspace(0,14, τ.shape[0]))

    fbar = np.array([A, B, C])
    fbar = 15*preprocessing.normalize(fbar)
    fbar = np.expand_dims(fbar, 0)
    f_observed = tf.stack([fbar[:, i, common_indices] for i in range(num_tfs)], axis=1)
    fbar = tfm.log((tfm.exp(fbar)-1))
    f_i = tfm.log(1+tfm.exp(fbar))
    f_observed += tf.random.normal([N_m], stddev=tf.sqrt(0.02*f_observed), dtype='float64')
    # Kinetics
    true_k_fbar = logistic(f64(np.array([2, 2, 2]).T)) #a was [0.1, 0.1, 0.1]
    if true_kbar is None:
        true_kbar = (np.array([[0.50563, 0.66, 0.893, 0.9273],
                            [0.6402, 0.6335, 0.7390, 0.7714],
                            [0.6202, 0.6935, 0.7990, 0.7114],
                            [0.5328, 0.5603, 0.6498, 0.9244],
                            [0.5328, 0.6603, 0.6798, 0.8244],
                            [0.5939, 0.5821, 0.77716, 0.8387],
                            [0.50, 0.68, 0.75716, 0.8587],
                            [0.58, 0.67, 0.57, 0.95],
                            [0.5553, 0.5734, 0.6462, 0.9068],
                            [0.5750, 0.5548, 0.6380, 0.7347],
                            [0.5373, 0.5277, 0.6319, 0.8608],
                            [0.5372, 0.5131, 0.8000, 0.9004],
                            [0.5145, 0.5818, 0.6801, 0.9129]]))
    if weights is None:
        w = 1*tf.ones((num_genes, num_tfs), dtype='float64')
        w_0 = tf.zeros(num_genes, dtype='float64')
    else:
        w = weights[0]
        w_0 = weights[1]


    # Genes
    temp_data = DataHolder((np.ones((1, num_genes, N_m)), np.ones((1, num_tfs, N_m))), None, time)
    temp_lik = likelihood_class(temp_data, opt)
    
    Δ = tf.constant(delays, dtype='float64')
    m_pred = temp_lik.predict_m(true_kbar, true_k_fbar, (w), fbar, (w_0), Δ)

    m_observed = tf.stack([m_pred.numpy()[:,i,common_indices] for i in range(num_genes)], axis=1)
    m_observed += tf.random.normal([N_m], stddev=tf.sqrt(0.02*m_observed), dtype='float64')
    data = (m_observed, f_observed)

    data = DataHolder(data, None, time)

    return data, fbar, (true_kbar, true_k_fbar)