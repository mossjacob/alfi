import tensorflow as tf
import torch
import pickle as pkl

from sklearn import preprocessing
from scipy.interpolate import interp1d
from tensorflow import math as tfm

from lafomo.datasets import DataHolder
from lafomo.mcmc import TranscriptionLikelihood
from lafomo.configuration import MCMCConfiguration

from lafomo.mcmc.models import TranscriptionMixedSampler
from lafomo.utilities.tf import discretise, logistic, inverse_positivity
from matplotlib import pyplot as plt

import numpy as np

f64 = np.float64

#%%

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


def get_artificial_dataset(num_genes=20, num_tfs=3):
    """
    Returns:
        nodelay, delay datasets
    """
    nodelay_path = '../data/articial_nodelay.pkl'
    delay_path = '../data/articial_delay.pkl'
    with open(nodelay_path, 'rb') as f:
        nodelay_dataset = pkl.load(f)
    with open(delay_path, 'rb') as f:
        delay_dataset = pkl.load(f)
    return nodelay_dataset, delay_dataset

    tf.random.set_seed(1)
    w = tf.random.normal([num_genes, num_tfs], mean=0.5, stddev=0.71, seed=42, dtype='float64')

    Δ_delay = tf.constant([0, 4, 10], dtype='float64')

    w_0 = tf.zeros(num_genes, dtype='float64')

    true_kbar = logistic((np.array([
        [1.319434062, 1.3962113525, 0.8245041865, 2.2684353378],
        [1.3080045137, 3.3992868747, 2.0189033658, 3.7460822389],
        [2.0189525448, 1.8480506624, 0.6805040228, 3.1039094120],
        [1.7758426875, 0.1907625023, 0.1925539427, 1.8306885751],
        [1.7207442227, 0.1252089546, 0.6297333943, 3.2567248923],
        [1.4878806850, 3.8623843570, 2.4816128746, 4.3931294404],
        [2.0853079514, 2.5115446790, 0.6560607356, 3.0945313562],
        [1.6144843688, 1.8651409657, 0.7785363895, 2.6845058360],
        [1.4858223122, 0.5396687493, 0.5842698019, 3.0026805243],
        [1.6610647522, 2.0486340884, 0.9863876546, 1.4300094581],
        [1.6027276189, 1.4320302060, 0.7175033248, 3.2151637970],
        [2.4912882714, 2.7935526605, 1.2438786874, 4.3944794204],
        [2.894114279, 1.4726280947, 0.7356719860, 2.2316019158],
     [1.7927833839, 1.0405867396, 0.4055775218, 2.9888350247],
     [1.0429721112, 0.1011544950, 0.7330443670, 3.1936843755],
     [1.2519286771, 2.0617880701, 1.0759649567, 3.9406060364],
     [1.4297185709, 1.3578824015, 0.6037986912, 2.6512418604],
     [1.9344878813, 1.4235867760, 0.8226320338, 4.2847217252],
     [1.4325562449, 1.1940752177, 1.0556928599, 4.1850449557],
     [0.8911103971, 1.3560009300, 0.5643954823, 3.4300182328],
     [1.0269654997, 1.0788097511, 0.5268448648, 4.4793299593],
     [0.8378220502, 1.8148234459, 1.0167440138, 4.4903387696]]
    )))
    true_kbar = true_kbar[:num_genes]

    opt = MCMCConfiguration(preprocessing_variance=False,
                            latent_data_present=True,
                            kinetic_exponential=True,
                            weights=True,
                            initial_step_sizes={'logistic': 1e-8, 'latents': 10},
                            delays=True)

    data, fbar, kinetics = artificial_dataset(opt, TranscriptionLikelihood, num_genes=num_genes,
                                              weights=(w, w_0), delays=Δ_delay.numpy(), t_end=10,
                                              true_kbar=true_kbar[:num_genes])
    true_kbar, true_k_fbar = kinetics
    f_i = inverse_positivity(fbar)
    t_observed, t_discretised, common_indices = data.t_observed, data.t_discretised, data.common_indices

    common_indices = common_indices.numpy()

    model = TranscriptionMixedSampler(data, opt)

    # Transcription factor
    plt.title('TFs')
    for i in range(num_tfs):
        plt.plot(t_discretised, f_i[0, i], label=f'TF {i}')
        plt.scatter(t_observed, data.f_obs[0, i], marker='x')
    plt.xticks(np.arange(0, 10))
    plt.legend()
    print(t_discretised.shape)


    lik = model.likelihood
    Δ_nodelay = tf.constant([0, 0, 0], dtype='float64')
    m_pred = lik.predict_m(true_kbar, true_k_fbar, (w), fbar, (w_0), Δ_delay)
    m_pred_nodelay = lik.predict_m(true_kbar, true_k_fbar, (w), fbar, (w_0), Δ_nodelay)
    m_observed_nodelay = tf.stack([m_pred_nodelay.numpy()[:,i,common_indices] for i in range(num_genes)], axis=1)
    m_observed = tf.stack([m_pred.numpy()[:,i,common_indices] for i in range(num_genes)], axis=1)

    p_nodelay = lik.calculate_protein(fbar, true_k_fbar, Δ_nodelay)
    p = lik.calculate_protein(fbar, true_k_fbar, Δ_delay)

    nodelay_dataset = (np.array(p_nodelay), np.array(m_observed_nodelay))
    delay_dataset = (np.array(p), np.array(m_observed))

    with open(nodelay_path, 'wb') as f:
        pkl.dump(nodelay_dataset, f)
    with open(delay_path, 'wb') as f:
        pkl.dump(delay_dataset, f)

    return nodelay_dataset, delay_dataset
