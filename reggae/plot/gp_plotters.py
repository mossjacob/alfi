import matplotlib.pyplot as plt
import numpy as np

from tensorflow import math as tfm

from reggae.data_loaders import scaled_barenco_data

plt.style.use('ggplot')


class Plotter:
    """
    This Plotter is designed for non-MCMC models.
    """

    def __init__(self, data, model, gene_names):
        self.data = data
        self.model = model
        self.num_genes = data.m_obs.shape[1]
        self.gene_names = gene_names

    def plot_genes(self, pred_t):
        mu, var = self.model.predict_m(pred_t)
        var = 2 * tfm.sqrt(var)
        for j in range(self.num_genes):
            plt.subplot(self.num_genes, 3, j + 1)
            plt.plot(pred_t, mu[j])
            plt.scatter(self.data.t, self.data.m_obs[0, j])
            plt.fill_between(pred_t, mu[j] + var[j], mu[j] - var[j], alpha=0.4)
            plt.ylim(-0.2, max(mu[j]) * 1.2)
        plt.tight_layout()
        return mu, var

    def plot_tf(self, pred_t):
        mu_post = self.model.predict_f(pred_t, )
        plt.plot(pred_t, mu_post)
        barencof, _ = scaled_barenco_data(mu_post)
        plt.scatter(self.data.t, barencof, marker='x')
        return mu_post

    def plot_kinetics(self):
        plt.subplot(3, 3, 1)
        B = self.model.mean_function.B.numpy()
        S = self.model.kernel.S.numpy()
        D = self.model.kernel.D.numpy()
        B_barenco = np.array([2.6, 1.5, 0.5, 0.2, 1.35])
        B_barenco = (B_barenco / np.mean(B_barenco) * np.mean(B))[[0, 4, 2, 3, 1]]
        S_barenco = (np.array([3, 0.8, 0.7, 1.8, 0.7]) / 1.8)[[0, 4, 2, 3, 1]]
        S_barenco = (S_barenco / np.mean(S_barenco) * np.mean(S))[[0, 4, 2, 3, 1]]
        D_barenco = (np.array([1.2, 1.6, 1.75, 3.2, 2.3]) * 0.8 / 3.2)[[0, 4, 2, 3, 1]]
        D_barenco = (D_barenco / np.mean(D_barenco) * np.mean(D))[[0, 4, 2, 3, 1]]

        data = [B, S, D]
        barenco_data = [B_barenco, S_barenco, D_barenco]
        vars = [0, 0, 0]  # [ S_mcmc, D_mcmc]
        labels = ['Basal rates', 'Sensitivities', 'Decay rates']

        plotnum = 331
        for A, B, var, label in zip(data, barenco_data, vars, labels):
            plt.subplot(plotnum)
            plotnum += 1
            plt.bar(np.arange(5) - 0.2, A, width=0.4, tick_label=self.gene_names)
            plt.bar(np.arange(5) + 0.2, B, width=0.4, color='blue', align='center')

            plt.title(label)
