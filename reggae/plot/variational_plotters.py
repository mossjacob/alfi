import matplotlib.pyplot as plt
import numpy as np

from reggae.data_loaders import scaled_barenco_data

plt.style.use('ggplot')


class Plotter:
    """
    This Plotter is designed for non-MCMC models.
    """

    def __init__(self, model, gene_names):
        self.model = model
        self.gene_names = gene_names

    def plot_kinetics(self):
        plt.figure(figsize=(8, 4))
        plt.subplot(3, 3, 1)
        B = np.squeeze(self.model.basal_rate.detach().numpy())
        S = np.squeeze(self.model.sensitivity.detach().numpy())
        D = np.squeeze(self.model.decay_rate.detach().numpy())
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
            plt.xticks(rotation=rotation)


    def plot_losses(self, trainer, last_x=50):
        plt.figure(figsize=(5, 2))
        plt.plot(np.sum(trainer.losses, axis=1)[-last_x:])
        plt.title('Total loss')
        plt.figure(figsize=(5, 2))
        plt.subplot(221)
        plt.plot(trainer.losses[-last_x:, 0])
        plt.title('Loss')
        plt.subplot(222)
        plt.plot(trainer.losses[-last_x:, 1])
        plt.title('KL-divergence')
