import matplotlib.pyplot as plt
import numpy as np
import torch

from reggae.data_loaders import scaled_barenco_data

plt.style.use('ggplot')


class Plotter:
    """
    This Plotter is designed for non-MCMC models.
    """

    def __init__(self, model, gene_names, t_inducing):
        self.model = model
        self.gene_names = gene_names
        self.t_inducing = t_inducing

    def _plot_barenco(self, mean):
        barenco_f, _ = scaled_barenco_data(mean)
        plt.scatter(np.linspace(0, 1, 7), barenco_f, marker='x', s=60, linewidth=2, label='Barenco et al.')

    def plot_tfs(self, ylim=(-2, 2), num_samples=7, plot_barenco=False, plot_inducing=False):
        tf_i = 0

        t_predict = torch.linspace(0, 1, 80)
        q_f = self.model.get_tfs(t_predict.reshape(-1))
        q_u = self.model.get_tfs(self.model.inducing_inputs)
        mean = self.model.G(q_f.mean).detach().numpy()  # (T)
        mean_u = self.model.G(q_u.mean).detach().numpy()
        std = torch.sqrt(q_f.variance)[tf_i].detach().numpy()
        std_u = torch.sqrt(q_u.variance[0]).detach().numpy()
        plt.figure(figsize=(5, 3))
        if plot_barenco:
            self._plot_barenco(mean)
        plt.fill_between(t_predict, mean + std, mean - std, color='orangered', alpha=0.5)
        plt.scatter(self.t_inducing, mean_u, marker='o', color='brown')
        for _ in range(num_samples):
            plt.plot(t_predict, self.model.G(q_f.sample()).detach(), alpha=0.3, color='gray')
        plt.plot(t_predict, mean, color='gray')

        if plot_inducing:
            S = torch.matmul(self.model.q_cholS, self.model.q_cholS.transpose(1, 2))
            std_u = torch.sqrt(torch.diagonal(S[0])).detach()
            u = torch.squeeze(self.model.q_m[tf_i].detach())
            print(std_u, u.shape, self.model.q_m.shape)
            plt.plot(self.t_inducing, u)
            plt.fill_between(self.t_inducing.view(-1), u + std_u, u - std_u, color='green', alpha=0.5)

        plt.title('Latent')
        plt.ylim(ylim)

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
            plt.xticks(rotation=45)


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
