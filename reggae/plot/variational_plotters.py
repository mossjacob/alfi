import matplotlib.pyplot as plt
import numpy as np
import torch

from reggae.data_loaders import scaled_barenco_data

plt.style.use('ggplot')


class Plotter:
    """
    This Plotter is designed for gp models.
    """

    def __init__(self, model, gene_names, t_inducing=None):
        self.model = model
        self.gene_names = gene_names
        self.num_genes = self.gene_names.shape[0]
        self.t_inducing = t_inducing
        self.variational = self.t_inducing is not None

    def _plot_barenco(self, mean):
        barenco_f, _ = scaled_barenco_data(mean)
        plt.scatter(np.linspace(0, 1, 7), barenco_f, marker='x', s=60, linewidth=2, label='Barenco et al.')

    def plot_tfs(self, ylim=(-2, 2), num_samples=7, plot_barenco=False, plot_inducing=False, extrap=0.5):
        tf_i = 0

        t_predict = torch.linspace(-extrap, 1+extrap, 80)
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
        for _ in range(num_samples):
            plt.plot(t_predict, self.model.G(q_f.sample()).detach(), alpha=0.3, color='gray')
        plt.plot(t_predict, mean, color='gray')

        if self.variational:
            plt.scatter(self.t_inducing, mean_u, marker='o', color='brown')

            if plot_inducing:
                S = torch.matmul(self.model.q_cholS, self.model.q_cholS.transpose(1, 2))
                std_u = torch.sqrt(torch.diagonal(S[0])).detach()
                u = torch.squeeze(self.model.q_m[tf_i].detach())
                print(std_u, u.shape, self.model.q_m.shape)
                plt.plot(self.t_inducing, u)
                plt.fill_between(self.t_inducing.view(-1), u + std_u, u - std_u, color='green', alpha=0.5)

        plt.title('Latent')
        plt.ylim(ylim)

    def plot_tf(self, pred_t):
        """Imported from old plotter"""
        mu_post = self.model.predict_f(pred_t, )
        plt.plot(pred_t, mu_post)
        barencof, _ = scaled_barenco_data(mu_post)
        plt.scatter(self.data.t, barencof, marker='x')
        return mu_post

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

    def plot_convergence(self, trainer):
        titles = ['basal', 'decay', 'sensitivity', 'lengthscale']
        datas = [np.array(trainer.basalrates)[:,:,0],
                 np.array(trainer.decayrates)[:,:,0],
                 np.array(trainer.sensitivities)[:,:,0],
                 np.array(trainer.lengthscales)]

        plt.figure(figsize=(5, 6))
        for i, (title, data) in enumerate(zip(titles, datas)):
            plt.subplot(411 + i)
            plt.title(title)
            # if data.ndim > 1:
            #     for j in range(data.shape[1]):

            plt.plot(data)

    def plot_genes(self, t_predict, t_scatter=None, y_scatter=None):
        """
        Parameters:
            t_predict: tensor (T*,)
            t_scatter: tensor (T,)
            y_scatter: tensor (J, T)
        """
        mu, var = self.model.predict_m(t_predict)
        print(mu.shape)
        var = 2 * torch.sqrt(var)
        plt.figure()
        for i in range(self.num_genes):
            plt.subplot(self.num_genes, 3, i + 1)
            plt.plot(t_predict, mu[i].detach())
            plt.fill_between(t_predict, mu[i] + var[i], mu[i] - var[i], alpha=0.4)

            if t_scatter is not None:
                plt.scatter(t_scatter, y_scatter[i])

            # plt.ylim(-0.2, max(mu[j]) * 1.2)
        plt.tight_layout()
        return mu, var
