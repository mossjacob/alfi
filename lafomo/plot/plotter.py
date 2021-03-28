import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lafomo.datasets import scaled_barenco_data
from lafomo.models import VariationalLFM

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(font="CMU Serif")


class Plotter:
    """
    This Plotter is designed for gp models.
    """

    def __init__(self, model, output_names, style='seaborn'):
        self.model = model
        self.output_names = output_names
        self.num_outputs = self.output_names.shape[0]
        self.num_replicates = self.model.num_outputs // self.num_outputs
        self.variational = isinstance(self.model, VariationalLFM)
        palette = sns.color_palette('colorblind')
        self.shade_color = palette[0]
        self.line_color = palette[0]
        self.scatter_color = palette[3]
        self.bar1_color = palette[3]
        self.bar2_color = palette[2]
        plt.style.use(style)
        sns.set(font="CMU Serif")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'CMU Serif'

    def _plot_barenco(self, mean):
        barenco_f, _ = scaled_barenco_data(mean)
        plt.scatter(np.linspace(0, 12, 7), barenco_f, marker='x', s=60, linewidth=2, label='Barenco et al.')

    def plot_outputs(self, t_predict, replicate=0, t_scatter=None, y_scatter=None, model_kwargs={}, ylim=None, max_plots=10):
        """
        Parameters:
            t_predict: tensor (T*,)
            t_scatter: tensor (T,)
            y_scatter: tensor (J, T)
            model_kwargs: dictionary of keyword arguments to send to the model predict_m function
        """
        q_m = self.model.predict_m(t_predict, **model_kwargs)
        mu = q_m.mean.detach().transpose(0, 1)  # (T, J)
        std = q_m.variance.detach().transpose(0, 1).sqrt()
        mu = mu.view(self.num_outputs, self.num_replicates, -1).transpose(0, 1)
        std = std.view(self.num_outputs, self.num_replicates, -1).transpose(0, 1)
        num_plots = min(max_plots, self.num_outputs)
        plt.figure(figsize=(6, 4 * np.ceil(num_plots / 3)))
        for i in range(num_plots):
            plt.subplot(num_plots, 3, i + 1)
            plt.title(self.output_names[i])
            plt.plot(t_predict, mu[replicate, i].detach(), color=self.line_color)
            plt.fill_between(t_predict,
                             mu[replicate, i] + 2*std[replicate, i],
                             mu[replicate, i] - 2*std[replicate, i],
                             color=self.shade_color, alpha=0.3)

            if t_scatter is not None:
                plt.scatter(t_scatter, y_scatter[replicate, i], color=self.scatter_color, marker='x')

            if ylim is None:
                lb = min(mu[replicate, i])
                lb -= 0.2 * lb
                ub = max(mu[replicate, i]) * 1.2
                plt.ylim(lb, ub)
        plt.tight_layout()
        return q_m

    def plot_latents(self, t_predict, ylim=None, num_samples=7, plot_barenco=False, plot_inducing=False):
        q_f = self.model.predict_f(t_predict.reshape(-1))
        mean = q_f.mean.detach().transpose(0, 1)  # (T)
        std = q_f.variance.detach().sqrt().transpose(0, 1)  # (T)
        plt.figure(figsize=(5, 3*mean.shape[0]))
        for i in range(mean.shape[0]):
            plt.subplot(mean.shape[0], 1, i+1)
            plt.plot(t_predict, mean[i], color=self.line_color)
            plt.fill_between(t_predict,
                             mean[i] + 2 * std[i],
                             mean[i] - 2 * std[i],
                             color=self.shade_color, alpha=0.3)
            for _ in range(num_samples):
                plt.plot(t_predict, q_f.sample().detach().transpose(0, 1)[i], alpha=0.3, color=self.line_color)

            if plot_barenco:
                self._plot_barenco(mean[i])
            if self.variational:
                inducing_points = self.model.inducing_points.detach()[0].squeeze()
                plt.scatter(inducing_points, np.zeros_like(inducing_points), marker='_', c='black', linewidths=4)

            if ylim is None:
                plt.ylim(min(mean[i])-0.3, max(mean[i])+0.3)
            else:
                plt.ylim(ylim)

        plt.title('Latent')
        return q_f

    def plot_double_bar(self, params, labels, ground_truths=None):
        real_bars = [None] * len(params) if ground_truths is None else ground_truths
        vars = [0] * len(params)
        fig, axes = plt.subplots(ncols=len(params), figsize=(8, 4))
        plotnum = 0
        num_bars = self.output_names.shape[0]
        for A, B, var, label in zip(params, real_bars, vars, labels):
            if B is None:
                axes[plotnum].bar(np.arange(num_bars), A, width=0.4, tick_label=self.output_names, color=self.bar1_color)
            else:
                axes[plotnum].bar(np.arange(num_bars) - 0.2, A, width=0.4, tick_label=self.output_names, color=self.bar1_color)
                axes[plotnum].bar(np.arange(num_bars) + 0.2, B, width=0.4, color=self.bar2_color, align='center')

            axes[plotnum].set_title(label)
            axes[plotnum].tick_params(axis='x', labelrotation=45)
            plotnum += 1

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
                 np.array(trainer.lengthscales)[:, 0, 0]]

        plt.figure(figsize=(5, 6))
        for i, (title, data) in enumerate(zip(titles, datas)):
            plt.subplot(411 + i)
            plt.title(title)
            # if data.ndim > 1:
            #     for j in range(data.shape[1]):

            plt.plot(data)
