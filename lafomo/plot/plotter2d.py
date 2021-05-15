import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .base_plotter import Plotter
from lafomo.datasets import scaled_barenco_data
from lafomo.models import VariationalLFM
from .colours import Colours


plt.style.use('seaborn')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(style='white', font="CMU Serif")


class Plotter2d(Plotter):
    """
    This Plotter is designed for 2D LFMs or LFOs.
    """

    def __init__(self, model, output_names, style='seaborn'):
        super().__init__(model, output_names, style=style)
        self.num_outputs = self.output_names.shape[0]
        self.num_replicates = self.model.num_outputs // self.num_outputs

    def plot_gp(self, gp,
                t_predict, t_scatter=None, y_scatter=None,
                num_samples=0,
                transform=lambda x:x,
                ylim=None,
                titles=None,
                max_plots=10,
                replicate=0,
                only_plot_index=None,
                ax=None,
                plot_inducing=False,
                color=Colours.line_color,
                shade_color=Colours.shade_color):
        """
        Parameters:
            gp: output distribution of LFM or associated GP models.
            t_predict: tensor (T*,) prediction input vector
            t_scatter: tensor (T,) target input vector
            y_scatter: tensor (J, T) target output vector
        """
        mean = gp.mean.detach().transpose(0, 1)  # (T, J)
        std = gp.variance.detach().transpose(0, 1).sqrt()
        num_plots = mean.shape[0]
        mean = mean.view(num_plots, self.num_replicates, -1).transpose(0, 1)[replicate].detach()
        std = std.view(num_plots, self.num_replicates, -1).transpose(0, 1)[replicate].detach()
        mean = transform(mean)
        # std = transform(std)
        num_plots = min(max_plots, num_plots)
        axes_given = ax is not None
        if not axes_given:
            fig = plt.figure(figsize=(6, 4 * np.ceil(num_plots / 3)))
        for i in range(num_plots):
            if only_plot_index is not None:
                i = only_plot_index
            if not axes_given:
                ax = fig.add_subplot(num_plots, min(num_plots, 3), i + 1)
            if titles is not None:
                ax.set_title(titles[i])

            ax.plot(t_predict, mean[i], color=color)
            ax.fill_between(t_predict,
                            mean[i] + 2 * std[i],
                            mean[i] - 2 * std[i],
                            color=shade_color, alpha=0.3)

            for _ in range(num_samples):
                ax.plot(t_predict, transform(gp.sample().detach()).transpose(0, 1)[i], alpha=0.3, color=color)

            if self.variational and plot_inducing:
                inducing_points = self.model.inducing_points.detach()[0].squeeze()
                ax.scatter(inducing_points, np.zeros_like(inducing_points), marker='_', c='black', linewidths=2)
            if t_scatter is not None:
                ax.scatter(t_scatter, y_scatter[replicate, i], color=Colours.scatter_color, marker='x')
            if ylim is None:
                lb = min(mean[i])
                lb -= 0.2 * lb
                ub = max(mean[i]) * 1.2
                ax.set_ylim(lb, ub)
            else:
                ax.set_ylim(ylim)

            if axes_given:
                break
        return gp

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
