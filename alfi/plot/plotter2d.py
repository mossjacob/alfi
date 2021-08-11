import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .base_plotter import Plotter
from alfi.datasets import scaled_barenco_data
from alfi.models import VariationalLFM, TrainMode, OrdinaryLFMNoPrecompute
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

    def plot_vector_gp(self, h, true_h,
                       time_ass=None,
                       ax=None,
                       figsize=(3, 3),
                       show=True,
                       save_name=None,
                       independent=True,
                       batch_index=None,
                       title='',
                       labels=None,
                       alpha=0.8,
                       cell_colors=None):
        with torch.no_grad():

            x1, x2 = h
            true_x1, true_x2 = true_h
            if not show:
                plt.ioff()

            if ax is None:
                plt.figure(figsize=figsize)
                ax = plt.subplot()
            plt.title(title)

            # Plot trajectory
            ax.plot(x1, x2, color='red')

            # Plot cell points
            indices = np.intersect1d(true_x1.nonzero(), true_x2.nonzero())

            if cell_colors is None:
                cell_colors = time_ass
            ax.scatter(true_x1[indices], true_x2[indices],
                       alpha=alpha, s=5, cmap='viridis', c=cell_colors[indices])

            # Plot inducing vectors
            inducing_points = self.model.inducing_points
            num_inducing = inducing_points.shape[1]
            if isinstance(self.model, OrdinaryLFMNoPrecompute):
                ax.scatter(self.model.initial_state[:, 0], self.model.initial_state[:, 1])

                self.model.set_mode(TrainMode.GRADIENT_MATCH)
                h_grad = self.model((inducing_points, None)).mean
                if independent:
                    ind_x1 = inducing_points[0, :, 0]
                    ind_x2 = inducing_points[1, :, 0]
                else:
                    ind_x1 = inducing_points[0, :, 0]
                    ind_x2 = inducing_points[0, :, 1]

            elif batch_index is not None:
                initial_state = self.model.initial_state.view(2, -1)
                ax.scatter(initial_state[0], initial_state[1])

                self.model.set_mode(TrainMode.NORMAL)
                self.model(inducing_points[0, :, 0])
                traj = self.model.current_trajectory
                ind = traj.view(2, -1, num_inducing)[:, batch_index]
                ind_x1 = ind[0]
                ind_x2 = ind[1]

                self.model.set_mode(TrainMode.GRADIENT_MATCH)
                h_grad = self.model((inducing_points, traj)).mean.view(num_inducing, 2, -1)
                h_grad = h_grad[..., batch_index]

            grad_x1 = h_grad[:, 0]
            grad_x2 = h_grad[:, 1]
            ax.quiver(ind_x1, ind_x2, grad_x1, grad_x2)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
            if save_name is not None:
                plt.savefig(save_name)
            if not show:
                plt.close(plt.gcf())

        return ax
