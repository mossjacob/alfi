import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .base_plotter import Plotter
from alfi.datasets import scaled_barenco_data
from alfi.models import VariationalLFM
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

    def plot_vector_gp(self, x1, x2, true_x1, true_x2,
                       time_ass=None,
                       ax=None,
                       figsize=(3, 3),
                       show=True,
                       save_name=None,
                       independent=True):
        if not show:
            plt.ioff()

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.subplot()
        plt.title('lotka')

        # Plot trajectory
        ax.plot(x1, x2, color='red')
        ax.scatter(self.model.initial_state[:, 0], self.model.initial_state[:, 1])

        # Plot cell points
        indices = np.intersect1d(true_x1[:].nonzero(),
                                 true_x2[:].nonzero())


        ax.scatter(true_x1[indices], true_x2[indices],
                   alpha=0.8, s=5, cmap='viridis', c=time_ass)

        # Plot inducing vectors
        inducing_points = self.model.inducing_points
        with torch.no_grad():
            out = self.model.gp_model(inducing_points).mean
        predator_grad = out[:, 0]
        prey_grad = out[:, 1]
        if independent:
            predator = inducing_points[0, :, 0]
            x2 = inducing_points[1, :, 0]
        else:
            predator = inducing_points[0, :, 0]
            x2 = inducing_points[0, :, 1]

        ax.quiver(predator, x2, predator_grad, prey_grad)

        ax.set_ylabel('unspliced')
        ax.set_xlabel('spliced')
        if save_name is not None:
            plt.savefig(save_name)
        if not show:
            plt.close(plt.gcf())
