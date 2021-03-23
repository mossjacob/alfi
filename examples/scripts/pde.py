# Spatio-temporal Transcriptomics

import torch
import numpy as np
from datetime import datetime
from os import path

from torch.nn import Parameter
from gpytorch.constraints import Interval
from matplotlib import pyplot as plt

from lafomo.models import MultiOutputGP, PartialLFM
from lafomo.models.pdes import ReactionDiffusion
from lafomo.trainer import VariationalTrainer
from lafomo.datasets import ToySpatialTranscriptomics, P53Data
from lafomo.configuration import VariationalConfiguration
from lafomo.utilities.torch import save, load
from lafomo.plot import Plotter

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt


def scatter_output(ax, tx, output, title=None):
    ax.set_title(title)
    ax.scatter(tx[0], tx[1], c=output)
    ax.set_xlabel('time')
    ax.set_ylabel('distance')
    ax.set_aspect('equal')


dataset = ToySpatialTranscriptomics(data_dir='../../../data/')
df = dataset.orig_data

num_inducing = 1100
inducing_points = torch.rand((1, num_inducing, 2))

print(inducing_points.shape)
gp_kwargs = dict(use_ard=True,
                 use_scale=False,
                 lengthscale_constraint=Interval(0.1, 0.3),
                 learn_inducing_locations=False)
gp_model = MultiOutputGP(inducing_points, 1, **gp_kwargs)
gp_model.double()
gp_model.covar_module.lengthscale = 0.3*0.3 * 2

t_range = (0.0, 1.0)
time_steps = 40
mesh_cells = 40
fenics_model = ReactionDiffusion(t_range, time_steps, mesh_cells)

config = VariationalConfiguration(
    initial_conditions=False,
    num_samples=25
)

sensitivity = Parameter(torch.ones((1, 1), dtype=torch.float64), requires_grad=False)
decay = Parameter(0.1*torch.ones((1, 1), dtype=torch.float64), requires_grad=False)
diffusion = Parameter(0.01*torch.ones((1, 1), dtype=torch.float64), requires_grad=False)
fenics_params = [sensitivity, decay, diffusion]

lfm = PartialLFM(1, gp_model, fenics_model, fenics_params, config)


class PDETrainer(VariationalTrainer):

    def debug_out(self, data_input, y_target, output):

        print(output.variance.max(), output.mean.shape, output.variance.shape)
        f_mean = output.mean.reshape(1, -1)

        fig, axes = plt.subplots(ncols=2)
        scatter_output(axes[0], data_input, f_mean.detach(), 'Prediction')
        scatter_output(axes[1], data_input, y_target, 'Actual')
        plt.savefig('./out' + str(datetime.now().timestamp()) + '.png')

    def print_extra(self):
        print(' s:', self.lfm.fenics_parameters[0][0].item(),
              'dif:', self.lfm.fenics_parameters[0][0].item(),
              'dec:', self.lfm.fenics_parameters[0][0].item())


optimizer = torch.optim.Adam(lfm.parameters(), lr=0.07)
trainer = PDETrainer(lfm, optimizer, dataset)

if __name__ == '__main__':
    if path.exists('./lfm-test.pt'):
        print('present')
        lfm = PartialLFM.load('test',
                              gp_cls=MultiOutputGP,
                              gp_args=[inducing_points, 1],
                              gp_kwargs=gp_kwargs,
                              lfm_args=[config, dataset])

        optimizer = torch.optim.Adam(lfm.parameters(), lr=0.07)
        trainer = PDETrainer(lfm, optimizer, dataset)

    trainer.train(40)
    lfm.save('test')

