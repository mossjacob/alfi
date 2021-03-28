import torch
import argparse
import numpy as np
import seaborn as sns
import yaml
import os

from pathlib import Path
from torch.nn import Parameter
from matplotlib import pyplot as plt
from os import path
from datetime import datetime
from gpytorch.mlls import ExactMarginalLogLikelihood

from lafomo.datasets import (
    P53Data, HafnerData, ToyTimeSeries,
    ToySpatialTranscriptomics, DrosophilaSpatialTranscriptomics
)
from lafomo.configuration import VariationalConfiguration
from lafomo.models import OrdinaryLFM, MultiOutputGP, ExactLFM, PartialLFM
from lafomo.models.pdes import ReactionDiffusion
from lafomo.plot import Plotter, plot_before_after
from lafomo.trainers import ExactTrainer, PDETrainer
from lafomo.utilities.fenics import interval_mesh


tight_kwargs = dict(bbox_inches='tight', pad_inches=0)
# ------Config------ #

with open("experiments/experiments.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

print('Config', config)

dataset_choices = list(config.keys())

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=dataset_choices, default=dataset_choices[0])

# ------Set up model initialisers------ #


def build_variational(dataset):
    pass

def build_exact(dataset):
    model = ExactLFM(dataset, dataset.variance.reshape(-1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.07)

    loss_fn = ExactMarginalLogLikelihood(model.likelihood, model)

    trainer = ExactTrainer(model, optimizer, dataset, loss_fn=loss_fn)
    plotter = Plotter(model, dataset.gene_names)
    model.likelihood.train()

    return model, trainer, plotter

def build_pde(dataset):
    data = next(iter(dataset))
    tx, y_target = data
    lengthscale = 10

    # Define mesh
    spatial = np.unique(tx[1, :])
    mesh = interval_mesh(spatial)

    # Define GP
    ts = tx[0, :].unique().sort()[0].numpy()
    xs = tx[1, :].unique().sort()[0].numpy()
    t_diff = ts[-1] - ts[0]
    x_diff = xs[-1] - xs[0]
    num_inducing = int(tx.shape[1] * 2/3)
    inducing_points = torch.stack([
        ts[0] + t_diff * torch.rand((1, num_inducing)),
        xs[0] + x_diff * torch.rand((1, num_inducing))
    ], dim=2)

    gp_kwargs = dict(use_ard=True,
                     use_scale=False,
                     # lengthscale_constraint=Interval(0.1, 0.3),
                     learn_inducing_locations=False,
                     initial_lengthscale=lengthscale)
    gp_model = MultiOutputGP(inducing_points, 1, **gp_kwargs)
    gp_model.double();

    t_range = (ts[0], ts[-1])
    time_steps = dataset.num_discretised
    fenics_model = ReactionDiffusion(t_range, time_steps, mesh)

    config = VariationalConfiguration(
        initial_conditions=False,
        num_samples=25
    )

    sensitivity = Parameter(torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    decay = Parameter(0.1*torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    diffusion = Parameter(0.01*torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    fenics_params = [sensitivity, decay, diffusion]

    lfm = PartialLFM(1, gp_model, fenics_model, fenics_params, config)
    optimizer = torch.optim.Adam(lfm.parameters(), lr=0.07)
    trainer = PDETrainer(lfm, optimizer, dataset)
    plotter = None
    return lfm, trainer, plotter

builders = {
    'variational': build_variational,
    'exact': build_exact,
    'partial': build_pde,
}


# ------Plotters------#
def plot_variational(lfm, trainer, plotter):
    plt.savefig(path.join(experiment_dir, 'outputs.pdf'), **tight_kwargs)
    plt.savefig(path.join(experiment_dir, 'latents.pdf'), **tight_kwargs)
    plt.savefig(path.join(experiment_dir, 'kinetics.pdf'), **tight_kwargs)

def plot_pde(lfm, trainer, plotter, filepath):
    tx = trainer.tx
    num_t = tx[0, :].unique().shape[0]
    num_x = tx[1, :].unique().shape[0]
    out = lfm(tx).mean
    out = out.detach().view(num_t, num_x)
    y_target = trainer.y_target
    ts = tx[0, :].unique().sort()[0].numpy()
    xs = tx[1, :].unique().sort()[0].numpy()
    t_diff = ts[-1] - ts[0]
    x_diff = xs[-1] - xs[0]
    extent = [ts[0], ts[-1], xs[0], xs[-1]]

    plot_before_after(
        out.transpose(0, 1),
        y_target.view(num_t, num_x).detach().transpose(0, 1),
        extent,
        titles=['Prediction', 'Ground truth']
    )
    plt.savefig(filepath / 'beforeafter.pdf', **tight_kwargs)


plotters = {
    'partial': plot_pde,

}

# ------Datasets------ #

def load_dataset(name):
    return {
        'p53': lambda: P53Data(replicate=0, data_dir='data'),
        'hafner': lambda: HafnerData(replicate=0, data_dir='data', extra_targets=False),
        'toy-spatial': lambda: ToySpatialTranscriptomics(data_dir='data'),
        'dros-kr': lambda: DrosophilaSpatialTranscriptomics(gene='kr', data_dir='data'),
        'toy': ToyTimeSeries(),
    }[name]()


# ------Exact------ #
# We run the exact experiments separately as it isn't appropriate for all datasets


if __name__ == "__main__":
    args = parser.parse_args()
    key = args.data

    print('Running experiments for dataset:', key)
    data_config = config[key]
    dataset = load_dataset(key)
    print(dataset)
    methods = data_config['methods']
    for method in methods:
        print('--and for method:', method)

        if method in builders:
            filepath = Path('experiments', key, method)
            filepath.mkdir(parents=True, exist_ok=True)

            model, trainer, plotter = builders[method](dataset)
            trainer.train(**methods[method])
            if method in plotters:
                plotters[method](model, trainer, plotter, filepath)

            model.save(str(filepath / 'savedmodel'))
        else:
            print('--ignoring method', method, 'since no builder implemented.')
