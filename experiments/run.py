import torch
import numpy as np
from torch.nn import Parameter
from matplotlib import pyplot as plt
import seaborn as sns
from os import path
import yaml
from gpytorch.mlls import ExactMarginalLogLikelihood
from lafomo.datasets import P53Data, HafnerData, ToyTimeSeries
from lafomo.configuration import VariationalConfiguration
from lafomo.models import OrdinaryLFM, MultiOutputGP, ExactLFM
from lafomo.plot import Plotter
from lafomo.trainer import TranscriptionalTrainer, ExactTrainer


# ------Config------ #

with open("experiments/experiments.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()
print(config)

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

builders = {
    'variational': build_variational,
    'exact': build_exact
}

# ------Datasets------ #
p53 = P53Data(replicate=0, data_dir='data')
hafner = HafnerData(replicate=0, data_dir='data', extra_targets=False)
artificial = ToyTimeSeries()

datasets = {
    'p53': p53,
    'hafner': hafner,
    'artificial': artificial
}

for key in config:
    print('Running experiments for dataset:', key)
    data_config = config[key]
    dataset = datasets[key]

    methods = data_config['methods']
    for method in methods:
        print('--and for method:', method)
        print(methods[method])

        if method in builders:
            # model, trainer, plotter = builders[method](dataset)
            # trainer.train(epochs=150, report_interval=10)
            pass
        else:
            print('--failed for method:', method, 'since no builder.')


datasets = [p53, hafner]


# ------Experiments------ #

methods = ['variational', 'mcmc']

experiments = ['linear', 'nonlinear']

# ------Exact------ #
# We run the exact experiments separately as it isn't appropriate for all datasets


