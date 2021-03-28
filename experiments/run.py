import torch
import argparse
import numpy as np
import seaborn as sns
import yaml
import os

from pathlib import Path
from matplotlib import pyplot as plt
from os import path
from datetime import datetime

from lafomo.datasets import (
    P53Data, HafnerData, ToyTimeSeries,
    ToySpatialTranscriptomics, DrosophilaSpatialTranscriptomics
)

from .partial import build_partial, plot_partial
from .variational import build_variational, plot_variational
from .exact import build_exact

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

builders = {
    'variational': build_variational,
    'exact': build_exact,
    'partial': build_partial,
}


# ------Plotters------#

plotters = {
    'partial': plot_partial,
    'variational': plot_variational,
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
