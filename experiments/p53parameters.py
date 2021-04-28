import argparse
import yaml
import seaborn as sns
import time
import numpy as np
import torch

from matplotlib import pyplot as plt
from pathlib import Path

from lafomo.datasets import P53Data
from lafomo.utilities.data import p53_ground_truth
from lafomo.plot import tight_kwargs

from .variational import build_variational, plot_variational


# ------Config------ #

with open("experiments/experiments.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

dataset_choices = list(config.keys())

parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default=5)


if __name__ == "__main__":
    args = parser.parse_args()
    data_config = config['p53']
    experiments = data_config['experiments']
    for experiment in experiments:
        if experiment['method'] == 'variational':
            break

    print(experiment)
    method = experiment['method']

    # Create experiments path
    filepath = Path('experiments', 'p53', method)
    filepath.mkdir(parents=True, exist_ok=True)

    kinetics_samples = list()
    for i in range(args.samples):
        dataset = P53Data(replicate=0, data_dir='data')

        # Construct model
        modelparams = experiment['model-params'] if 'model-params' in experiment else None
        model, trainer, plotter = build_variational(
            dataset, modelparams, checkpoint_dir=None)

        trainer.train(**experiment['train_params'], step_size=5e-1)

        kinetics = np.array(
            [x.detach().squeeze().numpy() for x in [model.basal_rate, model.sensitivity, model.decay_rate]])

        kinetics_samples.append(kinetics)

    samples = np.stack(kinetics_samples)
    print(samples.shape)
    kinetics = samples.mean(0)
    err = samples.std(0)
    print(kinetics.shape, err.shape)
    labels = ['Basal rates', 'Sensitivities', 'Decay rates']

    plotter.plot_double_bar(kinetics, labels, params_var=err, ground_truths=p53_ground_truth(),
                            figsize=(7.5, 2.6),
                            yticks=[
                                np.linspace(0, 0.12, 5),
                                np.linspace(0, 1.2, 4),
                                np.arange(0, 1.1, 0.2),
                            ])
    plt.tight_layout()

    plt.savefig(filepath / 'linear-kinetics2.pdf', **tight_kwargs)
