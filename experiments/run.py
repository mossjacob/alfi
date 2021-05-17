import argparse
import yaml
import seaborn as sns
import time
import numpy as np
import torch

from matplotlib import pyplot as plt
from pathlib import Path

from lafomo.datasets import (
    P53Data, HafnerData, ToyTranscriptomics, ToyTranscriptomicGenerator,
    HomogeneousReactionDiffusion, DrosophilaSpatialTranscriptomics,
    DeterministicLotkaVolterra, ReactionDiffusion
)
from lafomo.utilities.torch import get_mean_trace, is_cuda
try:
    from .partial import build_partial, plot_partial, pretrain_partial
except ImportError:
    build_partial, plot_partial, pretrain_partial = None, None, None
from .variational import build_variational, plot_variational
from .exact import build_exact, plot_exact
from .lotka import build_lotka, plot_lotka
from .lfo import build_dataset, build_lfo


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(font="CMU Serif")
class TerminalColours:
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'

# ------Config------ #

with open("experiments/experiments.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

dataset_choices = list(config.keys())

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=dataset_choices, default=dataset_choices[0])
parser.add_argument('--reload', type=bool, default=False)
parser.add_argument('--timer', type=bool, default=False)
parser.add_argument('--timer_samples', type=int, default=5)

datasets = {
    'p53': lambda: P53Data(replicate=0, data_dir='data'),
    'hafner': lambda: HafnerData(replicate=0, data_dir='data', extra_targets=False),
    'toy-spatial': lambda: HomogeneousReactionDiffusion(data_dir='data'),
    'dros-kr': lambda: DrosophilaSpatialTranscriptomics(gene='kr', data_dir='data', scale=False),
    'dros-kr-scaled': lambda: DrosophilaSpatialTranscriptomics(gene='kr', data_dir='data', scale=True),
    'toy': lambda: ToyTranscriptomicGenerator().generate_single(),
    'lotka': lambda: DeterministicLotkaVolterra(alpha=2./3, beta=4./3,
                                                gamma=1., delta=1.,
                                                steps=13, end_time=12, fixed_initial=0.8),
    'reaction-diffusion': lambda: build_dataset(
        ReactionDiffusion('data', nn_format=True, max_n=4000, ntest=50), ntest=50
    ),
}


# ------Set up model initialisers------ #
builders = {
    'variational': build_variational,
    'exact': build_exact,
    'partial': build_partial,
    'lotka': build_lotka,
    'lfo-2d': lambda *args, **kwargs: build_lfo(*args, **kwargs, block_dim=2)
}

plotters = {
    'exact': plot_exact,
    'partial': plot_partial,
    'variational': plot_variational,
    'lotka': plot_lotka,
}


train_pre_step = {
    'exact': lambda _, model, __, ___: model.likelihood.train(),
    'partial': pretrain_partial
}


def time_models(builder, dataset, filepath, modelparams, num_samples):
    times_with = list()
    loglosses_with = list()
    times_without = list()
    loglosses_without = list()

    for i in range(num_samples):
        # Without pretraining
        print(TerminalColours.GREEN, 'Without pretraining...', TerminalColours.END)
        model, trainer, plotter = builder(dataset, modelparams)
        trainer.plot_outputs = False

        t0 = time.time()
        model.pretrain(False)
        train_times = trainer.train(**experiment['train_params'])
        train_times = np.array(train_times)
        train_time = (train_times[:, 0] - t0) / 60
        logloss = train_times[:, 1]
        times_without.append(train_time)
        loglosses_without.append(logloss)
        model.save(str(filepath / f'model_without_{i}'))
        torch.save(get_mean_trace(trainer.parameter_trace), filepath / f'parameter_trace_without_{i}.pt')
        # With pretraining
        print(TerminalColours.GREEN, 'With pretraining...', TerminalColours.END)
        model, trainer, plotter = builder(dataset, modelparams)
        trainer.plot_outputs = False

        model.pretrain(True)
        pretrain_times, t_start = train_pre_step[method](dataset, model, trainer, modelparams)

        t1 = time.time()
        model.pretrain(False)
        train_times = trainer.train(**experiment['train_params'])
        pretrain_times = np.array(pretrain_times)
        train_times = np.array(train_times)
        pretrain_time = (pretrain_times[:, 0] - t_start) / 60
        train_time = (train_times[:, 0] - t1 + pretrain_time[-1]) / 60
        times_with.append(np.concatenate([pretrain_time, train_time]))
        loglosses_with.append(np.concatenate([pretrain_times[:, 1], train_times[:, 1]]))

        model.save(str(filepath / f'model_with_{i}'))
        torch.save(get_mean_trace(trainer.parameter_trace), filepath / f'parameter_trace_with_{i}.pt')

    loglosses_with = np.array(loglosses_with)
    loglosses_without = np.array(loglosses_without)
    times_with = np.array(times_with)
    times_without = np.array(times_without)
    np.save(str(filepath / 'time_with.npy'), times_with)
    np.save(str(filepath / 'loss_with.npy'), loglosses_with)
    np.save(str(filepath / 'time_without.npy'), times_without)
    np.save(str(filepath / 'loss_without.npy'), loglosses_without)


def run_model(method, dataset, model, trainer, plotter, filepath, save_filepath, modelparams):
    # Train model with optional initial step
    if method in train_pre_step:
        train_pre_step[method](dataset, model, trainer, modelparams)
    print(TerminalColours.GREEN, 'Training...', TerminalColours.END)
    times = trainer.train(**experiment['train_params'])
    np.save(save_filepath + '_times.npy', times)

    # Plot results of model
    if method in plotters:
        print(TerminalColours.GREEN, 'Running plotter...', TerminalColours.END)
        plotters[method](dataset, model, trainer, plotter, filepath, modelparams)
    else:
        print(TerminalColours.WARNING, 'Ignoring plotter for', method, 'since no plotter implemented.',
              TerminalColours.END)
    model.save(save_filepath)


if __name__ == "__main__":
    args = parser.parse_args()
    key = args.data

    print(TerminalColours.GREEN, 'Running experiments for dataset:', key, TerminalColours.END)
    data_config = config[key]
    dataset = datasets[key]()
    experiments = data_config['experiments']
    seen_methods = dict()
    for experiment in experiments:
        method = experiment['method']
        print(TerminalColours.GREEN, 'Constructing method:', method, TerminalColours.END)
        if method in builders:
            if method not in seen_methods:
                seen_methods[method] = 0

            # Create experiments path
            filepath = Path('experiments', key, method)
            filepath.mkdir(parents=True, exist_ok=True)
            save_filepath = str(filepath / (str(seen_methods[method]) + 'savedmodel'))

            # Construct model
            modelparams = experiment['model-params'] if 'model-params' in experiment else None
            reload = save_filepath if args.reload else None
            model, trainer, plotter = builders[method](
                dataset, modelparams, reload=reload, checkpoint_dir=filepath)
            model = model.cuda() if is_cuda() else model
            if args.timer:
                time_models(builders[method], dataset, filepath, modelparams, args.timer_samples)
            else:
                run_model(method, dataset, model, trainer, plotter, filepath, save_filepath, modelparams)
            seen_methods[method] += 1
            print(TerminalColours.GREEN, f'{method} completed successfully.', TerminalColours.END)
        else:
            print(TerminalColours.WARNING, 'Ignoring method', method, 'since no builder implemented.', TerminalColours.END)
