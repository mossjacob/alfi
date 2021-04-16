import argparse
import yaml
import seaborn as sns
from matplotlib import pyplot as plt
import time
import numpy as np

from pathlib import Path

from lafomo.datasets import (
    P53Data, HafnerData, ToyTimeSeries,
    ToySpatialTranscriptomics, DrosophilaSpatialTranscriptomics,
    DeterministicLotkaVolterra,
)
try:
    from .partial import build_partial, plot_partial, pretrain_partial
except ImportError:
    build_partial, plot_partial = None, None
from .variational import build_variational, plot_variational
from .exact import build_exact, plot_exact
from .lotka import build_lotka, plot_lotka

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

datasets = {
    'p53': lambda: P53Data(replicate=0, data_dir='data'),
    'hafner': lambda: HafnerData(replicate=0, data_dir='data', extra_targets=False),
    'toy-spatial': lambda: ToySpatialTranscriptomics(data_dir='data'),
    'dros-kr': lambda: DrosophilaSpatialTranscriptomics(gene='kr', data_dir='data', scale=True),
    'toy': lambda: ToyTimeSeries(),
    'lotka': lambda: DeterministicLotkaVolterra(alpha = 2./3, beta = 4./3,
                                                gamma = 1., delta = 1.,
                                                steps=13, end_time=12, fixed_initial=0.8),
}


# ------Set up model initialisers------ #
builders = {
    'variational': build_variational,
    'exact': build_exact,
    'partial': build_partial,
    'lotka': build_lotka,
}

plotters = {
    'exact': plot_exact,
    'partial': plot_partial,
    'variational': plot_variational,
    'lotka': plot_lotka,
}


train_pre_step = {
    'exact': lambda _, model, __: model.likelihood.train(),
    'partial': pretrain_partial
}

if __name__ == "__main__":
    args = parser.parse_args()
    key = args.data
    data_config = config[key]
    dataset = datasets[key]()
    experiments = data_config['experiments']
    print(TerminalColours.GREEN, 'Running experiments for dataset:', key, TerminalColours.END)
    for experiment in experiments:
        method = experiment['method']
        print(TerminalColours.GREEN, 'Constructing method:', method, TerminalColours.END)
        if method in builders:
            # Create experiments path
            filepath = Path('experiments', key, method)
            filepath.mkdir(parents=True, exist_ok=True)
            # Construct model
            modelparams = experiment['model-params'] if 'model-params' in experiment else None
            model, trainer, plotter = builders[method](dataset, modelparams)

            times_with = list()
            loglosses_with = list()
            times_without = list()
            loglosses_without = list()
            trainer.plot_outputs = False

            for i in range(5):
                # try without pretraining
                print(TerminalColours.GREEN, 'Without pretraining...', TerminalColours.END)

                t0 = time.time()
                model.pretrain(False)
                train_times = trainer.train(**experiment['train-params'])
                train_times = np.array(train_times)
                train_time = (train_times[:, 0] - t0) / 60
                logloss = train_times[:, 1]
                times_without.append(train_time)
                loglosses_without.append(logloss)

                # now with pretraining
                print(TerminalColours.GREEN, 'With pretraining...', TerminalColours.END)
                model.pretrain(True)
                pretrain_times, t_start = train_pre_step[method](dataset, model, trainer)

                t1 = time.time()
                model.pretrain(False)
                train_times = trainer.train(**experiment['train-params'])
                pretrain_times = np.array(pretrain_times)
                train_times = np.array(train_times)
                pretrain_time = (pretrain_times[:, 0] - t_start) / 60
                train_time = (train_times[:, 0] - t1 + pretrain_time[-1]) / 60
                times_with.append(np.concatenate([pretrain_time, train_time]))
                loglosses_with.append(np.concatenate([pretrain_times[:, 1], train_times[:, 1]]))

            loglosses_with = np.array(loglosses_with)
            loglosses_without = np.array(loglosses_without)
            times_with = np.array(times_with)
            times_without = np.array(times_without)
            np.save(str(filepath / 'traintime_with.npy'), times_with)
            np.save(str(filepath / 'trainloss_with.npy'), loglosses_with)
            np.save(str(filepath / 'traintime_without.npy'), times_without)
            np.save(str(filepath / 'trainloss_without.npy'), loglosses_without)

            print(TerminalColours.GREEN, f'{method} completed successfully.', TerminalColours.END)
        else:
            print(TerminalColours.WARNING, 'Ignoring method', method, 'since no builder implemented.', TerminalColours.END)
