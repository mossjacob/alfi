import argparse
import yaml
import seaborn as sns
from matplotlib import pyplot as plt

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
parser.add_argument('--reload', type=bool, default=False)

datasets = {
    'p53': lambda: P53Data(replicate=0, data_dir='data'),
    'hafner': lambda: HafnerData(replicate=0, data_dir='data', extra_targets=False),
    'toy-spatial': lambda: ToySpatialTranscriptomics(data_dir='data'),
    'dros-kr': lambda: DrosophilaSpatialTranscriptomics(gene='kr', data_dir='data'),
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
            model, trainer, plotter = builders[method](dataset, modelparams, reload=reload)

            # Train model with optional initial step
            if method in train_pre_step:
                train_pre_step[method](dataset, model, trainer)
            print(TerminalColours.GREEN, 'Training...', TerminalColours.END)
            trainer.train(**experiment['train-params'])

            # Plot results of model
            if method in plotters:
                print(TerminalColours.GREEN, 'Running plotter...', TerminalColours.END)
                plotters[method](dataset, model, trainer, plotter, filepath, modelparams)
            else:
                print(TerminalColours.WARNING, 'Ignoring plotter for', method, 'since no plotter implemented.', TerminalColours.END)
            model.save(save_filepath)
            seen_methods[method] += 1
            print(TerminalColours.GREEN, f'{method} completed successfully.', TerminalColours.END)
        else:
            print(TerminalColours.WARNING, 'Ignoring method', method, 'since no builder implemented.', TerminalColours.END)
