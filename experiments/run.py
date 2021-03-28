import argparse
import yaml
import seaborn as sns
from matplotlib import pyplot as plt

from pathlib import Path

from lafomo.datasets import (
    P53Data, HafnerData, ToyTimeSeries,
    ToySpatialTranscriptomics, DrosophilaSpatialTranscriptomics
)
try:
    from .partial import build_partial, plot_partial
except ImportError:
    build_partial, plot_partial = None, None
from .variational import build_variational, plot_variational
from .exact import build_exact, plot_exact


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(font="CMU Serif")

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


def load_dataset(name):
    return {
        'p53': lambda: P53Data(replicate=0, data_dir='data'),
        'hafner': lambda: HafnerData(replicate=0, data_dir='data', extra_targets=False),
        'toy-spatial': lambda: ToySpatialTranscriptomics(data_dir='data'),
        'dros-kr': lambda: DrosophilaSpatialTranscriptomics(gene='kr', data_dir='data'),
        'toy': ToyTimeSeries(),
    }[name]()


# ------Set up model initialisers------ #
builders = {
    'variational': build_variational,
    'exact': build_exact,
    'partial': build_partial,
}

plotters = {
    'exact': plot_exact,
    'partial': plot_partial,
    'variational': plot_variational,
}
train_pre_step = {
    'exact': lambda model: model.likelihood.train()
}

if __name__ == "__main__":
    args = parser.parse_args()
    key = args.data

    print('Running experiments for dataset:', key)
    data_config = config[key]
    dataset = load_dataset(key)
    methods = data_config['methods']
    for method_key in methods:
        print('--and for method:', method_key)
        method = methods[method_key]
        if method_key in builders:
            # Create experiments path
            filepath = Path('experiments', key, method_key)
            filepath.mkdir(parents=True, exist_ok=True)

            # Construct model
            modelparams = method['model-params'] if 'model-params' in method else None
            model, trainer, plotter = builders[method_key](dataset, modelparams)

            # Train model with optional initial step
            if method_key in train_pre_step:
                train_pre_step[method_key](model)
            trainer.train(**method['train-params'])

            # Plot results of model
            if method_key in plotters:
                plotters[method_key](dataset, model, trainer, plotter, filepath)
            else:
                print('--ignoring plotter for', method_key, 'since no plotter implemented.')
            model.save(str(filepath / 'savedmodel'))
        else:
            print('--ignoring method', method_key, 'since no builder implemented.')
