import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib import pyplot as plt

from alfi.models import ExactLFM
from alfi.plot import Plotter1d
from alfi.trainers import ExactTrainer
from alfi.datasets import P53Data

tight_kwargs = dict(bbox_inches='tight', pad_inches=0)


def build_exact(dataset, params, **kwargs):
    model = ExactLFM(dataset, dataset.variance.reshape(-1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.07)

    loss_fn = ExactMarginalLogLikelihood(model.likelihood, model)

    track_parameters = [
        'mean_module.raw_basal',
        'covar_module.raw_decay',
        'covar_module.raw_sensitivity',
        'covar_module.raw_lengthscale',
    ]
    print(dict(model.named_parameters()))

    trainer = ExactTrainer(model, [optimizer], dataset, loss_fn=loss_fn, track_parameters=track_parameters)
    plotter = Plotter1d(model, dataset.gene_names)
    model.likelihood.train()

    return model, trainer, plotter


def plot_exact(dataset, lfm, trainer, plotter, filepath, params):
    t_predict = torch.linspace(-1, 13, 80, dtype=torch.float64)

    plotter.plot_outputs(t_predict, t_scatter=dataset.t_observed, y_scatter=dataset.m_observed)
    plt.savefig(filepath / 'outputs.pdf', **tight_kwargs)

    plotter.plot_latents(t_predict, ylim=(-2, 3.2), num_samples=0)
    plt.savefig(filepath / 'latents.pdf', **tight_kwargs)

    labels = ['Basal rates', 'Sensitivities', 'Decay rates']
    track_parameters = [
        'mean_module.raw_basal',
        'covar_module.raw_sensitivity',
        'covar_module.raw_decay',
    ]
    kinetics = list()
    constraints = dict(lfm.named_constraints())
    for key in track_parameters:
        val = trainer.parameter_trace[key][-1].squeeze()
        if key + '_constraint' in constraints:
            val = constraints[key + '_constraint'].transform(val)
        kinetics.append(val.numpy())

    plotter.plot_double_bar(kinetics, labels, P53Data.params_ground_truth())
    plt.savefig(filepath / 'kinetics.pdf', **tight_kwargs)
