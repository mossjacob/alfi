import torch
import numpy as np
from torch.nn import Parameter
from matplotlib import pyplot as plt

from lafomo.configuration import VariationalConfiguration
from lafomo.models import MultiOutputGP, PartialLFM
from lafomo.models.pdes import ReactionDiffusion
from lafomo.plot import Plotter, plot_before_after
from lafomo.trainers import PDETrainer
from lafomo.utilities.fenics import interval_mesh
from lafomo.utilities.torch import cia, q2, smse, inv_softplus, softplus

tight_kwargs = dict(bbox_inches='tight', pad_inches=0)


def build_partial(dataset, params):
    data = next(iter(dataset))
    tx, y_target = data
    lengthscale = params['lengthscale']

    # Define mesh
    spatial = np.unique(tx[1, :])
    mesh = interval_mesh(spatial)

    # Define GP
    ts = tx[0, :].unique().sort()[0].numpy()
    xs = tx[1, :].unique().sort()[0].numpy()
    t_diff = ts[-1] - ts[0]
    x_diff = xs[-1] - xs[0]
    num_inducing = int(tx.shape[1] * 5/6)
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

    # Define LFM
    t_range = (ts[0], ts[-1])
    time_steps = dataset.num_discretised
    fenics_model = ReactionDiffusion(t_range, time_steps, mesh)

    config = VariationalConfiguration(
        initial_conditions=False,
        num_samples=25
    )

    sensitivity = Parameter(inv_softplus(params['sensitivity']) * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    decay = Parameter(inv_softplus(params['decay']) * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    diffusion = Parameter(inv_softplus(params['diffusion']) * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    # sensitivity = Parameter(1 * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    # decay = Parameter(0.1 * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    # diffusion = Parameter(0.01 * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    fenics_params = [sensitivity, decay, diffusion]

    lfm = PartialLFM(1, gp_model, fenics_model, fenics_params, config, num_training_points=int(0.9 * tx.shape[1]))
    optimizer = torch.optim.Adam(lfm.parameters(), lr=0.07)

    # As in Lopez-Lopera et al., we take 30% of data for training
    train_mask = torch.zeros_like(tx[0,:])
    train_mask[torch.randperm(tx.shape[1])[:int(0.3 * tx.shape[1])]] = 1
    track_parameters = list(lfm.fenics_named_parameters.keys()) + ['gp_model.covar_module.raw_lengthscale']
    trainer = PDETrainer(lfm, optimizer, dataset,
                         track_parameters=track_parameters,
                         train_mask=train_mask.bool())
    plotter = Plotter(lfm, dataset.gene_names)
    return lfm, trainer, plotter



def plot_partial(dataset, lfm, trainer, plotter, filepath):
    tx = trainer.tx
    num_t = tx[0, :].unique().shape[0]
    num_x = tx[1, :].unique().shape[0]
    f_mean = lfm(tx).mean.detach()
    f_var = lfm(tx).variance.detach()
    y_target = trainer.y_target[0]
    ts = tx[0, :].unique().sort()[0].numpy()
    xs = tx[1, :].unique().sort()[0].numpy()
    t_diff = ts[-1] - ts[0]
    x_diff = xs[-1] - xs[0]
    extent = [ts[0], ts[-1], xs[0], xs[-1]]

    with open(filepath / 'metrics.csv', 'w') as f:
        f.write('smse\tq2\tca\n')
        f.write('\t'.join([
            str(smse(y_target[~trainer.train_mask], f_mean[~trainer.train_mask]).mean().item()),
            str(q2(y_target[~trainer.train_mask], f_mean[~trainer.train_mask]).item()),
            str(cia(y_target[~trainer.train_mask], f_mean[~trainer.train_mask], f_var[~trainer.train_mask]).item())
        ]) + '\n')

    plot_before_after(
        f_mean.view(num_t, num_x).transpose(0, 1),
        y_target.view(num_t, num_x).detach().transpose(0, 1),
        extent,
        titles=['Prediction', 'Ground truth']
    )
    plt.savefig(filepath / 'beforeafter.pdf', **tight_kwargs)

    labels = ['Sensitivity', 'Decay', 'Diffusion']
    kinetics = list()
    for key in lfm.fenics_named_parameters.keys():
        kinetics.append(trainer.parameter_trace[key][-1].squeeze().numpy())

    plotter.plot_double_bar(kinetics, labels)
    plt.savefig(filepath / 'kinetics.pdf', **tight_kwargs)
