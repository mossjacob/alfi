import torch
import numpy as np
from torch.nn import Parameter
from matplotlib import pyplot as plt
from torch.optim import Adam
from gpytorch.optim import NGD
import gpytorch

from lafomo.configuration import VariationalConfiguration
from lafomo.models import MultiOutputGP, PartialLFM, generate_multioutput_rbf_gp
from lafomo.models.pdes import ReactionDiffusion
from lafomo.plot import Plotter, plot_spatiotemporal_data
from lafomo.trainers import PDETrainer, PartialPreEstimator
from lafomo.utilities.fenics import interval_mesh
from lafomo.utilities.torch import cia, q2, smse, inv_softplus, softplus, spline_interpolate_gradient

tight_kwargs = dict(bbox_inches='tight', pad_inches=0)


def build_partial(dataset, params, reload=None):
    data = next(iter(dataset))
    tx, y_target = data
    lengthscale = params['lengthscale']

    # Define mesh
    spatial = np.unique(tx[1, :])
    mesh = interval_mesh(spatial)

    # Define GP
    num_inducing = int(tx.shape[1] * 5/6)
    inducing_points = torch.stack([
        tx[0, torch.randperm(tx.shape[1])[:num_inducing]],
        tx[1, torch.randperm(tx.shape[1])[:num_inducing]]
    ], dim=1).unsqueeze(0)

    gp_kwargs = dict(learn_inducing_locations=False,
                     natural=params['natural'],
                     use_tril=True)

    gp_model = generate_multioutput_rbf_gp(1, inducing_points, ard_dims=2, zero_mean=False, gp_kwargs=gp_kwargs)
    gp_model.covar_module.lengthscale = lengthscale
    # lengthscale_constraint=Interval(0.1, 0.3),
    gp_model.double()

    # Define LFM
    def fenics_fn():
        # We calculate a mesh that contains all possible spatial locations in the dataset
        data = next(iter(dataset))
        tx, y_target = data

        # Define mesh
        spatial = np.unique(tx[1, :])
        mesh = interval_mesh(spatial)

        # Define fenics model
        ts = tx[0, :].unique().sort()[0].numpy()
        t_range = (ts[0], ts[-1])
        time_steps = dataset.num_discretised
        return ReactionDiffusion(t_range, time_steps, mesh)

    config = VariationalConfiguration(
        initial_conditions=False,
        num_samples=25
    )

    parameter_grad = params['parameter_grad'] if 'parameter_grad' in params else True
    sensitivity = Parameter(
        inv_softplus(torch.tensor(params['sensitivity'])) * torch.ones((1, 1), dtype=torch.float64),
        requires_grad=parameter_grad)
    decay = Parameter(
        inv_softplus(torch.tensor(params['decay'])) * torch.ones((1, 1), dtype=torch.float64),
        requires_grad=parameter_grad)
    diffusion = Parameter(
        inv_softplus(torch.tensor(params['diffusion'])) * torch.ones((1, 1), dtype=torch.float64),
        requires_grad=parameter_grad)
    # sensitivity = Parameter(1 * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    # decay = Parameter(0.1 * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    # diffusion = Parameter(0.01 * torch.ones((1, 1), dtype=torch.float64), requires_grad=True)
    fenics_params = [sensitivity, decay, diffusion]
    train_ratio = 0.3
    num_training = int(train_ratio * tx.shape[1])

    lfm = PartialLFM(1, gp_model, fenics_fn, fenics_params, config, num_training_points=num_training)
    if reload is not None:
        lfm = lfm.load(reload,
                       gp_model=lfm.gp_model,
                       lfm_args=[1, lfm.fenics_model_fn, lfm.fenics_parameters, config])

    if params['natural']:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.09)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.05)]

    # As in Lopez-Lopera et al., we take 30% of data for training
    train_mask = torch.zeros_like(tx[0, :])
    train_mask[torch.randperm(tx.shape[1])[:int(train_ratio * tx.shape[1])]] = 1
    track_parameters = list(lfm.fenics_named_parameters.keys()) + ['gp_model.covar_module.raw_lengthscale']
    warm_variational = params['warm_epochs'] if 'warm_epochs' in params else 10
    trainer = PDETrainer(lfm, optimizers, dataset,
                         clamp=params['clamp'],
                         track_parameters=track_parameters,
                         train_mask=train_mask.bool(),
                         warm_variational=warm_variational)
    plotter = Plotter(lfm, dataset.gene_names)
    return lfm, trainer, plotter


def pretrain_partial(dataset, lfm, trainer):
    tx = trainer.tx
    num_t = tx[0, :].unique().shape[0]
    num_x = tx[1, :].unique().shape[0]
    print(num_t, num_x)
    y_target = trainer.y_target[0]
    y_matrix = y_target.view(num_t, num_x)

    dy_t = list()
    for i in range(num_x):
        t = tx[0][::num_x]
        y = y_matrix[:, i].unsqueeze(-1)
        t_interpolate, y_interpolate, y_grad, _ = \
            spline_interpolate_gradient(t, y)
        plt.plot(t_interpolate, y_interpolate)
        dy_t.append(y_grad)
    dy_t = torch.stack(dy_t)

    d2y_x = list()
    dy_x = list()
    for i in range(num_t):
        t = tx[1][:num_x]
        y = y_matrix[i].unsqueeze(-1)
        t_interpolate, y_interpolate, y_grad, y_grad_2 = \
            spline_interpolate_gradient(t, y)
        d2y_x.append(y_grad_2)
        dy_x.append(y_grad)

    d2y_x = torch.stack(d2y_x)
    dy_x = torch.stack(dy_x)[..., ::10, 0].reshape(1, -1)
    d2y_x = d2y_x[..., ::10, 0].reshape(1, -1)
    dy_t = dy_t[..., ::10, 0].t().reshape(1, -1)

    def pde_func(y, u, sensitivity, decay, diffusion):
        # y (1, 1681) u (25, 1, 41, 41) s (25, 1)
        dy_t = (sensitivity * u.view(u.shape[0], -1) -
                decay * y.view(1, -1) +
                diffusion * d2y_x)
        return dy_t

    optimizers = [Adam(lfm.parameters(), lr=0.05)]

    pre_estimator = PartialPreEstimator(
        lfm, optimizers, dataset, pde_func,
        input_pair=(trainer.tx, trainer.y_target), target=dy_t.t()
    )

    lfm.pretrain(True)
    pre_estimator.train(150, report_interval=10)
    lfm.pretrain(False)


def plot_partial(dataset, lfm, trainer, plotter, filepath, params):
    lfm.eval()
    tx = trainer.tx
    num_t = tx[0, :].unique().shape[0]
    num_x = tx[1, :].unique().shape[0]
    f = lfm(tx)
    f_mean = f.mean.detach()
    f_var = f.variance.detach()
    y_target = trainer.y_target[0]
    ts = tx[0, :].unique().sort()[0].numpy()
    xs = tx[1, :].unique().sort()[0].numpy()
    extent = [ts[0], ts[-1], xs[0], xs[-1]]

    with open(filepath / 'metrics.csv', 'w') as f:
        f.write('smse\tq2\tca\n')
        f_mean_test = f_mean[~trainer.train_mask].squeeze()
        f_var_test = f_var[~trainer.train_mask].squeeze()
        f.write('\t'.join([
            str(smse(y_target[~trainer.train_mask], f_mean_test).mean().item()),
            str(q2(y_target[~trainer.train_mask], f_mean_test).item()),
            str(cia(y_target[~trainer.train_mask], f_mean_test, f_var_test).item())
        ]) + '\n')

    l_target = torch.tensor(dataset.orig_data[:, 2])
    l = lfm.gp_model(tx.t())
    l_mean = l.mean.detach()
    plot_spatiotemporal_data(
        [
            f_mean.view(num_t, num_x).t(),
            y_target.view(num_t, num_x).detach().t(),

            l_mean.view(num_t, num_x).t(),
            l_target.view(num_t, num_x).t()
        ],
        extent,
        titles=None
    )

    plt.savefig(filepath / 'beforeafter.pdf', **tight_kwargs)

    labels = ['Sensitivity', 'Decay', 'Diffusion']
    kinetics = list()
    for key in lfm.fenics_named_parameters.keys():
        kinetics.append(softplus(trainer.parameter_trace[key][-1].squeeze()).numpy())
    with open(filepath / 'kinetics.csv', 'w') as f:
        f.write('sensitivity\tdecay\tdiffusion\n')
        f.write('\t'.join(map(str, kinetics)) + '\n')
    #
    # plotter.plot_double_bar(kinetics, labels)
    # plt.savefig(filepath / 'kinetics.pdf', **tight_kwargs)
