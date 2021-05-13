import torch
import numpy as np
from torch.nn import Parameter
from matplotlib import pyplot as plt
from torch.optim import Adam
from gpytorch.optim import NGD
import gpytorch
import time

from lafomo.configuration import VariationalConfiguration
from lafomo.models import MultiOutputGP, PartialLFM, generate_multioutput_rbf_gp
from lafomo.models.pdes import ReactionDiffusion
from lafomo.plot import Plotter, plot_spatiotemporal_data, tight_kwargs
from lafomo.trainers import PDETrainer, PartialPreEstimator
from lafomo.utilities.fenics import interval_mesh
from lafomo.utilities.torch import cia, q2, smse, inv_softplus, softplus, spline_interpolate_gradient, get_mean_trace


def build_partial(dataset, params, reload=None, checkpoint_dir=None, **kwargs):
    data = next(iter(dataset))
    tx, y_target = data
    lengthscale = params['lengthscale']
    zero_mean = params['zero_mean'] if 'zero_mean' in params else False
    # Define mesh
    spatial = np.unique(tx[1, :])
    mesh = interval_mesh(spatial)

    # Define GP
    if tx.shape[1] > 1000:
        num_inducing = int(tx.shape[1] * 3/6)
    else:
        num_inducing = int(tx.shape[1] * 5/6)
    use_lhs = False
    if use_lhs:
        print('tx', tx.shape)
        from smt.sampling_methods import LHS
        ts = tx[0, :].unique().sort()[0].numpy()
        xs = tx[1, :].unique().sort()[0].numpy()
        xlimits = np.array([[ts[0], ts[-1]],[xs[0], xs[-1]]])
        sampling = LHS(xlimits=xlimits)
        inducing_points = torch.tensor(sampling(num_inducing)).unsqueeze(0)
    else:
        inducing_points = torch.stack([
            tx[0, torch.randperm(tx.shape[1])[:num_inducing]],
            tx[1, torch.randperm(tx.shape[1])[:num_inducing]]
        ], dim=1).unsqueeze(0)

    gp_kwargs = dict(learn_inducing_locations=False,
                     natural=params['natural'],
                     use_tril=True)

    gp_model = generate_multioutput_rbf_gp(1, inducing_points, ard_dims=2, zero_mean=zero_mean, gp_kwargs=gp_kwargs)
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
        num_samples=5
    )

    parameter_grad = params['parameter_grad'] if 'parameter_grad' in params else True
    sensitivity = Parameter(
        inv_softplus(torch.tensor(params['sensitivity'])) * torch.ones((1, 1), dtype=torch.float64),
        requires_grad=False)
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
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.09)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.05)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.07)]

    track_parameters = list(lfm.fenics_named_parameters.keys()) +\
                       list(map(lambda s: f'gp_model.{s}', dict(lfm.gp_model.named_hyperparameters()).keys()))

    # As in Lopez-Lopera et al., we take 30% of data for training
    train_mask = torch.zeros_like(tx[0, :])
    train_mask[torch.randperm(tx.shape[1])[:int(train_ratio * tx.shape[1])]] = 1


    warm_variational = params['warm_epochs'] if 'warm_epochs' in params else 10
    orig_data = dataset.orig_data.squeeze().t()
    trainer = PDETrainer(lfm, optimizers, dataset,
                         clamp=params['clamp'],
                         track_parameters=track_parameters,
                         train_mask=train_mask.bool(),
                         warm_variational=warm_variational,
                         checkpoint_dir=checkpoint_dir, lf_target=orig_data)
    plotter = Plotter(lfm, dataset.gene_names)
    return lfm, trainer, plotter


def pretrain_partial(dataset, lfm, trainer, modelparams):
    tx = trainer.tx
    num_t = tx[0, :].unique().shape[0]
    num_x = tx[1, :].unique().shape[0]
    y_target = trainer.y_target[0]
    y_matrix = y_target.view(num_t, num_x)

    dy_t = list()
    for i in range(num_x):
        t = tx[0][::num_x]
        y = y_matrix[:, i].unsqueeze(-1)
        t_interpolate, y_interpolate, y_grad, _ = \
            spline_interpolate_gradient(t, y)
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
                diffusion * 0)
        return dy_t

    train_ratio = 0.3
    num_training = int(train_ratio * tx.shape[1])
    print('num training', num_training)
    if modelparams['natural']:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.05)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.05)]

    pre_estimator = PartialPreEstimator(
        lfm, optimizers, dataset, pde_func,
        input_pair=(trainer.tx, trainer.y_target), target=dy_t.t(),
        train_mask=trainer.train_mask
    )

    lfm.pretrain(True)
    lfm.config.num_samples = 50
    t0 = time.time()
    times = pre_estimator.train(80, report_interval=10)
    lfm.pretrain(False)
    lfm.config.num_samples = 5
    return times, t0


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
    torch.save(get_mean_trace(trainer.parameter_trace), filepath / 'parameter_trace.pt')
    with open(filepath / 'metrics.csv', 'w') as f:
        f.write('smse\tq2\tca\n')
        f_mean_test = f_mean[~trainer.train_mask].squeeze()
        f_var_test = f_var[~trainer.train_mask].squeeze()
        f.write('\t'.join([
            str(smse(y_target[~trainer.train_mask], f_mean_test).mean().item()),
            str(q2(y_target[~trainer.train_mask], f_mean_test).item()),
            str(cia(y_target[~trainer.train_mask], f_mean_test, f_var_test).item())
        ]) + '\n')

    orig_data = dataset.orig_data.squeeze().t()
    num_t_orig = orig_data[:, 0].unique().shape[0]
    num_x_orig = orig_data[:, 1].unique().shape[0]

    l_target = orig_data[trainer.t_sorted, 2]
    l = lfm.gp_model(tx.t())
    l_mean = l.mean.detach()
    plot_spatiotemporal_data(
        [
            l_mean.view(num_t, num_x).t(),
            l_target.view(num_t_orig, num_x_orig).t(),
            f_mean.view(num_t_orig, num_x_orig).t(),
            y_target.view(num_t_orig, num_x_orig).detach().t(),
        ],
        extent,
        titles=['Latent (Prediction)', 'Latent (Target)', 'Output (Prediction)', 'Output (Target)'],
        cticks=None,  # [0, 100, 200]
        clim=[(l_target.min(), l_target.max())] * 2 + [(y_target.min(), y_target.max())] * 2,
    )
    plt.gca().get_figure().set_size_inches(15, 7)

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
