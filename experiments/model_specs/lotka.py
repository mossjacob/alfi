import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import gpytorch
from torch.nn import Parameter
from torch.optim import Adam
from gpytorch.optim import NGD
from gpytorch.constraints import Positive, Interval

from alfi.models import MultiOutputGP, generate_multioutput_gp
from alfi.utilities.torch import softplus
from alfi.plot import Plotter1d, Plotter2d, plot_phase, Colours
from alfi.configuration import VariationalConfiguration
from alfi.trainers import VariationalTrainer
from alfi.impl.odes import LotkaVolterra, LotkaVolterraState

tight_kwargs = dict(bbox_inches='tight', pad_inches=0)

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(style='white', font="CMU Serif")


def build_lotka(dataset, params, reload=None, **kwargs):
    x_min, x_max = min(dataset.times), max(dataset.times)

    periodic = params['kernel'] == 'periodic'
    use_natural = params['natural']
    use_lhs = 'lhs' in params and params['lhs']
    num_training = dataset[0][0].shape[0]
    lfm_kwargs = dict(num_training_points=num_training)
    if params['state']:
        num_tasks = 2
        num_outputs = 2
        num_samples = 50
        num_latents = 2
        num_inducing = 200
        lr = 0.04
        model_class = LotkaVolterraState
        initial_state = torch.tensor([dataset.predator[0], dataset.prey[0]], dtype=torch.float)
        lfm_kwargs['initial_state'] = initial_state
        data = torch.stack([dataset.predator, dataset.prey], dim=1)
        dataset.data = [(dataset.times, data.t())]
        data_min = data.min(dim=0).values - 0.2
        data_max = data.max(dim=0).values + 0.2
        if use_lhs:
            from smt.sampling_methods import LHS
            xlimits = np.array([
                [data_min[0], data_max[0]],
                [data_min[1], data_max[1]]])
            sampling = LHS(xlimits=xlimits)(num_inducing)
            inducing_points = torch.tensor(sampling, dtype=torch.float).unsqueeze(0)
        else:
            data_max -= data_min
            inducing_points = torch.stack([
                data_min[0] + data_max[0] * torch.rand((1, num_inducing)),
                data_min[1] + data_max[1] * torch.rand((1, num_inducing))
            ], dim=-1)
        inducing_points = inducing_points.repeat(num_latents, 1, 1)
        inducing_points = inducing_points.permute(2, 1, 0)
        track_parameters = []
        lengthscale_constraint = None
    else:
        num_tasks = 1
        num_outputs = 1
        num_samples = 150
        num_latents = 1
        num_inducing = 20
        lr = 0.025
        model_class = LotkaVolterra
        inducing_points = torch.linspace(x_min, x_max, num_inducing).repeat(num_latents, 1).view(
            num_latents, num_inducing, 1)
        track_parameters = ['raw_growth', 'raw_decay']
        lengthscale_constraint = Interval(1, 6)

    print('Num training points: ', num_training)
    config = VariationalConfiguration(latent_data_present=False, num_samples=num_samples)

    if periodic:
        mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        with torch.no_grad():
            mean_module.constant += 0.1

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(batch_shape=torch.Size([num_latents])),
            # * gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]))

        if type(covar_module.base_kernel) is gpytorch.kernels.ProductKernel:
            print(covar_module.base_kernel.kernels)
            covar_module.base_kernel.kernels[0].lengthscale = 3
            covar_module.base_kernel.kernels[0].period_length = 8
        else:
            covar_module.base_kernel.lengthscale = 1.3
            covar_module.base_kernel.period_length = 8
            track_parameters.append('gp_model.covar_module.base_kernel.raw_lengthscale')
            track_parameters.append('gp_model.covar_module.base_kernel.raw_period_length')
        gp_model = MultiOutputGP(mean_module, covar_module,
                                 inducing_points, num_latents,
                                 natural=use_natural)
        # covar_module.kernels[1].lengthscale = 2
    else:
        track_parameters.append('gp_model.covar_module.base_kernel.raw_lengthscale')
        gp_model = generate_multioutput_gp(
            num_tasks, inducing_points,
            ard_dims=1,
            zero_mean=False,
            use_scale=True, initial_lengthscale=2.,
            lengthscale_constraint=lengthscale_constraint,
            gp_kwargs=dict(natural=use_natural, independent=True)
        )

    if reload is not None:
        lfm = model_class.load(reload,
                                 gp_model=gp_model,
                                 lfm_args=[1, config],
                                 lfm_kwargs=lfm_kwargs)
    else:
        lfm = model_class(num_outputs, gp_model, config, **lfm_kwargs)

    if params['state']:
        plotter = Plotter2d(lfm, np.array(['predator', 'prey']))
    else:
        plotter = Plotter1d(lfm, np.array(['predator']))

    if use_natural:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=lr)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=lr)]

    trainer = VariationalTrainer(
        lfm,
        optimizers,
        dataset,
        warm_variational=50,
        track_parameters=track_parameters
    )

    return lfm, trainer, plotter


def plot_lotka(dataset, lfm, trainer, plotter, filepath, params):
    f_transform = softplus
    lfm.eval()
    t_interval = (0, 20)
    t_predict = torch.linspace(*t_interval, 100, dtype=torch.float32)

    if params['state']:
        with torch.no_grad():
            traj = lfm(t_predict).mean.t()
        x1 = traj[0].detach().squeeze()  # (num_genes, 100)
        x2 = traj[1].detach().squeeze()  # (num_genes, 100)
        true_x1 = dataset.predator  # (num_genes, num_cells)
        true_x2 = dataset.prey  # (num_genes, num_cells)
        time_ass = dataset.times / dataset.times.max()  # * np.array([[1., 1., 1.,]])

        plotter.plot_vector_gp(
            x1, x2, true_x1, true_x2,
            time_ass=time_ass,
            figsize=(6, 6),
        )
        plt.savefig(filepath / 'vectorfield.pdf', **tight_kwargs)
    else:
        t_scatter = dataset.data[0][0].unsqueeze(0).unsqueeze(0)
        y_scatter = dataset.data[0][1].unsqueeze(0).unsqueeze(0)

        q_m = lfm.predict_m(t_predict, step_size=1e-1)
        q_f = lfm.predict_f(t_predict)
        ylim = (-0.5, 3)
        fig, axes = plt.subplots(ncols=2,
                                 figsize=(8, 3),
                                 gridspec_kw=dict(width_ratios=[3, 1]))
        plotter.plot_gp(q_m, t_predict, num_samples=0,
                        t_scatter=t_scatter,
                        y_scatter=y_scatter,
                        ylim=ylim,
                        titles=None, ax=axes[0])
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Population')
        axes[0].set_xlim(*t_interval)
        # axes[0].legend()

        plotter.plot_gp(q_f, t_predict, num_samples=5,
                        transform=f_transform,
                        color=Colours.line2_color,
                        shade_color=Colours.shade2_color,
                        ylim=ylim,
                        titles=None, ax=axes[0])
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Prey population')
        axes[0].plot(dataset.times, dataset.prey, c=Colours.scatter_color, label='Target')
        axes[0].set_xticks([t_predict[0], t_predict[-1]])
        # axes[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=2, integer=True))
        axes[0].set_yticks([ylim[0], ylim[1]])
        axes[0].fill_between(t_scatter.squeeze(), ylim[0], ylim[1], alpha=0.2, color='gray')
        axes[0].get_lines()[0].set_label('Predator')
        axes[0].get_lines()[1].set_label('Prey')
        axes[0].legend()
        real_prey, real_pred = dataset.prey, dataset.predator
        prey = lfm.likelihood(lfm.gp_model(t_predict))
        predator = lfm(t_predict)

        prey_mean = prey.mean.detach().squeeze()
        predator_mean = predator.mean.detach().squeeze()
        x_samples = f_transform(prey.sample(torch.Size([50])).detach().squeeze())
        y_samples = predator.sample(torch.Size([50])).detach().squeeze()

        plot_phase(x_samples, y_samples,
                   x_mean=f_transform(prey_mean),
                   y_mean=predator_mean,
                   x_target=real_prey,
                   y_target=real_pred,
                   ax=axes[1])
        axes[1].set_xlabel('Prey population')
        axes[1].set_ylabel('Predator population')
        axes[1].set_yticks([0, 1])
        plt.tight_layout()

        plt.savefig(filepath / (params['kernel'] + '-combined.pdf'), **tight_kwargs)

        # labels = ['Initial', 'Grown rates', 'Decay rates']
        # kinetics = list()
        # for key in ['raw_initial', 'raw_growth', 'raw_decay']:
        #     kinetics.append(softplus(torch.tensor(trainer.parameter_trace[key][-1])).squeeze().numpy())
        #
        # plotter.plot_double_bar(kinetics, labels)
        # plt.savefig(filepath / 'kinetics.pdf', **tight_kwargs)
