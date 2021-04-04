import torch
import numpy as np
from matplotlib import pyplot as plt
import gpytorch
from torch.nn import Parameter
from torch.optim import Adam
from gpytorch.optim import NGD
from gpytorch.constraints import Positive, Interval

from lafomo.models import OrdinaryLFM, MultiOutputGP
from lafomo.utilities.torch import inv_softplus, softplus
from lafomo.plot import Plotter, plot_phase
from lafomo.configuration import VariationalConfiguration
from lafomo.trainers import VariationalTrainer

tight_kwargs = dict(bbox_inches='tight', pad_inches=0)


def build_lotka(dataset, params):
    num_tfs = 1
    x_min, x_max = min(dataset.times), max(dataset.times)

    num_latents = 1
    num_outputs = 1
    num_training = dataset[0][0].shape[0]
    num_inducing = 12

    print('Num training points: ', num_training)
    output_names = np.array(['pred', 'prey'])

    class LotkaVolterra(OrdinaryLFM):
        """Outputs are predator. Latents are prey"""

        def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
            super().__init__(num_outputs, gp_model, config, **kwargs)
            self.positivity = Positive()
            self.decay_constraint = Interval(0., 1.5)
            self.raw_decay = Parameter(
                self.positivity.inverse_transform(torch.ones(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
            self.raw_growth = Parameter(self.positivity.inverse_transform(
                0.5 * torch.ones(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
            self.raw_initial = Parameter(self.positivity.inverse_transform(
                0.3 + torch.zeros(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
            self.true_f = dataset.prey[::3].unsqueeze(0).repeat(self.config.num_samples, 1).unsqueeze(1)
            print(self.true_f.shape)

        @property
        def decay_rate(self):
            return self.decay_constraint.transform(self.raw_decay)

        @decay_rate.setter
        def decay_rate(self, value):
            self.raw_decay = self.decay_constraint.inverse_transform(value)

        @property
        def growth_rate(self):
            return softplus(self.raw_growth)

        @growth_rate.setter
        def growth_rate(self, value):
            self.raw_growth = inv_softplus(value)

        @property
        def initial_predators(self):
            return softplus(self.raw_initial)

        @initial_predators.setter
        def initial_predators(self, value):
            self.raw_initial = inv_softplus(value)

        def initial_state(self):
            return self.initial_predators

        def odefunc(self, t, h):
            """h is of shape (num_samples, num_outputs, 1)"""
            self.nfe += 1
            # if (self.nfe % 100) == 0:
            # print(t, self.t_index, self.f.shape, self.true_f.shape)
            # f shape (num_samples, num_outputs, num_times)
            f = self.f[:, :, self.t_index].unsqueeze(2)
            # f = self.true_f[:, :, self.t_index].unsqueeze(2)
            dh = self.growth_rate * h * f - self.decay_rate * h
            if t > self.last_t:
                self.t_index += 1
            self.last_t = t

            return dh

        def G(self, f):
            return softplus(f).repeat(1, self.num_outputs, 1)

    use_natural = params['natural']
    config = VariationalConfiguration(num_samples=70)
    inducing_points = torch.linspace(x_min, x_max, num_inducing).repeat(num_latents, 1).view(
        num_latents, num_inducing, 1)
    t_predict = torch.linspace(0, x_max, 151, dtype=torch.float32)

    periodic = params['kernel'] == 'periodic'
    mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
    with torch.no_grad():
        mean_module.constant -= 0.2
    track_parameters = ['raw_growth', 'raw_decay']

    if periodic:
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )  # * \
        # gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]))

        print(covar_module.base_kernel.period_length)
        covar_module.base_kernel.lengthscale = 3
        covar_module.base_kernel.period_length = 8
        track_parameters.append('gp_model.covar_module.base_kernel.raw_lengthscale')
        track_parameters.append('gp_model.covar_module.base_kernel.raw_period_length')
        # covar_module.kernels[1].lengthscale = 2
    else:
        covar_module = gpytorch.kernels.RBFKernel(
            batch_shape=torch.Size([num_latents]),
            lengthscale_constraint=Interval(1, 6))
        covar_module.lengthscale = 2
        track_parameters.append('gp_model.covar_module.raw_lengthscale')

    gp_model = MultiOutputGP(mean_module, covar_module,
                             inducing_points, num_latents,
                             natural=use_natural)
    lfm = LotkaVolterra(num_outputs, gp_model, config, num_training_points=num_training)

    plotter = Plotter(lfm, np.array(['predator']))

    if use_natural:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.02)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.02)]

    trainer = VariationalTrainer(
        lfm,
        optimizers,
        dataset,
        warm_variational=50,
        track_parameters=track_parameters
    )

    return lfm, trainer, plotter


def plot_lotka(dataset, lfm, trainer, plotter, filepath, params):
    lfm.eval()

    t_predict = torch.linspace(-5, 18, 100, dtype=torch.float32)
    t_scatter = dataset.data[0][0].unsqueeze(0).unsqueeze(0)
    y_scatter = dataset.data[0][1].unsqueeze(0).unsqueeze(0)

    q_m = lfm.predict_m(t_predict, step_size=1e-2)
    q_f = lfm.predict_f(t_predict)
    plotter.plot_gp(q_m, t_predict, num_samples=0,
                    t_scatter=t_scatter,
                    y_scatter=y_scatter,
                    titles=['Predator'])
    plt.title('')
    plt.savefig(filepath / (params['kernel'] + '-predator.pdf'), **tight_kwargs)

    plotter.plot_gp(q_f, t_predict,
                    transform=softplus,
                    t_scatter=dataset.times[::5],
                    y_scatter=dataset.prey[None, None, ::5],
                    ylim=(-0.9, 4),
                    titles=['Prey'])
    plt.savefig(filepath / (params['kernel'] + '-prey.pdf'), **tight_kwargs)

    real_prey, real_pred = dataset.prey, dataset.predator
    prey = lfm.likelihood(lfm.gp_model(t_predict))
    predator = lfm(t_predict)

    prey_mean = prey.mean.detach().squeeze()
    predator_mean = predator.mean.detach().squeeze()
    x_samples = softplus(prey.sample(torch.Size([50])).detach().squeeze())
    y_samples = predator.sample(torch.Size([50])).detach().squeeze()

    plot_phase(x_samples, y_samples,
               x_mean=softplus(prey_mean),
               y_mean=predator_mean,
               x_target=real_prey,
               y_target=real_pred)
    plt.xlabel('Prey population')
    plt.ylabel('Prey population')
    plt.tight_layout()
    plt.savefig(filepath / (params['kernel'] + '-phase.pdf'), **tight_kwargs)


    # labels = ['Initial', 'Grown rates', 'Decay rates']
    # kinetics = list()
    # for key in ['raw_initial', 'raw_growth', 'raw_decay']:
    #     kinetics.append(softplus(torch.tensor(trainer.parameter_trace[key][-1])).squeeze().numpy())
    #
    # plotter.plot_double_bar(kinetics, labels)
    # plt.savefig(filepath / 'kinetics.pdf', **tight_kwargs)
