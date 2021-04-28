import torch
import numpy as np
from torch.nn import Parameter
from matplotlib import pyplot as plt
from gpytorch.constraints import Positive
from gpytorch.optim import NGD
from torch.optim import Adam

from lafomo.configuration import VariationalConfiguration
from lafomo.models import OrdinaryLFM, generate_multioutput_rbf_gp
from lafomo.plot import Plotter
from lafomo.trainers import VariationalTrainer
from lafomo.utilities.data import p53_ground_truth

tight_kwargs = dict(bbox_inches='tight', pad_inches=0)


def build_variational(dataset, params, **kwargs):
    num_tfs = 1
    class TranscriptionLFM(OrdinaryLFM):
        def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
            super().__init__(num_outputs, gp_model, config, **kwargs)
            self.positivity = Positive()
            self.raw_decay = Parameter(0.1 + torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))
            self.raw_basal = Parameter(0.5 * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))
            self.raw_sensitivity = Parameter(torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))

        @property
        def decay_rate(self):
            return self.positivity.transform(self.raw_decay)

        @decay_rate.setter
        def decay_rate(self, value):
            self.raw_decay = self.positivity.inverse_transform(value)

        @property
        def basal_rate(self):
            return self.positivity.transform(self.raw_basal)

        @basal_rate.setter
        def basal_rate(self, value):
            self.raw_basal = self.positivity.inverse_transform(value)

        @property
        def sensitivity(self):
            return self.positivity.transform(self.raw_sensitivity)

        @sensitivity.setter
        def sensitivity(self, value):
            self.raw_sensitivity = self.decay_constraint.inverse_transform(value)

        def initial_state(self):
            return self.basal_rate / self.decay_rate

        def odefunc(self, t, h):
            """h is of shape (num_samples, num_outputs, 1)"""
            self.nfe += 1

            f = self.f
            if not self.pretrain_mode:
                f = self.f[:, :, self.t_index].unsqueeze(2)
                if t > self.last_t:
                    self.t_index += 1
                self.last_t = t

            dh = self.basal_rate + self.sensitivity * f - self.decay_rate * h
            return dh

    config = VariationalConfiguration(
        preprocessing_variance=dataset.variance,
        num_samples=80,
        initial_conditions=False
    )

    num_inducing = 12  # (I x m x 1)
    inducing_points = torch.linspace(0, 12, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)
    num_training = dataset.m_observed.shape[-1]
    use_natural = True
    gp_model = generate_multioutput_rbf_gp(num_tfs, inducing_points, gp_kwargs=dict(natural=use_natural))

    lfm = TranscriptionLFM(dataset.num_outputs, gp_model, config, num_training_points=num_training)
    plotter = Plotter(lfm, dataset.gene_names, style='seaborn')

    class P53ConstrainedTrainer(VariationalTrainer):
        def after_epoch(self):
            super().after_epoch()
            with torch.no_grad():
                self.lfm.basal_rate.clamp_(0, 20)
                self.lfm.decay_rate.clamp_(0, 20)
                sens = torch.tensor(1.)
                dec = torch.tensor(0.8)
                self.lfm.raw_sensitivity[3] = self.lfm.positivity.inverse_transform(sens)
                self.lfm.raw_decay[3] = self.lfm.positivity.inverse_transform(dec)

    track_parameters = [
        'raw_basal',
        'raw_decay',
        'raw_sensitivity',
        'gp_model.covar_module.raw_lengthscale',
    ]
    if use_natural:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.03)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.05)]
    trainer = P53ConstrainedTrainer(lfm, optimizers, dataset, track_parameters=track_parameters)

    return lfm, trainer, plotter


def plot_variational(dataset, lfm, trainer, plotter, filepath, params):
    lfm.eval()

    t_predict = torch.linspace(-1, 13, 80, dtype=torch.float32)

    labels = ['Basal rates', 'Sensitivities', 'Decay rates']
    kinetics = list()
    for key in ['raw_basal', 'raw_sensitivity', 'raw_decay']:
        kinetics.append(trainer.parameter_trace[key][-1].squeeze().numpy())

    plotter.plot_double_bar(kinetics, labels, p53_ground_truth())
    plt.savefig(filepath / 'kinetics.pdf', **tight_kwargs)

    plotter.plot_outputs(t_predict, replicate=0,
                         t_scatter=dataset.t_observed, y_scatter=dataset.m_observed,
                         model_kwargs=dict(step_size=1e-1))
    plt.savefig(filepath / 'outputs.pdf', **tight_kwargs)

    plotter.plot_latents(t_predict, ylim=(-1, 3), plot_barenco=False, plot_inducing=False)
    plt.savefig(filepath / 'latents.pdf', **tight_kwargs)
