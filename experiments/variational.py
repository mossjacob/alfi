import torch

from matplotlib import pyplot as plt
from torch.optim import Adam
from gpytorch.optim import NGD

from alfi.configuration import VariationalConfiguration
from alfi.models import OrdinaryLFM, generate_multioutput_rbf_gp
from alfi.plot import Plotter1d
from alfi.trainers import VariationalTrainer
from alfi.utilities.data import p53_ground_truth
from alfi.impl.odes import TranscriptionLFM

tight_kwargs = dict(bbox_inches='tight', pad_inches=0)


def build_variational(dataset, params, **kwargs):
    num_tfs = 1

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
    plotter = Plotter1d(lfm, dataset.gene_names, style='seaborn')

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
