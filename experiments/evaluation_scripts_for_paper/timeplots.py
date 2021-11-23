import torch
from torch.nn import Parameter
from matplotlib import pyplot as plt
from torch.optim import Adam
from gpytorch.optim import NGD
from gpytorch.constraints import Positive
import seaborn as sns
import numpy as np

from alfi.configuration import VariationalConfiguration
from alfi.models import OrdinaryLFM, generate_multioutput_gp
from alfi.trainers import VariationalTrainer
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from experiments.model_specs.variational import TranscriptionLFM

from alfi.datasets import P53Data
from alfi.models import ExactLFM
from alfi.trainers import ExactTrainer
from collections import namedtuple

""" Experiment for finding timings """


dataset = P53Data(replicate=0, data_dir='./data')
num_genes = 5
num_tfs = 1
config = VariationalConfiguration(
    preprocessing_variance=dataset.variance,
    num_samples=80,
    initial_conditions=False
)

t_end = dataset.t_observed[-1]
num_inducing = 20  # (I x m x 1)

class TempExactTrainer(ExactTrainer):
    def after_epoch(self):
        super().after_epoch()
        with torch.no_grad():
            sens = self.lfm.covar_module.sensitivity
            sens[3] = np.float64(1.)
            deca = self.lfm.covar_module.decay
            deca[3] = np.float64(0.8)
            self.lfm.covar_module.sensitivity = sens
            self.lfm.covar_module.decay = deca

class TempVariationalTrainer(VariationalTrainer):
    def after_epoch(self):
        with torch.no_grad():
            sens = torch.tensor(1.)
            dec = torch.tensor(0.8)
            self.lfm.raw_sensitivity[3] = self.lfm.positivity.inverse_transform(sens)
            self.lfm.raw_decay[3] = self.lfm.positivity.inverse_transform(dec)
        super().after_epoch()

if __name__ == '__main__':
    with open('experiments/times.txt', 'w') as f:
        num_genes = 5
        num_tfs = 1
        config = VariationalConfiguration(
            num_samples=80,
            initial_conditions=False
        )
        exact_times_list = list()
        varia_times_list = list()

        for i in range(10):
            print('Running ', i)
            #### Train Exact
            exact_lfm = ExactLFM(dataset, dataset.variance.reshape(-1))
            optimizer = torch.optim.Adam(exact_lfm.parameters(), lr=0.01)

            loss_fn = ExactMarginalLogLikelihood(exact_lfm.likelihood, exact_lfm)


            exact_trainer = TempExactTrainer(
                exact_lfm, [optimizer], dataset, loss_fn=loss_fn)

            exact_lfm.train()
            exact_lfm.likelihood.train()

            exact_times = exact_trainer.train(epochs=600, report_interval=50)
            exact_times = np.array(exact_times)
            exact_times_list.append(exact_times)

            #### Train variational

            inducing_points = torch.linspace(0, t_end, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)
            gp_model = generate_multioutput_gp(num_tfs, inducing_points, gp_kwargs=dict(natural=True))

            lfm = TranscriptionLFM(num_genes, gp_model, config)

            track_parameters = [
                'raw_basal',
                'raw_decay',
                'raw_sensitivity',
                'gp_model.covar_module.raw_lengthscale',
            ]
            num_training = dataset.m_observed.shape[-1]

            variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.1)
            parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.03)
            optimizers = [variational_optimizer, parameter_optimizer]
            trainer = TempVariationalTrainer(lfm, optimizers, dataset, track_parameters=track_parameters)

            lfm.train()
            varia_times = trainer.train(600, step_size=5e-1, report_interval=50)
            varia_times = np.array(varia_times)
            varia_times_list.append(varia_times)
            last_loss = trainer.losses[-1].sum()

        varia_times = np.stack(varia_times_list, axis=0)
        exact_times = np.stack(exact_times_list, axis=0)

        np.save('./experiments/exact_times.npy', exact_times)
        np.save('./experiments/varia_times.npy', varia_times)

        print('finished')
