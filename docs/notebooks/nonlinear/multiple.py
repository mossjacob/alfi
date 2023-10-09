
import torch
import numpy as np
from gpytorch.optim import NGD
from torch.optim import Adam
from torch.nn import Parameter
from matplotlib import pyplot as plt

from alfi.datasets import P53Data
from alfi.configuration import VariationalConfiguration
from alfi.models import OrdinaryLFM, TrainMode, generate_multioutput_gp
from alfi.plot import Plotter1d, Colours, tight_kwargs
from alfi.trainers import VariationalTrainer, PreEstimator
from alfi.datasets import ToyTranscriptomicGenerator
from docs.notebooks.nonlinear.temp_data import Data

dataset = torch.load('./test_dataset.pt')

#basal_rate, sensitivity, decay_rate, lengthscale=lengthscale)
# ground_truths = P53Data.params_ground_truth()
class ConstrainedTrainer(VariationalTrainer):
    def after_epoch(self):
        # with torch.no_grad():
        #     sens = torch.tensor(1.)
        #     dec = torch.tensor(0.8)
        #     self.model.raw_sensitivity[3] = self.model.positivity.inverse_transform(sens)
        #     self.model.raw_decay[3] = self.model.positivity.inverse_transform(dec)
        super().after_epoch()

num_genes = 5
num_tfs = 1

t_end = dataset.t_observed[-1]

from alfi.utilities import softplus
from torch import softmax
from gpytorch.constraints import Positive
class TranscriptionLFM(OrdinaryLFM):
    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        self.positivity = Positive()
        self.raw_decay = Parameter(
            self.positivity.inverse_transform(0.1 + torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        self.raw_basal = Parameter(
            self.positivity.inverse_transform(0.1 * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        self.raw_sensitivity = Parameter(
            self.positivity.inverse_transform(2*torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))

    def nonlinearity(self, f):
        return softplus(f)

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
        # if (self.nfe % 100) == 0:
        #     print(t)
        f = self.latent_gp
        if not (self.train_mode == TrainMode.GRADIENT_MATCH):
            f = f[:, :, self.t_index].unsqueeze(2)
            if t > self.last_t:
                self.t_index += 1
            self.last_t = t

        dh = self.basal_rate + self.sensitivity * f - self.decay_rate * h
        return dh


config = VariationalConfiguration(
    num_samples=80,
    initial_conditions=False
)

num_inducing = 20  # (I x m x 1)
t_predict = torch.linspace(0, t_end + 2, 80, dtype=torch.float32)
step_size = 5e-1
num_training = dataset.m_observed.shape[-1]
use_natural = False

print('hi')

track_parameters = [
    'raw_basal',
    'raw_decay',
    'raw_sensitivity',
    'gp_model.covar_module.raw_lengthscale',
]


def run(arg):
    inducing_points = torch.linspace(0, t_end, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)
    gp_model = generate_multioutput_gp(num_tfs, inducing_points, gp_kwargs=dict(natural=use_natural))
    lfm = TranscriptionLFM(num_genes, gp_model, config, num_training_points=num_training)
    if use_natural:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.09)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.02)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.05)]
    trainer = ConstrainedTrainer(lfm, optimizers, dataset, track_parameters=track_parameters)

    lfm.set_mode(TrainMode.NORMAL)
    lfm.loss_fn.num_data = num_training
    trainer.train(200, report_interval=1000, step_size=step_size);


from torch import multiprocessing
from collections import defaultdict
import time
if __name__ == '__main__':
    durations = defaultdict(list)
    for size in [64, 256, 1024, 4096, 16384]:
        for r in range(10):  # replicates
            start = time.time()

            with multiprocessing.Pool(20) as p:
                p.map(run, range(size))

            end = time.time()
            print(f'Iteration {r} for size {size} complete in {end - start} seconds')
            durations[size].append(end - start)
        torch.save(durations, f'./durations{size}.pt')
