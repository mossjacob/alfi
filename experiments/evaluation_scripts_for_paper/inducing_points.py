import torch
from torch.nn import Parameter
from matplotlib import pyplot as plt
from torch.optim import Adam
from gpytorch.optim import NGD
from gpytorch.constraints import Positive
import seaborn as sns
import numpy as np

from experiments.model_specs.variational import TranscriptionLFM
from alfi.datasets import P53Data
from alfi.configuration import VariationalConfiguration
from alfi.models import OrdinaryLFM, generate_multioutput_gp
from alfi.trainers import VariationalTrainer


""" 
Experiment for plotting the ideal number of inducing points 
"""


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


dataset = P53Data(replicate=0, data_dir='./data')
num_genes = 5
num_tfs = 1
config = VariationalConfiguration(
    preprocessing_variance=dataset.variance,
    num_samples=80,
    initial_conditions=False
)

def diff(lfm: TranscriptionLFM):
    B_exact, S_exact, D_exact = P53Data.params_ground_truth()
    B = lfm.basal_rate.detach().squeeze()
    D = lfm.decay_rate.detach().squeeze()
    S = lfm.sensitivity.detach().squeeze()
    mse = torch.square(B-B_exact) + torch.square(D-D_exact) + torch.square(S-S_exact)
    mse = mse.mean()

    return mse


with open('experiments/inducing_points.txt', 'w') as f:
    outputs = list()
    for i in range(5, 30):
        print('Running inducing points', i)
        num_inducing = i  # (I x m x 1)
        inducing_points = torch.linspace(0, 12, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)

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
        trainer = P53ConstrainedTrainer(lfm, optimizers, dataset, track_parameters=track_parameters)

        lfm.train()
        trainer.train(350, report_interval=50, step_size=1e-1)
        last_loss = trainer.losses[-1].sum()
        mse = diff(lfm)
        f.write(f'{i}\t{mse}\n')
        outputs.append((i, mse, last_loss))

    outputs = np.array(outputs)
    palette = sns.color_palette('colorblind')
    line_color = palette[0]
    plt.style.use('seaborn')
    sns.set(font="CMU Serif")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'CMU Serif'

    plt.figure(figsize=(4, 3))
    plt.plot(outputs[:, 0], outputs[:, 2])
    plt.xlabel('Inducing points')
    plt.ylabel('MSE')
    plt.savefig('experiments/inducingpoints_loss.pdf')

    plt.figure(figsize=(4, 3))
    plt.plot(outputs[:, 0], outputs[:, 1])
    plt.xlabel('Inducing points')
    plt.ylabel('MSE')
    plt.savefig('experiments/inducingpoints_line.pdf')

    plt.figure(figsize=(4, 3))
    plt.scatter(outputs[:, 0], outputs[:, 1], s=5)
    plt.xlabel('Inducing points')
    plt.ylabel('MSE')
    plt.savefig('experiments/inducingpoints_scat.pdf')
