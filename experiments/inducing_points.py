import torch
from torch.nn import Parameter
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from lafomo.datasets import P53Data
from lafomo.configuration import VariationalConfiguration
from lafomo.models import OrdinaryLFM, MultiOutputGP
from lafomo.trainer import TranscriptionalTrainer


""" Experiment for plotting the ideal inducing point """


class TranscriptionLFM(OrdinaryLFM):
    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration):
        super().__init__(num_outputs, gp_model, config)
        self.decay_rate = Parameter(0.1 + torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))
        self.basal_rate = Parameter(torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))
        self.sensitivity = Parameter(0.2 + torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64))

    def initial_state(self):
        return self.basal_rate / self.decay_rate

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)

        decay = self.decay_rate * h

        f = self.f[:, :, self.t_index].unsqueeze(2)

        h = self.basal_rate + self.sensitivity * f - decay
        if t > self.last_t:
            self.t_index += 1
        self.last_t = t
        return h


class P53ConstrainedTrainer(TranscriptionalTrainer):
    def extra_constraints(self):
        self.lfm.sensitivity[3] = np.float64(1.)
        self.lfm.decay_rate[3] = np.float64(0.8)


dataset = P53Data(replicate=0, data_dir='./data')
num_genes = 5
num_tfs = 1
config = VariationalConfiguration(
    preprocessing_variance=dataset.variance,
    num_samples=80,
    kernel_scale=False,
    initial_conditions=False
)



def diff(lfm: TranscriptionLFM):
    B_exact = np.array([0.0649, 0.0069, 0.0181, 0.0033, 0.0869])
    D_exact = np.array([0.2829, 0.3720, 0.3617, 0.8000, 0.3573])
    S_exact = np.array([0.9075, 0.9748, 0.9785, 1.0000, 0.9680])

    B = lfm.basal_rate.detach()
    D = lfm.basal_rate.detach()
    S = lfm.basal_rate.detach()
    mse = torch.square(B-B_exact) + torch.square(D-D_exact) + torch.square(S-S_exact)
    mse = mse.mean()

    return mse


with open('experiments/inducing_points.txt', 'w') as f:
    outputs = list()
    for i in range(5, 30):
        print('Running inducing points', i)
        num_inducing = i  # (I x m x 1)
        inducing_points = torch.linspace(0, 12, num_inducing).repeat(num_tfs, 1).view(num_tfs, num_inducing, 1)

        gp_model = MultiOutputGP(inducing_points, num_tfs)
        lfm = TranscriptionLFM(num_genes, gp_model, config)

        optimizer = torch.optim.Adam(lfm.parameters(), lr=0.03)
        trainer = P53ConstrainedTrainer(lfm, optimizer, dataset)

        lfm.train()
        trainer.train(350, report_interval=9, step_size=1e-1)
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
