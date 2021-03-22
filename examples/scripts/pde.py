#%% md

# Spatio-temporal Transcriptomics

#%%

import torch
import numpy as np

# from lafomo.variational.kernels import SpatioTemporalRBF
from lafomo.models import ReactionDiffusion, MultiOutputGP
from lafomo.datasets import ToySpatialTranscriptomics, P53Data
from lafomo.configuration import VariationalConfiguration
from lafomo.utilities.torch import save, load
from lafomo.plot import Plotter

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt


def plot_output(ax, output, title=None):
    ax.set_title(title)
    ax.plot(output)
    ax.set_xlabel('distance')
    ax.set_ylabel('y')
def scatter_output(ax, tx, output, title=None):
    ax.set_title(title)
    ax.scatter(tx[0], tx[1], c=output)
    ax.set_xlabel('time')
    ax.set_ylabel('distance')
    ax.set_aspect('equal')

config = VariationalConfiguration(
    initial_conditions=False,
    num_samples=25, learn_inducing=False
)

dataset = ToySpatialTranscriptomics(data_dir='./data/')
df = dataset.orig_data

#%%
num_inducing = 41
inducing_points = torch.linspace(0, 12, num_inducing)
inducing_points = inducing_points.unsqueeze(-1).repeat(1, 1, 2)  # (I x m x 1)
print(inducing_points.shape)

from gpytorch.constraints import Interval
gp_model = MultiOutputGP(inducing_points, 1, use_ard=True, lengthscale_constraint=Interval(0, 0.4))
gp_model.double()

# Now let's see some samples from the GP
data = next(iter(dataset))
tx, y_target = data
print(tx.dtype)

gp_model.covar_module.lengthscale = 0.3

out = gp_model(tx.transpose(0, 1))

print(out.mean.shape)
plt.imshow(out.mean.squeeze().detach().view(41, 41))
#%%

from torch.nn import Parameter
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal

from lafomo.datasets import LFMDataset
from lafomo.models import VariationalLFM


class PartialLFM(VariationalLFM):
    def __init__(self,
                 gp_model: ApproximateGP,
                 config: VariationalConfiguration,
                 dataset: LFMDataset,
                 dtype=torch.float64):
        super().__init__(gp_model, config, dataset, dtype)
        if self.config.initial_conditions:
            raise Exception('Initial conditions are not implemented for PartialLFM.')

        T = 1.0            # final time
        self.time_steps = 40
        self.mesh_cells = 40
        self.fenics_module = ReactionDiffusion(T/self.time_steps, self.mesh_cells)
        self.sensitivity = Parameter(torch.ones((1, 1), dtype=torch.float64), requires_grad=False)
        self.decay = Parameter(0.1*torch.ones((1, 1), dtype=torch.float64), requires_grad=False)
        self.diffusion = Parameter(0.01*torch.ones((1, 1), dtype=torch.float64), requires_grad=False)

    def forward(self, tx, step_size=1e-1, return_samples=False):
        """
        tx : torch.Tensor
            Shape (2, num_times)
        h : torch.Tensor the initial state of the ODE
            Shape (num_genes, 1)
        Returns
        -------
        Returns evolved h across times t.
        Shape (num_genes, num_points).
        """
        self.nfe = 0
        outputs = list()

        # Get GP outputs
        q_u = self.gp_model(tx.transpose(0, 1))
        u = q_u.rsample(torch.Size([self.config.num_samples])).permute(0, 2, 1)
        print(u.shape)
        u = self.G(u)  # (S, num_outputs, t)

        y_prev = torch.zeros((self.config.num_samples, self.mesh_cells + 1), requires_grad=False, dtype=torch.float64)
        t_index = 0

        # u = torch.tensor(df['U']).unsqueeze(0).repeat(self.config.num_samples, 1, 1)

        # Integrate forward from the initial positions h0.

        t = df['t'].values[:41]
        for n in range(self.time_steps + 1):
            # print(t_index)
            # u_n = df[df['t'] == t[t_index]]['U'].values
            # u_n = torch.tensor(u_n, requires_grad=False).unsqueeze(0).repeat(self.options.num_samples, 1)

            u_n = u[:,0,t_index::41]  # (S, t)
            # print(u_n.shape, u_nn.shape)
            # plt.plot(u[0].detach())

            sensitivity = self.sensitivity.repeat(self.config.num_samples, 1)
            decay = self.decay.repeat(self.config.num_samples, 1)
            diffusion = self.diffusion.repeat(self.config.num_samples, 1)

            y_prev = self.fenics_module(y_prev, u_n,
                                        sensitivity, decay, diffusion)

            # y_prev shape (N, 21)
            t_index += 1
            outputs.append(y_prev)

        outputs = torch.stack(outputs).permute(1, 2, 0)

        print('out', outputs.shape)

        ##
        if return_samples:
            return outputs
        f_mean = outputs.mean(dim=0).view(1, -1)  # shape (batch, times, distance)
        # h_var = torch.var(h_samples, dim=1).squeeze(-1).permute(1, 0) + 1e-7
        f_var = outputs.var(dim=0).view(1, -1) + 1e-7
        print(f_mean.shape, f_var.shape)
        # TODO: make distribution something less constraining
        f_covar = torch.diag_embed(f_var)
        print(f_covar.shape)
        batch_mvn = MultivariateNormal(f_mean, f_covar)
        return MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)
##
        # print(f_mean.shape)
        ####
        # h_samples = odeint(self.odefunc, h0, t, )  # (T, S, num_outputs, 1)
        #
        # if return_samples:
        #     return h_samples
        #
        # h_out = torch.mean(h_samples, dim=1).transpose(0, 1)
        # h_std = torch.std(h_samples, dim=1).transpose(0, 1)
        #
        # if compute_var:
        #     return self.decode(h_out), h_std
        # return self.decode(h_out)

    def G(self, u):
        return u


from lafomo.trainer import VariationalTrainer
from lafomo.utilities.torch import is_cuda
from datetime import datetime

class PDETrainer(VariationalTrainer):

    def single_epoch(self, step_size=1e-1):
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        for i, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()


            tx, y_target = data
            tx = tx.cuda() if is_cuda() else tx
            y_target = y_target.cuda() if is_cuda() else y_target
            # Assume that the batch of t s are the same
            tx = tx[0]

            output = self.lfm(tx, step_size=step_size)
            print(output.variance.max(), output.mean.shape, output.variance.shape)
            f_mean = output.mean.reshape(1, -1)

            fig, axes = plt.subplots(ncols=2)
            print(y_target.shape)
            scatter_output(axes[0], tx, f_mean.detach(), 'Prediction')
            scatter_output(axes[1], tx, y_target, 'Actual')
            plt.savefig('./out'+str(datetime.now().timestamp())+'.png')
            # Calc loss and backprop gradients
            print(output.mean.shape, y_target.permute(1, 0).shape)
            log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target.permute(1, 0))

            loss = - (log_likelihood - kl_divergence)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_ll += log_likelihood.item()
            epoch_kl += kl_divergence.item()

        return epoch_loss, (-epoch_ll, epoch_kl)

    def print_extra(self):
        print(' s:', self.lfm.sensitivity[0].item(),
              'dif:', self.lfm.diffusion[0].item(),
              'dec:', self.lfm.decay[0].item())


t_inducing = torch.linspace(0, 1, 40, dtype=torch.float64).view(1, -1).repeat(2, 1)

num_latents = 1

# kernel = SpatioTemporalRBF(num_latents, initial_lengthscale=0.3/2)

lfm = PartialLFM(gp_model, config, dataset)
optimizer = torch.optim.Adam(lfm.parameters(), lr=0.07)
trainer = PDETrainer(lfm, optimizer, dataset)

#%%

# import pandas as pd
# from os import path
# df = pd.read_csv(path.join('../../../data', 'demToy1GPmRNA.csv'))

print(df)
print(df['t'].values)

#%%
from os import path
if __name__ == '__main__':
    if path.exists('./lfm-test.pt'):
        print('present')
        lfm = PartialLFM.load('test',
                              gp_cls=MultiOutputGP,
                              gp_args=[inducing_points, 1],
                              gp_kwargs=dict(use_ard=True, lengthscale_constraint=Interval(0, 0.3)),
                              lfm_args=[config, dataset])

        optimizer = torch.optim.Adam(lfm.parameters(), lr=0.07)
        trainer = PDETrainer(lfm, optimizer, dataset)

    trainer.train(40)
    lfm.save('test')

