from typing import List

import torch
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

from lafomo.utilities.torch import is_cuda, discretise
from lafomo.models import PartialLFM
from lafomo.plot import plot_spatiotemporal_data
from .variational import VariationalTrainer
from lafomo.utilities.torch import cia, q2, smse, softplus


class PDETrainer(VariationalTrainer):

    def __init__(self, lfm: PartialLFM, optimizers: List[torch.optim.Optimizer], dataset, clamp=False, lf_target=None, **kwargs):
        super().__init__(lfm, optimizers, dataset, **kwargs)
        self.debug_iteration = 0
        self.clamp = clamp
        self.plot_outputs = True
        self.plot_outputs_iter = 10
        self.prot_q2_best = 0
        self.mrna_q2_best = 0
        self.cia = (0, 0)
        self.disc = 1
        data = next(iter(dataset))
        data_input, y = data
        tx, y_target = data_input, y

        # 1. Ensure that time axis must be in ascending order:
        # We use mergesort to maintain relative order
        self.t_sorted = np.argsort(tx[0, :], kind='mergesort')
        tx = tx[:, self.t_sorted]
        self.num_t_orig = tx[0].unique().shape[0]
        self.num_x_orig = tx[1].unique().shape[0]
        y_target = y_target[:, self.t_sorted]

        # 2. Discretise time
        if hasattr(dataset, 'num_discretised'):
            num_discretised = dataset.num_discretised
            self.disc = dataset.disc
        else:
            num_discretised = 40
        time = discretise(tx[0, :], num_discretised=num_discretised)
        time = torch.tensor(time)

        # 3. Discretise space
        spatial_grid = self.discretise_spatial(tx)
        spatial = torch.tensor(spatial_grid)

        # 4. Reconstruct dataset
        new_t = time.repeat(spatial.shape[0], 1).transpose(0, 1).reshape(-1)
        # t_mask = new_t == tx[0, :]

        new_x = spatial.repeat(time.shape[0])
        # x_mask = new_x == tx[1, :]

        # mask = torch.stack([t_mask, x_mask])
        self.tx = torch.stack([new_t, new_x])
        self.y_target = y_target.cuda() if is_cuda() else y_target
        self.tx = self.tx.cuda() if is_cuda() else self.tx
        self.t_sorted = self.t_sorted.cuda() if is_cuda() else self.t_sorted
        self.lf_target = lf_target[self.t_sorted, 2]
        self.num_t = self.tx[0, :].unique().shape[0]
        self.num_x = self.tx[1, :].unique().shape[0]

    def discretise_spatial(self, tx):
        # whilst maintaining an inverse mapping to mask the output for
        # calculating the loss. shape (T, X_unique).
        # For each time, for which spatial points do we have data for
        spatial = torch.unique(tx[1, :])
        range = spatial[-1] - spatial[0]
        x_dp = spatial[1] - spatial[0]
        print('x dp is set to', x_dp)
        num_discretised = int(range / x_dp)
        spatial_grid = discretise(spatial, num_discretised=num_discretised)
        return spatial_grid

    def single_epoch(self, step_size=1e-1, epoch=0, **kwargs):
        [optim.zero_grad() for optim in self.optimizers]
        output = self.lfm(self.tx, step_size=step_size, step=self.disc)
        y_target = self.y_target.t()

        self.debug_out(self.tx, y_target, output)
        log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target, mask=self.train_mask)
        loss = - (log_likelihood - kl_divergence)

        loss.backward()
        if epoch >= self.warm_variational:
            [optim.step() for optim in self.optimizers]
        else:
            print(epoch, 'warming up')
            self.optimizers[0].step()

        # Now we are warmed up, start training non variational parameters in the next epoch.
        if epoch + 1 == self.warm_variational:
            for param in self.lfm.nonvariational_parameters():
                param.requires_grad = True

        return loss.item(), (-log_likelihood.item(), kl_divergence.item())

    def debug_out(self, data_input, y_target, output):
        if (self.debug_iteration % 5) != 0:
            self.debug_iteration += 1
            return
        print('Mean output variance:', output.variance.mean().item())
        if self.train_mask is not None:
            with torch.no_grad():
                log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target, mask=~self.train_mask)
                test_loss = - (log_likelihood - kl_divergence)
            print('Test loss:', test_loss.item())
        with torch.no_grad():
            f_mean = output.mean
            f_var = output.variance
            prot_q2 = q2(y_target.squeeze(), f_mean.squeeze())
            prot_cia = cia(y_target.squeeze(), f_mean.squeeze(), f_var.squeeze())

            gp = self.lfm.gp_model(self.tx.t())
            lf_target = self.lf_target
            num_t = self.num_t
            num_x = self.num_x
            f_mean = gp.mean.view(num_t, num_x)[::self.disc].reshape(-1)
            f_var = gp.variance.view(num_t, num_x)[::self.disc].reshape(-1)

            mrna_q2 = q2(lf_target.squeeze(), f_mean.squeeze())
            mrna_cia = cia(lf_target.squeeze(), f_mean.squeeze(), f_var.squeeze())
            if (mrna_q2 + prot_q2) > (self.mrna_q2_best + self.prot_q2_best):
                self.mrna_q2_best = mrna_q2
                self.prot_q2_best = prot_q2
                self.cia = (mrna_cia, prot_cia)
            print(f'prot Q2: {prot_q2.item():.03f}')
            print(f'prot Q2: {prot_cia.item():.03f}')
            print('mrna Q2', mrna_q2.item())
            print('mrna CA', mrna_cia.item())

        if self.plot_outputs and (self.debug_iteration % self.plot_outputs_iter) == 0:
            ts = self.tx[0, :].unique().numpy()
            xs = self.tx[1, :].unique().numpy()
            extent = [ts[0], ts[-1], xs[0], xs[-1]]

            num_t = self.num_t_orig
            num_x = self.num_x_orig
            f_mean = output.mean.reshape(num_t, num_x).detach()
            y_target = y_target.reshape(num_t, num_x)
            axes = plot_spatiotemporal_data(
                [f_mean.t(), y_target.t()],
                extent, titles=['Prediction', 'Ground truth']
            )
            xy = self.lfm.inducing_points.detach()[0]
            axes[0].scatter(xy[:, 0], xy[:, 1], facecolors='none', edgecolors='r', s=3)

            plt.savefig(str(datetime.now().timestamp()) + '.png')
            self.lfm.save('currentmodel')
        self.debug_iteration += 1

    def print_extra(self):
        print(' s:', softplus(self.lfm.fenics_parameters[0][0]).item(),
              'dec:', softplus(self.lfm.fenics_parameters[1][0]).item(),
              'diff:', softplus(self.lfm.fenics_parameters[2][0]).item())

    def after_epoch(self):
        super().after_epoch()
        if self.clamp:
            with torch.no_grad():
                self.lfm.fenics_parameters[2].clamp_(-15, -2.25)
                self.lfm.fenics_parameters[1].clamp_(-15, -2.25)
                self.lfm.fenics_parameters[0].clamp_(-15, -2.25)
