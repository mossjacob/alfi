from typing import List
import torch

from alfi.trainers import Trainer
from alfi.models import VariationalLFM
from alfi.utilities.torch import is_cuda, spline_interpolate_gradient, savgol_filter_gradient

from . import EMTrainer


class EMPreEstimator(Trainer):
    def __init__(self,
                 lfm: VariationalLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 em_trainer: EMTrainer,
                 **kwargs):
        super().__init__(lfm, optimizers, dataset, **kwargs)
        self.e_step = em_trainer.e_step

    def single_epoch(self, epoch=0, step_size=1e-1, **kwargs):
        data = next(iter(self.data_loader))
        y = data.permute(0, 2, 1)  # (O, C, 1)
        y = y.cuda() if is_cuda() else y
        cells = y[:-1] if self.lfm.config.latent_data_present else y
        num_outputs = self.lfm.num_outputs
        u_y = cells[:num_outputs // 2]  # (num_genes, num_cells)
        s_y = cells[num_outputs // 2:]  # (num_genes, num_cells)

        # First step: sort the cells in the current assignment order
        sorted_t, sorted_ind = self.lfm.time_assignments_indices.sort()
        sorted_t = self.lfm.timepoint_choices[sorted_t]
        u_y = u_y[:, sorted_ind].squeeze(-1)
        s_y = s_y[:, sorted_ind].squeeze(-1)

        # Second step: bucket the cells based on their time assigment
        u_y_bucketed = list()
        s_y_bucketed = list()
        for t in self.lfm.timepoint_choices:
            a = sorted_t == t
            u_y_bucketed.append(u_y[:, a].mean(-1))
            s_y_bucketed.append(s_y[:, a].mean(-1))
        u_y_bucketed = torch.stack(u_y_bucketed, dim=1)
        s_y_bucketed = torch.stack(s_y_bucketed, dim=1)

        # Third step: interpolate the buckets to ensure regular time spacing
        t_interpolate, u_y_interp, _, _ = spline_interpolate_gradient(
            self.lfm.timepoint_choices,
            u_y_bucketed.unsqueeze(-1),
            num_disc=2)
        _, s_y_interp, _, _ = spline_interpolate_gradient(
            self.lfm.timepoint_choices,
            s_y_bucketed.unsqueeze(-1),
            num_disc=2)

        # Fourth step: denoise and get the first derivative
        u_y_denoised, du = savgol_filter_gradient(t_interpolate, u_y_interp.squeeze(-1))
        s_y_denoised, ds = savgol_filter_gradient(t_interpolate, s_y_interp.squeeze(-1))
        du, ds, u_y_denoised, s_y_denoised = [
            torch.tensor(x) for x in [du, ds, u_y_denoised, s_y_denoised]]
        if du.ndim < 2:
            du, ds, u_y_denoised, s_y_denoised = [
                x.unsqueeze(0) for x in [du, ds, u_y_denoised, s_y_denoised]]
        du, ds, u_y_denoised, s_y_denoised = [
            x[:, ::3] for x in [du, ds, u_y_denoised, s_y_denoised]]
        data_interpolated_gradient = torch.cat([du, ds], dim=0).t()
        data_interpolated = torch.cat([u_y_denoised, s_y_denoised], dim=0)
        t_interpolate = t_interpolate[::3]

        # Run E step (time assignment)
        if (epoch % 5) == 0:
            print('running e step', epoch)
            with torch.no_grad():
                self.lfm.pretrain(False)
                self.lfm(self.lfm.timepoint_choices, step_size=step_size)
                self.e_step(cells)

        # Run M step (gradient matching)
        [optim.zero_grad() for optim in self.optimizers]
        self.lfm.pretrain(True)
        cell_trajectory_ordered = self.lfm((t_interpolate, data_interpolated))

        log_likelihood, kl_divergence, _ = self.lfm.loss_fn(cell_trajectory_ordered, data_interpolated_gradient)
        total_loss = (-log_likelihood + kl_divergence)
        total_loss.backward()

        [optim.step() for optim in self.optimizers]
        return total_loss, (-log_likelihood, kl_divergence)
