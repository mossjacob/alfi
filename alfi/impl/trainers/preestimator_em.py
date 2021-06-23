from typing import List
import torch

from alfi.trainers import PreEstimator
from alfi.models import VariationalLFM
from alfi.utilities.torch import is_cuda, spline_interpolate_gradient

from . import EMTrainer


class EMPreEstimator(PreEstimator):
    def __init__(self,
                 lfm: VariationalLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 em_trainer: EMTrainer,
                 **kwargs):
        super().__init__(lfm, optimizers, dataset, **kwargs)
        self.e_step = em_trainer.e_step

    def single_epoch(self, epoch=0, **kwargs):
        data = next(iter(self.data_loader))
        print(data.shape)
        [optim.zero_grad() for optim in self.optimizers]
        y = data.permute(0, 2, 1)  # (O, C, 1)
        y = y.cuda() if is_cuda() else y
        cells = y[:-1] if self.lfm.config.latent_data_present else y
        # Run E step (time assignment)
        self.e_step(cells)

        # Run M step (gradient matching)
        num_outputs = self.lfm.num_outputs
        u_y = cells[:num_outputs//2]  # (num_genes, num_cells)
        s_y = cells[num_outputs//2:]  # (num_genes, num_cells)
        print(u_y.shape)
        sorted_t, sorted_ind = self.lfm.time_assignments_indices.sort()

        u_y = u_y[:, sorted_ind]
        s_y = s_y[:, sorted_ind]
        u_y_interp = spline_interpolate_gradient(sorted_t, )
        data_interpolated =
        cell_trajectory_ordered = ode_func(t)
        loss (cell_trajectory_ordered, data_interpolated)

        loss backward