from typing import List

import torch

from .trainer import Trainer
from lafomo.models import VariationalLFM
from lafomo.utilities.torch import is_cuda, spline_interpolate_gradient


class ParameterPreEstimator(Trainer):
    def __init__(self,
                 lfm: VariationalLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 **kwargs):
        super().__init__(lfm, optimizers, dataset, **kwargs)
        num_intermediate = 9
        data = next(iter(self.data_loader))
        t, y = data[0][0], data[1]

        t_interpolate, y_interpolate, y_grad, _ = spline_interpolate_gradient(
            t, data[1].permute(1, 0), num_intermediate)

        self.y_interpolate = y_interpolate.t()
        self.input_pair = (t_interpolate, self.y_interpolate)
        self.target = y_grad

    def single_epoch(self, epoch=0, **kwargs):
        assert self.lfm.pretrain_mode
        [optim.zero_grad() for optim in self.optimizers]
        # y = y.cuda() if is_cuda() else y

        output = self.lfm(self.input_pair)
        y_target = self.target
        log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target)

        loss = - (log_likelihood - kl_divergence)
        loss.backward()
        [optim.step() for optim in self.optimizers]

        return loss.item(), (-log_likelihood.item(), kl_divergence.item())
