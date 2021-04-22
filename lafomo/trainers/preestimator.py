from typing import List, Callable
from matplotlib import pyplot as plt
import torch

from .trainer import Trainer
from lafomo.models import VariationalLFM
try:
    from lafomo.models import PartialLFM
except:
    class PartialLFM:
        pass
from lafomo.utilities.torch import is_cuda, spline_interpolate_gradient


class PreEstimator(Trainer):
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
        self.model_kwargs = {}

    def single_epoch(self, epoch=0, **kwargs):
        assert self.lfm.pretrain_mode
        [optim.zero_grad() for optim in self.optimizers]
        # y = y.cuda() if is_cuda() else y

        output = self.lfm(self.input_pair, **self.model_kwargs)
        y_target = self.target
        log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target, mask=self.train_mask)

        loss = - (log_likelihood - kl_divergence)
        loss.backward()
        [optim.step() for optim in self.optimizers]
        self.debug_out(output, y_target)
        return loss.item(), (-log_likelihood.item(), kl_divergence.item())

    def debug_out(self, output, y_target):
        pass


class PartialPreEstimator(PreEstimator):
    def __init__(self,
                 lfm: PartialLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 pde_func: Callable,
                 input_pair,
                 target,
                 **kwargs):
        super(PreEstimator, self).__init__(lfm, optimizers, dataset, **kwargs)

        self.pde_func = pde_func
        self.input_pair = input_pair
        self.target = target
        self.model_kwargs = dict(pde_func=self.pde_func)

    def debug_out(self, output, y_target):
        # plt.figure()
        # plt.imshow(output.mean.detach().view(41, 41).t())
        # plt.colorbar()
        pass