from typing import List, Callable
import torch

from alfi.trainers import Trainer
from alfi.models import VariationalLFM, TrainMode
try:
    from alfi.models import PartialLFM
except:
    class PartialLFM:
        pass
from alfi.utilities.torch import spline_interpolate_gradient


class PreEstimator(Trainer):
    def __init__(self,
                 model: VariationalLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 **kwargs):
        super().__init__(model, optimizers, dataset, **kwargs)
        num_intermediate = 9  # this determines how granular the interpolation is
        data = next(iter(self.data_loader))
        t, y = data[0][0], data[1]

        t_interpolate, y_interpolate, y_grad, _ = spline_interpolate_gradient(
            t, data[1].permute(1, 0), num_intermediate)

        self.y_interpolate = y_interpolate.t()
        self.input_pair = (t_interpolate, self.y_interpolate)
        self.target = y_grad
        self.model_kwargs = {}

    def single_epoch(self, epoch=0, **kwargs):
        assert self.model.train_mode == TrainMode.GRADIENT_MATCH
        [optim.zero_grad() for optim in self.optimizers]
        # y = y.cuda() if is_cuda() else y

        output = self.model(self.input_pair, **self.model_kwargs)
        y_target = self.target
        log_likelihood, kl_divergence, _ = self.model.loss_fn(output, y_target, mask=self.train_mask)

        loss = - (log_likelihood - kl_divergence)
        loss.backward()
        [optim.step() for optim in self.optimizers]
        self.debug_out(output, y_target)
        return loss.item(), (-log_likelihood.item(), kl_divergence.item())

    def debug_out(self, output, y_target):
        pass


class PartialPreEstimator(PreEstimator):
    def __init__(self,
                 model: PartialLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 pde_func: Callable,
                 input_pair,
                 target,
                 **kwargs):
        super(PreEstimator, self).__init__(model, optimizers, dataset, **kwargs)

        self.pde_func = pde_func
        self.input_pair = input_pair
        self.target = target
        disc = dataset.disc if hasattr(dataset, 'disc') else 1

        self.model_kwargs = dict(step=disc, pde_func=self.pde_func)

    def debug_out(self, output, y_target):
        # plt.figure()
        # plt.imshow(output.mean.detach().view(41, 41).t())
        # plt.colorbar()
        pass