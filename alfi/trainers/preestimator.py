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
import numpy as np

class PreEstimator(Trainer):
    def __init__(self,
                 model: VariationalLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 **kwargs):
        super().__init__(model, optimizers, dataset, **kwargs)
        data = next(iter(self.data_loader))
        t, y = data[0][0], data[1]

        t_interpolate, self.y_interpolate, self.target = self.interpolate(t, y)
        self.input_pair = (t_interpolate, self.y_interpolate)
        self.model_kwargs = {}

    def interpolate(self, time, y, num_intermediate=9):
        """
        :param time: pseudotime
        :param y: output
        :param num_intermediate: this determines how granular the interpolation is
        :return:
        """
        print('interpolating', y.shape)
        # y[y <= 0] = np.nan
        if y.ndim > 2:
            ts = list()
            ys = list()
            t_interp = torch.linspace(time.min(), time.max(), 200)
            y_interpolate = np.zeros((y.shape[0], y.shape[1], 200))
            y_grad = torch.zeros((y.shape[0], y.shape[1], 200))
            t_interpolate = t_interp#.unsqueeze(0).repeat((y.shape[0], 1))
            print(t_interpolate.shape)
            time_uniq, ind = np.unique(time, return_index=True)
            print(time)
            for i in range(y.shape[0]):
                _t_interpolate, _y_interpolate, _y_grad, _ = spline_interpolate_gradient(
                    time[ind], y[i:i+1, ..., ind].transpose(-1, -2), num_intermediate, x_interpolate=t_interp)
                y_interpolate[i, :] = _y_interpolate.transpose(-1, -2)
                y_grad[i, :] = _y_grad.transpose(-1, -2)

        else:
            t_interpolate, y_interpolate, y_grad, _ = spline_interpolate_gradient(
                time, y.transpose(-1, -2), num_intermediate)

            y_interpolate = y_interpolate.transpose(-1, -2)
        return t_interpolate, y_interpolate, y_grad

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