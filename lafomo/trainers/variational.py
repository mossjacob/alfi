from typing import List

import torch

from lafomo.utilities.torch import is_cuda

from .trainer import Trainer
from lafomo.models import VariationalLFM


class VariationalTrainer(Trainer):
    """
    Parameters:
        batch_size: in the case of the transcriptional regulation model, we train the entire gene set as a batch
    """
    def __init__(self,
                 lfm: VariationalLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 warm_variational=-1,
                 **kwargs):
        super().__init__(lfm, optimizers, dataset, batch_size=lfm.num_outputs, **kwargs)
        self.warm_variational = warm_variational
        if warm_variational >= 0:
            for param in self.lfm.nonvariational_parameters():
                param.requires_grad = False

    def single_epoch(self, step_size=1e-1, epoch=0, pretrain_target=None, **kwargs):
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        for i, data in enumerate(self.data_loader):
            [optim.zero_grad() for optim in self.optimizers]
            data_input, y = data
            data_input = data_input.cuda() if is_cuda() else data_input
            y = y.cuda() if is_cuda() else y
            # Assume that the batch of t s are the same
            data_input, y = data_input[0], y
            if self.lfm.pretrain_mode:
                output = self.lfm((data_input, y))
                y_target = pretrain_target.t()
            else:
                output = self.lfm(data_input, step_size=step_size)
                y_target = y.t()

            self.debug_out(data_input, y, output)
            log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y_target)

            loss = - (log_likelihood - kl_divergence)

            loss.backward()
            if epoch >= self.warm_variational:
                [optim.step() for optim in self.optimizers]
            else:
                self.optimizers[0].step()

            epoch_loss += loss.item()
            epoch_ll += log_likelihood.item()
            epoch_kl += kl_divergence.item()
            # if (epoch % 10) == 0:
            #     print(dict(self.lfm.gp_model.named_variational_parameters()))
            # Now we are warmed up, start training non variational parameters in the next epoch.
            if epoch + 1 == self.warm_variational:
                for param in self.lfm.nonvariational_parameters():
                    param.requires_grad = True

        return epoch_loss, (-epoch_ll, epoch_kl)

    def debug_out(self, data_input, y_target, output):
        pass
