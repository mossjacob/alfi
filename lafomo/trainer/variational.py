import torch
import numpy as np
import gpytorch
from torch.utils.data.dataloader import DataLoader

from lafomo.utilities.torch import is_cuda
from lafomo.datasets import LFMDataset
from lafomo.models import LFM

from .trainer import Trainer


class VariationalTrainer(Trainer):
    """
    Parameters:
        batch_size: in the case of the transcriptional regulation model, we train the entire gene set as a batch
    """
    def __init__(self, lfm, optimizer: torch.optim.Optimizer, dataset):
        super().__init__(lfm, optimizer, dataset, batch_size=lfm.num_outputs)

    def single_epoch(self, step_size=1e-1):
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        for i, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            data_input, y = data
            data_input = data_input.cuda() if is_cuda() else data_input
            y = y.cuda() if is_cuda() else y
            # Assume that the batch of t s are the same
            data_input, y = data_input[0], y

            output = self.lfm(data_input, step_size=step_size)
            self.debug_out(data_input, y, output)

            log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y.permute(1, 0))

            loss = - (log_likelihood - kl_divergence)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_ll += log_likelihood.item()
            epoch_kl += kl_divergence.item()

        return epoch_loss, (-epoch_ll, epoch_kl)

    def debug_out(self, data_input, y_target, output):
        pass
