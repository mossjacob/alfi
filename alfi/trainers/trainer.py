import time
from abc import abstractmethod
from typing import List

from alfi.models import LFM
import torch
import numpy as np
import gpytorch
from torch.utils.data.dataloader import DataLoader

from alfi.utilities.torch import is_cuda
from alfi.datasets import LFMDataset


class Trainer:
    """
    An abstract LFM trainer. Subclasses must implement the `single_epoch` function.

    Parameters
    ----------
    lfm: The Latent Force Model.
    optimizers: list of `torch.optim.Optimizer`s. For when natural gradients are used for variational models.
    dataset: Dataset where t_observed (D, T), m_observed (J, T).
    give_output: whether the trainers should give the first output (y_0) as initial value to the model `forward()`
    track_parameters: the keys into `named_parameters()` of parameters that the trainer should track. The
                      tracked parameters can be accessed from `parameter_trace`
    train_mask: boolean mask
    """
    def __init__(self,
                 lfm: LFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset: LFMDataset,
                 batch_size=1,
                 give_output=False,
                 track_parameters=None,
                 train_mask=None,
                 checkpoint_dir=None):
        self.lfm = lfm
        self.num_epochs = 0
        self.optimizers = optimizers
        self.use_natural_gradient = len(self.optimizers) > 1
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.losses = None
        self.give_output = give_output
        self.train_mask = train_mask
        self.checkpoint_dir = checkpoint_dir
        self.parameter_trace = None
        if track_parameters is not None:
            named_params = dict(lfm.named_parameters())
            self.parameter_trace = {key: [named_params[key].detach()] for key in track_parameters}

    def train(self, epochs=20, report_interval=1, **kwargs):
        self.lfm.train()

        losses = list()
        times = list()
        end_epoch = self.num_epochs+epochs

        for epoch in range(epochs):
            epoch_loss, split_loss = self.single_epoch(epoch=self.num_epochs, **kwargs)
            t = time.time()
            times.append((t, epoch_loss))
            if (epoch % report_interval) == 0:
                print('Epoch %03d/%03d - Loss: %.2f (' % (
                    self.num_epochs + 1, end_epoch, epoch_loss), end='')
                print(' '.join(map(lambda l: '%.2f' % l, split_loss)), end='')

                if isinstance(self.lfm, gpytorch.models.GP):
                    kernel = self.lfm.covar_module
                    print(f') λ: {str(kernel.lengthscale.view(-1).detach().numpy())}', end='')
                elif hasattr(self.lfm, 'gp_model'):
                    print(f') kernel: {self.lfm.summarise_gp_hyp()}', end='')
                else:
                    print(')', end='')
                self.print_extra()
                if self.checkpoint_dir is not None:
                    self.lfm.save(self.checkpoint_dir / f'epoch{epoch}')
            losses.append(split_loss)

            self.after_epoch()
            self.num_epochs += 1

        losses = torch.tensor(losses).cpu().numpy()
        if self.losses is None:
            self.losses = np.empty((0, losses.shape[1]))
        self.losses = np.concatenate([self.losses, losses], axis=0)
        return times

    @abstractmethod
    def single_epoch(self, epoch=0, **kwargs):
        raise NotImplementedError

    def set_optimizers(self, optimizers):
        self.optimizers = optimizers

    def print_extra(self):
        print('')

    def after_epoch(self):
        if self.parameter_trace is not None:
            params = dict(self.lfm.named_parameters())
            for key in params:
                if key in self.parameter_trace:
                    self.parameter_trace[key].append(params[key].detach().clone())
