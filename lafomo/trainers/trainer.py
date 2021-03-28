from lafomo.models import LFM
import torch
import numpy as np
import gpytorch
from torch.utils.data.dataloader import DataLoader

from lafomo.utilities.torch import is_cuda
from lafomo.datasets import LFMDataset


class Trainer:
    """
    Trainer

    Parameters
    ----------
    lfm: .
    optimizer:
    dataset: Dataset where t_observed (D, T), m_observed (J, T).
    inducing timepoints.
    give_output: whether the trainers should give the first output (y_0) as initial value to the model `forward()`
    """
    def __init__(self,
                 lfm: LFM,
                 optimizer: torch.optim.Optimizer,
                 dataset: LFMDataset, batch_size=1, give_output=False, track_parameters=None):
        self.lfm = lfm
        self.num_epochs = 0
        self.kl_mult = 0
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.losses = np.empty((0, 2))
        self.give_output = give_output
        self.parameter_trace = None
        if track_parameters is not None:
            named_params = dict(lfm.named_parameters())
            self.parameter_trace = {key: [named_params[key].detach()] for key in track_parameters}

    def initial_value(self, y):
        initial_value = torch.zeros((self.batch_size, 1), dtype=torch.float64)
        initial_value = initial_value.cuda() if is_cuda() else initial_value
        if self.give_output:
            initial_value = y[0]
        return initial_value.repeat(self.lfm.config.num_samples, 1, 1)  # Add batch dimension for sampling

    def train(self, epochs=20, report_interval=1, **kwargs):
        self.lfm.train()

        losses = list()
        end_epoch = self.num_epochs+epochs

        for epoch in range(epochs):
            epoch_loss, split_loss = self.single_epoch(**kwargs)

            if (epoch % report_interval) == 0:
                print('Epoch %03d/%03d - Loss: %.2f (' % (
                    self.num_epochs + 1, end_epoch, epoch_loss), end='')
                print(' '.join(map(lambda l: '%.2f' % l, split_loss)), end='')

                if isinstance(self.lfm, gpytorch.models.GP):
                    kernel = self.lfm.covar_module
                else:
                    kernel = self.lfm.gp_model.covar_module
                print(f') λ: {str(kernel.lengthscale.view(-1).detach().numpy())}', end='')
                self.print_extra()

            losses.append(split_loss)

            self.after_epoch()
            self.num_epochs += 1

        losses = np.array(losses)
        self.losses = np.concatenate([self.losses, losses], axis=0)

    def single_epoch(self, **kwargs):
        raise NotImplementedError

    def print_extra(self):
        print('')

    def after_epoch(self):
        if self.parameter_trace is not None:
            params = dict(self.lfm.named_parameters())
            for key in params:
                if key in self.parameter_trace:
                    self.parameter_trace[key].append(params[key].detach())
