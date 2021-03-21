from lafomo.variational.models import VariationalLFM
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
    de_model: .
    optimizer:
    dataset: Dataset where t_observed (T,), m_observed (J, T).
    inducing timepoints.
    give_output: whether the trainer should give the first output (y_0) as initial value to the model `forward()`
    """
    def __init__(self,
                 gp_model: gpytorch.models.GP,
                 de_model: VariationalLFM,
                 optimizer: torch.optim.Optimizer,
                 dataset: LFMDataset, batch_size=1, give_output=False):
        self.gp_model = gp_model
        self.de_model = de_model
        self.num_epochs = 0
        self.kl_mult = 0
        self.optimizer = optimizer
        self.t_observed = dataset.data[0][0].view(-1)
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.losses = np.empty((0, 2))
        self.give_output = give_output

    def initial_value(self, y):
        initial_value = torch.zeros((self.batch_size, 1), dtype=torch.float64)
        initial_value = initial_value.cuda() if is_cuda() else initial_value
        if self.give_output:
            initial_value = y[0]
        return initial_value.repeat(self.de_model.config.num_samples, 1, 1)  # Add batch dimension for sampling

    def train(self, epochs=20, report_interval=1, rtol=1e-5, atol=1e-6):
        losses = list()
        end_epoch = self.num_epochs+epochs

        for epoch in range(epochs):
            epoch_loss, split_loss = self.single_epoch(rtol, atol)

            if (epoch % report_interval) == 0:
                print('Epoch %03d/%03d - Loss: %.2f (' % (
                    self.num_epochs + 1, end_epoch, epoch_loss), end='')
                for loss in split_loss:
                    print('%.2f  ' % loss, end='')

                print(f') λ: {self.gp_model.covar_module.lengthscale.item()}', end='')
                self.print_extra()

            losses.append(split_loss)

            self.after_epoch()
            self.num_epochs += 1

        losses = np.array(losses)
        self.losses = np.concatenate([self.losses, losses], axis=0)

    def single_epoch(self, rtol, atol):
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        for i, data in enumerate(self.data_loader):

            self.optimizer.zero_grad()
            t, y = data
            t = t.cuda() if is_cuda() else t
            y = y.cuda() if is_cuda() else y
            # Assume that the batch of t s are the same
            t, y = t[0].view(-1), y

            # with ef.scan():
            initial_value = self.initial_value(y)
            y_mean, y_var = self.de_model(t, initial_value, rtol=rtol, atol=atol)
            y_mean = y_mean.squeeze()
            y_var = y_var.squeeze()
            # Calc loss and backprop gradients
            mult = 1
            if self.num_epochs <= 10:
                mult = self.num_epochs/10

            ll, kl = self.de_model.elbo(y, y_mean, y_var, kl_mult=mult)
            total_loss = -ll + kl

            total_loss.backward()
            self.optimizer.step()
            epoch_loss += total_loss.item()
            epoch_ll += ll.item()
            epoch_kl += kl.item()

        return epoch_loss, (-epoch_ll, epoch_kl)

    def print_extra(self):
        print('')

    def after_epoch(self):
        pass


class TranscriptionalTrainer(Trainer):
    """
    TranscriptionalTrainer
    Parameters:
        batch_size: in the case of the transcriptional regulation model, we train the entire gene set as a batch
    """
    def __init__(self, de_model: VariationalLFM, optimizer: torch.optim.Optimizer, dataset: LFMDataset, batch_size=None):
        if batch_size is None:
            batch_size = de_model.num_outputs
        super(TranscriptionalTrainer, self).__init__(de_model, optimizer, dataset, batch_size=batch_size)
        self.basalrates = list()
        self.decayrates = list()
        self.lengthscales = list()
        self.sensitivities = list()
        self.mus = list()
        self.cholS = list()

    def print_extra(self):
        print(' b: %.2f d %.2f s: %.2f' % (
            self.de_model.basal_rate[0].item(),
            self.de_model.decay_rate[0].item(),
            self.de_model.sensitivity[0].item()
        ))

    def after_epoch(self):
        self.basalrates.append(self.de_model.basal_rate.detach().clone().numpy())
        self.decayrates.append(self.de_model.decay_rate.detach().clone().numpy())
        self.sensitivities.append(self.de_model.sensitivity.detach().clone().numpy())
        self.lengthscales.append(self.de_model.kernel.lengthscale.detach().clone().numpy())
        self.cholS.append(self.de_model.q_cholS.detach().clone())
        self.mus.append(self.de_model.q_m.detach().clone())
        with torch.no_grad():
            # TODO can we replace these with parameter transforms like we did with lengthscale
            self.de_model.sensitivity.clamp_(0, 20)
            self.de_model.basal_rate.clamp_(0, 20)
            self.de_model.decay_rate.clamp_(0, 20)
            self.extra_constraints()
            # self.model.inducing_inputs.clamp_(0, 1)
            self.de_model.q_m[0, 0] = 0.

    def extra_constraints(self):
        pass


class P53ConstrainedTrainer(TranscriptionalTrainer):
    def extra_constraints(self):
        self.de_model.sensitivity[3] = np.float64(1.)
        self.de_model.decay_rate[3] = np.float64(0.8)
