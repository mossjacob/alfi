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
    de_model: .
    optimizer:
    dataset: Dataset where t_observed (T,), m_observed (J, T).
    inducing timepoints.
    give_output: whether the trainer should give the first output (y_0) as initial value to the model `forward()`
    """
    def __init__(self,
                 lfm: LFM,
                 optimizer: torch.optim.Optimizer,
                 dataset: LFMDataset, batch_size=1, give_output=False):
        self.lfm = lfm
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
        return initial_value.repeat(self.lfm.config.num_samples, 1, 1)  # Add batch dimension for sampling

    def train(self, epochs=20, report_interval=1, rtol=1e-5, atol=1e-6):
        self.lfm.train()

        losses = list()
        end_epoch = self.num_epochs+epochs

        for epoch in range(epochs):
            epoch_loss, split_loss = self.single_epoch(rtol, atol)

            if (epoch % report_interval) == 0:
                print('Epoch %03d/%03d - Loss: %.2f (' % (
                    self.num_epochs + 1, end_epoch, epoch_loss), end='')
                for loss in split_loss:
                    print('%.2f  ' % loss, end='')

                print(f') λ: {self.lfm.gp_model.covar_module.lengthscale.item()}', end='')
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
            y_mean, y_var = self.lfm(t, initial_value, rtol=rtol, atol=atol)
            y_mean = y_mean.squeeze()
            y_var = y_var.squeeze()
            # Calc loss and backprop gradients
            mult = 1
            if self.num_epochs <= 10:
                mult = self.num_epochs/10

            ll, kl = self.lfm.elbo(y, y_mean, y_var, kl_mult=mult)
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


class ExactTrainer(Trainer):
    def single_epoch(self, **kwargs):
        epoch_loss = 0

        self.optimizer.zero_grad()
        # Output from model
        output = self.model(self.model.train_t)
        # print(output.mean.shape)
        # plt.imshow(output.covariance_matrix.detach())
        # plt.colorbar()
        # Calc loss and backprop gradients
        loss = -self.mll(output, self.model.train_y.squeeze())
        loss.backward()
        self.optimizer.step()
        epoch_loss += loss.item()

        return epoch_loss, None

    def print_extra(self):
        print('')
        self.model.covar_module.lengthscale.item(),
        self.model.likelihood.noise.item()

    def after_epoch(self):
        with torch.no_grad():
            sens = self.model.sensitivity
            sens[3] = np.float64(1.)
            deca = self.model.decay_rate
            deca[3] = np.float64(0.8)
            self.model.sensitivity = sens
            self.model.decay_rate = deca


class VariationalTrainer(Trainer):
    """
    Parameters:
        batch_size: in the case of the transcriptional regulation model, we train the entire gene set as a batch
    """
    def __init__(self, lfm, optimizer: torch.optim.Optimizer, dataset):
        super().__init__(lfm, optimizer, dataset, batch_size=lfm.num_outputs)

    def single_epoch(self, rtol, atol):
        data = next(iter(self.data_loader))

        self.optimizer.zero_grad()
        t, y = data
        t = t.cuda() if is_cuda() else t
        y = y.cuda() if is_cuda() else y
        # Assume that the batch of t s are the same
        t, y = t[0].view(-1), y

        output = self.lfm(t, step_size=1e-1)

        # print('gout', g_output.event_shape, g_output.batch_shape)
        #  log_likelihood - kl_divergence + log_prior - added_loss
        # print(y.shape)
        log_likelihood, kl_divergence, _ = self.lfm.loss_fn(output, y.permute(1, 0))

        loss = - (log_likelihood - kl_divergence)

        loss.backward()
        self.optimizer.step()

        return loss, (-log_likelihood, kl_divergence)


class TranscriptionalTrainer(VariationalTrainer):
    """
    TranscriptionalTrainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basalrates = list()
        self.decayrates = list()
        self.lengthscales = list()
        self.sensitivities = list()
        self.mus = list()
        self.cholS = list()

    def after_epoch(self):
        self.basalrates.append(self.lfm.basal_rate.detach().clone().numpy())
        self.decayrates.append(self.lfm.decay_rate.detach().clone().numpy())
        self.sensitivities.append(self.lfm.sensitivity.detach().clone().numpy())
        self.lengthscales.append(self.lfm.kernel.lengthscale.detach().clone().numpy())
        self.cholS.append(self.lfm.q_cholS.detach().clone())
        self.mus.append(self.lfm.q_m.detach().clone())
        with torch.no_grad():
            # TODO can we replace these with parameter transforms like we did with lengthscale
            self.lfm.sensitivity.clamp_(0, 20)
            self.lfm.basal_rate.clamp_(0, 20)
            self.lfm.decay_rate.clamp_(0, 20)
            self.extra_constraints()
            # self.model.inducing_inputs.clamp_(0, 1)
            # self.lfm.q_m[0, 0] = 0.

    def extra_constraints(self):
        pass


class P53ConstrainedTrainer(TranscriptionalTrainer):
    def extra_constraints(self):
        self.lfm.sensitivity[3] = np.float64(1.)
        self.lfm.decay_rate[3] = np.float64(0.8)
