import torch

from reggae.utilities import inv_softplus, LFMDataset, is_cuda
from torch.utils.data.dataloader import DataLoader

import numpy as np
from matplotlib import pyplot as plt


class Trainer:
    """
    Trainer

    Parameters
    ----------
    model: .
    optimizer:
    dataset: Dataset where t_observed (T,), m_observed (J, T).
    inducing timepoints.
    give_output: whether the trainer should give the first output (y_0) as initial value to the model `forward()`
    """
    def __init__(self, model, optimizer: torch.optim.Optimizer, dataset: LFMDataset, batch_size=1, give_output=False):
        self.num_epochs = 0
        self.kl_mult = 0
        self.optimizer = optimizer
        self.model = model
        self.t_observed = dataset.data[0][0].view(-1)
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.losses = np.empty((0, 2))
        self.give_output = give_output

    def train(self, epochs=20, report_interval=1, plot_interval=20, rtol=1e-5, atol=1e-6, num_samples=5):
        losses = list()
        end_epoch = self.num_epochs+epochs
        plt.figure(figsize=(4, 2.3))
        for epoch in range(epochs):
            for i, data in enumerate(self.data_loader):
                t, y = data
                t = t.cuda() if is_cuda() else t
                y = y.cuda() if is_cuda() else y
                # Assume that the batch of t s are the same
                t, y = t[0].view(-1), y
                self.optimizer.zero_grad()

                # with ef.scan():
                initial_value = torch.zeros((self.batch_size, 1), dtype=torch.float64)
                initial_value = initial_value.cuda() if is_cuda() else initial_value
                if self.give_output:
                    initial_value = y[0]
                output = self.model(t, initial_value, rtol=rtol, atol=atol, num_samples=num_samples)
                output = torch.squeeze(output)
                # Calc loss and backprop gradients
                mult = 1
                if self.num_epochs <= 10:
                    mult = self.num_epochs/10

                ll, kl = self.model.elbo(y, output, mult)
                total_loss = -ll + kl
                total_loss.backward()
                self.optimizer.step()

                if (epoch % report_interval) == 0:
                    print('Epoch %d/%d - Loss: %.2f (%.2f %.2f) λ: %.3f' % (
                        self.num_epochs + 1, end_epoch,
                        total_loss.item(),
                        -ll.item(), kl.item(),
                        self.model.lengthscale[0].item(),
                    ), end='')
                    self.print_extra()

            losses.append((ll.item(), kl.item()))
            self.after_epoch()

            if (epoch % plot_interval) == 0:
                plt.plot(self.t_observed, output[0].cpu().detach().numpy(), label='epoch'+str(epoch))
            self.num_epochs += 1
        plt.legend()

        losses = np.array(losses)
        self.losses = np.concatenate([self.losses, losses], axis=0)


        return output

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
    def __init__(self, model, optimizer: torch.optim.Optimizer, dataset: LFMDataset, batch_size=5):
        super(TranscriptionalTrainer, self).__init__(model, optimizer, dataset, batch_size=batch_size)
        self.basalrates = list()
        self.decayrates = list()
        self.lengthscales = list()
        self.sensitivities = list()
        self.mus = list()
        self.cholS = list()

    def print_extra(self):
        print(' b: %.2f d %.2f s: %.2f' % (
            self.model.basal_rate[0].item(),
            self.model.decay_rate[0].item(),
            self.model.sensitivity[0].item()
        ))

    def after_epoch(self):
        self.basalrates.append(self.model.basal_rate.detach().clone().numpy())
        self.decayrates.append(self.model.decay_rate.detach().clone().numpy())
        self.sensitivities.append(self.model.sensitivity.detach().clone().numpy())
        self.lengthscales.append(self.model.lengthscale.detach().clone().numpy())
        self.cholS.append(self.model.q_cholS.detach().clone())
        self.mus.append(self.model.q_m.detach().clone())
        with torch.no_grad():
            # self.model.raw_lengthscale.clamp_(-2.5, inv_softplus(torch.tensor(1., dtype=torch.float64))) # TODO is this needed?
            # TODO can we replace these with parameter transforms like we did with lengthscale
            self.model.sensitivity.clamp_(0.4, 8)
            self.model.basal_rate.clamp_(0, 8)
            self.model.decay_rate.clamp_(0, 8)
            self.model.sensitivity[3] = np.float64(1.)
            self.model.decay_rate[3] = np.float64(0.8)
            self.model.q_m[0, 0] = 0.

