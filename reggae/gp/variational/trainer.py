import torch

from reggae.utilities import inv_softplus

import numpy as np
from matplotlib import pyplot as plt


class Trainer:
    """
    Trainer

    Parameters
    ----------
    model: .
    optimizer:
    dataset: (t_observed, m_observed) where t_observed (T,), m_observed (J, T).
    inducing timepoints.
    """
    def __init__(self, model, optimizer: torch.optim.Optimizer, dataset: tuple):
        self.num_epochs = 0
        self.kl_mult = 0
        self.optimizer = optimizer
        self.model = model
        self.t_observed, self.m_observed = dataset
        self.num_genes = self.m_observed.shape[0]

        self.losses = np.empty((0, 2))
        self.basalrates = list()
        self.decayrates = list()
        self.lengthscales = list()
        self.mus = list()
        self.cholS = list()


    def train(self, epochs=20, report_interval=1, plot_interval=20, rtol=1e-5, atol=1e-6, num_samples=5):
        losses = list()
        end_epoch = self.num_epochs+epochs
        plt.figure(figsize=(4, 2.3))
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # with ef.scan():
            initial_value = torch.zeros((self.num_genes, 1), dtype=torch.float64) #, dtype=torch.float64
            output, kl = self.model(self.t_observed, initial_value, rtol=rtol, atol=atol, num_samples=num_samples)
            output = torch.squeeze(output)

            # print(model.q_cholS)
            # Calc loss and backprop gradients
            loss = -1*torch.sum(self.model.log_likelihood(self.m_observed, output))
            mult = 1
            if self.num_epochs <= 10:
                mult = self.num_epochs/10
            kl *= mult
            total_loss = loss + kl
            total_loss.backward()
            self.optimizer.step()

            if (epoch % report_interval) == 0:
                print('Epoch %d/%d - Loss: %.2f (%.2f %.2f) b: %.2f d %.2f s: %.2f λ: %.3f' % (
                    self.num_epochs + 1, end_epoch,
                    total_loss.item(),
                    loss.item(), kl.item(),
                    self.model.basal_rate[0].item(),
                    self.model.decay_rate[0].item(),
                    self.model.sensitivity[0].item(),
                    self.model.lengthscale.squeeze().item()
                ))

            self.basalrates.append(self.model.basal_rate.detach().numpy())
            self.decayrates.append(self.model.decay_rate.detach().numpy())
            self.lengthscales.append(self.model.lengthscale.squeeze().item())
            self.cholS.append(self.model.q_cholS.detach().clone())
            self.mus.append(self.model.q_m.detach().clone())
            losses.append((loss.item(), kl.item()))
            with torch.no_grad():
                self.model.raw_lengthscale.clamp_(-2, inv_softplus(torch.tensor(1., dtype=torch.float64))) # TODO is this needed?
                # TODO can we replace these with parameter transforms like we did with lengthscale
                self.model.sensitivity.clamp_(0.4, 8)
                self.model.basal_rate.clamp_(0, 8)
                self.model.decay_rate.clamp_(0, 8)
                self.model.sensitivity[3] = np.float64(1.)
                self.model.decay_rate[3] = np.float64(0.8)

            if (epoch % plot_interval) == 0:
                plt.plot(self.t_observed, output[0].detach().numpy(), label='epoch'+str(epoch))
            self.num_epochs += 1
        plt.legend()

        losses = np.array(losses)
        self.losses = np.concatenate([self.losses, losses], axis=0)

        return output
