import torch
import numpy as np


from .trainer import Trainer


class ExactTrainer(Trainer):
    def __init__(self, *args, loss_fn):
        super().__init__(*args)
        self.loss_fn = loss_fn
        self.losses = np.empty((0, 1))

    def single_epoch(self, **kwargs):
        epoch_loss = 0

        self.optimizer.zero_grad()
        # Output from model
        output = self.lfm(self.lfm.train_t)
        # print(output.mean.shape)
        # plt.imshow(output.covariance_matrix.detach())
        # plt.colorbar()
        # Calc loss and backprop gradients
        loss = -self.loss_fn(output, self.lfm.train_y.squeeze())
        loss.backward()
        self.optimizer.step()
        epoch_loss += loss.item()

        return epoch_loss, [epoch_loss]

    def print_extra(self):
        print('')
        self.lfm.covar_module.lengthscale.item(),
        self.lfm.likelihood.noise.item()

    def after_epoch(self):
        with torch.no_grad():
            sens = self.lfm.sensitivity
            sens[3] = np.float64(1.)
            deca = self.lfm.decay_rate
            deca[3] = np.float64(0.8)
            self.lfm.sensitivity = sens
            self.lfm.decay_rate = deca
