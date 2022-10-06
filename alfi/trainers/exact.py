import numpy as np

from alfi.trainers import Trainer


class ExactTrainer(Trainer):
    def __init__(self, *args, loss_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.losses = np.empty((0, 1))

    def single_epoch(self, **kwargs):
        epoch_loss = 0

        [optim.zero_grad() for optim in self.optimizers]
        # Output from model
        output = self.model(self.model.train_t)
        # print(output.mean.shape)
        # plt.imshow(output.covariance_matrix.detach())
        # plt.colorbar()
        # Calc loss and backprop gradients
        loss = -self.loss_fn(output, self.model.train_y.squeeze())
        loss.backward()
        [optim.step() for optim in self.optimizers]
        epoch_loss += loss.item()

        return epoch_loss, [epoch_loss]

    def print_extra(self):
        self.model.covar_module.lengthscale.item(),
        self.model.likelihood.noise.item()
        super().print_extra()
