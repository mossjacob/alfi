import gpytorch
import torch
import numpy as np

from reggae.gp.exact import AnalyticalLFM


class Trainer:
    def __init__(self, model: AnalyticalLFM, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def train(self, epochs=100, report_interval=10):
        self.model.train()
        self.model.likelihood.train()

        for epoch in range(epochs):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(self.model.train_t)
            # print(output.mean.shape)
            # plt.imshow(output.covariance_matrix.detach())
            # plt.colorbar()
            # Calc loss and backprop gradients
            loss = -self.mll(output, self.model.train_y.squeeze())
            loss.backward()
            if (epoch % report_interval) == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    epoch + 1, epochs, loss.item(),
                    self.model.covar_module.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
            self.optimizer.step()
            with torch.no_grad():
                sens = self.model.sensitivity
                sens[3] = np.float64(1.)
                deca = self.model.decay_rate
                deca[3] = np.float64(0.8)
                self.model.sensitivity = sens
                self.model.decay_rate = deca

        # Get into evaluation (predictive posterior) mode and predict
        self.model.eval()
        self.model.likelihood.eval()
