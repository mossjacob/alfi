import torch

from typing import List

from torch.distributions import Normal
from torch.optim.lr_scheduler import StepLR
from torch.optim import Optimizer
from torch.nn.functional import mse_loss, softplus

from alfi.nn import LpLoss
from alfi.utilities.torch import is_cuda
from alfi.models import NeuralOperator
from alfi.trainers import Trainer


class NeuralOperatorTrainer(Trainer):
    """
    Parameters:
        batch_size: in the case of the transcriptional regulation model, we train the entire gene set as a batch
    """
    def __init__(self,
                 lfm: NeuralOperator,
                 optimizers: List[Optimizer],
                 train_loader,
                 test_loader,
                 warm_variational=-1,
                 **kwargs):
        super().__init__(lfm, optimizers, train_loader.dataset, **kwargs)
        self.warm_variational = warm_variational
        self.optimizer = optimizers[0]
        self.train_loader = train_loader
        self.test_loader = test_loader
        gamma = 0.5
        step_size = 100

        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        self.loss_fn = LpLoss(size_average=True)

    def to_normal(self, p_y_pred):
        mu = p_y_pred[..., 0]
        sigma = softplus(p_y_pred[..., 1]) + 1e-6
        return Normal(mu, sigma)

    def single_epoch(self, step_size=1e-1, epoch=0, **kwargs):
        self.model.train()
        train_mse = 0
        train_l2 = 0
        train_loss = 0
        train_params_mse = 0
        batch_size = self.train_loader.batch_size
        for x, y, params in self.train_loader:
            if is_cuda():
                x, y, params = x.cuda(), y.cuda(), params.cuda()

            self.optimizer.zero_grad()
            additional_loss = 0

            p_y_pred = self.model(x)
            if self.model.params:
                p_y_pred, params_out = p_y_pred
                params_mse = mse_loss(params_out, params.view(batch_size, -1), reduction='mean')
                train_params_mse += params_mse.item()
                additional_loss = 10 * params_mse

            p_y_pred = self.to_normal(p_y_pred)

            mse = mse_loss(p_y_pred.mean, y.squeeze(-1), reduction='mean')
            l2 = self._loss(p_y_pred, y.squeeze(-1))
            total_loss = l2 + additional_loss

            total_loss.backward()  # use the l2 relative loss
            self.optimizer.step()

            train_mse += mse.item()
            train_l2 += l2.item()
            train_loss += total_loss.item()

        self.scheduler.step()
        self.model.eval()

        test_l2 = 0.0
        test_mse = 0.0
        test_params_mse = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for x, y, params in self.test_loader:
                # x, y = x.cuda(), y.cuda()
                if is_cuda():
                    x, y, params = x.cuda(), y.cuda(), params.cuda()
                additional_loss = 0

                p_y_pred = self.model(x)
                if self.model.params:
                    p_y_pred, params_out = p_y_pred
                    params_mse = mse_loss(params_out, params.view(self.test_loader.batch_size, -1), reduction='mean')
                    test_params_mse += params_mse.item()
                    additional_loss = 10 * params_mse

                p_y_pred = self.to_normal(p_y_pred)

                mse = mse_loss(p_y_pred.mean, y.squeeze(-1), reduction='mean')
                l2 = self._loss(p_y_pred, y.squeeze(-1))
                total_loss = l2 + additional_loss

                test_mse += mse.item()
                test_l2 += l2.item()
                test_loss += total_loss.item()

        train_mse /= len(self.train_loader)
        train_l2 /= len(self.train_loader)
        train_params_mse /= len(self.train_loader)
        train_loss /= len(self.train_loader)
        test_l2 /= len(self.test_loader)
        test_mse /= len(self.test_loader)
        test_params_mse /= len(self.test_loader)
        test_loss /= len(self.test_loader)

        return train_loss, (test_loss, train_l2, test_l2, train_params_mse, test_params_mse)

    def _loss(self, p_y_pred, y_target):
        """
        Computes Neural Operator loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        """
        nll = -p_y_pred.log_prob(y_target).mean()
        return nll
