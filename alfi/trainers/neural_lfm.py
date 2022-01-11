import torch

from typing import List
from torch.optim.lr_scheduler import StepLR
from torch.optim import Optimizer
from torch.nn.functional import mse_loss
from torch.distributions.kl import kl_divergence

from alfi.nn import LpLoss
from alfi.utilities.torch import is_cuda
from alfi.utilities.data import context_target_split as cts
from alfi.models import NeuralOperator
from torte import Trainer


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

    def single_epoch(self, step_size=1e-1, epoch=0, num_context=7, num_target=4, **kwargs):
        self.lfm.train()
        train_mse = 0
        train_l2 = 0
        train_loss = 0
        train_params_mse = 0
        batch_size = self.train_loader.batch_size
        for x, y, params in self.train_loader:
            if is_cuda():
                x, y, params = x.cuda(), y.cuda(), params.cuda()
            x_context, y_context, x_target, y_target = cts(x, y, num_context, num_target)

            x_context = x_context
            y_context = y_context
            x_target = x_target
            y_target = y_target
            self.optimizer.zero_grad()
            # out, params_out = self.lfm(x_context, y_context, x_target, y_target)
            p_y_pred, y_params, q_target, q_context = self.lfm(x_context, y_context, x_target, y_target)

            loss = self._loss(p_y_pred, y_target.squeeze(-1), q_target, q_context)
            mse = mse_loss(p_y_pred.mean, y.squeeze(-1), reduction='mean')
            # l2 = self.loss_fn(out, y.view(batch_size, -1))
            params_mse = mse_loss(y_params, params.view(batch_size, -1), reduction='mean')
            # total_loss = l2 + params_mse
            total_loss = loss + params_mse
            total_loss.backward()  # use the l2 relative loss
            self.optimizer.step()

            train_mse += mse.item()
            # train_l2 += l2.item()
            train_loss += total_loss.item()
            train_params_mse += params_mse.item()

        self.scheduler.step()
        self.lfm.eval()

        test_l2 = 0.0
        test_mse = 0.0
        test_params_mse = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for x, y, params in self.test_loader:
                # x, y = x.cuda(), y.cuda()
                x_context, y_context, x_target, y_target = cts(x, y, num_context, num_target)
                x_context = x_context
                y_context = y_context
                x_target = x_target
                y_target = y_target
                p_y_pred, y_params = self.lfm(x_context, y_context, x_target, y_target)
                loss = -p_y_pred.log_prob(y_target.squeeze(-1)).mean(dim=0).sum()
                test_mse += mse_loss(p_y_pred.mean, y.squeeze(-1), reduction='mean')
                # test_mse += mse_loss(out[..., 0:1], y, reduction='mean')
                # test_l2 += self.loss_fn(out, y.view(self.test_loader.batch_size, -1)).item()
                params_mse = mse_loss(y_params, params.view(self.test_loader.batch_size, -1), reduction='mean')
                test_params_mse += params_mse
                test_loss += loss + params_mse

        train_mse /= len(self.train_loader)
        train_l2 /= len(self.train_loader)
        train_params_mse /= len(self.train_loader)
        train_loss /= len(self.train_loader)
        test_l2 /= len(self.test_loader)
        test_mse /= len(self.test_loader)
        test_params_mse /= len(self.test_loader)
        test_loss /= len(self.test_loader)

        return train_loss, (test_loss, train_l2, test_l2, train_params_mse, test_params_mse)

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        nll = -p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return nll + kl
