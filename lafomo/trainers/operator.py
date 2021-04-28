import torch

from typing import List
from torch.optim.lr_scheduler import StepLR
from torch.optim import Optimizer
from torch.nn.functional import mse_loss

from lafomo.nn import LpLoss
from lafomo.utilities.torch import is_cuda
from lafomo.models import NeuralOperator
from lafomo.trainers import Trainer


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

        self.loss_fn = LpLoss(size_average=False)

    def single_epoch(self, step_size=1e-1, epoch=0, **kwargs):
        self.lfm.train()
        train_mse = 0
        train_l2 = 0
        batch_size = self.train_loader.batch_size
        for x, y in self.train_loader:
            if is_cuda():
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            out = self.lfm(x)

            mse = mse_loss(out, y, reduction='mean')
            # mse.backward()
            l2 = self.loss_fn(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            self.optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        self.scheduler.step()
        self.lfm.eval()
        test_l2 = 0.0
        test_mse = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                # x, y = x.cuda(), y.cuda()

                out = self.lfm(x)
                test_mse += mse_loss(out, y, reduction='mean')
                test_l2 += self.loss_fn(
                    out.view(self.test_loader.batch_size, -1),
                    y.view(self.test_loader.batch_size, -1)).item()

        train_mse /= len(self.train_loader)
        train_l2 /= len(self.train_loader)
        test_l2 /= len(self.test_loader)
        test_mse /= len(self.test_loader)

        return train_l2, (train_mse, test_mse, test_l2)
