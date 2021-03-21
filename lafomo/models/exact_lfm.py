import torch
import gpytorch

from torch.distributions import MultivariateNormal

from .lfm import LFM
from lafomo.kernels import SIMKernel
from lafomo.means import SIMMean
from lafomo.utilities.data import flatten_dataset, LFMDataset


class ExactLFM(LFM, gpytorch.models.ExactGP):
    def __init__(self, dataset: LFMDataset, variance):
        train_t, train_y = flatten_dataset(dataset)
        super().__init__(train_t, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        # self.gp_model = self
        self.num_outputs = dataset.num_outputs
        self.block_size = int(train_t.shape[0] / self.num_outputs)
        self.train_t = train_t.view(-1, 1)
        self.train_y = train_y.view(-1, 1)
        self.covar_module = SIMKernel(self.num_outputs, torch.tensor(variance, requires_grad=False))
        initial_basal = torch.mean(train_y.view(5, 7), dim=1) * self.covar_module.decay
        self.mean_module = SIMMean(self.covar_module, self.num_outputs, initial_basal)

    @property
    def basal_rate(self):
        return self.mean_module.basal

    @property
    def sensitivity(self):
        return self.covar_module.sensitivity

    @sensitivity.setter
    def sensitivity(self, val):
        self.covar_module.sensitivity = val

    @property
    def decay_rate(self):
        return self.covar_module.decay

    @decay_rate.setter
    def decay_rate(self, val):
        self.covar_module.decay = val

    # def train(self, mode: bool = True):
    #     super().train(mode)
    #     self.likelihood.train()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_m(self, pred_t) -> torch.distributions.MultivariateNormal:
        Kxx = self.covar_module(self.train_t, self.train_t)
        K_inv = torch.inverse(Kxx.evaluate())
        pred_t_blocked = pred_t.repeat(self.num_outputs)
        K_xxstar = self.covar_module(self.train_t, pred_t_blocked).evaluate()
        K_xstarx = torch.transpose(K_xxstar, 0, 1).type(torch.float64)
        K_xstarxK_inv = torch.matmul(K_xstarx, K_inv)
        KxstarxKinvY = torch.matmul(K_xstarxK_inv, self.train_y)
        mean = KxstarxKinvY.view(self.num_outputs, pred_t.shape[0])

        K_xstarxstar = self.covar_module(pred_t_blocked, pred_t_blocked).evaluate()
        var = K_xstarxstar - torch.matmul(K_xstarxK_inv, torch.transpose(K_xstarx, 0, 1))
        var = torch.diagonal(var, dim1=0, dim2=1).view(self.num_outputs, pred_t.shape[0])
        mean = mean.transpose(0, 1)
        var = var.transpose(0, 1)
        var = torch.diag_embed(var)
        return MultivariateNormal(mean, var)

    def predict_f(self, pred_t) -> MultivariateNormal:
        Kxx = self.covar_module(self.train_t, self.train_t)
        K_inv = torch.inverse(Kxx.evaluate())

        Kxf = self.covar_module.K_xf(self.train_t, pred_t).type(torch.float64)
        KfxKxx = torch.matmul(torch.transpose(Kxf, 0, 1), K_inv)
        mean = torch.matmul(KfxKxx, self.train_y).view(-1).unsqueeze(0)

        #Kff-KfxKxxKxf
        Kff = self.covar_module.K_ff(pred_t, pred_t)  # (100, 500)
        var = Kff - torch.matmul(KfxKxx, Kxf)
        # var = torch.diagonal(var, dim1=0, dim2=1).view(-1)
        var = var.unsqueeze(0)
        print(var.shape, var.min())
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(var[0].detach())
        plt.colorbar()
        var += 1e-2*torch.eye(var.shape[-1])
        print(mean.shape, var.shape)
        print(torch.diagonal(var, dim1=1, dim2=2).min())
        # print(torch.cholesky(var + 1e-2 * torch.eye(80)))

        batch_mvn = gpytorch.distributions.MultivariateNormal(mean, var)
        print(batch_mvn)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)
