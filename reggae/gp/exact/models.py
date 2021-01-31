import torch
import gpytorch

from .kernels import SIMKernel, SIMMean
from ..models import LFM


class AnalyticalLFM(LFM, gpytorch.models.ExactGP):
    def __init__(self, train_t, train_y, num_genes, likelihood, variance):
        super().__init__(train_t, train_y, likelihood)
        self.num_genes = num_genes
        self.train_t = train_t.view(-1, 1)
        self.train_y = train_y.view(-1, 1)
        self.covar_module = SIMKernel(num_genes, variance)
        initial_basal = torch.mean(train_y.view(5, 7), dim=1) * self.covar_module.decay
        print(initial_basal)
        self.mean_module = SIMMean(self.covar_module, num_genes, initial_basal)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_m(self, pred_t, compute_var=True):
        Kxx = self.covar_module(self.train_t, self.train_t)
        K_inv = torch.inverse(Kxx.evaluate())
        K_xxstar = self.covar_module.K_xstarx(self.train_t, pred_t)
        K_xstarx = torch.transpose(K_xxstar, 0, 1).type(torch.float64)
        K_xstarxK_inv = torch.matmul(K_xstarx, K_inv)
        KxstarxKinvY = torch.matmul(K_xstarxK_inv, self.train_y)
        mu = KxstarxKinvY.view(self.num_genes, pred_t.shape[0])
        if compute_var:
            K_xstarxstar = self.covar_module.K_xstarx(pred_t, pred_t)
            print(K_xstarxstar.shape)
            var = K_xstarxstar - torch.matmul(K_xstarxK_inv, torch.transpose(K_xstarx, 0, 1))
            var = torch.diagonal(var, dim1=0, dim2=1).view(self.num_genes, pred_t.shape[0])
            return mu, var
        return mu

    def predict_f(self, pred_t):
        Kxx = self.covar_module(self.train_t, self.train_t)
        print('kxx', Kxx.shape, Kxx.dtype)
        K_inv = torch.inverse(Kxx.evaluate())

        Kxf = self.covar_module.K_xf(self.train_t, pred_t.view(-1, 1)).type(torch.float64)
        KfxKxx = torch.matmul(torch.transpose(Kxf, 0, 1), K_inv)
        print(KfxKxx.shape, self.train_y.shape)
        print(torch.matmul(KfxKxx, self.train_y).shape)
        return torch.matmul(KfxKxx, self.train_y).view(-1)
