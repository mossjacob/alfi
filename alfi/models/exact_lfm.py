import torch
import gpytorch

from torch.distributions import MultivariateNormal

from .lfm import LFM
from alfi.kernels import SIMKernel
from alfi.means import SIMMean
from alfi.utilities.data import flatten_dataset
from alfi.datasets import LFMDataset


class ExactLFM(LFM, gpytorch.models.ExactGP):
    def __init__(self, dataset: LFMDataset, variance):
        train_t, train_y = flatten_dataset(dataset)
        super().__init__(train_t, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())

        self.num_outputs = dataset.num_outputs
        self.train_t = train_t.view(-1, 1)
        self.train_y = train_y.view(-1, 1)
        self.covar_module = SIMKernel(self.num_outputs, torch.tensor(variance, requires_grad=False))
        initial_basal = torch.mean(train_y.view(self.num_outputs, -1), dim=1) * self.covar_module.decay
        self.mean_module = SIMMean(self.covar_module, self.num_outputs, initial_basal)

    @property
    def decay_rate(self):
        return self.covar_module.decay

    @decay_rate.setter
    def decay_rate(self, val):
        self.covar_module.decay = val

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_m(self, pred_t, jitter=1e-5) -> torch.distributions.MultivariateNormal:
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
        var += jitter * torch.eye(var.shape[-1])
        return MultivariateNormal(mean, var)

    def predict_f(self, pred_t, jitter=1e-3) -> MultivariateNormal:
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
        # For some reason a full covariance doesn't work, for now just take the variance: (TODO)
        var = torch.diagonal(var, dim1=1, dim2=2)
        var = torch.diag_embed(var)
        var += jitter * torch.eye(var.shape[-1])

        batch_mvn = gpytorch.distributions.MultivariateNormal(mean, var)
        print(batch_mvn)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath+'lfm.pt')

    @classmethod
    def load(cls,
             filepath,
             lfm_args=[], lfm_kwargs={}):
        lfm_state_dict = torch.load(filepath+'lfm.pt')
        lfm = cls(*lfm_args, **lfm_kwargs)
        lfm.load_state_dict(lfm_state_dict)
        return lfm
