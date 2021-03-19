import torch
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

import numpy as np

from lafomo.utilities.torch import softplus, inv_softplus
from lafomo import LFM
from lafomo.configuration import VariationalConfiguration
from lafomo.datasets import LFMDataset


class VariationalLFM(LFM):
    """
    Variational inducing point approximation for Latent Force Models.

    Parameters
    ----------
    num_latents : int : the number of latent GPs (for example, the number of TFs)
    fixed_variance : tensor : variance if the preprocessing variance is known, otherwise learnt.
    t_inducing : tensor of shape (..., T_u) : the inducing timepoints. Preceding dimensions are for multi-dimensional inputs
    """
    def __init__(self,
                 num_latents: int,
                 config: VariationalConfiguration,
                 kernel: torch.nn.Module,
                 t_inducing,
                 dataset: LFMDataset,
                 dtype=torch.float64):
        super().__init__()
        self.num_outputs = dataset.num_outputs
        self.options = config
        self.num_inducing = t_inducing.shape[-1]
        self.num_observed = dataset[0][0].shape[0]
        self.inducing_inputs = Parameter(torch.tensor(t_inducing), requires_grad=config.learn_inducing)
        self.dtype = dtype
        self.kernel = kernel

        q_m = torch.rand((num_latents, self.num_inducing, 1), dtype=dtype)
        q_S = self.kernel(self.inducing_inputs)
        q_cholS = torch.cholesky(q_S)
        self.q_m = Parameter(q_m)
        self.q_cholS = Parameter(q_cholS)

        if config.preprocessing_variance is not None:
            self.likelihood_variance = Parameter(torch.tensor(config.preprocessing_variance), requires_grad=False)
        else:
            self.raw_likelihood_variance = Parameter(torch.ones((self.num_outputs, self.num_observed), dtype=dtype))

        if config.initial_conditions:
            self.initial_conditions = Parameter(torch.tensor(torch.zeros(self.num_outputs, 1)), requires_grad=True)

    @property
    def likelihood_variance(self):
        return softplus(self.raw_likelihood_variance)

    @likelihood_variance.setter
    def likelihood_variance(self, value):
        self.raw_likelihood_variance = inv_softplus(value)

    def get_latents(self, t):
        """
        Parameters:
            t: shape (T*,)
        """
        Ksm = self.kernel(t, self.inducing_inputs)  # (I, T*, Tu)
        α = torch.cholesky_solve(Ksm.permute([0, 2, 1]), self.L, upper=False).permute([0, 2, 1])  # (I, T*, Tu)
        m_s = torch.matmul(α, self.q_m)  # (I, T*, 1)
        m_s = torch.squeeze(m_s, 2)
        Kss = self.kernel(t)  # (I, T*, T*) this is always scale=1
        S_Kmm = self.S - self.Kmm  # (I, Tu, Tu)
        AS_KA = torch.matmul(torch.matmul(α, S_Kmm), torch.transpose(α, 1, 2))  # (I, T*, T*)
        S_s = (Kss + AS_KA)  # (I, T*, T*)

        if S_s.shape[2] > 1:
            if True:
                jitter = 1e-5 * torch.eye(S_s.shape[1], dtype=S_s.dtype)
                S_s = S_s + jitter
            q_f = MultivariateNormal(m_s, S_s)
        else:
            q_f = Normal(m_s, S_s.squeeze(2))

        return q_f

    def predict_m(self, t_predict, **kwargs):
        """
        Calls self on input `t_predict`
        """
        initial_value = torch.zeros((self.options.num_samples, self.num_outputs, 1), dtype=self.dtype)
        mean, var = self(t_predict.view(-1), initial_value, **kwargs)
        var = var.squeeze().detach()
        mean = mean.squeeze().detach()
        return mean, var

    def predict_f(self, t_predict):
        """
        Returns the latents
        """
        q_f = self.get_latents(t_predict)
        return q_f

    def log_likelihood(self, y_true, f_mean, f_var):
        """
        Computes the expected log density of the data given a Gaussian
        distribution for the function values, q(y^hat) = N(y_mean, y_var)
        Returns p(y|y^hat) = ∫ log(p(y=Y|y)) q(y) dy.

        Parameters:
            y: target
            h: predicted
            data_index: in case the likelihood terms rely on the data index, e.g. variance
        """
        sq_diff = torch.square(f_mean - y_true)
        print(sq_diff.min(), sq_diff.max())
        log_lik = torch.sum(
            - 0.5 * np.log(2 * np.pi) - torch.log(self.likelihood_variance)
            - 0.5 * (sq_diff + f_var) / self.likelihood_variance
        )
        return log_lik

    def kl_divergence(self):
        KL = -0.5 * self.num_inducing # CHECK * self.num_latents

        # log(det(S)): Uses that sqrt(det(X)) = det(X^(1/2)) and that det of triangular matrix
        # is the product of the diagonal entries (i.e. sum of their logarithm).
        q_cholS = torch.tril(self.q_cholS)

        logdetS = torch.sum(torch.log(torch.diagonal(q_cholS, dim1=1, dim2=2)**2))  # log(det(S))
        logdetK = torch.sum(torch.log(torch.diagonal(self.L, dim1=1, dim2=2)**2))   # log(det(Kmm))

        trKS = torch.cholesky_solve(self.S, self.L, upper=False)  # tr(inv_Kmm * S):
        trKS = torch.sum(torch.diagonal(trKS, dim1=1, dim2=2))

        Kinv_m = torch.cholesky_solve(self.q_m, self.L, upper=False)  # m^T Kuu^(-1) m: cholesky_solve(b, chol)
        m_Kinv_m = torch.matmul(torch.transpose(self.q_m, 1, 2), Kinv_m)  # (1,1,1)
        m_Kinv_m = torch.squeeze(m_Kinv_m)
        KL += 0.5 * (logdetK - logdetS + trKS + m_Kinv_m)
        KL = torch.sum(KL)
        ## Use this code to check:
        # print('kl', KL)
        # plt.imshow(self.S[0].detach())
        # p = MultivariateNormal(torch.zeros((1, self.num_inducing), dtype=torch.float64), self.Kmm)
        # q = MultivariateNormal(torch.squeeze(self.q_m, 2), self.S)
        # KL2 = torch.distributions.kl_divergence(q, p)
        # print('kl2', KL2)
        ##
        return KL

    def elbo(self, y_true, y_mean, y_var, kl_mult=1):
        return self.log_likelihood(y_true, y_mean, y_var), kl_mult * self.kl_divergence()
