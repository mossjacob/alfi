from abc import abstractmethod

import torch
from torchdiffeq import odeint
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from lafomo.utilities import softplus, inv_softplus
from lafomo.data_loaders import LFMDataset
from lafomo.gp.variational.options import VariationalOptions
from lafomo.gp import LFM
from lafomo.gp.kernels import RBF


class VariationalLFM(LFM):
    """
    Variational inducing point approximation of Latent Force Models.
    Must override the `odefunc` function which encodes the ODE. This odefunc may call
    `get_latents` to get the values of the latent function at arbitrary time `t`.

    Parameters
    ----------
    num_outputs : int : the number of outputs, size of h vector (for example, the number of genes)
    num_latents : int : the number of latent functions (for example, the number of TFs)
    fixed_variance : tensor : variance if the preprocessing variance is known, otherwise learnt.
    t_inducing : tensor of shape (T_u) : the inducing timepoints.
    """
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, options: VariationalOptions, dtype=torch.float64):
        super(VariationalLFM, self).__init__()
        self.num_outputs = num_outputs
        self.num_latents = num_latents
        self.options = options
        self.num_inducing = t_inducing.shape[0]
        self.num_observed = dataset[0][0].shape[0]
        self.inducing_inputs = Parameter(torch.tensor(t_inducing), requires_grad=options.learn_inducing)
        self.num_samples = options.num_samples
        self.dtype = dtype
        self.kernel = RBF(num_outputs=num_latents, scale=options.kernel_scale, dtype=dtype)

        q_m = torch.rand((self.num_latents, self.num_inducing, 1), dtype=dtype)
        q_S = self.kernel(self.inducing_inputs)
        q_cholS = torch.cholesky(q_S)
        self.q_m = Parameter(q_m)
        self.q_cholS = Parameter(q_cholS)

        if options.preprocessing_variance is not None:
            self.likelihood_variance = Parameter(torch.tensor(options.preprocessing_variance), requires_grad=False)
        else:
            self.raw_likelihood_variance = Parameter(torch.ones((self.num_outputs, self.num_observed), dtype=dtype))

        if options.initial_conditions:
            self.initial_conditions = Parameter(torch.tensor(torch.zeros(self.num_outputs, 1)), requires_grad=True)
        self.nfe = 0

    @property
    def likelihood_variance(self):
        return softplus(self.raw_likelihood_variance)

    @likelihood_variance.setter
    def likelihood_variance(self, value):
        self.raw_likelihood_variance = inv_softplus(value)

    def initial_state(self, h):
        if self.options.initial_conditions:
            h = self.initial_conditions.repeat(h.shape[0], 1, 1)
        return h

    def forward(self, t, h, rtol=1e-4, atol=1e-6, compute_var=False):
        """
        t : torch.Tensor
            Shape (num_times)
        h : torch.Tensor the initial state of the ODE
            Shape (num_genes, 1)
        Returns
        -------
        Returns evolved h across times t.
        Shape (num_genes, num_points).
        """
        self.nfe = 0

        # Precompute variables
        self.Kmm = self.kernel(self.inducing_inputs)
        self.L = torch.cholesky(self.Kmm)
        q_cholS = torch.tril(self.q_cholS)
        self.S = torch.matmul(q_cholS, torch.transpose(q_cholS, 1, 2))

        # Integrate forward from the initial positions h0.
        h0 = self.initial_state(h)
        h_samples = odeint(self.odefunc, h0, t, method='dopri5', rtol=rtol, atol=atol)  # (T, S, num_outputs, 1)

        h_out = torch.mean(h_samples, dim=1).transpose(0, 1)
        h_std = torch.std(h_samples, dim=1).transpose(0, 1)

        if compute_var:
            return self.decode(h_out), h_std
        return self.decode(h_out)

    def decode(self, h_out):
        return h_out

    def predict_m(self, t_predict, **kwargs):
        """
        Calls self on input `t_predict`
        """
        initial_value = torch.zeros((self.num_samples, self.num_outputs, 1), dtype=self.dtype)
        outputs, var = self(t_predict.view(-1), initial_value, compute_var=True, **kwargs)
        var = torch.squeeze(var).detach()
        outputs = torch.squeeze(outputs).detach()
        return outputs, var

    def predict_f(self, t_predict):
        """
        Returns the latents
        """
        q_f = self.get_latents(t_predict)
        return q_f

    @abstractmethod
    def odefunc(self, t, h):
        """
        Parameters:
            h: shape (num_samples, num_outputs, 1)
        """
        pass

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
        # print('ss', m_s.shape, S_s.shape)

        if S_s.shape[2] > 1:
            if True:
                jitter = 1e-5 * torch.eye(S_s.shape[1], dtype=S_s.dtype)
                S_s = S_s + jitter
            q_f = MultivariateNormal(m_s, S_s)
        else:
            q_f = Normal(m_s, torch.squeeze(S_s, 2))

        return q_f

    def log_likelihood(self, y, h, data_index=None):
        """
        Parameters:
            y: target
            h: predicted
            data_index: in case the likelihood terms rely on the data index, e.g. variance
        """
        sq_diff = torch.square(y - h)
        variance = self.likelihood_variance  # add PUMA variance
        log_lik = -0.5*torch.log(2*3.1415926*variance) - 0.5*sq_diff/variance
        log_lik = torch.sum(log_lik)
        return log_lik #* self.num_tfs * self.num_observed # TODO: check if we need this multiplier

    def kl_divergence(self):
        KL = -0.5 * self.num_inducing # CHECK * self.num_latents

        # log(det(S)): Uses that sqrt(det(X)) = det(X^(1/2)) and that det of triangular matrix
        # is the product of the diagonal entries (i.e. sum of their logarithm).
        q_cholS = torch.tril(self.q_cholS)

        logdetS = torch.sum(torch.log(torch.diagonal(q_cholS, dim1=1, dim2=2)**2))

        # log(det(Kmm)): (already checked, seems working)
        logdetK = torch.sum(torch.log(torch.diagonal(self.L, dim1=1, dim2=2)**2))
        # tr(inv_Kmm * S):
        # trKS_1 = torch.matmul(self.inv_Kmm, self.S)
        trKS = torch.cholesky_solve(self.S, self.L, upper=False)
        trKS = torch.sum(torch.diagonal(trKS, dim1=1, dim2=2))

        # m^T Kuu^(-1) m: cholesky_solve(b, chol)
        Kinv_m = torch.cholesky_solve(self.q_m, self.L, upper=False)
        m_Kinv_m = torch.matmul(torch.transpose(self.q_m, 1, 2), Kinv_m)# (1,1,1)
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

    def elbo(self, y, h, kl_mult=1, data_index=None):
        return self.log_likelihood(y, h, data_index=data_index), kl_mult * self.kl_divergence()
