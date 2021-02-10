from abc import abstractmethod

import torch
from torchdiffeq import odeint
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from reggae.utilities import softplus, inv_softplus, cholesky_inverse, LFMDataset

from reggae.gp import LFM


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
    t_observed: tensor of shape (T) : the observed timepoints, i.e., the timepoints that the ODE solver should output
    """
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, fixed_variance=None, extra_points=1, dtype=torch.float64, learn_inducing=False):
        super(VariationalLFM, self).__init__()
        self.num_outputs = num_outputs
        self.num_latents = num_latents
        self.num_inducing = t_inducing.shape[0]
        self.num_observed = dataset[0][0].shape[0]
        self.inducing_inputs = Parameter(torch.tensor(t_inducing), requires_grad=learn_inducing)
        self.extra_points = extra_points
        self.dtype = dtype
        self.raw_lengthscale = Parameter(inv_softplus(0.2 * torch.ones((num_latents), dtype=dtype)))
        self.raw_scale = Parameter(torch.ones((num_latents), dtype=dtype))

        q_m = torch.rand((self.num_latents, self.num_inducing, 1), dtype=dtype)
        q_S = self.rbf(self.inducing_inputs)
        q_cholS = torch.cholesky(q_S)
        self.q_m = Parameter(q_m)
        self.q_cholS = Parameter(q_cholS)

        if fixed_variance is not None:
            self.likelihood_variance = Parameter(torch.tensor(fixed_variance), requires_grad=False)
        else:
            self.raw_likelihood_variance = Parameter(torch.ones((self.num_outputs, self.num_observed), dtype=dtype))
        self.nfe = 0

    @property
    def lengthscale(self):
        return softplus(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self.raw_lengthscale = inv_softplus(value)

    @property
    def scale(self):
        return softplus(self.raw_scale)

    @scale.setter
    def scale(self, value):
        self.raw_scale = inv_softplus(value)

    @property
    def likelihood_variance(self):
        return softplus(self.raw_likelihood_variance)

    @likelihood_variance.setter
    def likelihood_variance(self, value):
        self.raw_likelihood_variance = inv_softplus(value)

    def rbf(self, x: torch.Tensor, x2: torch.Tensor=None):
        """
        TODO: move this to another file
        Radial basis function kernel.
        Parameters:
            x: tensor
            x2: if None, then x2 becomes x
        Returns:
             K of shape (I, |x|, |x2|)
        """
        add_jitter = x2 is None
        if x2 is None:
            x2 = x
        x = x.view(-1)
        x2 = x2.view(-1)
        sq_dist = torch.square(x.view(-1, 1)-x2)
        sq_dist = sq_dist.repeat(self.num_latents, 1, 1)
        sq_dist = torch.div(sq_dist, 2*self.lengthscale.view((-1, 1, 1)))
        K = self.scale.view(-1, 1, 1) * torch.exp(-sq_dist)
        if add_jitter:
            jitter = 1e-4 * torch.eye(x.shape[0], dtype=K.dtype, device=K.device)
            K += jitter

        return K

    def initial_state(self, h):
        return h

    def forward(self, t, h, rtol=1e-4, atol=1e-6, num_samples=5):
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

        # 1: Likelihood step: Sample from variational distribution
        self.Kmm = self.rbf(self.inducing_inputs)

        self.L = torch.cholesky(self.Kmm)
        self.inv_Kmm = cholesky_inverse(self.L)
        q_cholS = torch.tril(self.q_cholS)
        self.S = torch.matmul(q_cholS, torch.transpose(q_cholS, 1, 2))

        h0 = self.initial_state(h)
        # Integrate forward from the initial positions h.
        h_avg = 0
        for _ in range(num_samples):
            h_avg += odeint(self.odefunc, h0, t, method='dopri5', rtol=rtol, atol=atol) / num_samples # shape (num_genes, num_times, 1

        h_out = torch.transpose(h_avg, 0, 1)
        return self.decode(h_out)

    def decode(self, h_out):
        return h_out

    def predict_m(self, t_predict, **kwargs):
        """
        Calls self on input `t_predict`
        """
        initial_value = torch.zeros((self.num_outputs, 1), dtype=self.dtype)
        outputs = self(t_predict.view(-1), initial_value, **kwargs)
        outputs = torch.squeeze(outputs).detach()
        return outputs, torch.zeros_like(outputs, requires_grad=False) #TODO: send back variance!

    @abstractmethod
    def odefunc(self, t, h):
        """
        Parameters:

        """
        pass

    def get_latents(self, t):
        """
        Parameters:
            t: shape (T*,)
        """
        ## Uncomment this if calling get_tfs on a fresh model
        # self.Kmm = self.rbf(self.inducing_inputs)
        # L = torch.cholesky(self.Kmm)
        # self.inv_Kmm = cholesky_inverse(L)
        # q_cholS = torch.tril(self.q_cholS)
        # self.S = torch.matmul(q_cholS, torch.transpose(q_cholS, 1, 2))
        ##
        if t.shape[0] == 1 and self.extra_points > 0:
            # t_new = torch.ones(3)
            # t_new[0] = t[0]-0.05
            # t_new[2] = t[0]+0.05
            # t_new[1] = t[0]
            t_l = list()
            for i in range(self.extra_points, 0, -1):
                t_l.append(t[0]-0.05*i)
            for i in range(self.extra_points+1):
                t_l.append(t[0]+0.05*i)
            t = torch.tensor(t_l, device=t.device).reshape(-1)
            # t = torch.tensor([t[0]-0.05, t[0], t[0]+0.05]).reshape(-1)
            # print(t)
        Ksm = self.rbf(t, self.inducing_inputs)  # (I, T*, Tu)
        α = torch.matmul(Ksm, self.inv_Kmm)  # (I, T*, Tu)
        m_s = torch.matmul(α, self.q_m)  # (I, T*, Tu)
        Kss = self.rbf(t)  # (I, T*, T*)
        S_Kmm = self.S - self.Kmm # (I, Tu, Tu)
        AS_KA = torch.matmul(torch.matmul(α, S_Kmm), torch.transpose(α, 1, 2)) # (I, T*, T*)
        S_s = (Kss + AS_KA) # (I, T*, T*)
        # plt.figure()
        # plt.imshow(self.S[0].detach())
        # plt.plot(torch.squeeze(m_s[0], 1).detach())
        if S_s.shape[2] > 1:
            if True:
                jitter = 1e-5 * torch.eye(S_s.shape[1], dtype=S_s.dtype)
                S_s = S_s + jitter
            q_f = MultivariateNormal(torch.squeeze(m_s, 2), S_s)
        else:
            q_f = Normal(torch.squeeze(m_s), torch.squeeze(S_s))

        return q_f

    def log_likelihood(self, y, h):
        sq_diff = torch.square(y - h)
        variance = self.likelihood_variance # add PUMA variance, 0th replicate
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
        trKS = torch.matmul(self.inv_Kmm, self.S)
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

    def elbo(self, y, h, kl_mult=1):
        return self.log_likelihood(y, h), kl_mult * self.kl_divergence()






