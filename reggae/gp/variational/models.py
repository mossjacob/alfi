from abc import abstractmethod

import torch
from torch import nn
from torchdiffeq import odeint
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from reggae.utilities import softplus, inv_softplus, cholesky_inverse


class VariationalLFM(nn.Module):
    """
    Description blah
    Parameters
    ----------
    num_genes : int
    Number of genes.
    num_tfs : int
    known_variance : bool = variance tensor if the preprocessing variance is known, otherwise learnt.
    t_inducing : int
    inducing timepoints.
    """
    def __init__(self, num_genes, num_tfs, t_inducing, known_variance=None):
        super(VariationalLFM, self).__init__()
        self.num_genes = num_genes
        self.num_tfs = num_tfs
        self.num_inducing = t_inducing.shape[0]
        self.inducing_inputs = torch.tensor(t_inducing, requires_grad=False)

        self.decay_rate = Parameter(1*torch.ones((self.num_genes, 1), dtype=torch.float64))
        self.basal_rate = Parameter(0.2*torch.ones((self.num_genes, 1), dtype=torch.float64))
        self.sensitivity = Parameter(2*torch.ones((self.num_genes, 1), dtype=torch.float64))
        # self.w = Parameter(torch.ones((self.num_genes, self.num_tfs), dtype=torch.float64))
        # self.w_0 = Parameter(torch.ones((self.num_tfs), dtype=torch.float64))

        self.nfe = 0
        self.raw_lengthscale = Parameter(0.5*torch.ones((num_tfs), dtype=torch.float64))
        self.v = Parameter(torch.ones((num_tfs), dtype=torch.float64))
        q_m = torch.rand((self.num_tfs, self.num_inducing, 1), dtype=torch.float64)
        q_K = self.rbf(self.inducing_inputs)

        # q_K = torch.eye(self.num_inducing, dtype=torch.float64).view(1, self.num_inducing, self.num_inducing)
        q_K = q_K.repeat(self.num_tfs, 1, 1)
        q_cholK = torch.cholesky(q_K)
        self.q_m = Parameter(q_m)
        self.q_cholS = Parameter(q_cholK)

        if known_variance is None:
            self.likelihood_variance = Parameter(torch.ones())
        else:
            self.likelihood_variance = torch.tensor(known_variance, requires_grad=False)
        # self.likelihood_variance = Parameter(torch.ones((self.num_genes, self.num_inducing), dtype=torch.float64))


    @property
    def lengthscale(self):
        return softplus(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self.raw_lengthscale = inv_softplus(value)

    def rbf(self, x: torch.Tensor, x2: torch.Tensor=None):
        """
        Radial basis function kernel.
        @param x:
        @param x2: if None, then x2 becomes x
        @return: K of shape (I, |x|, |x2|)
        """
        add_jitter = x2 is None
        if x2 is None:
            x2 = x
        x = x.view(-1)
        x2 = x2.view(-1)
        sq_dist = torch.square(x.view(-1, 1)-x2)
        sq_dist = sq_dist.repeat(self.num_tfs, 1, 1)
        sq_dist = torch.div(sq_dist, 2*self.lengthscale.view((-1, 1, 1)))
        K =  self.v.view(-1, 1, 1) * torch.exp(-sq_dist)
        if add_jitter:
            jitter = 1e-5 * torch.eye(x.shape[0])
            K += jitter
        return K


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
        num_times = t.size()

        # 1: Likelihood step: Sample from variational distribution
        self.Kmm = self.rbf(self.inducing_inputs)

        L = torch.cholesky(self.Kmm)
        self.inv_Kmm = cholesky_inverse(L)
        q_cholS = torch.tril(self.q_cholS)
        self.S = torch.matmul(q_cholS, torch.transpose(q_cholS, 1, 2))

        # Integrate forward from the initial positions h.
        h_avg = 0
        for _ in range(num_samples):
            h_avg += odeint(self.odefunc, h, t, method='dopri5', rtol=rtol, atol=atol) / num_samples # shape (num_genes, num_times, 1

        # 2: KL term:
        # above: make cholesky
        KL = -0.5 * self.num_tfs * self.num_inducing # CHECK

        # log(det(S)): (already checked, seems working)
        # Uses that sqrt(det(X)) = det(X^(1/2)) and that det of triangular matrix
        # is the product of the diagonal entries (i.e. sum of their logarithm).
        logdetS = torch.sum(torch.log(torch.diagonal(q_cholS, dim1=1, dim2=2)**2))

        # log(det(Kmm)): (already checked, seems working)
        logdetK = torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)**2))
        # print('logdetK', logdetK.item(), torch.logdet(self.Kmm).item())
        # print('logdetS', logdetS.item(), torch.logdet(self.S).item())
        # tr(inv_Kmm * S):
        trKS = torch.matmul(self.inv_Kmm, self.S)
        trKS = torch.sum(torch.diagonal(trKS, dim1=1, dim2=2))

        # m^T Kuu^(-1) m:
        # cholesky_solve(b, chol)
        Kinv_m = torch.cholesky_solve(self.q_m, L, upper=False)
        m_Kinv_m = torch.matmul(torch.transpose(self.q_m, 1, 2), Kinv_m)# (1,1,1)
        m_Kinv_m = torch.squeeze(m_Kinv_m)
        KL += 0.5 * (logdetK - logdetS + trKS + m_Kinv_m)

        ## Use this code to check:
        # print('kl', KL)
        # plt.imshow(self.S[0].detach())
        # p = MultivariateNormal(torch.zeros((1, self.num_inducing), dtype=torch.float64), self.Kmm)
        # q = MultivariateNormal(torch.squeeze(self.q_m, 2), self.S)
        # KL2 = torch.distributions.kl_divergence(q, p)
        # print('kl2', KL2)
        ##
        return torch.transpose(h_avg, 0, 1), KL


    def odefunc(self, t, h):
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)
        # h is of shape (num_genes, 1)
        decay = torch.multiply(self.decay_rate.view(-1), h.view(-1)).view(-1, 1)

        q_f = self.get_tfs(t.reshape(-1))
        # Reparameterisation trick
        f = q_f.rsample() # TODO: multiple samples?
        midpoint = 1
        Gp = self.G(f)[midpoint] #get the midpoint

        # print(Gp.shape)
        # print(self.basal_rate, Gp, decay)

        return self.basal_rate + self.sensitivity * Gp - decay

    def get_tfs(self, t):
        """t: shape (T*,)"""
        ## Uncomment this if calling get_tfs on a fresh model
        # self.Kmm = self.rbf(self.inducing_inputs)
        # L = torch.cholesky(self.Kmm)
        # self.inv_Kmm = cholesky_inverse(L)
        # q_cholS = torch.tril(self.q_cholS)
        # self.S = torch.matmul(q_cholS, torch.transpose(q_cholS, 1, 2))
        ##
        if t.shape[0] == 1:
            t = torch.tensor([t[0]-0.05, t, t[0]+0.05])
            # print(t)
        Ksm = self.rbf(t, self.inducing_inputs) # (I, T*, T)
        α = torch.matmul(Ksm, self.inv_Kmm) # (I, T*, T)
        m_s = torch.matmul(α, self.q_m) # (I, T*, T*)
        Kss = self.rbf(t) # (I, T*, T*)
        # I think Kss is always v: rbf with one timepoint is = ve^0
        S_Kmm = self.S - self.Kmm # (I, T, T)
        AS_KA = torch.matmul(torch.matmul(α, S_Kmm), torch.transpose(α, 1, 2)) # (I, T*, T*)
        S_s = (Kss + AS_KA) # (I, T*, T*)
        # print(Kss, AS_KA)
        # plt.figure()
        # plt.imshow(self.S[0].detach())
        # plt.plot(torch.squeeze(m_s[0], 1).detach())
        if S_s.shape[2] > 1:
            q_f = MultivariateNormal(torch.squeeze(m_s, 2), S_s)
        else:
            q_f = Normal(torch.squeeze(m_s), torch.squeeze(S_s))

        return q_f

    @abstractmethod
    def G(self, f):
        """
        Parameters:
            f: (I, T)
        """
        pass

    def train(self, mode=True):
        # self.gp.train(mode)
        # self.gp.likelihood.train(mode)
        super().train(mode)

    def log_likelihood(self, y, h):
        # print(self.likelihood_variance)
        sq_diff = torch.square(y - h)
        variance = self.likelihood_variance # add PUMA variance, 0th replicate
        log_lik = -0.5*torch.log(2*3.1415926*variance) - 0.5*sq_diff/variance
        log_lik = torch.sum(log_lik)
        return log_lik * self.num_tfs * self.num_inducing
        # return MultivariateNormal(y, torch.exp(self.likelihood_variance)).log_prob(h)


class SingleLinearLFM(VariationalLFM):

    def G(self, pf):
        return torch.squeeze(f, dim=0)


class MultiLFM(VariationalLFM):

    def G(self, f):
        p_pos = softplus(p)
        interactions = torch.matmul(self.w, torch.log(p_pos+1e-50)) + self.w_0 #(TODO)
        return torch.sigmoid(interactions) # TF Activation Function (sigmoid)
        return p
