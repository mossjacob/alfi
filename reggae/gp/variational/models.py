from abc import abstractmethod

import torch
from torchdiffeq import odeint
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from reggae.utilities import softplus, inv_softplus, cholesky_inverse, LFMDataset

from ..models import LFM


class VariationalLFM(LFM):
    """
    Variational inducing point approximation of Latent Force Models.
    Must override the `odefunc` function which encodes the ODE. This odefunc may call
    `get_latents` to get the values of the latent function at arbitrary time `t`.

    Parameters
    ----------
    num_outputs : int : the number of GP outputs (for example, the number of genes)
    num_latents : int : the number of latent functions (for example, the number of TFs)
    fixed_variance : tensor : variance if the preprocessing variance is known, otherwise learnt.
    t_inducing : tensor of shape (T_u) : the inducing timepoints.
    t_observed: tensor of shape (T) : the observed timepoints, i.e., the timepoints that the ODE solver should output
    """
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, fixed_variance=None, extra_points=1):
        super(VariationalLFM, self).__init__()
        self.num_outputs = num_outputs
        self.num_latents = num_latents
        self.num_inducing = t_inducing.shape[0]
        self.num_observed = dataset[0][0].shape[0]
        self.inducing_inputs = torch.tensor(t_inducing, requires_grad=False)
        self.extra_points = extra_points

        self.raw_lengthscale = Parameter(0.5 * torch.ones((num_latents), dtype=torch.float64))
        self.raw_scale = Parameter(torch.ones((num_latents), dtype=torch.float64))

        q_m = torch.rand((self.num_latents, self.num_inducing, 1), dtype=torch.float64)
        q_S = self.rbf(self.inducing_inputs)
        q_cholS = torch.cholesky(q_S)
        self.q_m = Parameter(q_m)
        self.q_cholS = Parameter(q_cholS)

        if fixed_variance is not None:
            self.likelihood_variance = torch.tensor(fixed_variance, requires_grad=False)
        else:
            self.raw_likelihood_variance = Parameter(torch.ones((self.num_outputs, self.num_observed), dtype=torch.float64))
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

        # 1: Likelihood step: Sample from variational distribution
        self.Kmm = self.rbf(self.inducing_inputs)

        self.L = torch.cholesky(self.Kmm)
        self.inv_Kmm = cholesky_inverse(self.L)
        q_cholS = torch.tril(self.q_cholS)
        self.S = torch.matmul(q_cholS, torch.transpose(q_cholS, 1, 2))

        # Integrate forward from the initial positions h.
        h_avg = 0
        for _ in range(num_samples):
            h_avg += odeint(self.odefunc, h, t, method='dopri5', rtol=rtol, atol=atol) / num_samples # shape (num_genes, num_times, 1

        return torch.transpose(h_avg, 0, 1)

    def predict_m(self, t_predict, **kwargs):
        """
        Calls self on input `t_predict`
        """
        initial_value = torch.zeros((self.num_outputs, 1), dtype=torch.float64)
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
            t = torch.tensor(t_l).reshape(-1)
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


class ReactionDiffusionLFM(VariationalLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, fixed_variance=None):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, fixed_variance=fixed_variance)
        self.translation_rate = Parameter(1 * torch.ones((self.num_outputs, 1), dtype=torch.float64))
        self.decay_rate = Parameter(1 * torch.ones((self.num_outputs, 1), dtype=torch.float64))
        self.diffusion_rate = Parameter(1 * torch.ones((self.num_outputs, 1), dtype=torch.float64))


class MLPLFM(VariationalLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, extra_points=2):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, fixed_variance=None, extra_points=extra_points)
        h_dim = 20  # number of hidden units
        ode_layers = [nn.Linear(num_latents, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, num_outputs)]

        self.mlp = nn.Sequential(*ode_layers)

    def odefunc(self, t, h):
        """
        h shape (num_outputs, 1)
        """
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)

        q_f = self.get_latents(t.reshape(-1))

        # Reparameterisation trick
        f = q_f.rsample()
        if self.extra_points > 0:
            f = f[:, self.extra_points]  # get the midpoint

        y = self.mlp(f)
        y = torch.unsqueeze(y, 1)
        return y


class TranscriptionalRegulationLFM(VariationalLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, fixed_variance=None, extra_points=2):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, fixed_variance=fixed_variance, extra_points=extra_points)
        self.decay_rate = Parameter(1 * torch.ones((self.num_outputs, 1), dtype=torch.float64))
        self.basal_rate = Parameter(0.2 * torch.ones((self.num_outputs, 1), dtype=torch.float64))
        self.sensitivity = Parameter(2 * torch.ones((self.num_outputs, 1), dtype=torch.float64))

    def odefunc(self, t, h):
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)
        # h is of shape (num_genes, 1)

        decay = torch.multiply(self.decay_rate.view(-1), h.view(-1)).view(-1, 1)

        q_f = self.get_latents(t.reshape(-1))
        # Reparameterisation trick
        f = q_f.rsample() # TODO: multiple samples?
        Gp = self.G(f)
        if self.extra_points > 0:
            Gp = Gp[:, self.extra_points] # get the midpoint
            Gp = torch.unsqueeze(Gp, 1)
        # print(Gp.shape)
        # print(self.basal_rate.shape, Gp.shape, decay.shape)
        # print((self.basal_rate + self.sensitivity * Gp - decay).shape)
        return self.basal_rate + self.sensitivity * Gp - decay


    @abstractmethod
    def G(self, f):
        """
        Parameters:
            f: (I, T)
        """
        pass


class SingleLinearLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return f.repeat(self.num_outputs, 1)


class NonLinearLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return softplus(f).repeat(self.num_outputs, 1)


class ExponentialLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return torch.exp(f).repeat(self.num_outputs, 1)


class MultiLFM(TranscriptionalRegulationLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, t_observed, fixed_variance=None):
        super().__init__(num_outputs, num_latents, t_inducing, t_observed, fixed_variance=fixed_variance)
        self.w = Parameter(torch.ones((self.num_outputs, self.num_latents), dtype=torch.float64))
        self.w_0 = Parameter(torch.ones((self.num_outputs,1), dtype=torch.float64))

    def G(self, f):
        p_pos = softplus(f)  # (I, extras)
        interactions = torch.matmul(self.w, torch.log(p_pos+1e-50)) + self.w_0  # (J,I)(I,e)+(J,1)
        return torch.sigmoid(interactions) # TF Activation Function (sigmoid)
