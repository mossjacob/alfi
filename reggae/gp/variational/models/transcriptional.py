from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

from .model import VariationalLFM
from reggae.utilities import softplus
from reggae.data_loaders import LFMDataset


class TranscriptionalRegulationLFM(VariationalLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, **kwargs):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, **kwargs)
        self.decay_rate = Parameter(0.1 + torch.rand((self.num_outputs, 1), dtype=torch.float64))
        self.basal_rate = Parameter(torch.rand((self.num_outputs, 1), dtype=torch.float64))
        self.sensitivity = Parameter(0.2 + torch.rand((self.num_outputs, 1), dtype=torch.float64))

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)

        decay = torch.multiply(self.decay_rate.squeeze(), h.squeeze(-1)).view(self.num_samples, -1, 1)

        q_f = self.get_latents(t.reshape(-1))

        # Reparameterisation trick
        f = q_f.rsample([self.num_samples])  # (S, I, t)

        f = self.G(f)  # (S, num_outputs, t)

        return self.basal_rate + self.sensitivity * f - decay


    @abstractmethod
    def G(self, f):
        """
        Parameters:
            f: (I, T)
        """
        pass

    def log_likelihood(self, y, h, data_index=0):
        sq_diff = torch.square(y - h)
        variance = self.likelihood_variance[data_index]  # add PUMA variance
        log_lik = -0.5*torch.log(2*3.1415926*variance) - 0.5*sq_diff/variance
        log_lik = torch.sum(log_lik)
        return log_lik


class SingleLinearLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return f.repeat(1, self.num_outputs, 1)


class NonLinearLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return softplus(f).repeat(1, self.num_outputs, 1)


class ExponentialLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return torch.exp(f).repeat(1, self.num_outputs, 1)


class MultiLFM(TranscriptionalRegulationLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, t_observed, fixed_variance=None):
        super().__init__(num_outputs, num_latents, t_inducing, t_observed, fixed_variance=fixed_variance)
        self.w = Parameter(torch.ones((self.num_outputs, self.num_latents), dtype=torch.float64))
        self.w_0 = Parameter(torch.ones((self.num_outputs, 1), dtype=torch.float64))

    def G(self, f):
        p_pos = softplus(f)  # (S, I, extras)
        interactions = torch.matmul(self.w, torch.log(p_pos+1e-50)) + self.w_0  # (J,I)(I,e)+(J,1)
        return torch.sigmoid(interactions)  # TF Activation Function (sigmoid)


class PoissonLFM(TranscriptionalRegulationLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, fixed_variance=None):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, fixed_variance=fixed_variance)

    """Adds poison to the latent forces"""
    def G(self, λ):
        # λ (I, points) is the parameter of the poison distribution
        print('lam shape', λ.shape)
        # f = Poisson(λ).rsample() #  not implemented - no reparam trick for Poisson implemented
        # so we use an approximation as a N(λ, λ)
        f = Normal(λ, λ).rsample()
        return f.repeat(self.num_outputs, 1)
