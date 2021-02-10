from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

from .model import VariationalLFM
from reggae.utilities import softplus, LFMDataset


class TranscriptionalRegulationLFM(VariationalLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, extra_points=2, **kwargs):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, extra_points=extra_points, **kwargs)
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
        f = self.G(f)
        if self.extra_points > 0:
            f = f[:, self.extra_points] # get the midpoint
            f = torch.unsqueeze(f, 1)
        # print(f.shape)
        # print(self.basal_rate.shape, f.shape, decay.shape)
        # print((self.basal_rate + self.sensitivity * f - decay).shape)
        return self.basal_rate + self.sensitivity * f - decay


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
        self.w_0 = Parameter(torch.ones((self.num_outputs, 1), dtype=torch.float64))

    def G(self, f):
        p_pos = softplus(f)  # (I, extras)
        interactions = torch.matmul(self.w, torch.log(p_pos+1e-50)) + self.w_0  # (J,I)(I,e)+(J,1)
        return torch.sigmoid(interactions)  # TF Activation Function (sigmoid)


class PoissonLFM(TranscriptionalRegulationLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, fixed_variance=None):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, fixed_variance=fixed_variance, extra_points=0)

    """Adds poison to the latent forces"""
    def G(self, λ):
        # λ (I, points) is the parameter of the poison distribution
        print('lam shape', λ.shape)
        # f = Poisson(λ).rsample() #  not implemented - no reparam trick for Poisson implemented
        # so we use an approximation as a N(λ, λ)
        f = Normal(λ, λ).rsample()
        return f.repeat(self.num_outputs, 1)
