from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

from .model import VariationalLFM
from lafomo.options import VariationalOptions
from lafomo.utilities.torch import softplus
from lafomo.datasets import LFMDataset


class TranscriptionalRegulationLFM(VariationalLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, options: VariationalOptions, **kwargs):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, options, **kwargs)
        self.decay_rate = Parameter(0.1 + torch.rand((self.num_outputs, 1), dtype=torch.float64))
        self.basal_rate = Parameter(torch.rand((self.num_outputs, 1), dtype=torch.float64))
        self.sensitivity = Parameter(0.2 + torch.rand((self.num_outputs, 1), dtype=torch.float64))

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)

        decay = self.decay_rate * h

        q_f = self.get_latents(t.reshape(-1))

        # Reparameterisation trick
        f = q_f.rsample([self.options.num_samples])  # (S, I, t)

        f = self.G(f)  # (S, num_outputs, t)

        return self.basal_rate + self.sensitivity * f - decay

    @abstractmethod
    def G(self, f):
        """
        Parameters:
            f: (I, T)
        """
        pass

    def predict_f(self, t_predict):
        # Sample from the latent distribution
        q_f = self.get_latents(t_predict.reshape(-1))
        f = q_f.sample([500])  # (S, I, t)
        # This is a hack to wrap the latent function with the nonlinearity. Note we use the same variance.
        f = torch.mean(self.G(f), dim=0)[0]
        return torch.distributions.multivariate_normal.MultivariateNormal(f, scale_tril=q_f.scale_tril)

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
    def __init__(self, num_outputs, num_latents, t_inducing, dataset, options):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, options)
        self.w = Parameter(torch.ones((self.num_outputs, self.num_latents), dtype=torch.float64))
        self.w_0 = Parameter(torch.ones((self.num_outputs, 1), dtype=torch.float64))

    def G(self, f):
        p_pos = softplus(f)  # (S, I, extras)
        interactions = torch.matmul(self.w, torch.log(p_pos+1e-50)) + self.w_0  # (J,I)(I,e)+(J,1)
        return torch.sigmoid(interactions)  # TF Activation Function (sigmoid)

    def predict_f(self, t_predict):
        q_f = self.get_latents(t_predict.reshape(-1))
        f = softplus(q_f.sample([500]))
        return torch.distributions.multivariate_normal.MultivariateNormal(f, scale_tril=q_f.scale_tril)


class PoissonLFM(TranscriptionalRegulationLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, options):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, options=options)

    """Adds poison to the latent forces"""
    def G(self, λ):
        # λ (I, points) is the parameter of the poison distribution
        print('lam shape', λ.shape)
        # f = Poisson(λ).rsample() #  not implemented - no reparam trick for Poisson implemented
        # so we use an approximation as a N(λ, λ)
        f = Normal(λ, λ).rsample()
        return f.repeat(self.num_outputs, 1)
