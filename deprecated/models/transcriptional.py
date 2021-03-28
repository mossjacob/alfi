
import torch
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

from lafomo.variational.models import OrdinaryLFM
from lafomo.configuration import VariationalConfiguration
from lafomo.utilities.torch import softplus
from lafomo.datasets import LFMDataset


class MultiLFM(TranscriptionalRegulationLFM):
    def __init__(self, num_latents, config, kernel, t_inducing, dataset):
        super().__init__(num_latents, config, kernel, t_inducing, dataset)
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
    def __init__(self, num_latents, config, kernel, t_inducing, dataset: LFMDataset):
        super().__init__(num_latents, config, kernel, t_inducing, dataset)

    """Adds poison to the latent forces"""
    def G(self, λ):
        # λ (I, points) is the parameter of the poison distribution
        print('lam shape', λ.shape)
        # f = Poisson(λ).rsample() #  not implemented - no reparam trick for Poisson implemented
        # so we use an approximation as a N(λ, λ)
        f = Normal(λ, λ).rsample()
        return f.repeat(self.num_outputs, 1)
