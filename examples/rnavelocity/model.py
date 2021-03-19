import torch
from torch.nn import Parameter

from lafomo.variational.models import VariationalLFM
from lafomo.datasets import LFMDataset
from lafomo.configuration import VariationalConfiguration


class RNAVelocityLFM(VariationalLFM):
    def __init__(self, num_genes, num_latents, t_inducing, dataset: LFMDataset, options: VariationalConfiguration, **kwargs):
        super().__init__(num_genes*2, num_latents, t_inducing, dataset, options, **kwargs)
        self.transcription_rate = Parameter(torch.rand((num_genes, 1), dtype=torch.float64))
        self.splicing_rate = Parameter(torch.rand((num_genes, 1), dtype=torch.float64))
        self.decay_rate = Parameter(0.1 + torch.rand((num_genes, 1), dtype=torch.float64))
        self.num_cells = dataset[0][0].shape[0]
        ### Initialise random time assignments
        self.time_assignments = torch.rand(self.num_cells, requires_grad=False)

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        if (self.nfe % 10) == 0:
            print(t)
        self.nfe += 1
        num_samples = h.shape[0]
        num_outputs = h.shape[1]
        h = h.view(num_samples, num_outputs//2, 2)
        u = h[:, :, 0].unsqueeze(-1)
        s = h[:, :, 1].unsqueeze(-1)
        du = self.transcription_rate - self.splicing_rate * u
        ds = self.splicing_rate * u - self.decay_rate * s

        # q_f = self.get_latents(t.reshape(-1))
        # # Reparameterisation trick
        # f = q_f.rsample([self.num_samples])  # (S, I, t)
        # f = self.G(f)  # (S, num_outputs, t)

        h_t = torch.cat([du, ds], dim=1)
        return h_t

    def G(self, f):
        """
        Parameters:
            f: (I, T)
        """
        return f

    def predict_f(self, t_predict):
        # Sample from the latent distribution
        q_f = self.get_latents(t_predict.reshape(-1))
        f = q_f.sample([500])  # (S, I, t)
        # This is a hack to wrap the latent function with the nonlinearity. Note we use the same variance.
        f = torch.mean(self.G(f), dim=0)[0]
        return torch.distributions.multivariate_normal.MultivariateNormal(f, scale_tril=q_f.scale_tril)
