import torch
from torch.nn import Parameter

from lafomo.models import OrdinaryLFM
from lafomo.configuration import VariationalConfiguration


class RNAVelocityLFM(OrdinaryLFM):
    def __init__(self, num_cells, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        num_genes = num_outputs // 2
        self.transcription_rate = Parameter(3 * torch.rand(torch.Size([num_genes, 1]), dtype=torch.float64))
        self.splicing_rate = Parameter(3 * torch.rand(torch.Size([num_genes, 1]), dtype=torch.float64))
        self.decay_rate = Parameter(1 * torch.rand(torch.Size([num_genes, 1]), dtype=torch.float64))
        self.num_cells = num_cells
        ### Initialise random time assignments
        self.time_assignments = torch.rand(self.num_cells, requires_grad=False)

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        # if (self.nfe % 10) == 0:
        #     print(t)
        self.nfe += 1
        num_samples = h.shape[0]
        num_outputs = h.shape[1]
        h = h.view(num_samples, num_outputs//2, 2)
        u = h[:, :, 0].unsqueeze(-1)
        s = h[:, :, 1].unsqueeze(-1)

        f = self.f[:, :, self.t_index].unsqueeze(2)

        du = f - self.splicing_rate * u
        ds = self.splicing_rate * u - self.decay_rate * s

        h_t = torch.cat([du, ds], dim=1)

        if t > self.last_t:
            self.t_index += 1
        self.last_t = t

        return h_t

    def G(self, f):
        """
        Parameters:
            f: (I, T)
        """
        # nn linear
        return f.repeat(1, self.num_outputs//2//10, 1)  # (S, I, t)

    def predict_f(self, t_predict):
        # Sample from the latent distribution
        q_f = self.get_latents(t_predict.reshape(-1))
        f = q_f.sample([500])  # (S, I, t)
        # This is a hack to wrap the latent function with the nonlinearity. Note we use the same variance.
        f = torch.mean(self.G(f), dim=0)[0]
        return torch.distributions.multivariate_normal.MultivariateNormal(f, scale_tril=q_f.scale_tril)
