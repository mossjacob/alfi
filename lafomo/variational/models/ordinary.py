from abc import abstractmethod

import torch
from torchdiffeq import odeint

from lafomo.datasets import LFMDataset
from lafomo.configuration import VariationalConfiguration
from lafomo.variational.models import VariationalLFM


class OrdinaryLFM(VariationalLFM):
    """
    Variational approximation for an LFM based on an ordinary differential equation (ODE).
    Inheriting classes must override the `odefunc` function which encodes the ODE. This odefunc
    may call `get_latents` to get the values of the latent function at arbitrary time `t`.
    """
    def __init__(self, options: VariationalConfiguration, kernel, t_inducing, dataset: LFMDataset, dtype=torch.float64):
        super().__init__(options, kernel, t_inducing, dataset, dtype)
        self.nfe = 0

    def initial_state(self, h):
        if self.options.initial_conditions:
            h = self.initial_conditions.repeat(h.shape[0], 1, 1)
        return h

    def forward(self, t, h, rtol=1e-4, atol=1e-6, compute_var=False, return_samples=False):
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

        if return_samples:
            return h_samples

        h_out = torch.mean(h_samples, dim=1).transpose(0, 1)
        h_std = torch.std(h_samples, dim=1).transpose(0, 1)

        if compute_var:
            return self.decode(h_out), h_std
        return self.decode(h_out)

    def decode(self, h_out):
        return h_out

    @abstractmethod
    def odefunc(self, t, h):
        """
        Parameters:
            h: shape (num_samples, num_outputs, 1)
        """
        pass
