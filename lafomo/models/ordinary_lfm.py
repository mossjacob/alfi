from abc import abstractmethod

import torch
import gpytorch
from torchdiffeq import odeint

from .variational_lfm import VariationalLFM
from lafomo.configuration import VariationalConfiguration
from lafomo.utilities.torch import is_cuda

class OrdinaryLFM(VariationalLFM):
    """
    Variational approximation for an LFM based on an ordinary differential equation (ODE).
    Inheriting classes must override the `odefunc` function which encodes the ODE.
    """

    def __init__(self, gp_model, config: VariationalConfiguration, dataset, dtype=torch.float64):
        super().__init__(gp_model, config, dataset, dtype=dtype)
        self.nfe = 0
        self.f = None

    def initial_state(self):
        initial_state = torch.zeros(torch.Size([self.num_outputs, 1]), dtype=torch.float64)
        initial_state = initial_state.cuda() if is_cuda() else initial_state
        return initial_state.repeat(self.config.num_samples, 1, 1)  # Add batch dimension for sampling
        # if self.config.initial_conditions: TODO:
        #     h = self.initial_conditions.repeat(h.shape[0], 1, 1)

    def forward(self, t, step_size=1e-1, return_samples=False):
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

        # Get GP outputs
        t_f = torch.arange(t.min(), t.max()+step_size/3, step_size/3)
        # print('feeding gp with ', t_f.shape)
        q_f = self.gp_model(t_f)

        # Integrate forward from the initial positions h0.
        h0 = self.initial_state()
        h0 = h0.unsqueeze(0).repeat(self.config.num_samples, 1, 1)
        self.f = q_f.rsample(torch.Size([self.config.num_samples])).permute(0, 2, 1)  # (S, I, T)
        self.f = self.G(self.f)
        # print('de_model forward', self.f.shape)
        # print(self.f.shape)
        self.t_index = 0
        self.last_t = self.f.min()-1

        h_samples = odeint(self.odefunc, h0, t, method='rk4', options=dict(step_size=step_size)) # (T, S, num_outputs, 1)

        self.f = None
        # self.t_index = None
        # self.last_t = None
        if return_samples:
            return h_samples

        h_out = torch.mean(h_samples, dim=1).squeeze(-1).permute(1, 0) # shape was (#outputs, #T, 1) .permute(1, 0, 2)
        h_var = torch.var(h_samples, dim=1).squeeze(-1).permute(1, 0) + 1e-7

        h_out = self.decode(h_out)
        # print('h_out', h_out.shape, h_var.shape)
        # TODO: make distribution something less constraining
        # print(h_var.min(), h_var)
        h_covar = torch.diag_embed(h_var)

        batch_mvn = gpytorch.distributions.MultivariateNormal(h_out, h_covar)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)

    def decode(self, h_out):
        return h_out

    @abstractmethod
    def odefunc(self, t, h):
        """
        Parameters:
            h: shape (num_samples, num_outputs, 1)
        """
        pass

    def G(self, f):
        return f.repeat(1, self.num_outputs, 1)  # (S, I, t)
