from abc import abstractmethod

import torch
from torch.distributions import Distribution
from torchdiffeq import odeint

from . import OrdinaryLFM, TrainMode
from alfi.configuration import VariationalConfiguration


class OrdinaryStateLFM(OrdinaryLFM):
    """
    Variational approximation for an LFM based on an ordinary differential equation (ODE).
    The state element means that the GP model is given state as input rather than time.
    Inheriting classes must override the `odefunc` function which encodes the ODE.
    """
    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        self.nfe = 0
        self.f = None

    def forward(self, t, step_size=1e-1, return_samples=False, **kwargs):
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
        if self.train_mode == TrainMode.GRADIENT_MATCH:
            t_f = t[0]
            t_output = t_f
            h0 = t[1].unsqueeze(0).repeat(self.config.num_samples, 1, 1)
        else:
            t_f = torch.arange(t.min(), t.max()+step_size/3, step_size/3)
            t_output = t
            h0 = self.initial_state()
            # h0 = h0.unsqueeze(0).repeat(self.config.num_samples, 1, 1)
            # h0 = h0.view(h0.shape[0]//2, 2)

        if self.train_mode == TrainMode.GRADIENT_MATCH:
            h_samples = self.odefunc(t_f, h0)
            h_samples = h_samples.permute(2, 0, 1)
        else:
            # Integrate forward from the initial positions h0.
            h_samples = odeint(self.odefunc, h0, t, method='rk4', options=dict(step_size=step_size)) # (T, S, num_outputs, 1)

        # self.t_index = None
        # self.last_t = None
        if return_samples:
            return h_samples

        dist = self.build_output_distribution(t_output, h_samples)
        self.f = None
        return dist

    def build_output_distribution(self, t, h_samples) -> Distribution:
        h_mean = h_samples.mean(dim=1).squeeze(-1).transpose(0, 1)  # shape was (#outputs, #T, 1)
        h_var = h_samples.var(dim=1).squeeze(-1).transpose(0, 1) + 1e-7

        # TODO: make distribution something less constraining
        if self.config.latent_data_present:
            # todo: make this
            f = self.gp_model(t).rsample(torch.Size([self.config.num_samples])).permute(0, 2, 1)
            # f = self.nonlinearity(f)
            f_mean = f.mean(dim=0)
            f_var = f.var(dim=0) + 1e-7
            h_mean = torch.cat([h_mean, f_mean], dim=0)
            h_var = torch.cat([h_var, f_var], dim=0)

        h_covar = DiagLazyTensor(h_var)  # (num_tasks, t, t)
        batch_mvn = gpytorch.distributions.MultivariateNormal(h_mean, h_covar)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)

    @abstractmethod
    def odefunc(self, t, h):
        """
        Parameters:
            h: shape (num_samples, num_outputs, 1)
        """
        pass

    def sample_latents(self, t, num_samples=1):
        q_f = self.gp_model(t)
        return self.nonlinearity(q_f.sample(torch.Size([num_samples])))

