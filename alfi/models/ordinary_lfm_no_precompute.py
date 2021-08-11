from abc import ABC

import torch
from torchdiffeq import odeint

from . import OrdinaryLFM, TrainMode
from alfi.configuration import VariationalConfiguration


class OrdinaryLFMNoPrecompute(OrdinaryLFM, ABC):
    """
    Variational approximation for an LFM based on an ordinary differential equation (ODE).
    NoPrecompute means that this class does not precompute the GP at timepoints.
    This enables giving the state as input rather than time.
    Inheriting classes must override the `odefunc` function which encodes the ODE.
    """
    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        self.nfe = 0
        self.f = None

    @property
    def initial_state(self):
        return self._initial_state.unsqueeze(0)\
                                  .repeat(self.config.num_samples, 1)

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
            h0 = t[0]
            t_output = t[1]
            h_samples = self.odefunc(None, h0, return_mean=True).unsqueeze(1)
        else:
            t_output = t
            h0 = self.initial_state

            # Integrate forward from the initial positions h0.
            h_samples = odeint(self.odefunc, h0, t, method='rk4', options=dict(step_size=step_size)) # (T, S, num_outputs, 1)

        if return_samples:
            return h_samples

        dist = self.build_output_distribution(t_output, h_samples)
        self.f = None
        return dist

    def sample_latents(self, t, num_samples=1):
        q_f = self.gp_model(t)
        return self.nonlinearity(q_f.sample(torch.Size([num_samples])))

