import torch
from torch.nn import Parameter
from torch.distributions.utils import _standard_normal
from gpytorch.constraints import Positive, Interval
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.lazy import DiagLazyTensor

from alfi.configuration import VariationalConfiguration
from alfi.models import OrdinaryLFM, OrdinaryLFMNoPrecompute
from alfi.utilities.torch import softplus, inv_softplus


class LotkaVolterra(OrdinaryLFM):
    """Outputs are predator. Latents are prey"""

    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        self.positivity = Positive()
        self.decay_constraint = Interval(0., 1.5)
        self.raw_decay = Parameter(
            self.positivity.inverse_transform(torch.ones(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        self.raw_growth = Parameter(self.positivity.inverse_transform(
            0.5 * torch.ones(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        self.raw_initial = Parameter(self.decay_constraint.inverse_transform(
            0.3 + torch.zeros(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        # self.true_f = dataset.prey[::3].unsqueeze(0).repeat(self.config.num_samples, 1).unsqueeze(1)

    @property
    def decay_rate(self):
        return self.decay_constraint.transform(self.raw_decay)

    @decay_rate.setter
    def decay_rate(self, value):
        self.raw_decay = self.decay_constraint.inverse_transform(value)

    @property
    def growth_rate(self):
        return softplus(self.raw_growth)

    @growth_rate.setter
    def growth_rate(self, value):
        self.raw_growth = inv_softplus(value)

    @property
    def initial_state(self):
        return softplus(self.raw_initial)

    @initial_state.setter
    def initial_state(self, value):
        self.raw_initial = inv_softplus(value)

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        # print(t, self.t_index, self.f.shape)
        # f shape (num_samples, num_outputs, num_times)
        f = self.f[:, :, self.t_index].unsqueeze(2)
        if t > self.last_t:
            self.t_index += 1
        self.last_t = t

        dh = self.growth_rate * h * f - self.decay_rate * h
        return dh

    def mix(self, f):
        return softplus(f).repeat(1, self.num_outputs, 1)


class LotkaVolterraState(OrdinaryLFMNoPrecompute):

    def __init__(self, *args, nonzero_mask=None, initial_state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nonlinearity = softplus
        self.nonzero_mask = nonzero_mask
        # self.num_data = num_data
        if initial_state is None:
            self.initial_state = torch.tensor([0, 0], dtype=torch.float)
        else:
            self.initial_state = initial_state

    def odefunc(self, t, h, return_mean=False, **kwargs):
        """
        Parameters:
            h shape  (S, D)
        """
        self.nfe += 1
        # Sample from the GP
        # dh = self.gp_model(h).rsample(torch.Size([1]))[0]
        dh = self.gp_model(h)
        # reparam trick:
        mean = dh.mean
        if return_mean:
            return mean
        var = dh.variance
        z = _standard_normal(mean.shape, dtype=mean.dtype, device=mean.device)
        dh = mean + z * (var + 1e-7).sqrt()  # shape (S, D)
        return dh

    def build_output_distribution(self, t, h_samples) -> MultitaskMultivariateNormal:
        """
        Parameters:
            h_samples: shape (T, S, D)
        """
        h_mean = h_samples.mean(dim=1).transpose(0, 1)  # shape was (#outputs, #T, 1)
        h_var = h_samples.var(dim=1).transpose(0, 1) + 1e-7
        # h_mean shape: (D, T)
        self.current_trajectory = h_mean

        if self.config.latent_data_present:
            # todo: make this
            f = self.gp_model(self.timepoint_choices).rsample(torch.Size([self.config.num_samples])).permute(0, 2, 1)
            f = self.nonlinearity(f)
            f_mean = f.mean(dim=0)
            f_var = f.var(dim=0) + 1e-7
            filler = torch.zeros(f_mean.shape[0], h_mean.shape[1]-self.timepoint_choices.shape[0])
            f_mean = torch.cat([f_mean, filler], dim=1)
            f_var = torch.cat([f_var, filler], dim=1)
            h_mean = torch.cat([h_mean, f_mean], dim=0)
            h_var = torch.cat([h_var, f_var], dim=0)

        h_covar = DiagLazyTensor(h_var)
        batch_mvn = MultivariateNormal(h_mean, h_covar)
        return MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)

    def predict_f(self, t_predict, **kwargs):
        raise NotImplementedError('This model has no latent forces')
