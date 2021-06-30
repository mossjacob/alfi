import torch
from torch.nn import Parameter
from gpytorch.constraints import Positive

from alfi.configuration import VariationalConfiguration
from alfi.models import OrdinaryLFM, TrainMode


class TranscriptionLFM(OrdinaryLFM):
    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration,
                 initial_basal=None, initial_decay=None, initial_sensitivity=None, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        self.positivity = Positive()
        initial_basal = 0.1 if initial_basal is None else initial_basal
        initial_decay = 0.3 if initial_decay is None else initial_decay
        initial_sensitivity = 1 if initial_sensitivity is None else initial_sensitivity
        self.raw_decay = Parameter(self.positivity.inverse_transform(
            initial_decay + torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        self.raw_basal = Parameter(self.positivity.inverse_transform(
            initial_basal * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        self.raw_sensitivity = Parameter(self.positivity.inverse_transform(
            initial_sensitivity * torch.rand(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))

    @property
    def decay_rate(self):
        return self.positivity.transform(self.raw_decay)

    @decay_rate.setter
    def decay_rate(self, value):
        self.raw_decay = self.positivity.inverse_transform(value)

    @property
    def basal_rate(self):
        return self.positivity.transform(self.raw_basal)

    @basal_rate.setter
    def basal_rate(self, value):
        self.raw_basal = self.positivity.inverse_transform(value)

    @property
    def sensitivity(self):
        return self.positivity.transform(self.raw_sensitivity)

    @sensitivity.setter
    def sensitivity(self, value):
        self.raw_sensitivity = self.decay_constraint.inverse_transform(value)

    def initial_state(self):
        return self.basal_rate / self.decay_rate

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        self.nfe += 1

        f = self.f
        if not (self.train_mode == TrainMode.PRETRAIN):
            f = self.f[:, :, self.t_index].unsqueeze(2)
            if t > self.last_t:
                self.t_index += 1
            self.last_t = t

        dh = self.basal_rate + self.sensitivity * f - self.decay_rate * h
        return dh
