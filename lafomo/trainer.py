import torch
import numpy as np
import gpytorch
from torch.utils.data.dataloader import DataLoader

from lafomo.utilities.torch import is_cuda
from lafomo.datasets import LFMDataset
from lafomo.models import LFM


class TranscriptionalTrainer(VariationalTrainer):
    """
    TranscriptionalTrainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basalrates = list()
        self.decayrates = list()
        self.lengthscales = list()
        self.sensitivities = list()
        # self.mus = list()
        # self.cholS = list()

    def after_epoch(self):
        self.basalrates.append(self.lfm.basal_rate.detach().clone().numpy())
        self.decayrates.append(self.lfm.decay_rate.detach().clone().numpy())
        # self.sensitivities.append(self.lfm.sensitivity.detach().clone().numpy())
        self.lengthscales.append(self.lfm.gp_model.covar_module.lengthscale.detach().clone().numpy())
        # self.cholS.append(self.lfm.q_cholS.detach().clone())
        # self.mus.append(self.lfm.q_m.detach().clone())
        with torch.no_grad():
            # TODO can we replace these with parameter transforms like we did with lengthscale
            # self.lfm.sensitivity.clamp_(0, 20)
            self.lfm.basal_rate.clamp_(0, 20)
            self.lfm.decay_rate.clamp_(0, 20)
            self.extra_constraints()
            # self.model.inducing_inputs.clamp_(0, 1)
            # self.lfm.q_m[0, 0] = 0.

    def extra_constraints(self):
        pass
